"""Evaluate frozen standalone release checkpoints, including the untouched final split.

The release policy binds four named artifacts before evaluation. No training,
checkpoint reselection, altered references, or post-hoc threshold changes.
"""
from __future__ import annotations
import argparse
import copy
import gc
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import run_quality as quality
from merged_checkpoint import merged_model_provenance
from run_recovery_quality import baseline_provenance

MODEL_IDS = ("original_step0", "calibrated_5888_step0", "original_step64_merged", "calibrated_5888_step64_merged")
MODEL_IDENTITIES = {
    "original_step0": ("original", "original", 0),
    "calibrated_5888_step0": ("pruned", "pruned", 0),
    "original_step64_merged": ("merged_lora", "original", 64),
    "calibrated_5888_step64_merged": ("merged_lora", "pruned", 64),
}
PRECISION_POLICY = {"parameter_dtype": "torch.bfloat16", "autocast_device_type": "cuda",
    "autocast_dtype": "torch.bfloat16", "autocast_enabled": True,
    "mode": "eval; all parameters frozen; standalone checkpoint without adapter branch",
    "ppl_loss": "existing float32 cross entropy and float64 token-loss accumulation"}


def validate_release_policy(path):
    policy = quality.read_json(path)
    if (policy.get("schema_version") != 1 or policy.get("status") != "frozen"
            or set(policy.get("models", {})) != set(MODEL_IDS)
            or policy.get("final_selection_allowed") is not False
            or policy.get("precision_policy") != PRECISION_POLICY):
        raise ValueError("Require the fixed four-artifact release policy")
    for label, expected in MODEL_IDENTITIES.items():
        if policy["models"][label].get("checkpoint_kind") != expected[0]:
            raise ValueError("Release label has the wrong checkpoint kind")
    expected_code = policy.get("evaluation_source_sha256", {})
    for name in ("run_release_quality.py", "merged_checkpoint.py", "run_quality.py", "run_recovery_quality.py", "cmrc_official.py", "eval_ppl.py"):
        if expected_code.get(name) != quality.sha256_file(Path(__file__).with_name(name)):
            raise ValueError("Evaluation source changed after release policy was frozen")
    return policy


def release_provenance(directory, revision, policy_path, model_id):
    policy = validate_release_policy(policy_path)
    expected = policy["models"][model_id]
    if revision != policy["revision"] or Path(expected["directory"]).resolve() != directory.resolve():
        raise ValueError("Unexpected model directory or revision")
    if model_id.endswith("_merged"):
        provenance, pruning = merged_model_provenance(directory, revision)
    else:
        provenance, pruning = quality.model_provenance(directory, revision)
        provenance = baseline_provenance(provenance)
    actual_identity = (provenance.get("checkpoint_kind"), provenance.get("base_checkpoint_kind"), provenance.get("training_step"))
    if actual_identity != MODEL_IDENTITIES[model_id]:
        raise ValueError("Actual model does not match its release label/base kind/training step")
    if (provenance["checkpoint_kind"] != expected["checkpoint_kind"]
            or provenance["files_sha256"] != expected["files_sha256"]
            or provenance["manifest_sha256"] != expected["manifest_sha256"]):
        raise ValueError("Release checkpoint identity or files changed")
    return provenance, pruning


def load_release_data(directory, split, policy_path):
    policy = validate_release_policy(policy_path)
    if split not in ("validation", "final"):
        raise ValueError("Release evaluation supports validation or final only")
    manifest_path = directory / "manifest.json"
    if quality.sha256_file(manifest_path) != policy["data_manifest_sha256"]:
        raise ValueError("Release dataset manifest changed")
    manifest = quality.read_json(manifest_path)
    qa_file, lm_file = f"{split}_qa.jsonl", f"{split}_lm.jsonl"
    hashes = quality.verify_files(directory, manifest["files_sha256"], required=(qa_file, lm_file))
    examples = quality.read_jsonl(directory / qa_file)
    documents = quality.read_jsonl(directory / lm_file)
    quality.validate_examples(examples, documents, manifest["protocol"])
    counts = (len(examples), sum(len(row["token_ids"])-1 for row in documents))
    if counts != ((100,8192) if split == "validation" else (200,16384)):
        raise ValueError("Frozen release question/LM token budget differs")
    if manifest["protocol"]["acceptance"] != policy["acceptance"]:
        raise ValueError("Quality thresholds changed after release freeze")
    return manifest, examples, documents, {"directory": str(directory.resolve()),
        "manifest_sha256": quality.sha256_file(manifest_path), "files_sha256": hashes,
        "qa_file": qa_file, "lm_file": lm_file, "qa_count": len(examples),
        "lm_document_count": len(documents), "lm_input_token_count": sum(len(item["token_ids"]) for item in documents)}


def check_export_probe(model, directory, provenance, device):
    if provenance["checkpoint_kind"] != "merged_lora":
        return {"status": "not_applicable", "reason": "Unmerged base"}
    import torch
    from safetensors.torch import load_file
    manifest = provenance["merge_manifest"]
    expected = load_file(str(directory / "merge_probes.safetensors"), device="cpu")["after_merge"]
    for index, probe in enumerate(manifest["probes"]):
        ids = torch.tensor([probe["input_ids"]], device=device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            actual = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1).logits.float().cpu()
        if not torch.equal(actual, expected[index:index+1]):
            raise ValueError("Standalone export probe differs from saved merged logits")
    return {"status": "passed", "probe_exact": True, "probe_count": len(manifest["probes"]), "probe_max_abs_error": 0.0}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "data-dir", "output-dir", "policy"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--split", choices=("validation", "final"), required=True)
    parser.add_argument("--model-id", choices=MODEL_IDS, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.set_defaults(adapter_dir=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    for source in (args.model_dir, args.adapter_dir):
        if source is not None and args.output_dir.resolve().is_relative_to(source.resolve()):
            raise ValueError("Evaluation output must be outside base and adapter directories")
    quality.create_output(args.output_dir)
    report_path = args.output_dir / "quality.json"
    report = {"status": "starting", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "split": args.split, "source_model_revision": args.revision,
              "model_directory": str(args.model_dir.resolve()),
              "adapter_directory": str(args.adapter_dir.resolve()) if args.adapter_dir else None,
              "runner_sha256": quality.sha256_file(Path(__file__)),
              "scope": "Frozen release comparison; standalone BF16 checkpoints; validation and one fixed final assessment.",
              "release_policy_sha256": quality.sha256_file(args.policy), "model_id": args.model_id,
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "packages": {**quality.package_versions(), "peft": importlib.metadata.version("peft")}}}
    started = time.perf_counter()
    model = torch = device = None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        from cmrc_official import score_examples, scoring_provenance
        from eval_ppl import evaluate_ppl

        report["stage"] = "data_validation"
        manifest, examples, documents, data_report = load_release_data(args.data_dir, args.split, args.policy)
        report["data"], report["data_manifest"] = data_report, manifest
        report["protocol"] = {**copy.deepcopy(manifest["protocol"]), "inference_precision": PRECISION_POLICY}
        report["helper_sha256"] = {name: quality.sha256_file(Path(__file__).with_name(name))
                                   for name in ("run_quality.py", "run_recovery_quality.py", "run_release_quality.py", "merged_checkpoint.py", "cmrc_official.py", "eval_ppl.py")}
        report["scoring_provenance"] = scoring_provenance()
        canonical_dir = Path(manifest["tokenizer"]["directory"])
        report["canonical_tokenizer_files_sha256"] = quality.verify_files(
            canonical_dir, manifest["tokenizer"]["files_sha256"], required=("tokenizer.json", "tokenizer_config.json"))
        canonical = AutoTokenizer.from_pretrained(canonical_dir, local_files_only=True, trust_remote_code=False)
        candidate_tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
        report["tokenizer_compatibility"] = quality.check_tokenizers(canonical, candidate_tokenizer)
        del candidate_tokenizer
        vocab_ids = set(canonical.get_vocab().values())
        for item in examples + documents:
            if any(token not in vocab_ids for token in item.get("prompt_token_ids", item.get("token_ids"))):
                raise ValueError(f"{item['id']}: token ID outside frozen vocabulary")
        report["stage"] = "checkpoint_validation"
        base_provenance, pruning_manifest = release_provenance(args.model_dir, args.revision, args.policy, args.model_id)
        base_config = quality.read_json(args.model_dir / "config.json")
        if (base_config["model_type"] != "qwen3" or base_config["num_hidden_layers"] != 28
                or base_config["hidden_size"] != 2048):
            raise ValueError("Real evaluation supports the local 28-layer Qwen3-1.7B bases")
        report["provenance"] = base_provenance
        canonical_generation = quality.read_json(canonical_dir / "generation_config.json")
        generation_config = GenerationConfig(
            max_new_tokens=manifest["protocol"]["qa"]["max_new_tokens"], do_sample=False, use_cache=True,
            pad_token_id=canonical.pad_token_id, eos_token_id=canonical_generation["eos_token_id"],
            bos_token_id=canonical_generation.get("bos_token_id"))
        report["generation_config"] = generation_config.to_dict()
        report["decode"] = {"use_model_defaults": False, "batch_size": 1, "skip_special_tokens": True,
                            "clean_up_tokenization_spaces": False, "answer_extraction": "strip_only",
                            "prompt_truncation": False, "seed": 42}
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 is required; no CPU fallback for real evaluation")
        torch.cuda.set_device(args.device)
        device = torch.device(f"cuda:{args.device}")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        properties = torch.cuda.get_device_properties(device)
        report["environment"].update({
            "torch_cuda_build": torch.version.cuda, "cuda_device": str(device), "gpu_name": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
            "selected_dtype": "torch.bfloat16", "attention_implementation": "sdpa",
            "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32, "precision_policy": PRECISION_POLICY})
        report["stage"] = "model_loading"
        report["host_memory_before_load"] = quality.host_memory()
        torch.cuda.reset_peak_memory_stats(device)
        print(f"Loading one base on {device}; adapter={bool(args.adapter_dir)}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", use_safetensors=True,
            low_cpu_mem_usage=True, device_map={"": str(device)}, local_files_only=True, trust_remote_code=False)
        base_report = quality.inspect_model(model, pruning_manifest, str(device))
        model.requires_grad_(False)
        model.eval()
        model.gradient_checkpointing_disable()
        report["model"] = {**base_report, "base_parameter_count": base_report["unique_parameter_count"],
            "adapter_parameter_count": 0, "adapter_merged": base_provenance["checkpoint_kind"] == "merged_lora", "all_parameters_frozen": True}
        report["loading"] = {"local_files_only": True, "trust_remote_code": False, "use_safetensors": True,
                             "low_cpu_mem_usage": True, "device_map": {"": str(device)}, "adapter_merged": report["model"]["adapter_merged"]}
        report["load_peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["load_peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["host_memory_after_load"] = quality.host_memory()
        for key in ("eos_token_id", "bos_token_id"):
            if getattr(model.generation_config, key) != getattr(generation_config, key):
                raise ValueError(f"Checkpoint generation {key} differs from frozen original")
        report["stage"] = "export_probe_validation"
        report["reload_validation"] = check_export_probe(model, args.model_dir, base_provenance, device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        report["stage"] = "qa_generation"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            report["qa"] = quality.run_qa(model, canonical, examples, generation_config, device,
                                           args.output_dir / "qa_predictions.jsonl")
        report["qa_peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["qa_peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["stage"] = "qa_scoring"
        report["qa_metrics"] = score_examples(examples, {row["id"]: row["answer"] for row in report["qa"]})
        report["qa_output_diagnostics"] = {
            "empty_answer_count": sum(not row["answer"] for row in report["qa"]),
            "max_new_tokens_stop_count": sum(row["stop_reason"] == "max_new_tokens" for row in report["qa"])}
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        report["stage"] = "language_model_evaluation"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            report["lm"] = evaluate_ppl(model, documents, device, window=512, stride=256)
        report["lm_peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["lm_peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["status"] = report["stage"] = "completed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        partial = args.output_dir / "qa_predictions.jsonl"
        if partial.is_file():
            report["qa"] = quality.read_jsonl(partial) if partial.stat().st_size else []
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = time.perf_counter() - started
        report["host_memory_at_exit"] = quality.host_memory()
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Report: {report_path}", flush=True)
        model = None
        if torch is not None and device is not None:
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
