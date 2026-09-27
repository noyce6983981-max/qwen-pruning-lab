"""Evaluate an unchanged local base or its bound LoRA with one precision policy.

Reuses the frozen QA/PPL routines. No merge, training, download, or final split.
"""
from __future__ import annotations

import copy
import gc
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import run_quality as quality


TARGETS = ("gate_proj", "up_proj", "down_proj")
PRECISION_POLICY = {
    "base_parameter_dtype": "torch.bfloat16",
    "adapter_parameter_dtype_when_present": "torch.float32",
    "autocast_device_type": "cuda", "autocast_dtype": "torch.bfloat16",
    "autocast_enabled_for_base_and_adapter": True,
    "autocast_scope": "all QA generation and PPL forwards; also saved checkpoint probe",
    "mode": "eval; all parameters frozen; no merge",
    "ppl_loss": "existing float32 cross entropy and float64 token-loss accumulation",
}


def source_identity(provenance):
    kind = provenance["checkpoint_kind"]
    if kind not in ("original", "pruned") or provenance.get("recovery_applied", False):
        raise ValueError("Expected an original or unrecovered pruned base")
    source = provenance["download_manifest"] if kind == "original" else provenance["source"]
    return kind, source["model_id"], source["revision"]


def baseline_provenance(base):
    kind, model_id, revision = source_identity(base)
    return {**copy.deepcopy(base), "base_checkpoint_kind": kind,
            "source_model_id": model_id, "source_model_revision": revision,
            "training_step": 0, "recovery_applied": False, "adapter_present": False}


def _check_lora_config(config):
    expected = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8,
                "lora_alpha": 16, "lora_dropout": 0.05, "bias": "none"}
    if any(config.get(key) != value for key, value in expected.items()) or config.get("init_lora_weights") is not True:
        raise ValueError("Adapter must use the fixed standard MLP LoRA configuration")
    if set(config.get("target_modules") or []) != set(TARGETS):
        raise ValueError("Unexpected adapter target modules")
    for key in ("modules_to_save", "rank_pattern", "alpha_pattern", "layers_to_transform",
                "layers_pattern", "target_parameters", "exclude_modules", "layer_replication",
                "trainable_token_indices", "loftq_config", "eva_config", "corda_config", "megatron_config"):
        if config.get(key):
            raise ValueError(f"Unsupported adapter configuration: {key}")
    if any(config.get(key, False) for key in ("use_dora", "use_rslora", "use_qalora", "fan_in_fan_out", "lora_bias")):
        raise ValueError("Only standard unmerged LoRA with no extra bias is supported")


def _expected_adapter_shapes(config):
    expected = {}
    for index in range(config["num_hidden_layers"]):
        for target in TARGETS:
            inputs, outputs = ((config["intermediate_size"], config["hidden_size"])
                               if target == "down_proj" else
                               (config["hidden_size"], config["intermediate_size"]))
            prefix = f"base_model.model.model.layers.{index}.mlp.{target}"
            expected[f"{prefix}.lora_A.weight"] = [8, inputs]
            expected[f"{prefix}.lora_B.weight"] = [outputs, 8]
    return expected


def validate_training_run(directory, manifest_path, manifest):
    """Only select checkpoints from the fully completed, fixed-budget run."""
    root = Path(manifest["formal_training_run"])
    if not root.is_absolute() or root.resolve() != directory.resolve().parent:
        raise ValueError("Formal training run must be the adapter directory's actual parent")
    run_path = root / "run.json"
    run = quality.read_json(run_path)
    if (run.get("status") != "completed" or run.get("mode") != "train"
            or run.get("base_parameters_unchanged") is not True
            or len(run.get("steps", [])) != 64 or run.get("sample_count") != 512
            or run.get("supervised_token_count") != 6762):
        raise ValueError("Checkpoint requires a completed formal run: 64 steps, 512 rows, 6762 targets, unchanged base")
    for key in ("base_provenance", "settings", "data", "training_policy_sha256", "source_model_revision"):
        if not manifest.get(key) or run.get(key) != manifest[key]:
            raise ValueError(f"Formal run and checkpoint differ: {key}")
    if run.get("source_sha256") != manifest.get("training_source_sha256"):
        raise ValueError("Formal run and checkpoint training code hashes differ")
    checkpoints = run.get("checkpoints", [])
    if [row.get("step") for row in checkpoints] != [16, 32, 64]:
        raise ValueError("Formal run checkpoint schedule must be 16/32/64")
    matching = [row for row in checkpoints if row["step"] == manifest["training_step"]]
    if len(matching) != 1:
        raise ValueError("Checkpoint is not part of the completed training schedule")
    checkpoint = matching[0]
    if (Path(checkpoint["directory"]).resolve() != directory.resolve()
            or checkpoint["manifest_sha256"] != quality.sha256_file(manifest_path)):
        raise ValueError("Checkpoint manifest hash/path differs from the completed run")
    seen = [identifier for row in run["steps"] for identifier in row["sample_ids"]]
    if (len(seen) != 512 or len(set(seen)) != 512
            or sum(row["supervised_token_count"] for row in run["steps"]) != 6762):
        raise ValueError("Formal run actual sample/token budget differs")
    previous = run["steps"][:manifest["training_step"]]
    if (manifest.get("samples_seen") != sum(len(row["sample_ids"]) for row in previous)
            or manifest.get("supervised_tokens_seen") != sum(row["supervised_token_count"] for row in previous)
            or checkpoint.get("samples_seen") != manifest["samples_seen"]
            or checkpoint.get("supervised_tokens_seen") != manifest["supervised_tokens_seen"]
            or manifest.get("checkpoint_rng_preserved") is not True or checkpoint.get("rng_preserved") is not True):
        raise ValueError("Checkpoint partial training budget or RNG record differs")
    return {"path": str(run_path.resolve()), "sha256": quality.sha256_file(run_path),
            "status": "completed", "optimizer_steps": 64, "sample_count": 512,
            "supervised_token_count": 6762, "base_parameters_unchanged": True}


def validate_adapter(adapter_dir, base, revision, base_config, evaluation_data=None):
    """Bind actual local adapter files to an independently hashed base report."""
    import torch
    from safetensors import safe_open

    directory = Path(adapter_dir)
    choices = [directory / name for name in ("checkpoint_manifest.json", "recovery_manifest.json")
               if (directory / name).is_file()]
    if len(choices) != 1:
        raise ValueError("Adapter directory requires exactly one checkpoint/recovery manifest")
    manifest_path = choices[0]
    manifest = quality.read_json(manifest_path)
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "completed"
            or manifest.get("checkpoint_kind") != "recovered_lora"):
        raise ValueError("Adapter manifest must identify a completed recovered_lora checkpoint")
    step = manifest.get("training_step")
    if type(step) is not int or step <= 0:
        raise ValueError("Recovered adapter requires a positive training_step")
    if manifest.get("source_model_revision") != revision:
        raise ValueError("Adapter source revision differs")
    identity = source_identity(base)
    bound = manifest["base_provenance"]
    if identity != source_identity(bound) or identity[2] != revision:
        raise ValueError("Adapter base identity differs")
    hashes = base["files_sha256"]
    if (not hashes or "config.json" not in hashes or not any(name.endswith(".safetensors") for name in hashes)
            or bound.get("files_sha256") != hashes or bound.get("manifest_sha256") != base["manifest_sha256"]):
        raise ValueError("Adapter is bound to different base files or base manifest")
    settings = manifest["settings"]
    if (settings.get("rank") != 8 or settings.get("alpha") != 16 or settings.get("dropout") != 0.05
            or set(settings.get("target_modules") or []) != set(TARGETS)):
        raise ValueError("Adapter training settings disagree with the fixed LoRA configuration")
    data = manifest["data"]
    if not data.get("manifest_sha256") or not {"recovery.jsonl", "groups.json"}.issubset(data.get("files_sha256", {})):
        raise ValueError("Missing recovery data hash chain")
    if evaluation_data is not None:
        if (data["manifest_sha256"] != evaluation_data["manifest_sha256"]
                or any(evaluation_data["files_sha256"].get(name) != digest
                       for name, digest in data["files_sha256"].items())):
            raise ValueError("Recovery and evaluation data provenance differ")
    run_provenance = validate_training_run(directory, manifest_path, manifest)
    required = {"adapter_config.json", "adapter_model.safetensors"}
    expected_files = manifest["adapter_files_sha256"]
    probe = manifest.get("probe")
    if probe is not None:
        if (probe.get("file") != "probe.safetensors" or not quality.token_ids_valid(probe.get("input_ids"))
                or any(token >= base_config["vocab_size"] for token in probe["input_ids"])):
            raise ValueError("Invalid checkpoint probe specification")
        required.add("probe.safetensors")
    if not isinstance(expected_files, dict) or set(expected_files) != required:
        raise ValueError("Adapter file inventory must match configuration, weights, and optional probe")
    actual_files = quality.verify_files(directory, expected_files, required=required)
    config = quality.read_json(directory / "adapter_config.json")
    _check_lora_config(config)
    expected_shapes = _expected_adapter_shapes(base_config)
    with safe_open(directory / "adapter_model.safetensors", framework="pt", device="cpu") as tensors:
        if set(tensors.keys()) != set(expected_shapes):
            raise ValueError("Adapter tensor inventory differs from the exact MLP A/B inventory")
        for name, shape in expected_shapes.items():
            tensor = tensors.get_tensor(name)
            if list(tensor.shape) != shape or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Invalid adapter shape, dtype, or nonfinite tensor: {name}")
    if probe is not None:
        with safe_open(directory / "probe.safetensors", framework="pt", device="cpu") as tensors:
            if set(tensors.keys()) != {"logits"}:
                raise ValueError("Probe must contain only logits")
            tensor = tensors.get_tensor("logits")
            if (list(tensor.shape) != [1, 1, base_config["vocab_size"]]
                    or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all())):
                raise ValueError("Invalid checkpoint probe tensor")
    provenance = {
        "checkpoint_kind": "recovered_lora", "base_checkpoint_kind": identity[0],
        "source_model_id": identity[1], "source_model_revision": revision,
        "recovery_applied": True, "adapter_present": True, "training_step": step,
        "base_provenance": copy.deepcopy(base), "adapter_directory": str(directory.resolve()),
        "manifest_path": str(manifest_path.resolve()), "manifest_sha256": quality.sha256_file(manifest_path),
        "adapter_files_sha256": actual_files, "recovery_manifest": manifest,
        "formal_training_run": run_provenance,
    }
    return manifest, provenance


def inspect_evaluation_model(model, base_report, adapter_manifest, device):
    """Report real resident storage; validate base and A/B separately after load."""
    import torch

    parameters = list(model.named_parameters())
    adapter = [(name, p) for name, p in parameters if ".lora_A." in name or ".lora_B." in name]
    base = [(name, p) for name, p in parameters if ".lora_A." not in name and ".lora_B." not in name]
    if any(p.requires_grad or p.grad is not None or str(p.device) != str(device) for _, p in parameters):
        raise ValueError("Evaluation requires all parameters frozen on the selected device")
    if model.training or any(module.training for module in model.modules()):
        raise ValueError("Evaluation requires eval mode for every module")
    if any(p.dtype != torch.bfloat16 for _, p in base) or any(p.dtype != torch.float32 for _, p in adapter):
        raise ValueError("Expected BF16 base and FP32 adapter parameters")
    base_count = sum(p.numel() for _, p in base)
    base_bytes = sum(p.numel() * p.element_size() for _, p in base)
    if base_count != base_report["unique_parameter_count"] or base_bytes != base_report["unique_parameter_storage_bytes"]:
        raise ValueError("Base parameter count/storage changed after adapter loading")
    module_count = 0
    if adapter_manifest is None:
        if adapter:
            raise ValueError("Unbound adapter parameters found in baseline")
    else:
        expected_shapes = _expected_adapter_shapes(base_report)
        expected_loaded = {name.replace(".weight", ".default.weight"): shape for name, shape in expected_shapes.items()}
        if {name for name, _ in adapter} != set(expected_loaded):
            raise ValueError("Loaded adapter A/B tensor inventory differs")
        for name, p in adapter:
            if list(p.shape) != expected_loaded[name] or not bool(torch.isfinite(p).all()):
                raise ValueError("Loaded adapter has wrong shape or nonfinite values")
        expected_modules = {name.rsplit(".lora_", 1)[0] for name in expected_shapes}
        actual_modules = {name: module for name, module in model.named_modules() if hasattr(module, "lora_A")}
        if set(actual_modules) != expected_modules:
            raise ValueError("Loaded adapter target module inventory differs")
        for module in actual_modules.values():
            if (module.merged or module.disable_adapters or module.active_adapters != ["default"]
                    or module.scaling["default"] != 2 or module.lora_dropout["default"].p != 0.05):
                raise ValueError("Loaded adapter is merged, disabled, or uses unexpected scaling/dropout")
        module_count = len(actual_modules)
    adapter_count = sum(p.numel() for _, p in adapter)
    adapter_bytes = sum(p.numel() * p.element_size() for _, p in adapter)
    return {**copy.deepcopy(base_report), "class": type(model).__name__,
            "unique_parameter_count": base_count + adapter_count,
            "unique_parameter_storage_bytes": base_bytes + adapter_bytes,
            "base_parameter_count": base_count, "base_parameter_storage_bytes": base_bytes,
            "adapter_parameter_count": adapter_count, "adapter_parameter_storage_bytes": adapter_bytes,
            "parameter_dtypes": sorted({str(p.dtype) for _, p in parameters}),
            "all_parameters_frozen": True, "adapter_module_count": module_count,
            "adapter_tensor_count": len(adapter), "adapter_merged": False}


def check_saved_probe(model, adapter_dir, manifest, device):
    import torch
    from safetensors.torch import load_file

    probe = manifest.get("probe")
    if probe is None:
        return {"status": "not_provided", "scope": "No saved training probe in checkpoint manifest"}
    expected = load_file(str(Path(adapter_dir) / probe["file"]), device="cpu")["logits"]
    ids = torch.tensor([probe["input_ids"]], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                enabled=device.type == "cuda"):
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
        actual = output.logits.detach().float().cpu()
    error = float((actual - expected).abs().max())
    if not torch.equal(actual, expected):
        raise ValueError(f"Checkpoint reload probe differs: max_abs_error={error}")
    return {"status": "passed", "probe_exact": True, "probe_max_abs_error": error,
            "input_ids": probe["input_ids"], "scope": "Last-position full-vocabulary eval/autocast logits"}


def build_parser():
    parser = quality.build_parser()
    parser.description = __doc__
    parser.add_argument("--adapter-dir", type=Path)
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
              "scope": "Frozen local CMRC subset; unmerged LoRA/base under the same autocast precision policy.",
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "packages": {**quality.package_versions(), "peft": importlib.metadata.version("peft")}}}
    started = time.perf_counter()
    model = torch = device = None
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        from cmrc_official import score_examples, scoring_provenance
        from eval_ppl import evaluate_ppl

        report["stage"] = "data_validation"
        manifest, examples, documents, data_report = quality.load_frozen_data(args.data_dir, args.split)
        report["data"], report["data_manifest"] = data_report, manifest
        report["protocol"] = {**copy.deepcopy(manifest["protocol"]), "inference_precision": PRECISION_POLICY}
        report["helper_sha256"] = {name: quality.sha256_file(Path(__file__).with_name(name))
                                   for name in ("run_quality.py", "run_recovery_quality.py", "cmrc_official.py", "eval_ppl.py")}
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
        base_provenance, pruning_manifest = quality.model_provenance(args.model_dir, args.revision)
        base_config = quality.read_json(args.model_dir / "config.json")
        if (base_config["model_type"] != "qwen3" or base_config["num_hidden_layers"] != 28
                or base_config["hidden_size"] != 2048):
            raise ValueError("Real evaluation supports the local 28-layer Qwen3-1.7B bases")
        adapter_manifest = None
        if args.adapter_dir:
            adapter_manifest, report["provenance"] = validate_adapter(
                args.adapter_dir, base_provenance, args.revision, base_config, data_report)
        else:
            report["provenance"] = baseline_provenance(base_provenance)
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
        if args.adapter_dir:
            model = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=False,
                                              autocast_adapter_dtype=True, local_files_only=True)
        model.requires_grad_(False)
        model.eval()
        model.gradient_checkpointing_disable()
        report["model"] = inspect_evaluation_model(model, base_report, adapter_manifest, device)
        report["loading"] = {"local_files_only": True, "trust_remote_code": False, "use_safetensors": True,
                             "low_cpu_mem_usage": True, "device_map": {"": str(device)}, "adapter_merged": False}
        report["load_peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["load_peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["host_memory_after_load"] = quality.host_memory()
        for key in ("eos_token_id", "bos_token_id"):
            if getattr(model.generation_config, key) != getattr(generation_config, key):
                raise ValueError(f"Checkpoint generation {key} differs from frozen original")
        if adapter_manifest is not None:
            report["stage"] = "saved_probe_validation"
            report["reload_validation"] = check_saved_probe(model, args.adapter_dir, adapter_manifest, device)
        else:
            report["reload_validation"] = {"status": "not_applicable", "reason": "No adapter"}
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
