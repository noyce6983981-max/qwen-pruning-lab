"""Export a selected step64 LoRA as standalone BF16, audit weights and reload."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_quality as quality
from recovery_data import load_recovery_data
from run_recovery_quality import validate_adapter, check_saved_probe
from train_recovery import write_json


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.inference_mode()
def collect_probes(model, probe_ids, device):
    model.eval()
    result = []
    for ids in probe_ids:
        inputs = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                           use_cache=False, logits_to_keep=1).logits.float().cpu()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Nonfinite export probe")
        result.append(logits)
    return torch.cat(result, dim=0)


def probe_difference(before, after):
    difference = (before - after).abs()
    return {"max_abs_logit_difference": float(difference.max()),
            "mean_abs_logit_difference": float(difference.mean()),
            "argmax_agreement_count": int((before.argmax(-1) == after.argmax(-1)).sum()),
            "probe_count": before.shape[0], "exact": bool(torch.equal(before, after)),
            "scope": "Eight recovery examples, last-position vocabulary logits; not all inputs or task quality."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "adapter-dir", "data-dir", "selection", "output-dir", "report-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("original", "calibrated_5888"))
    parser.add_argument("--revision", required=True)
    args = parser.parse_args(argv)
    for destination in (args.output_dir, args.report_dir):
        if any(destination.resolve().is_relative_to(source.resolve()) for source in (args.model_dir, args.adapter_dir)):
            raise ValueError("Never write inside source weights")
        if destination.exists():
            raise FileExistsError(destination)
    selection = quality.read_json(args.selection)
    chosen = selection["selected"][args.arm]["checkpoint"]
    selected_report = quality.read_json(Path(chosen["report_path"]))
    if (chosen["training_step"] != 64 or not chosen["numeric_quality_gate_passed"]
            or quality.sha256_file(Path(chosen["report_path"])) != chosen["report_sha256"]
            or Path(selected_report["adapter_directory"]).resolve() != args.adapter_dir.resolve()
            or Path(selected_report["model_directory"]).resolve() != args.model_dir.resolve()):
        raise ValueError("Export must match the hash-bound selected step64 report")
    args.report_dir.mkdir(parents=True)
    args.output_dir.mkdir(parents=True)
    report = {"schema_version": 1, "status": "starting", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoint_kind": "merged_lora", "source_model_revision": args.revision,
              "selection_sha256": quality.sha256_file(args.selection), "parameter_dtype": "torch.bfloat16",
              "training_step": 64, "source_sha256": {name: quality.sha256_file(Path(__file__).with_name(name))
                  for name in ("export_lora.py", "merged_checkpoint.py", "run_recovery_quality.py")},
              "merge_method": "PEFT 0.17.1 safe_merge: FP32 delta = 2*B@A, cast delta to BF16 then add to BF16 base; eval dropout disabled.",
              "environment": quality.package_versions()}
    model = None
    started = time.perf_counter()
    try:
        base, pruning = quality.model_provenance(args.model_dir, args.revision)
        manifest, source = validate_adapter(args.adapter_dir, base, args.revision,
                                             quality.read_json(args.model_dir / "config.json"))
        report.update(source_adapter=source, source_pruning_manifest=pruning,
                      base_checkpoint_kind=base["checkpoint_kind"])
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
        _, rows, data = load_recovery_data(args.data_dir, tokenizer)
        if data != manifest["data"]:
            raise ValueError("Probe recovery data differs from training")
        # Fixed before reading any final sample. No final data is used for merge.
        probes = rows[:4] + sorted(rows[4:], key=lambda row: -row["sequence_length"])[:4]
        report["probes"] = [{"id": row["id"], "input_ids": row["input_ids"]} for row in probes]
        probe_ids = [row["input_ids"] for row in probes]
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 required")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats(device)
        def load():
            return AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16,
                attn_implementation="sdpa", device_map={"": str(device)}, local_files_only=True,
                trust_remote_code=False, use_safetensors=True, low_cpu_mem_usage=True)
        model = load()
        base_inventory = quality.inspect_model(model, pruning, str(device))
        before_hashes = {name: tensor_hash(value) for name, value in model.named_parameters()}
        model = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=False, autocast_adapter_dtype=True)
        model.requires_grad_(False)
        model.eval()
        report["adapter_reload_validation"] = check_saved_probe(model, args.adapter_dir, manifest, device)
        before = collect_probes(model, probe_ids, device)
        expected_hashes = dict(before_hashes)
        merged_names = []
        # Independent per-layer prediction of PEFT's actual BF16 merge arithmetic.
        with torch.no_grad():
            for name, module in model.get_base_model().named_modules():
                if not hasattr(module, "lora_A"):
                    continue
                if module.active_adapters != ["default"] or module.scaling["default"] != 2:
                    raise ValueError("Unexpected active adapter")
                value = module.get_base_layer().weight.detach().clone()
                value += module.get_delta_weight("default").to(value.dtype)
                key = name + ".weight"
                expected_hashes[key] = tensor_hash(value)
                merged_names.append(key)
                del value
        if len(merged_names) != 84:
            raise ValueError("Expected exactly 84 merged MLP weights")
        module = None
        model = model.merge_and_unload(safe_merge=True)
        model.requires_grad_(False)
        model.eval()
        actual = {name: tensor_hash(value) for name, value in model.named_parameters()}
        if actual != expected_hashes:
            raise ValueError("Merged weights differ from expected BF16 arithmetic or unchanged non-MLP weights")
        report["weight_audit"] = {"all_expected_hashes_match": True, "parameter_tensor_count": len(actual),
            "merged_mlp_tensor_count": len(merged_names), "non_mlp_unchanged": True,
            "expected_sha256": expected_hashes, "actual_sha256": actual}
        after = collect_probes(model, probe_ids, device)
        report["merge_difference"] = probe_difference(before, after)
        inventory = quality.inspect_model(model, pruning, str(device))
        if inventory["unique_parameter_count"] != base_inventory["unique_parameter_count"]:
            raise ValueError("Merged export still contains extra parameters")
        report["parameter_count"] = inventory["unique_parameter_count"]
        report["intermediate_size"] = inventory["intermediate_size"]
        model.config.use_cache = True
        model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(args.output_dir)
        save_file({"before_merge": before, "after_merge": after}, args.output_dir / "merge_probes.safetensors")
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(args.output_dir, dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": str(device)}, local_files_only=True,
            trust_remote_code=False, use_safetensors=True, low_cpu_mem_usage=True)
        model.requires_grad_(False)
        reloaded = collect_probes(model, probe_ids, device)
        if not torch.equal(reloaded, after):
            raise ValueError("Saved merged export differs after reload")
        reloaded_hashes = {name: tensor_hash(value) for name, value in model.named_parameters()}
        if reloaded_hashes != actual:
            raise ValueError("Saved merged parameter tensors differ after reload")
        other_tokenizer = AutoTokenizer.from_pretrained(args.output_dir, local_files_only=True, trust_remote_code=False)
        report["tokenizer_validation"] = quality.check_tokenizers(tokenizer, other_tokenizer)
        report["reload_validation"] = {"status": "passed", "probe_exact": True,
            "probe_count": len(probes), "parameter_hashes_exact": True, "probe_max_abs_error": 0.0}
        report["files_sha256"] = {path.name: quality.sha256_file(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()}
        report["status"] = "completed"
        write_json(args.output_dir / "merge_manifest.json", report)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.report_dir / "export.json", report)
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(json.dumps({"output": str(args.output_dir.resolve()), "parameters": report["parameter_count"],
                      "merge_difference": report["merge_difference"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
