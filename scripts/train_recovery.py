"""One fixed 512-example LoRA recovery run, or a separate longest-row memory check.

Only recovery data is read. Checkpoints are saved at policy-defined steps and
evaluated later in separate processes. No trial adapter is accepted or resumed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

from recovery_data import load_recovery_data
from recovery_training import batch_groups, train_accumulated_group
from run_quality import (check_tokenizers, inspect_model, model_provenance,
                         package_versions, read_json, sha256_file, verify_files)
from trial_lora import (SETTINGS, attach_adapter, is_adapter, load_base,
                        parameter_fingerprints, probe_logits)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_policy(policy):
    if policy.get("schema_version") != 1:
        raise ValueError("Unsupported training policy")
    config = policy["settings"]
    fixed = {**SETTINGS, "accumulation_steps": 8, "optimizer_steps": 64, "max_length": 512}
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("Training settings differ from the bounded first recovery design")
    if (config.get("checkpoint_steps") != [16, 32, 64] or config.get("epoch_count") != 1
            or config.get("betas") != [0.9, 0.999] or config.get("eps") != 1e-8
            or config.get("lr_schedule") != "constant"):
        raise ValueError("Unexpected checkpoint/optimizer policy")
    if set(policy["arms"]) != {"original", "calibrated_5888"}:
        raise ValueError("Expected original and calibrated_5888 arms")
    return config


def rng_hashes():
    return {"cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            "cuda": hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()}


def save_checkpoint(model, output_dir, step, report, probe_ids, device):
    directory = output_dir / f"step-{step:04d}"
    directory.mkdir(exist_ok=False)
    rng_before = rng_hashes()
    # Save/probe operations must not change the dropout RNG used by later steps.
    with torch.random.fork_rng(devices=[device.index or 0]):
        logits = probe_logits(model, probe_ids, device)
        model.save_pretrained(directory, safe_serialization=True, save_embedding_layers=False)
        save_file({"logits": logits.contiguous()}, directory / "probe.safetensors")
        after_adapter = parameter_fingerprints(model, adapter=True)
    preserved = rng_before == rng_hashes()
    if not preserved:
        raise ValueError("Checkpointing changed training RNG")
    files = {name: sha256_file(directory / name) for name in
             ("adapter_config.json", "adapter_model.safetensors", "probe.safetensors")}
    manifest = {"schema_version": 1, "status": "completed", "checkpoint_kind": "recovered_lora",
                "training_step": step, "source_model_revision": report["source_model_revision"],
                "base_provenance": report["base_provenance"], "adapter_files_sha256": files,
                "settings": report["settings"], "data": report["data"],
                "training_policy_sha256": report["training_policy_sha256"],
                "training_source_sha256": report["source_sha256"],
                "samples_seen": sum(len(s["sample_ids"]) for s in report["steps"]),
                "supervised_tokens_seen": sum(s["supervised_token_count"] for s in report["steps"]),
                "adapter_parameter_hashes": after_adapter,
                "probe": {"file": "probe.safetensors", "input_ids": probe_ids,
                          "scope": "One 32-token probe, last-position full-vocabulary logits, eval and BF16 autocast."},
                "checkpoint_rng_preserved": preserved, "evaluation_status": "not_yet_evaluated",
                "formal_training_run": str(output_dir.resolve())}
    write_json(directory / "checkpoint_manifest.json", manifest)
    model.train()
    return {"step": step, "directory": str(directory.resolve()),
            "manifest_sha256": sha256_file(directory / "checkpoint_manifest.json"),
            "samples_seen": manifest["samples_seen"],
            "supervised_tokens_seen": manifest["supervised_tokens_seen"], "rng_preserved": preserved}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("original", "calibrated_5888"))
    parser.add_argument("--mode", required=True, choices=("memory", "train"))
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--memory-report", type=Path)
    args = parser.parse_args(argv)
    policy = read_json(args.policy)
    config = validate_policy(policy)
    model_dir = Path(policy["arms"][args.arm]["model_directory"])
    if args.output_dir.resolve().is_relative_to(model_dir.resolve()):
        raise ValueError("Do not write inside a base checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / "run.json"
    report = {"schema_version": 1, "status": "starting", "stage": "preflight", "mode": args.mode,
              "arm": args.arm, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "settings": config, "source_model_revision": policy["revision"],
              "training_policy_sha256": sha256_file(args.policy),
              "source_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in
                                ("train_recovery.py", "recovery_training.py", "recovery_data.py", "trial_lora.py", "run_quality.py")},
              "environment": package_versions(), "steps": [], "checkpoints": [],
              "scope": "Eight longest-row memory check; discard changes" if args.mode == "memory" else
                       "Fresh LoRA, 512 examples once, 64 updates; validation in separate processes; final locked"}
    started = time.perf_counter()
    model = optimizer = None
    try:
        import importlib.metadata
        report["environment"]["peft"] = importlib.metadata.version("peft")
        data_manifest = read_json(args.data_dir / "manifest.json")
        canonical_dir = Path(data_manifest["tokenizer"]["directory"])
        verify_files(canonical_dir, data_manifest["tokenizer"]["files_sha256"])
        tokenizer = AutoTokenizer.from_pretrained(canonical_dir, local_files_only=True, trust_remote_code=False)
        candidate_tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        report["tokenizer_compatibility"] = check_tokenizers(tokenizer, candidate_tokenizer)
        del candidate_tokenizer
        _, rows, data = load_recovery_data(args.data_dir, tokenizer)
        report["data"] = data
        if data != policy["training_data"] or data["source_model_revision"] != policy["revision"]:
            raise ValueError("Recovery dataset changed after policy was recorded")
        if len(rows) != 512 or sum(r["supervised_token_count"] for r in rows) != 6762:
            raise ValueError("Unexpected frozen recovery budget")
        provenance, pruning = model_provenance(model_dir, policy["revision"])
        if provenance != policy["arms"][args.arm]["base_provenance"]:
            raise ValueError("Base checkpoint changed after policy was recorded")
        report["base_provenance"] = provenance
        if args.mode == "train":
            if args.memory_report is None:
                raise ValueError("A successful independent memory check is required")
            memory = read_json(args.memory_report)
            if (memory["status"] != "completed" or memory["mode"] != "memory" or memory["arm"] != args.arm
                    or memory["data"] != data or memory["base_provenance"] != provenance
                    or memory["settings"] != config or memory["source_sha256"] != report["source_sha256"]
                    or memory["training_policy_sha256"] != report["training_policy_sha256"]):
                raise ValueError("Memory check does not match the formal training setup")
            report["memory_check"] = {"path": str(args.memory_report.resolve()), "sha256": sha256_file(args.memory_report)}
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 is required")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        report["gpu"] = {"name": torch.cuda.get_device_name(device),
                         "total_bytes": torch.cuda.get_device_properties(device).total_memory}
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        model = load_base(model_dir, device)
        report["base_structure"] = inspect_model(model, pruning, str(device))
        if (model.config.num_hidden_layers != 28 or model.config.hidden_size != 2048
                or model.config.intermediate_size != policy["arms"][args.arm]["expected_base_width"]):
            raise ValueError("Unexpected base architecture")
        # Reset immediately before adapter construction so earlier loading cannot consume this seed.
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        model, report["adapter"] = attach_adapter(model)
        report["adapter"]["settings"] = config
        if any(p.dtype != torch.bfloat16 for n, p in model.named_parameters() if not is_adapter(n)):
            raise ValueError("Base precision changed during adapter construction")
        before_base = parameter_fingerprints(model)
        before_adapter = parameter_fingerprints(model, adapter=True)
        report["base_parameter_hashes_before"] = before_base
        report["adapter_parameter_hashes_before"] = before_adapter
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=config["learning_rate"], betas=tuple(config["betas"]),
                                      eps=config["eps"], weight_decay=config["weight_decay"])
        if args.mode == "memory":
            selected = sorted(rows, key=lambda r: -r["sequence_length"])[:8]
            groups = [selected]
            report["longest_rows"] = [{"id": r["id"], "length": r["sequence_length"]} for r in selected]
        else:
            groups = batch_groups(rows, size=config["accumulation_steps"])
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        report["stage"] = "training"
        write_json(report_path, report)
        for step, group in enumerate(groups, 1):
            metrics = train_accumulated_group(model, group, optimizer, device,
                                              max_grad_norm=config["gradient_clip"])
            metrics["step"] = step
            report["steps"].append(metrics)
            if args.mode == "train" and step in config["checkpoint_steps"]:
                report["checkpoints"].append(save_checkpoint(model, args.output_dir, step, report,
                                                              rows[0]["input_ids"][:32], device))
            if step == 1 or step % 8 == 0 or args.mode == "memory":
                print(f"{args.arm} {args.mode} step {step}/{len(groups)} loss={metrics['weighted_loss']:.6f} tokens={metrics['supervised_token_count']}", flush=True)
            write_json(report_path, report)
        torch.cuda.synchronize(device)
        report["training_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["training_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        optimizer = None
        gc.collect()
        torch.cuda.empty_cache()
        after_base = parameter_fingerprints(model)
        after_adapter = parameter_fingerprints(model, adapter=True)
        report["base_parameter_hashes_after"] = after_base
        report["base_parameters_unchanged"] = before_base == after_base
        report["adapter_parameter_hashes_after"] = after_adapter
        report["changed_adapter_tensors"] = [n for n in before_adapter if before_adapter[n] != after_adapter[n]]
        if not report["base_parameters_unchanged"] or not report["changed_adapter_tensors"]:
            raise ValueError("Base changed or adapter did not update")
        seen = [identifier for s in report["steps"] for identifier in s["sample_ids"]]
        report["sample_count"] = len(seen)
        report["supervised_token_count"] = sum(s["supervised_token_count"] for s in report["steps"])
        if args.mode == "train":
            if seen != [r["id"] for r in rows] or report["supervised_token_count"] != 6762:
                raise ValueError("Formal run did not consume exactly the fixed budget/order")
            if len(report["steps"]) != 64 or [c["step"] for c in report["checkpoints"]] != [16, 32, 64]:
                raise ValueError("Incomplete training/checkpoint schedule")
        report["status"] = "completed"
        report["stage"] = "completed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        optimizer = model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        report["elapsed_seconds"] = time.perf_counter() - started
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(report_path, report)
        print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
