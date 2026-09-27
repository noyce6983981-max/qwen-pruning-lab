"""Ten-step LoRA engineering trial, not formal recovery or quality evaluation.

One local Qwen3 base at a time. Never overwrite a base checkpoint, merge weights,
read validation/final examples, or resume these trial adapters for formal training.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from recovery_data import load_recovery_data, select_trial_examples
from run_quality import (check_tokenizers, inspect_model, model_provenance,
                         package_versions, read_json, sha256_file, verify_files)


SETTINGS = {"rank": 8, "alpha": 16, "dropout": 0.05, "learning_rate": 1e-4,
            "weight_decay": 0.0, "gradient_clip": 1.0, "batch_size": 1,
            "accumulation_steps": 1, "optimizer_steps": 10, "max_length": 256,
            "seed": 20260927, "target_modules": ["gate_proj", "up_proj", "down_proj"],
            "gradient_checkpointing_use_reentrant": False, "use_cache": False}


def is_adapter(name):
    return ".lora_A." in name or ".lora_B." in name


def parameter_fingerprints(model, adapter=False):
    """Hash one small CPU chunk at a time, avoiding a second full base in RAM."""
    result = {}
    for name, parameter in model.named_parameters():
        if is_adapter(name) != adapter:
            continue
        value = parameter.detach()
        if not value.is_contiguous():
            raise ValueError(f"Expected contiguous trial parameter: {name}")
        digest = hashlib.sha256()
        digest.update(json.dumps([list(value.shape), str(value.dtype)]).encode())
        flat = value.view(-1)
        for start in range(0, flat.numel(), 2**20):
            chunk = flat[start:start + 2**20].cpu().contiguous().view(torch.uint8)
            digest.update(chunk.numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def attach_adapter(base):
    expected = {f"model.layers.{i}.mlp.{name}" for i in range(base.config.num_hidden_layers)
                for name in SETTINGS["target_modules"]}
    actual = {name for name, module in base.named_modules()
              if name.endswith(tuple(SETTINGS["target_modules"]))}
    if actual != expected:
        raise ValueError("Unexpected MLP module inventory before LoRA injection")
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=SETTINGS["rank"],
                        lora_alpha=SETTINGS["alpha"], lora_dropout=SETTINGS["dropout"],
                        target_modules=SETTINGS["target_modules"], bias="none",
                        init_lora_weights=True, use_rslora=False)
    model = get_peft_model(base, config, autocast_adapter_dtype=True)
    trainable = []
    for name, parameter in model.named_parameters():
        if is_adapter(name):
            # Explicit FP32 optimizer/master parameters even with a BF16 base.
            parameter.data = parameter.data.float()
            if not parameter.requires_grad:
                raise ValueError(f"Adapter is unexpectedly frozen: {name}")
            trainable.append((name, parameter))
        elif parameter.requires_grad:
            raise ValueError(f"Unexpected trainable base parameter: {name}")
    if len(trainable) != 2 * len(expected):
        raise ValueError("Unexpected number of LoRA A/B tensors")
    expected_count = base.config.num_hidden_layers * 3 * SETTINGS["rank"] * (
        base.config.hidden_size + base.config.intermediate_size)
    if sum(p.numel() for _, p in trainable) != expected_count:
        raise ValueError("LoRA trainable parameter count mismatch")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if not model.is_gradient_checkpointing:
        raise ValueError("Gradient checkpointing did not enable")
    return model, {"target_module_count": len(expected), "trainable_tensor_count": len(trainable),
                   "trainable_parameter_count": expected_count, "adapter_dtype": "torch.float32",
                   "base_requires_grad": False, "settings": dict(SETTINGS)}


def batch_tensors(row, device):
    inputs = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([row["labels"]], dtype=torch.long, device=device)
    if inputs.shape != labels.shape or inputs.ndim != 2 or inputs.shape[1] < 2:
        raise ValueError("Invalid training input/label shape")
    mask = torch.tensor([row["attention_mask"]], dtype=torch.long, device=device)
    if mask.shape != inputs.shape or not bool(mask.eq(1).all()):
        raise ValueError("This batch=1 trial uses unpadded complete examples")
    count = int(labels[:, 1:].ne(-100).sum())
    if count <= 0 or count != row["supervised_token_count"]:
        raise ValueError("Supervised token count after causal shift differs")
    return {"input_ids": inputs, "attention_mask": mask, "labels": labels, "use_cache": False}


def autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=device.type == "cuda")


def train_step(model, row, optimizer, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with autocast(device):
        outputs = model(**batch_tensors(row, device))
        loss = outputs.loss
    if not bool(torch.isfinite(loss)):
        raise ValueError("Non-finite training loss")
    loss_value = float(loss.detach())
    loss.backward()
    del outputs, loss
    grads = {"A_nonzero_tensor_count": 0, "B_nonzero_tensor_count": 0}
    trainable = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise ValueError(f"Frozen base has a gradient: {name}")
            continue
        if not is_adapter(name) or parameter.grad is None:
            raise ValueError(f"Missing or unexpected trainable gradient: {name}")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise ValueError(f"Non-finite gradient: {name}")
        key = "A_nonzero_tensor_count" if ".lora_A." in name else "B_nonzero_tensor_count"
        grads[key] += int(bool(torch.count_nonzero(parameter.grad)))
        trainable.append(parameter)
    norm = torch.nn.utils.clip_grad_norm_(trainable, SETTINGS["gradient_clip"], error_if_nonfinite=True)
    if float(norm) <= 0:
        raise ValueError("All trainable gradients are zero")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if any(not bool(torch.isfinite(p).all()) for p in trainable):
        raise ValueError("Non-finite adapter after optimizer update")
    return {"id": row["id"], "sequence_length": row["sequence_length"],
            "supervised_token_count": row["supervised_token_count"],
            "loss": loss_value, "gradient_norm_before_clip": float(norm), **grads}


@torch.inference_mode()
def training_sample_loss(model, rows, device):
    """Same trial examples, eval mode: diagnostics only, never validation scores."""
    model.eval()
    per_item = []
    total, count = 0.0, 0
    for row in rows:
        with autocast(device):
            output = model(**batch_tensors(row, device))
        loss = float(output.loss)
        if not torch.isfinite(torch.tensor(loss)):
            raise ValueError("Non-finite diagnostic loss")
        n = row["supervised_token_count"]
        per_item.append({"id": row["id"], "mean_answer_nll": loss, "supervised_token_count": n})
        total += loss * n
        count += n
        del output
    return {"mean_answer_nll": total / count, "supervised_token_count": count, "per_item": per_item,
            "scope": "Same training examples; not held-out validation or general quality."}


@torch.inference_mode()
def probe_logits(model, ids, device):
    model.eval()
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    with autocast(device):
        output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False,
                       logits_to_keep=1)
    result = output.logits.detach().float().cpu().clone()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Non-finite probe logits")
    return result


def load_base(path, device):
    return AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": str(device)},
        low_cpu_mem_usage=True, use_safetensors=True, local_files_only=True,
        trust_remote_code=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args(argv)
    if args.output_dir.resolve().is_relative_to(args.model_dir.resolve()):
        raise ValueError("Trial output must not be inside the base checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "starting", "stage": "preflight", "schema_version": 1,
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Ten-step engineering trial; not formal recovery; adapters must not seed formal training.",
              "settings": dict(SETTINGS), "model_directory": str(args.model_dir.resolve()),
              "source_model_revision": args.revision,
              "source_sha256": {name: sha256_file(Path(__file__).with_name(name))
                                for name in ("trial_lora.py", "recovery_data.py", "run_quality.py")},
              "environment": {**package_versions(), "peft": importlib.metadata.version("peft")},
              "steps": []}
    model = optimizer = None
    try:
        manifest = read_json(args.data_dir / "manifest.json")
        canonical_dir = Path(manifest["tokenizer"]["directory"])
        verify_files(canonical_dir, manifest["tokenizer"]["files_sha256"])
        canonical = AutoTokenizer.from_pretrained(canonical_dir, local_files_only=True, trust_remote_code=False)
        candidate = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
        report["tokenizer_compatibility"] = check_tokenizers(canonical, candidate)
        del candidate
        _, rows, data_provenance = load_recovery_data(args.data_dir, canonical)
        if data_provenance["source_model_revision"] != args.revision:
            raise ValueError("Recovery data tokenizer revision differs from requested base revision")
        selected = select_trial_examples(rows, count=10, max_length=256)
        report["data"] = data_provenance
        report["trial_examples"] = [{k: row[k] for k in ("id", "sequence_length", "prompt_length",
                                                         "supervised_token_count")} for row in selected]
        report["selected_ids_sha256"] = hashlib.sha256(json.dumps([r["id"] for r in selected]).encode()).hexdigest()
        provenance, pruning = model_provenance(args.model_dir, args.revision)
        if provenance.get("recovery_applied", False):
            raise ValueError("Trial requires an unrecovered base")
        report["base_provenance"] = provenance
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This real-model trial requires CUDA BF16")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        report["gpu"] = {"name": torch.cuda.get_device_name(device),
                         "total_bytes": torch.cuda.get_device_properties(device).total_memory}
        torch.manual_seed(SETTINGS["seed"])
        torch.cuda.manual_seed_all(SETTINGS["seed"])
        torch.cuda.reset_peak_memory_stats(device)
        report["stage"] = "loading_one_base"
        print("Loading one BF16 base for ten-step LoRA trial", flush=True)
        model = load_base(args.model_dir, device)
        report["base_structure"] = inspect_model(model, pruning, str(device))
        if model.config.num_hidden_layers != 28 or model.config.hidden_size != 2048:
            raise ValueError("Real trial supports the local 28-layer Qwen3-1.7B bases")
        probe_ids = selected[0]["input_ids"][:32]
        base_probe = probe_logits(model, probe_ids, device)
        model, report["adapter"] = attach_adapter(model)
        if any(p.dtype != torch.bfloat16 for n, p in model.named_parameters() if not is_adapter(n)):
            raise ValueError("LoRA injection changed a base parameter dtype")
        zero_probe = probe_logits(model, probe_ids, device)
        report["initial_noop_max_abs_error"] = float((base_probe - zero_probe).abs().max())
        if not torch.equal(base_probe, zero_probe):
            raise ValueError("Default zero-B adapter changed initial probe")
        del base_probe, zero_probe
        report["stage"] = "before_training_checks"
        before_base = parameter_fingerprints(model)
        before_adapter = parameter_fingerprints(model, adapter=True)
        report["training_examples_before"] = training_sample_loss(model, selected, device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=SETTINGS["learning_rate"], weight_decay=0.0)
        torch.manual_seed(SETTINGS["seed"])
        torch.cuda.manual_seed_all(SETTINGS["seed"])
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        report["stage"] = "ten_optimizer_updates"
        for index, row in enumerate(selected, 1):
            step = train_step(model, row, optimizer, device)
            step["step"] = index
            report["steps"].append(step)
            print(f"step {index}/10 loss={step['loss']:.6f} grad_norm={step['gradient_norm_before_clip']:.6f}", flush=True)
        torch.cuda.synchronize(device)
        report["training_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["training_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        optimizer = None
        gc.collect()
        torch.cuda.empty_cache()
        report["training_examples_after"] = training_sample_loss(model, selected, device)
        after_base = parameter_fingerprints(model)
        after_adapter = parameter_fingerprints(model, adapter=True)
        report["base_parameter_hashes_before"] = before_base
        report["base_parameter_hashes_after"] = after_base
        report["base_parameters_unchanged"] = before_base == after_base
        report["adapter_hashes_before"] = before_adapter
        report["adapter_hashes_after"] = after_adapter
        changed = [n for n in before_adapter if before_adapter[n] != after_adapter[n]]
        report["changed_adapter_tensors"] = changed
        if not report["base_parameters_unchanged"] or not changed:
            raise ValueError("Base changed or adapter failed to update")
        if not any(".lora_A." in n for n in changed) or not any(".lora_B." in n for n in changed):
            raise ValueError("Both A and B should have changed by the end of ten steps")
        report["stage"] = "adapter_save"
        trained_probe = probe_logits(model, probe_ids, device)
        adapter_path = args.output_dir / "adapter"
        model.save_pretrained(adapter_path, safe_serialization=True, save_embedding_layers=False)
        report["adapter_files_sha256"] = {p.name: sha256_file(p) for p in sorted(adapter_path.iterdir()) if p.is_file()}
        # Drop every reference to the first model before loading another copy.
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        report["allocated_bytes_before_sequential_reload"] = torch.cuda.memory_allocated(device)
        report["stage"] = "sequential_base_and_adapter_reload"
        reload_provenance, _ = model_provenance(args.model_dir, args.revision)
        if reload_provenance["files_sha256"] != provenance["files_sha256"]:
            raise ValueError("Base files changed before adapter reload")
        model = PeftModel.from_pretrained(load_base(args.model_dir, device), adapter_path,
                                          is_trainable=False, autocast_adapter_dtype=True,
                                          local_files_only=True)
        reloaded_probe = probe_logits(model, probe_ids, device)
        error = float((trained_probe - reloaded_probe).abs().max())
        exact = torch.equal(trained_probe, reloaded_probe)
        state_exact = parameter_fingerprints(model, adapter=True) == after_adapter
        report["reload_check"] = {"adapter_hashes_exact": state_exact, "probe_exact": exact,
                                  "probe_max_abs_error": error, "probe_input_ids": probe_ids,
                                  "scope": "Last-position full-vocabulary logits on one 32-token probe, eval mode."}
        if not exact or not state_exact:
            raise ValueError("Saved adapter reload differs from trained adapter")
        report["status"] = "completed"
        report["stage"] = "completed"
        report["formal_recovery_started"] = False
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
        (args.output_dir / "trial.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Report: {args.output_dir / 'trial.json'}", flush=True)


if __name__ == "__main__":
    main()
