"""Observe original Qwen3 gated MLP activations on frozen calibration documents.

No scoring, pruning, training, generation, downloads, or validation/final inputs.
Stores per-channel sum of squares and derived RMS, never full activations.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from run_quality import (create_output, host_memory, inspect_model, json_hash,
                         model_provenance, package_versions, read_json, read_jsonl,
                         sha256_file, token_ids_valid, tokenizer_signature, verify_files)


class MLPActivationCollector:
    """Context-managed, transactional observation of down-projection inputs.

    One successful backbone forward commits one observation per layer. Each
    failed forward discards all its pending statistics. No hook changes inputs.
    """

    def __init__(self, model):
        if model.config.model_type != "qwen3" or model.config.hidden_act != "silu":
            raise ValueError("Only Qwen3 with SiLU gated MLP is supported")
        if model.training:
            raise ValueError("Put the model in eval mode before observation")
        self.model = model
        self.layers = list(model.model.layers)
        if len(self.layers) != model.config.num_hidden_layers or not self.layers:
            raise ValueError("Unexpected Qwen3 layer count")
        self.widths = [layer.mlp.down_proj.in_features for layer in self.layers]
        self.sums = [torch.zeros(width, dtype=torch.float64) for width in self.widths]
        self.counts = [0] * len(self.layers)
        self.calls = [0] * len(self.layers)
        self.handles = []
        self._mask = None
        self._pending = None

    def __enter__(self):
        if self.handles:
            raise RuntimeError("Collector context is already active")
        try:
            for index, layer in enumerate(self.layers):
                self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._hook(index)))
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._mask = None
        self._pending = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def _hook(self, index):
        def observe(module, inputs):
            if self._pending is None or self._mask is None:
                raise RuntimeError("Use collector.forward while observation hooks are installed")
            if index in self._pending:
                raise RuntimeError("Layer called more than once during one backbone forward")
            z = inputs[0].detach()
            if z.ndim != 3 or z.shape[:2] != self._mask.shape or z.shape[-1] != self.widths[index]:
                raise ValueError("Activation shape does not match the token mask and channel width")
            # Select before arithmetic: multiplying padded NaNs by zero is unsafe.
            valid = z[self._mask].float()
            if not torch.isfinite(valid).all().item():
                raise ValueError(f"Nonfinite valid activation in layer {index}")
            sums = valid.square().sum(dim=0, dtype=torch.float64).cpu()
            if not torch.isfinite(sums).all().item():
                raise ValueError(f"Nonfinite sum of squares in layer {index}")
            self._pending[index] = sums
            # Returning None preserves the original input to down_proj.
        return observe

    def forward(self, input_ids, attention_mask=None):
        if not self.handles:
            raise RuntimeError("Use the collector as a context manager")
        if self.model.training:
            raise ValueError("Model must remain in eval mode")
        if input_ids.ndim != 2 or input_ids.numel() == 0 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a nonempty [batch, sequence] long tensor")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape or attention_mask.device != input_ids.device:
            raise ValueError("attention_mask must match input_ids shape and device")
        if not ((attention_mask == 0) | (attention_mask == 1)).all().item():
            raise ValueError("attention_mask must contain only binary 0/1 values")
        mask = attention_mask.bool()
        if not mask.any(dim=1).all().item():
            raise ValueError("Every sequence must contain at least one valid token")
        if self._pending is not None:
            raise RuntimeError("Nested collection is unsupported")
        count = int(mask.sum().item())
        self._mask = mask
        self._pending = {}
        try:
            # Reset positions/context per document; also supports padded batches in tests.
            positions = (mask.long().cumsum(-1) - 1).masked_fill(~mask, 0)
            with torch.inference_mode():
                output = self.model.model(input_ids=input_ids, attention_mask=attention_mask,
                                          position_ids=positions, use_cache=False,
                                          output_hidden_states=False, output_attentions=False,
                                          return_dict=True).last_hidden_state
            if len(self._pending) != len(self.layers):
                raise RuntimeError("Not all MLP layers were observed")
            updated = [self.sums[i] + self._pending[i] for i in range(len(self.layers))]
            if not all(torch.isfinite(value).all().item() for value in updated):
                raise ValueError("Nonfinite accumulated sum of squares")
            self.sums = updated
            self.counts = [value + count for value in self.counts]
            self.calls = [value + 1 for value in self.calls]
            return output
        finally:
            self._mask = None
            self._pending = None

    def statistics(self):
        if not all(self.counts):
            raise ValueError("No successfully collected observations")
        return [{"layer": i, "width": self.widths[i], "token_count": self.counts[i],
                 "forward_calls": self.calls[i], "sum_squares": value.clone(),
                 "rms": (value / self.counts[i]).sqrt()}
                for i, value in enumerate(self.sums)]


def load_calibration_data(directory: Path):
    manifest = read_json(directory / "manifest.json")
    expected = manifest["files_sha256"]
    required = ("calibration.jsonl", "groups.json")
    if any(name not in expected for name in required):
        raise ValueError("Missing frozen calibration/groups hash")
    hashes = verify_files(directory, {name: expected[name] for name in required}, required)
    rows = read_jsonl(directory / "calibration.jsonl")
    assignments = read_json(directory / "groups.json")["assignments"]
    groups = set()
    for row in rows:
        group = row.get("group_id")
        if not isinstance(group, str) or not group or group in groups:
            raise ValueError(f"{row['id']}: missing or duplicate calibration group")
        if row.get("source") != "train" or assignments.get(group) != "calibration":
            raise ValueError(f"{row['id']}: document is not assigned to train calibration")
        groups.add(group)
        ids = row.get("token_ids")
        if not token_ids_valid(ids) or len(ids) != 256:
            raise ValueError(f"{row['id']}: calibration requires exactly 256 valid token IDs")
        original_count = row.get("original_token_count")
        if type(original_count) is not int or original_count < 256:
            raise ValueError(f"{row['id']}: invalid original_token_count")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"{row['id']}: missing decoded calibration text")
    counts = manifest["counts"]["calibration"]
    if (any(type(counts.get(key)) is not int or counts[key] < 1 for key in ("rows", "groups"))
            or counts["rows"] != len(rows) or counts["groups"] != len(groups)):
        raise ValueError("Calibration counts disagree with frozen manifest")
    return manifest, rows, {
        "directory": str(directory.resolve()),
        "manifest_sha256": sha256_file(directory / "manifest.json"),
        "files_sha256": hashes, "document_count": len(rows),
        "tokens_per_document": 256, "valid_tokens_per_layer": len(rows) * 256,
        "sample_order": [{key: row.get(key) for key in (
            "id", "group_id", "source", "document_id", "context_sha256", "original_token_count")}
                         for row in rows],
        "token_ids_sha256": json_hash([row["token_ids"] for row in rows]),
        "context_hash_definition": "Full source paragraph; text is the decoded 256-token prefix.",
    }


def validate_calibration_tokens(rows, tokenizer, vocab_size: int):
    vocabulary = set(tokenizer.get_vocab().values())
    for row in rows:
        if not token_ids_valid(row.get("token_ids")):
            raise ValueError(f"{row['id']}: invalid token IDs")
        if any(token not in vocabulary or not 0 <= token < vocab_size for token in row["token_ids"]):
            raise ValueError(f"{row['id']}: token ID outside tokenizer/model vocabulary")
        decoded = tokenizer.decode(row["token_ids"], skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        if decoded != row["text"]:
            raise ValueError(f"{row['id']}: decoded frozen token IDs differ from text")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    create_output(args.output_dir)
    report = {"schema_version": 1, "status": "starting",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "model_directory": str(args.model_dir.resolve()), "revision": args.revision,
              "runner_sha256": sha256_file(Path(__file__)),
              "helpers_sha256": {"run_quality.py": sha256_file(Path(__file__).with_name("run_quality.py"))},
              "environment": {"python": platform.python_version(), "packages": package_versions()},
              "protocol": {"activation": "z = SiLU(gate_proj(x)) * up_proj(x); down_proj input",
                           "batch_size": 1, "model_dtype": "bfloat16", "attention": "sdpa",
                           "square_dtype": "float32", "reduction_dtype": "float64",
                           "accumulator_device": "cpu", "accumulator_dtype": "float64",
                           "rms_definition": "sqrt(sum_squares / valid_token_count)",
                           "all_valid_positions_counted_including_first": True,
                           "use_cache": False, "gradients": False, "weights_updated": False,
                           "full_activations_saved": False, "chat_template": False,
                           "document_context": "reset; frozen raw token IDs, no added special tokens"},
              "scope": "Activation observation only; no channel ranking, pruning, training or quality claim."}
    started = time.perf_counter()
    model = collector = None
    device = None
    try:
        from safetensors.torch import load_file, save_file
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        import transformers.models.qwen3.modeling_qwen3 as qwen3_source

        report["stage"] = "input_validation"
        print("Validating frozen calibration inputs and local original checkpoint", flush=True)
        manifest, rows, report["data"] = load_calibration_data(args.data_dir)
        if manifest["tokenizer"]["source_model_revision"] != args.revision:
            raise ValueError("Frozen calibration tokenizer revision differs from requested model")
        if (args.model_dir / "pruning_manifest.json").exists():
            raise ValueError("This calibration run requires the original unpruned model")
        report["provenance"], _ = model_provenance(args.model_dir, args.revision)
        report["tokenizer_files_sha256"] = verify_files(
            args.model_dir, manifest["tokenizer"]["files_sha256"],
            required=("tokenizer.json", "tokenizer_config.json"))
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
        config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
        validate_calibration_tokens(rows, tokenizer, config.vocab_size)
        report["tokenizer_signature_sha256"] = json_hash(tokenizer_signature(tokenizer))
        report["qwen3_implementation_sha256"] = sha256_file(Path(qwen3_source.__file__))
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the real-model run")
        torch.cuda.set_device(args.device)
        device = torch.device(f"cuda:{args.device}")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA bfloat16 support is required")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        report["environment"].update({"gpu": torch.cuda.get_device_name(device),
                                      "torch_cuda_build": torch.version.cuda, "device": str(device),
                                      "seed": 42, "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32})
        report["host_memory_before_load"] = host_memory()
        torch.cuda.reset_peak_memory_stats(device)
        report["stage"] = "loading"
        print("Loading one original model (BF16/SDPA)", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", use_safetensors=True,
            local_files_only=True, trust_remote_code=False, low_cpu_mem_usage=True,
            device_map={"": str(device)}).eval()
        report["model"] = inspect_model(model, None, str(device))
        report["load_peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        report["stage"] = "collecting"
        collect_started = time.perf_counter()
        report["completed_documents"] = 0
        with MLPActivationCollector(model) as collector:
            for index, row in enumerate(rows, 1):
                ids = torch.tensor([row["token_ids"]], dtype=torch.long, device=device)
                output = collector.forward(ids)
                del output, ids
                report["completed_documents"] = index
                print(f"Calibration {index}/{len(rows)}: {row['id']} ({index * 256} valid tokens/layer)", flush=True)
            statistics = collector.statistics()
        torch.cuda.synchronize(device)
        report["collection_seconds"] = time.perf_counter() - collect_started
        report["collection_peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["collection_peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["stage"] = "saving_statistics"
        tensors, summaries = {}, []
        for item in statistics:
            if item["token_count"] != len(rows) * 256 or item["forward_calls"] != len(rows):
                raise ValueError("Layer token/forward counts do not match the calibration budget")
            index = item["layer"]
            for name in ("sum_squares", "rms"):
                tensors[f"layer.{index}.{name}"] = item[name]
            rms = item["rms"]
            summaries.append({key: item[key] for key in ("layer", "width", "token_count", "forward_calls")}
                             | {"rms_min": rms.min().item(), "rms_median": rms.median().item(),
                                "rms_max": rms.max().item(), "zero_rms_channels": int((rms == 0).sum().item())})
        tensor_path = args.output_dir / "activation_stats.safetensors"
        save_file(tensors, str(tensor_path), metadata={"schema_version": "1", "kind": "qwen3_mlp_activation_stats"})
        reloaded = load_file(str(tensor_path))
        if set(reloaded) != set(tensors) or not all(torch.equal(value, reloaded[name]) for name, value in tensors.items()):
            raise RuntimeError("Saved activation statistics did not reload exactly")
        report["statistics"] = {"file": tensor_path.name, "sha256": sha256_file(tensor_path),
                                "bytes": tensor_path.stat().st_size, "reload_exact": True,
                                "layers": summaries}
        report["status"] = report["stage"] = "completed"
        print(f"Complete: {len(statistics)} layers, {len(rows) * 256} valid tokens per layer", flush=True)
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if collector is not None:
            report["completed_layer_token_counts"] = collector.counts
            report["completed_layer_forward_calls"] = collector.calls
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output_dir / "collection.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if collector is not None:
            collector.close()
        collector = model = None
        gc.collect()
        if device is not None:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
