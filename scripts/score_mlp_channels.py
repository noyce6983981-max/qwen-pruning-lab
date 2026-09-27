"""Rank original Qwen3 MLP channels using frozen activation RMS and down weights.

CPU only: read one down-projection tensor at a time from local safetensors.
No model forward, GPU, pruning, training or quality evaluation occurs here.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from run_quality import (create_output, model_provenance, package_versions,
                         read_json, sha256_file, within)


@torch.no_grad()
def compute_channel_scores(down_weight: torch.Tensor, rms: torch.Tensor) -> dict:
    if (not down_weight.is_floating_point() or not rms.is_floating_point()
            or down_weight.ndim != 2 or rms.ndim != 1
            or not down_weight.numel() or rms.numel() != down_weight.shape[1]):
        raise ValueError("Expected floating down weights [outputs, channels] and RMS [channels]")
    weights = down_weight.detach().to(device="cpu", dtype=torch.float32)
    activations = rms.detach().to(device="cpu", dtype=torch.float64)
    if (not torch.isfinite(weights).all() or not torch.isfinite(activations).all()
            or (activations < 0).any()):
        raise ValueError("Weights and nonnegative RMS must be finite")
    means = weights.abs().sum(dim=0, dtype=torch.float64) / weights.shape[0]
    scores = means * activations
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite channel score")
    return {"mean_abs_weight": means, "score": scores,
            "ascending_indices": torch.argsort(scores, stable=True)}


def positive_int(value) -> bool:
    return type(value) is int and value > 0


def load_activation_statistics(directory: Path):
    report = read_json(directory / "collection.json")
    if (type(report.get("schema_version")) is not int or report["schema_version"] != 1
            or report.get("status") != "completed"
            or report.get("stage") != "completed"
            or report.get("provenance", {}).get("checkpoint_kind") != "original"):
        raise ValueError("Expected completed original-model activation collection schema 1")
    protocol = report.get("protocol", {})
    expected = {"activation": "z = SiLU(gate_proj(x)) * up_proj(x); down_proj input",
                "model_dtype": "bfloat16", "rms_definition": "sqrt(sum_squares / valid_token_count)",
                "weights_updated": False}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("Unsupported activation collection protocol")
    if protocol.get("weights_updated") is not False:
        raise ValueError("Activation collection must not update weights")
    model = report["model"]
    if any(not positive_int(model.get(key)) for key in (
            "num_hidden_layers", "hidden_size", "intermediate_size")):
        raise ValueError("Invalid model dimensions in activation report")
    depth, width = model["num_hidden_layers"], model["intermediate_size"]
    data = report["data"]
    documents, length, count = (data.get(key) for key in (
        "document_count", "tokens_per_document", "valid_tokens_per_layer"))
    if (not all(positive_int(value) for value in (documents, length, count))
            or length != 256 or count != documents * length
            or type(report.get("completed_documents")) is not int
            or report["completed_documents"] != documents):
        raise ValueError("Inconsistent completed calibration budget")
    stats = report["statistics"]
    if stats.get("reload_exact") is not True:
        raise ValueError("Activation file must have passed exact save/reload validation")
    tensor_path = within(directory, stats["file"])
    if sha256_file(tensor_path) != stats["sha256"]:
        raise ValueError("Activation statistics SHA256 mismatch")
    tensors = load_file(str(tensor_path), device="cpu")
    keys = {f"layer.{i}.{name}" for i in range(depth) for name in ("sum_squares", "rms")}
    if set(tensors) != keys or len(stats["layers"]) != depth:
        raise ValueError("Unexpected layer count or activation tensor keys")
    for i, layer in enumerate(stats["layers"]):
        expected_layer = {"layer": i, "width": width, "token_count": count, "forward_calls": documents}
        if any(type(layer.get(key)) is not int or layer[key] != value
               for key, value in expected_layer.items()):
            raise ValueError(f"Layer {i} metadata differs from the collection budget/model")
        sums, rms = (tensors[f"layer.{i}.{name}"] for name in ("sum_squares", "rms"))
        for value in (sums, rms):
            if (value.shape != (width,) or value.dtype != torch.float64
                    or not torch.isfinite(value).all() or (value < 0).any()):
                raise ValueError(f"Layer {i} statistics must be finite nonnegative float64 vectors")
        if not torch.equal(rms, (sums / count).sqrt()):
            raise ValueError(f"Layer {i} RMS does not match stored sum of squares and count")
    return report, tensors


def validate_checkpoint(directory: Path, collection: dict) -> tuple[dict, dict, dict]:
    if (directory / "pruning_manifest.json").exists():
        raise ValueError("Score the same original checkpoint used for activation collection")
    provenance, _ = model_provenance(directory, collection["revision"])
    if provenance["files_sha256"] != collection["provenance"]["files_sha256"]:
        raise ValueError("Checkpoint hashes differ from the activation collection source")
    config = read_json(directory / "config.json")
    if config.get("model_type") != "qwen3" or config.get("hidden_act") != "silu":
        raise ValueError("Only the original Qwen3 SiLU gated MLP is supported")
    if any(config[key] != collection["model"][key] for key in (
            "hidden_size", "intermediate_size", "num_hidden_layers")):
        raise ValueError("Checkpoint dimensions differ from the activation collection")
    index = read_json(directory / "model.safetensors.index.json")["weight_map"]
    for i in range(config["num_hidden_layers"]):
        key = f"model.layers.{i}.mlp.down_proj.weight"
        shard = index.get(key)
        if not isinstance(shard, str) or shard not in provenance["files_sha256"]:
            raise ValueError(f"Missing verified weight shard for {key}")
        within(directory, shard)
    return provenance, config, index


def channel_example(channel: int, rms: torch.Tensor, scored: dict) -> dict:
    return {"channel_index": channel, "activation_rms": rms[channel].item(),
            "mean_abs_down_weight": scored["mean_abs_weight"][channel].item(),
            "score": scored["score"][channel].item()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    create_output(args.output_dir)
    torch.set_num_threads(1)
    started = time.perf_counter()
    report = {"schema_version": 1, "status": "starting",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "runner_sha256": sha256_file(Path(__file__)),
              "helpers_sha256": {"run_quality.py": sha256_file(Path(__file__).with_name("run_quality.py"))},
              "environment": {"python": platform.python_version(), "packages": package_versions(),
                              "device": "cpu", "torch_threads": 1},
              "method": {"name": "Qwen3 MLP-only calibration score adaptation",
                         "formula": "mean_i(abs(W_down[i,j])) * sqrt(sum_t(z[t,j]^2) / valid_token_count)",
                         "weight_abs_dtype": "float32", "weight_reduction_and_score_dtype": "float64",
                         "ranking": "ascending within each layer; stable ties use original channel index",
                         "activation_source": "all layers of the unchanged original model",
                         "reference": "https://github.com/CASIA-LMC-Lab/FLAP/blob/3bb57db3449dd2fa04a5c2192de80e87e33be2b1/lib/prune.py",
                         "normalization_reference": "https://github.com/CASIA-LMC-Lab/FLAP/blob/3bb57db3449dd2fa04a5c2192de80e87e33be2b1/lib/layerwrapper.py",
                         "normalization_difference": "FLAP uses sqrt(sum_squares/sample_count); this uses sqrt(sum_squares/token_count). Same weights/activations give a common within-layer factor.",
                         "scope": "Scoring-form adaptation only, not reproduction of the full FLAP/wanda_sp algorithm"},
              "weights_modified": False, "pruning_applied": False, "model_forward_calls": 0}
    try:
        report["stage"] = "validating_inputs"
        collection, activations = load_activation_statistics(args.calibration_dir)
        provenance, config, index = validate_checkpoint(args.model_dir, collection)
        report["source"] = {"revision": collection["revision"], "model_directory": str(args.model_dir.resolve()),
                            "provenance": provenance,
                            "weight_index_sha256": sha256_file(args.model_dir / "model.safetensors.index.json"),
                            "calibration_directory": str(args.calibration_dir.resolve()),
                            "collection_report_sha256": sha256_file(args.calibration_dir / "collection.json"),
                            "activation_stats_sha256": collection["statistics"]["sha256"],
                            "calibration_data": collection["data"]}
        report["model"] = {key: config[key] for key in ("hidden_size", "intermediate_size", "num_hidden_layers")}
        report["per_layer"] = []
        tensors = {}
        report["stage"] = "scoring"
        for i in range(config["num_hidden_layers"]):
            key = f"model.layers.{i}.mlp.down_proj.weight"
            with safe_open(str(within(args.model_dir, index[key])), framework="pt", device="cpu") as weights_file:
                weights = weights_file.get_tensor(key)
                if (weights.shape != (config["hidden_size"], config["intermediate_size"])
                        or weights.dtype != torch.bfloat16):
                    raise ValueError(f"Unexpected down-projection shape or dtype at layer {i}")
                rms = activations[f"layer.{i}.rms"]
                scored = compute_channel_scores(weights, rms)
                del weights
            for name, values in scored.items():
                tensors[f"layer.{i}.{name}"] = values
            ascending = scored["ascending_indices"].tolist()
            layer = {"layer": i, "width": len(ascending),
                     "score_min": scored["score"].min().item(), "score_max": scored["score"].max().item(),
                     "zero_score_count": int((scored["score"] == 0).sum().item()),
                     "first_three_original_channels": [channel_example(j, rms, scored) for j in range(min(3, len(rms)))],
                     "lowest_five": [channel_example(j, rms, scored) for j in ascending[:5]],
                     "highest_five": [channel_example(j, rms, scored) for j in reversed(ascending[-5:])]}
            report["per_layer"].append(layer)
            print(f"Scored layer {i + 1}/{config['num_hidden_layers']}: {len(ascending)} channels", flush=True)
        tensor_path = args.output_dir / "channel_scores.safetensors"
        save_file(tensors, str(tensor_path), metadata={"schema_version": "1", "kind": "qwen3_mlp_channel_scores"})
        reloaded = load_file(str(tensor_path))
        if set(reloaded) != set(tensors) or not all(torch.equal(value, reloaded[key]) for key, value in tensors.items()):
            raise RuntimeError("Channel scores/rankings did not reload exactly")
        report["output"] = {"file": tensor_path.name, "sha256": sha256_file(tensor_path),
                            "bytes": tensor_path.stat().st_size, "reload_exact": True}
        report["status"] = report["stage"] = "completed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report_path = args.output_dir / "scoring.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
