"""MLP channel pruning for one local original Qwen3-1.7B checkpoint.

Default: joint squared-weight magnitude. --scores-dir selects previously saved
calibration scores bound to the same original weights. No recovery training or
claim to reproduce a complete paper. --smoke-test checks the default pruning
implementation on a tiny random CPU model; calibrated routing has CPU unit tests.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import torch
import torch_pruning as tp
from transformers import AutoConfig, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM


MODEL_ID = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "work" / "models" / "Qwen3-1.7B"
PROBE_IDS = list(range(17, 33))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def package_versions() -> dict:
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "torch-pruning", "accelerate", "safetensors")
    }


def score_stats(values: torch.Tensor) -> dict:
    return {
        "min": values.min().item(), "max": values.max().item(),
        "mean": values.mean().item(), "sum": values.sum().item(),
    }


def validate_shapes(model, width: int) -> None:
    if type(model) is not Qwen3ForCausalLM or model.config.model_type != "qwen3":
        raise ValueError("Only the built-in Qwen3ForCausalLM class is supported")
    if model.config.intermediate_size != width:
        raise ValueError("Model config has an unexpected intermediate_size")
    if len(model.model.layers) != model.config.num_hidden_layers:
        raise ValueError("Model layer count differs from its config")
    hidden = model.config.hidden_size
    for index, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        expected = ((mlp.gate_proj, (width, hidden)), (mlp.up_proj, (width, hidden)),
                    (mlp.down_proj, (hidden, width)))
        for linear, shape in expected:
            if tuple(linear.weight.shape) != shape or linear.bias is not None:
                raise ValueError(f"Layer {index} has an unexpected MLP shape or bias")


@torch.no_grad()
def joint_channel_scores(mlp) -> torch.Tensor:
    # Only one matrix is converted to FP32 at a time. Return the small score
    # vector to CPU for a stable sort with original channel index as tie-break.
    scores = mlp.gate_proj.weight.float().square().sum(dim=1)
    scores += mlp.up_proj.weight.float().square().sum(dim=1)
    scores += mlp.down_proj.weight.float().square().sum(dim=0)
    scores = scores.detach().cpu()
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite channel importance scores")
    return scores


@torch.no_grad()
def prune_model(model, keep_width: int, channel_scores=None) -> dict:
    before_width = model.config.intermediate_size
    validate_shapes(model, before_width)
    if type(keep_width) is not int or not 0 < keep_width < before_width:
        raise ValueError("keep-width must be positive and smaller than the original MLP width")
    # Validate every supplied layer before any structural mutation. Keep the
    # caller's tensors and channel coordinates unchanged.
    prepared_scores = None
    if channel_scores is not None:
        if not isinstance(channel_scores, (list, tuple)) or len(channel_scores) != len(model.model.layers):
            raise ValueError("Calibration scores must contain exactly one vector per layer")
        prepared_scores = []
        for index, values in enumerate(channel_scores):
            if (not isinstance(values, torch.Tensor) or values.device.type != "cpu" or not values.is_floating_point()
                    or values.shape != (before_width,)
                    or not torch.isfinite(values).all() or (values < 0).any()):
                raise ValueError(f"Invalid calibration scores for layer {index}")
            prepared_scores.append(values.detach().cpu().clone())
    model.requires_grad_(False)
    before_count = parameter_count(model)
    layers = []
    for index, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        scores = joint_channel_scores(mlp) if prepared_scores is None else prepared_scores[index]
        ranking = torch.argsort(scores, stable=True).tolist()
        removed_ranked = ranking[:before_width - keep_width]
        removed = sorted(removed_ranked)
        removed_set = set(removed)
        kept = [channel for channel in range(before_width) if channel not in removed_set]
        tp.prune_linear_out_channels(mlp.gate_proj, idxs=removed)
        tp.prune_linear_out_channels(mlp.up_proj, idxs=removed)
        tp.prune_linear_in_channels(mlp.down_proj, idxs=removed)
        mlp.intermediate_size = keep_width
        layers.append({
            "layer": index,
            "removed_indices": removed,
            "kept_indices": kept,
            "score_stats_all": score_stats(scores),
            "score_stats_removed": score_stats(scores[removed]),
            "score_stats_kept": score_stats(scores[kept]),
        })
        print(f"Pruned MLP {index + 1}/{len(model.model.layers)}: {before_width} -> {keep_width}", flush=True)
    model.config.intermediate_size = keep_width
    # Torch-Pruning reconstructs nn.Parameter objects; freeze again afterwards.
    model.requires_grad_(False)
    validate_shapes(model, keep_width)
    after_count = parameter_count(model)
    expected_removed = len(layers) * 3 * model.config.hidden_size * (before_width - keep_width)
    if before_count - after_count != expected_removed:
        raise AssertionError("Unexpected parameter reduction")
    return {
        "width_before": before_width, "width_after": keep_width,
        "architecture": {
            "class": type(model).__name__, "hidden_size": model.config.hidden_size,
            "num_hidden_layers": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
            "num_key_value_heads": model.config.num_key_value_heads,
            "head_dim": model.config.head_dim,
        },
        "mlp_channel_removal_fraction": (before_width - keep_width) / before_width,
        "parameters_before": before_count, "parameters_after": after_count,
        "parameters_removed": before_count - after_count,
        "parameter_removal_fraction": (before_count - after_count) / before_count,
        "per_layer": layers,
    }


@torch.inference_mode()
def probe_logits(model, device: torch.device) -> torch.Tensor:
    input_ids = torch.tensor([PROBE_IDS], dtype=torch.long, device=device)
    result = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    logits = result.logits.detach().cpu()
    if not torch.isfinite(logits).all():
        raise AssertionError("Non-finite probe logits")
    return logits


def load_model(path: Path, device: torch.device, dtype, backend: str):
    return Qwen3ForCausalLM.from_pretrained(
        path, dtype=dtype, use_safetensors=True, low_cpu_mem_usage=True,
        device_map={"": str(device)}, attn_implementation=backend,
        local_files_only=True, trust_remote_code=False,
    ).eval().requires_grad_(False)


def validate_source(path: Path, revision: str) -> dict:
    manifest_path = path / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != REVISION or revision != REVISION:
        raise ValueError("The source must be the pinned original Qwen3-1.7B snapshot")
    if (path / "pruning_manifest.json").exists():
        raise ValueError("A pruned checkpoint cannot be used as the original source")
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if (type(config) is not Qwen3Config or config.model_type != "qwen3"
            or config.architectures != ["Qwen3ForCausalLM"]
            or config.hidden_size != 2048 or config.intermediate_size != 6144
            or config.num_hidden_layers != 28):
        raise ValueError("Source config does not match the original 28-layer, 2048/6144 Qwen3-1.7B")
    index_path = path / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    required = set(weight_map.values()) | {"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"}
    records = {entry["name"]: entry for entry in manifest["files"]}
    if not required.issubset(records):
        raise ValueError("Download manifest is missing required source files")
    files = []
    for name, record in sorted(records.items()):
        if Path(name).name != name:
            raise ValueError("Source manifest file names must not contain directories")
        local_file = path / name
        if not local_file.is_file() or local_file.stat().st_size != record["bytes"]:
            raise ValueError(f"Missing source file or size mismatch: {name}")
        files.append({"name": name, "bytes": local_file.stat().st_size, "sha256": sha256_file(local_file)})
    return {
        "model_id": MODEL_ID, "revision": revision, "model_directory": str(path),
        "url": f"https://huggingface.co/{MODEL_ID}/tree/{revision}",
        "download_manifest_sha256": sha256_file(manifest_path),
        "hash_provenance": "SHA256 values computed by streaming the actual local source files; linked to the pinned download manifest. These hashes are not independent upstream hash attestations.",
        "files": files,
    }


def verify_reload(save_dir: Path, device, dtype, backend, expected_logits, expected_count, width) -> dict:
    reloaded = load_model(save_dir, device, dtype, backend)
    try:
        validate_shapes(reloaded, width)
        if parameter_count(reloaded) != expected_count:
            raise AssertionError("Reloaded parameter count differs")
        actual = probe_logits(reloaded, device)
        error = (actual.float() - expected_logits.float()).abs().max().item()
        # Same weights, dtype, backend, input and device: require exact output.
        torch.testing.assert_close(actual, expected_logits, atol=0, rtol=0)
        return {
            "status": "passed", "probe_token_ids": PROBE_IDS,
            "use_cache": False, "attention_implementation": backend,
            "max_absolute_error": error, "atol": 0, "rtol": 0,
            "parameter_count": parameter_count(reloaded),
            "definition": "Compares the pruned model before saving against the reloaded pruned model; does not compare pruned quality against the original model.",
        }
    finally:
        del reloaded
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def saved_files(save_dir: Path) -> list[dict]:
    return [
        {"name": path.name, "bytes": path.stat().st_size}
        for path in sorted(save_dir.iterdir())
        if path.is_file() and path.name != "pruning_manifest.json"
    ]


def run_smoke(report: dict, output_dir: Path, keep_width: int) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(42)
    device = torch.device("cpu")
    config = Qwen3Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=128, tie_word_embeddings=False,
        use_cache=False, bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).cpu().float().eval().requires_grad_(False)
    masked = copy.deepcopy(model)
    report.update(prune_model(model, keep_width))
    with torch.no_grad():
        for layer_record in report["per_layer"]:
            masked.model.layers[layer_record["layer"]].mlp.down_proj.weight[:, layer_record["removed_indices"]] = 0
    expected = probe_logits(model, device)
    masked_logits = probe_logits(masked, device)
    torch.testing.assert_close(expected, masked_logits, rtol=1e-5, atol=1e-6)
    report["masked_reference_check"] = {
        "status": "passed", "rtol": 1e-5, "atol": 1e-6,
        "max_absolute_error": (expected - masked_logits).abs().max().item(),
    }
    with tempfile.TemporaryDirectory(prefix="tiny-pruning-", dir=output_dir) as directory:
        path = Path(directory)
        model.save_pretrained(path, safe_serialization=True, max_shard_size="512MB")
        report["saved_files"] = saved_files(path)
        del model, masked
        gc.collect()
        report["reload_validation"] = verify_reload(path, device, torch.float32, "eager", expected, report["parameters_after"], keep_width)
    report["checkpoint_retained"] = False


def run_real(args, report: dict, save_dir: Path) -> None:
    report["source"] = validate_source(args.model_dir, args.revision)
    calibrated_scores = None
    if args.scores_dir is not None:
        from calibrated_pruning_scores import load_calibrated_scores
        calibrated_scores, report["calibration_scoring"] = load_calibrated_scores(args.scores_dir, report["source"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real-model run")
    torch.cuda.set_device(args.device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This run requires BF16 support on the selected CUDA device")
    device = torch.device(f"cuda:{args.device}")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    backend = "sdpa"
    report["gpu"] = {
        "name": torch.cuda.get_device_name(device), "device": str(device),
        "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "cuda_build": torch.version.cuda, "attention_implementation": backend,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    torch.cuda.reset_peak_memory_stats(device)
    model = load_model(args.model_dir, device, torch.bfloat16, backend)
    if {str(parameter.device) for parameter in model.parameters()} != {str(device)}:
        raise AssertionError("Model parameters must reside on the single requested GPU")
    report.update(prune_model(model, args.keep_width, calibrated_scores))
    expected = probe_logits(model, device)
    model.save_pretrained(save_dir, safe_serialization=True, max_shard_size="512MB")
    tokenizer.save_pretrained(save_dir)
    if (args.model_dir / "LICENSE").is_file():
        shutil.copyfile(args.model_dir / "LICENSE", save_dir / "LICENSE")
    report["saved_files"] = saved_files(save_dir)
    report["prune_and_save_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    # Do not retain layer/tensor aliases or two full GPU models during reload.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    report["allocated_bytes_before_reload"] = torch.cuda.memory_allocated(device)
    report["reload_validation"] = verify_reload(save_dir, device, torch.bfloat16, backend, expected, report["parameters_after"], args.keep_width)
    report["checkpoint_retained"] = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--save-dir", type=Path, help="New directory for the pruned checkpoint; required outside smoke-test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-width", type=int, help="Default: 5632 for real model; 96 for smoke-test")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--scores-dir", type=Path, help="Completed channel scoring directory; omit for joint weight magnitude")
    args = parser.parse_args()
    args.model_dir = args.model_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.keep_width = args.keep_width if args.keep_width is not None else (96 if args.smoke_test else 5632)
    if args.smoke_test and args.scores_dir is not None:
        parser.error("--scores-dir requires the real original model; calibrated tiny models are checked in unit tests")
    if args.output_dir == args.model_dir or args.model_dir in args.output_dir.parents:
        parser.error("output-dir must not be inside the original model directory")
    if not args.smoke_test and args.save_dir is None:
        parser.error("--save-dir is required for a real-model run")
    save_dir = None
    if args.save_dir is not None:
        if args.smoke_test:
            parser.error("smoke-test uses a temporary checkpoint; omit --save-dir")
        save_dir = args.save_dir.resolve()
        if save_dir.exists() or save_dir == args.model_dir or args.model_dir in save_dir.parents:
            parser.error("save-dir must be a new directory outside the original model directory")
        if save_dir == args.output_dir or save_dir in args.output_dir.parents:
            parser.error("output-dir must be separate from, and not nested inside, save-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "pruning_manifest.json").exists():
        parser.error("output-dir already contains a pruning manifest; choose a new run directory")
    report = {
        "status": "starting", "checkpoint_kind": "pruned",
        "run_kind": "cpu_tiny_smoke_test" if args.smoke_test else "real_model_pruning",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"model_id": "random_tiny_qwen3" if args.smoke_test else MODEL_ID,
                   "revision": None if args.smoke_test else args.revision,
                   "model_directory": None if args.smoke_test else str(args.model_dir)},
        "runner_path": str(Path(__file__).resolve()), "runner_sha256": sha256_file(Path(__file__)),
        "method": "joint_l2_squared",
        "score_formula": "score[j] = sum_i(gate_proj.weight[j,i]^2) + sum_i(up_proj.weight[j,i]^2) + sum_i(down_proj.weight[i,j]^2)",
        "score_dtype": "float32", "ranking": "Ascending score; stable CPU sort breaks ties by original channel index. Kept channels retain original order.",
        "recovery_applied": False,
        "scope": "Weight-only MLP channel-pruning baseline. No activation calibration, Taylor score, recovery training, quality guarantee, speedup claim, or complete paper reproduction.",
        "dtype": "float32" if args.smoke_test else "bfloat16",
        "python": platform.python_version(), "platform": platform.platform(), "packages": package_versions(),
        "save_directory": str(save_dir) if save_dir else None,
        "save_max_shard_size": "512MB; individual tensors larger than the limit remain whole",
    }
    if args.scores_dir is not None:
        report.update({
            "method": "calibrated_rms_mean_abs",
            "score_formula": "score[j] = mean_i(abs(down_proj.weight[i,j])) * activation_rms[j]",
            "score_dtype": "float64; saved BF16 weights converted to FP32 abs, FP64 mean and multiplication",
            "scope": "Qwen3 MLP-only calibration-score adaptation. Statistics from the unchanged original model; uniform per-layer width. No recovery training, quality guarantee or full FLAP reproduction.",
            "helper_sha256": {"calibrated_pruning_scores.py": sha256_file(Path(__file__).with_name("calibrated_pruning_scores.py"))},
        })
    owned_save_dir = False
    try:
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=False)
            owned_save_dir = True
        if args.smoke_test:
            run_smoke(report, args.output_dir, args.keep_width)
        else:
            run_real(args, report, save_dir)
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        (args.output_dir / "pruning_manifest.json").write_text(payload, encoding="utf-8")
        if owned_save_dir:
            (save_dir / "pruning_manifest.json").write_text(payload, encoding="utf-8")
        print(f"Pruning status: {report['status']}; report: {args.output_dir / 'pruning_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
