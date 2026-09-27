"""Validate saved calibration rankings before using them for real MLP pruning.

The caller supplies the original checkpoint inventory already verified by
prune_qwen_mlp.validate_source. This module reads only small scoring artifacts;
it never loads the model or uses CUDA, and it does not recompute importance.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

from run_quality import read_json, sha256_file, within


MODEL_DIMENSIONS = {"num_hidden_layers": 28, "hidden_size": 2048, "intermediate_size": 6144}
EXPECTED_METHOD = {
    "name": "Qwen3 MLP-only calibration score adaptation",
    "formula": "mean_i(abs(W_down[i,j])) * sqrt(sum_t(z[t,j]^2) / valid_token_count)",
    "weight_abs_dtype": "float32",
    "weight_reduction_and_score_dtype": "float64",
    "ranking": "ascending within each layer; stable ties use original channel index",
    "activation_source": "all layers of the unchanged original model",
}


def _mapping(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _sha256(value, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _file_name(value, label: str) -> str:
    if (not isinstance(value, str) or not value or value in (".", "..")
            or "/" in value or "\\" in value or ":" in value):
        raise ValueError(f"{label} must be a plain file name")
    return value


def _hash_mapping(value, label: str) -> dict:
    value = _mapping(value, label)
    if not value:
        raise ValueError(f"{label} must not be empty")
    return {_file_name(name, label): _sha256(digest, f"{label}.{name}")
            for name, digest in value.items()}


def _verified_source_files(source: dict) -> dict:
    records = source.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Verified source files must be a nonempty list")
    hashes = {}
    for item in records:
        item = _mapping(item, "Source file record")
        name = _file_name(item.get("name"), "Source file name")
        if name in hashes:
            raise ValueError(f"Duplicate verified source file: {name}")
        hashes[name] = _sha256(item.get("sha256"), f"Source file {name}")
    if (not {"config.json", "model.safetensors.index.json"}.issubset(hashes)
            or not any(name.endswith(".safetensors") for name in hashes)):
        raise ValueError("Verified source must include config, weight index and safetensors")
    return hashes


def validate_scoring_report(report: dict, source: dict) -> None:
    """Validate schema, source identity, method and the recorded hash chain."""
    report = _mapping(report, "Scoring report")
    source = _mapping(source, "Verified source")
    if (type(report.get("schema_version")) is not int or report["schema_version"] != 1
            or report.get("status") != "completed" or report.get("stage") != "completed"):
        raise ValueError("Expected completed scoring report schema 1")
    if (report.get("weights_modified") is not False or report.get("pruning_applied") is not False
            or type(report.get("model_forward_calls")) is not int or report["model_forward_calls"] != 0):
        raise ValueError("Scoring must not modify/prune weights or perform model forwards")
    method = _mapping(report.get("method"), "Scoring method")
    if any(method.get(key) != value for key, value in EXPECTED_METHOD.items()):
        raise ValueError("Scoring method, formula, precision or activation source is unsupported")
    model = _mapping(report.get("model"), "Scoring model dimensions")
    if any(type(model.get(key)) is not int or model[key] != value
           for key, value in MODEL_DIMENSIONS.items()):
        raise ValueError("Scores must describe the original 28-layer, 2048/6144 Qwen3 model")
    layers = report.get("per_layer")
    if not isinstance(layers, list) or len(layers) != MODEL_DIMENSIONS["num_hidden_layers"]:
        raise ValueError("Scoring per_layer metadata must contain exactly 28 layers")
    for index, layer in enumerate(layers):
        layer = _mapping(layer, f"Layer {index} metadata")
        if (type(layer.get("layer")) is not int or layer["layer"] != index
                or type(layer.get("width")) is not int
                or layer["width"] != MODEL_DIMENSIONS["intermediate_size"]):
            raise ValueError(f"Layer {index} metadata has incorrect index or width")

    scoring_source = _mapping(report.get("source"), "Scoring source")
    revision = source.get("revision")
    if not isinstance(revision, str) or not revision or scoring_source.get("revision") != revision:
        raise ValueError("Scoring source revision differs from verified source")
    source_hashes = _verified_source_files(source)
    provenance = _mapping(scoring_source.get("provenance"), "Scoring source provenance")
    if provenance.get("checkpoint_kind") != "original":
        raise ValueError("Scoring provenance must describe the original checkpoint")
    score_hashes = _hash_mapping(provenance.get("files_sha256"), "Scoring checkpoint hashes")
    required = {"config.json"} | {name for name in source_hashes if name.endswith(".safetensors")}
    if "generation_config.json" in source_hashes:
        required.add("generation_config.json")
    if not required.issubset(score_hashes):
        raise ValueError("Scoring checkpoint hashes omit config or verified checkpoint files")
    for name, digest in score_hashes.items():
        if source_hashes.get(name) != digest:
            raise ValueError(f"Scoring checkpoint hash differs from verified source: {name}")
    index_hash = _sha256(scoring_source.get("weight_index_sha256"), "Scoring weight index hash")
    if index_hash != source_hashes["model.safetensors.index.json"]:
        raise ValueError("Scoring weight index hash differs from verified source")
    if "download_manifest_sha256" in source:
        expected = _sha256(source["download_manifest_sha256"], "Verified download manifest hash")
        if provenance.get("manifest_sha256") != expected:
            raise ValueError("Scoring download manifest differs from verified source")
    if "download_manifest" in provenance:
        download = _mapping(provenance["download_manifest"], "Scoring download manifest")
        if download.get("revision") != revision:
            raise ValueError("Scoring download manifest has a different source revision")

    _sha256(report.get("runner_sha256"), "Scoring runner hash")
    _hash_mapping(report.get("helpers_sha256"), "Scoring helper hashes")
    _sha256(scoring_source.get("collection_report_sha256"), "Source collection report hash")
    _sha256(scoring_source.get("activation_stats_sha256"), "Source activation statistics hash")
    data = _mapping(scoring_source.get("calibration_data"), "Calibration data provenance")
    _sha256(data.get("manifest_sha256"), "Calibration manifest hash")
    _sha256(data.get("token_ids_sha256"), "Calibration token IDs hash")
    data_hashes = _hash_mapping(data.get("files_sha256"), "Calibration data hashes")
    if not {"calibration.jsonl", "groups.json"}.issubset(data_hashes):
        raise ValueError("Calibration provenance is missing calibration/groups hashes")
    output = _mapping(report.get("output"), "Scoring output")
    _sha256(output.get("sha256"), "Score tensor hash")
    if (output.get("reload_exact") is not True
            or type(output.get("bytes")) is not int or output["bytes"] <= 0):
        raise ValueError("Score tensors must have exact reload validation and a positive byte count")
    if not isinstance(output.get("file"), str) or not output["file"]:
        raise ValueError("Scoring output file must be a nonempty relative path")


def _validated_tensors(tensors: dict) -> list[torch.Tensor]:
    depth, width = MODEL_DIMENSIONS["num_hidden_layers"], MODEL_DIMENSIONS["intermediate_size"]
    expected = {f"layer.{index}.{name}" for index in range(depth)
                for name in ("mean_abs_weight", "score", "ascending_indices")}
    if set(tensors) != expected:
        raise ValueError("Score tensor keys must contain exactly three vectors for each of 28 layers")
    scores = []
    for index in range(depth):
        for name in ("mean_abs_weight", "score"):
            value = tensors[f"layer.{index}.{name}"]
            if (value.device.type != "cpu" or value.dtype != torch.float64 or value.shape != (width,)
                    or not torch.isfinite(value).all().item() or (value < 0).any().item()):
                raise ValueError(f"Layer {index} {name} must be a finite nonnegative CPU float64 vector of width 6144")
        score = tensors[f"layer.{index}.score"]
        ascending = tensors[f"layer.{index}.ascending_indices"]
        if (ascending.device.type != "cpu" or ascending.dtype != torch.int64
                or ascending.shape != (width,)
                or not torch.equal(ascending, torch.argsort(score, stable=True))):
            raise ValueError(f"Layer {index} ranking differs from stable ascending score order")
        scores.append(score.clone().contiguous())
    return scores


def load_calibrated_scores(scores_dir: Path, source: dict) -> tuple[list[torch.Tensor], dict]:
    """Return validated CPU scores and a self-contained audit trail for pruning.

    Linked calibration hashes are preserved from the completed scorer report;
    this does not require their original absolute paths to remain available.
    """
    scores_dir = Path(scores_dir).resolve()
    report_path = scores_dir / "scoring.json"
    report = read_json(report_path)
    validate_scoring_report(report, source)
    output = report["output"]
    tensor_path = within(scores_dir, output["file"])
    if tensor_path.stat().st_size != output["bytes"]:
        raise ValueError("Score tensor file size differs from scoring report")
    if sha256_file(tensor_path) != output["sha256"]:
        raise ValueError("Score tensor SHA256 mismatch")
    scores = _validated_tensors(load_file(str(tensor_path), device="cpu"))
    provenance = {
        "scoring_report_path": str(report_path),
        "scoring_report_sha256": sha256_file(report_path),
        "score_tensor_path": str(tensor_path),
        "score_tensor_sha256": output["sha256"],
        "score_tensor_bytes": output["bytes"],
        "method": copy.deepcopy(report["method"]),
        "model": copy.deepcopy(report["model"]),
        "scoring_runner_sha256": report["runner_sha256"],
        "scoring_helpers_sha256": copy.deepcopy(report["helpers_sha256"]),
        "source": copy.deepcopy(report["source"]),
        "validation": {
            "source_checkpoint_files_match": True,
            "source_weight_index_matches": True,
            "score_tensor_hash_matches": True,
            "stable_rankings_match_scores": True,
            "linked_calibration_artifacts_reread": False,
        },
    }
    return scores, provenance
