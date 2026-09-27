"""Verify a standalone BF16 LoRA export without requiring its source directories."""
from pathlib import Path

from run_quality import read_json, sha256_file, verify_files


def merged_model_provenance(directory: Path, revision: str):
    directory = Path(directory)
    path = directory / "merge_manifest.json"
    manifest = read_json(path)
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "completed"
            or manifest.get("checkpoint_kind") != "merged_lora"
            or manifest.get("source_model_revision") != revision
            or manifest.get("parameter_dtype") != "torch.bfloat16"
            or manifest.get("training_step") != 64
            or manifest.get("base_checkpoint_kind") not in ("original", "pruned")
            or manifest.get("reload_validation", {}).get("status") != "passed"
            or manifest["reload_validation"].get("probe_exact") is not True
            or manifest.get("weight_audit", {}).get("all_expected_hashes_match") is not True):
        raise ValueError("Incomplete or incompatible merged checkpoint")
    if any((directory / name).exists() for name in
           ("adapter_config.json", "adapter_model.safetensors", "download_manifest.json", "pruning_manifest.json")):
        raise ValueError("Merged checkpoint must have an unambiguous standalone identity")
    files = manifest["files_sha256"]
    required = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "merge_probes.safetensors"}
    weights = {p.name for p in directory.glob("*.safetensors")} - {"merge_probes.safetensors"}
    if not weights or weights != {name for name in files if name.endswith(".safetensors")} - {"merge_probes.safetensors"}:
        raise ValueError("Merged weights differ from manifest inventory")
    hashes = verify_files(directory, files, required=required | weights)
    config = read_json(directory / "config.json")
    if (config.get("model_type") != "qwen3" or config.get("num_hidden_layers") != 28
            or config.get("hidden_size") != 2048
            or config.get("intermediate_size") != manifest["intermediate_size"]):
        raise ValueError("Exported architecture differs from manifest")
    source = manifest["source_adapter"]
    if (source.get("checkpoint_kind") != "recovered_lora" or source.get("training_step") != 64
            or source.get("source_model_revision") != revision
            or source.get("base_checkpoint_kind") != manifest["base_checkpoint_kind"]):
        raise ValueError("Merged source adapter provenance differs")
    pruning = manifest.get("source_pruning_manifest")
    if (manifest["base_checkpoint_kind"] == "pruned") != (pruning is not None):
        raise ValueError("Missing or unexpected source pruning manifest")
    if pruning is not None and (pruning["width_after"] != config["intermediate_size"]
                               or pruning["parameters_after"] != manifest["parameter_count"]):
        raise ValueError("Merged pruning structure differs from source")
    return {"checkpoint_kind": "merged_lora", "base_checkpoint_kind": manifest["base_checkpoint_kind"],
            "source_model_revision": revision, "source_model_id": source["source_model_id"],
            "recovery_applied": True, "adapter_present": False, "adapter_merged": True,
            "training_step": 64, "manifest_path": str(path.resolve()),
            "manifest_sha256": sha256_file(path), "merge_manifest_sha256": sha256_file(path),
            "files_sha256": hashes, "merge_manifest": manifest}, pruning
