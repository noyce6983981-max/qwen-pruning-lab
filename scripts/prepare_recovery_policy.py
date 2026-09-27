"""Prepare a new path-bound recovery policy before a fresh, fixed-budget run.

Only tokenizer files, checkpoint provenance, and recovery/groups data are read.
No historical policy, trial adapter, validation/final examples, model execution,
or training is used. This creates a new record, not a backdated registration.
"""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from recovery_data import load_recovery_data
from run_quality import check_tokenizers, model_provenance, read_json, sha256_file, verify_files


def formal_settings():
    from trial_lora import SETTINGS
    return {**copy.deepcopy(SETTINGS), "accumulation_steps": 8, "optimizer_steps": 64, "max_length": 512,
            "epoch_count": 1, "sample_order": "frozen recovery file order, no shuffle",
            "checkpoint_steps": [16, 32, 64],
            "loss_normalization": "sum supervised-token NLL / sum supervised tokens within accumulation group",
            "optimizer": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "lr_schedule": "constant"}


def selection_policy():
    return {
        "candidate_steps": [0, 16, 32, 64],
        "numeric_gates_relative_to_new_original_step0": {
            "max_f1_drop_percentage_points": 5, "max_ppl_relative_increase": 0.1},
        "ordering": ["highest validation F1 among numerically passing candidates",
                     "lowest PPL when F1 exactly tied",
                     "earliest training step when both exactly tied"],
        "apply_separately_to_each_arm": True, "step0_is_no_training_fallback": True,
        "qualitative_review": "Inspect proposed choice and changed answers for widespread repetition/question copying and factual regressions; assistant review is not independent human acceptance. Keep all reports, do not change references or thresholds.",
        "if_no_trained_candidate_passes": "Keep step0 if it passes; report recovery failed gates. Do not extend budget or relax thresholds.",
        "comparison": "Report all checkpoints and matched-step arm differences, not only separately best scores.",
        "final_split": "Never used for training, checkpoint choice or debugging.",
    }


def _bound_base(directory, revision, kind, width):
    directory = Path(directory)
    expected_manifest = "download_manifest.json" if kind == "original" else "pruning_manifest.json"
    manifests = {name for name in ("download_manifest.json", "pruning_manifest.json", "merge_manifest.json", "adapter_config.json")
                 if (directory / name).exists()}
    if manifests != {expected_manifest}:
        raise ValueError(f"Expected an unambiguous unrecovered {kind} base")
    provenance, pruning = model_provenance(directory, revision)
    if provenance.get("checkpoint_kind") != kind or provenance.get("recovery_applied", False):
        raise ValueError("Policy preparation requires original and unrecovered pruned bases")
    config = read_json(directory / "config.json")
    if (config.get("model_type") != "qwen3" or config.get("num_hidden_layers") != 28
            or config.get("hidden_size") != 2048 or config.get("intermediate_size") != width):
        raise ValueError(f"Expected Qwen3-1.7B with MLP width {width}")
    if kind == "pruned" and (provenance.get("method") != "calibrated_rms_mean_abs"
                             or provenance.get("width_before") != 6144 or provenance.get("width_after") != 5888):
        raise ValueError("The fixed calibrated_5888 arm requires the calibrated 6144-to-5888 artifact")
    return {"model_directory": str(directory.resolve()), "base_provenance": provenance,
            "expected_base_width": width}


def prepare_policy(original_model, pruned_model, data_dir, revision):
    """Validate current inputs on CPU and build a fresh policy compatible with the trainer."""
    from transformers import AutoTokenizer

    original_model, pruned_model, data_dir = map(Path, (original_model, pruned_model, data_dir))
    if original_model.resolve() == pruned_model.resolve():
        raise ValueError("Original and pruned model directories must differ")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("A pinned source revision is required")
    arms = {"original": _bound_base(original_model, revision, "original", 6144),
            "calibrated_5888": _bound_base(pruned_model, revision, "pruned", 5888)}
    original_source = arms["original"]["base_provenance"]["download_manifest"]
    pruned_source = arms["calibrated_5888"]["base_provenance"]["source"]
    if (original_source.get("model_id") != "Qwen/Qwen3-1.7B"
            or pruned_source.get("model_id") != original_source.get("model_id")
            or original_source.get("revision") != revision or pruned_source.get("revision") != revision):
        raise ValueError("The two bases must share the pinned Qwen3-1.7B source")
    source_files = pruned_source.get("files")
    expected_original = arms["original"]["base_provenance"]["files_sha256"]
    if (pruned_source.get("download_manifest_sha256") != arms["original"]["base_provenance"]["manifest_sha256"]
            or not isinstance(source_files, list) or not source_files):
        raise ValueError("Pruned source requires the original file SHA256 inventory")
    source_hashes = {item["name"]: item.get("sha256") for item in source_files}
    if (len(source_hashes) != len(source_files)
            or any(source_hashes.get(name) != digest for name, digest in expected_original.items())):
        raise ValueError("Pruned checkpoint was made from different original base files")
    data_manifest = read_json(data_dir / "manifest.json")
    if data_manifest["tokenizer"]["source_model_revision"] != revision:
        raise ValueError("Recovery tokenizer revision differs from the requested source")
    verify_files(original_model, data_manifest["tokenizer"]["files_sha256"],
                 required=("tokenizer.json", "tokenizer_config.json"))
    original_tokenizer = AutoTokenizer.from_pretrained(original_model, local_files_only=True, trust_remote_code=False)
    pruned_tokenizer = AutoTokenizer.from_pretrained(pruned_model, local_files_only=True, trust_remote_code=False)
    compatibility = check_tokenizers(original_tokenizer, pruned_tokenizer)
    _, rows, data = load_recovery_data(data_dir, original_tokenizer)
    if (len(rows) != 512 or data.get("row_count") != 512 or data.get("group_count") != 433
            or sum(row["supervised_token_count"] for row in rows) != 6762
            or data.get("supervised_token_count") != 6762 or data.get("source_model_revision") != revision):
        raise ValueError("This trainer requires the frozen 512-row / 433-group / 6762-target recovery budget")
    return {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "One bounded formal recovery and original-model LoRA control; no hyperparameter search; final locked.",
        "recording_stage": "New user-requested policy preparation before a new formal training run. This is not the historical experiment policy and does not claim earlier preregistration or that existing published results were unknown.",
        "revision": revision, "settings": formal_settings(), "training_data": data, "arms": arms,
        "tokenizer_compatibility": compatibility,
        "preparation": {"runner_sha256": sha256_file(Path(__file__)),
                        "inputs": "Current supplied base/tokenizer files and recovery-only manifest/groups; no historical policy or trial resume.",
                        "supported_scope": "Qwen3-1.7B original width6144 and calibrated width5888; fixed first recovery budget."},
        "initialization": "Fresh corresponding base + newly initialized standard zero-B LoRA; no trial adapter resume. Reset same seed before adapter construction and training.",
        "memory_check": "Each arm loads fresh base and performs one accumulated update on eight longest actual recovery rows; discard all changes. Check finite losses/gradients and base unchanged. No claim of a full 512-token stress test.",
        "selection": selection_policy(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-model", type=Path, required=True)
    parser.add_argument("--pruned-model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite policy: {args.output}")
    if any(args.output.resolve().is_relative_to(path.resolve())
           for path in (args.original_model, args.pruned_model, args.data_dir)):
        raise ValueError("Write the new policy outside model and frozen data directories")
    policy = prepare_policy(args.original_model, args.pruned_model, args.data_dir, args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(policy, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Prepared new policy: {args.output.resolve()}\nSHA256: {sha256_file(args.output)}")


if __name__ == "__main__":
    main()
