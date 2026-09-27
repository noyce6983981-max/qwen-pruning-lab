"""Bind four deployment artifacts and evaluation code before touching final data."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from merged_checkpoint import merged_model_provenance
from run_quality import model_provenance, read_json, sha256_file
from run_release_quality import MODEL_IDS, MODEL_IDENTITIES, PRECISION_POLICY


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("original-model", "pruned-model", "original-merged", "pruned-merged", "data-dir", "selection", "review", "output"):
        parser.add_argument("--"+argument, required=True, type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = read_json(args.selection)
    review = read_json(args.review)
    if (summary.get("status") != "completed" or review.get("status") not in
            {"completed_assistant_review", "completed_operator_review"}
            or review["bindings"]["numeric_summary"]["sha256"] != sha256_file(args.selection)):
        raise ValueError("Require the completed, bound selection and validation review")
    if review["status"] == "completed_operator_review":
        notes = review.get("notes")
        reviewer = review.get("reviewer")
        if (review.get("operator_review_self_declared") is not True
                or not isinstance(notes, str) or not notes.strip()
                or not isinstance(reviewer, dict) or reviewer.get("type") != "operator_self_declared"
                or review.get("actual_reading_verified") is not False
                or review.get("factual_correctness_automatically_verified") is not False
                or review.get("independent_review_performed") is not False
                or review.get("human_acceptance_performed") is not False):
            raise ValueError("Operator notes must remain an explicit, unverified self-declaration")
    for arm in ("original", "calibrated_5888"):
        chosen = summary["selected"][arm]["checkpoint"]
        if chosen["training_step"] != 64 or not chosen["numeric_quality_gate_passed"]:
            raise ValueError("This release recipe requires the two selected step64 checkpoints")
        if sha256_file(Path(chosen["report_path"])) != chosen["report_sha256"]:
            raise ValueError("Selected validation report changed after summary/review")
    directories = [args.original_model, args.pruned_model, args.original_merged, args.pruned_merged]
    models = {}
    for index, (label, directory) in enumerate(zip(MODEL_IDS, directories)):
        provenance, _ = (merged_model_provenance if index >= 2 else model_provenance)(directory, args.revision)
        actual_identity = (provenance["checkpoint_kind"], provenance.get("base_checkpoint_kind", provenance["checkpoint_kind"]),
                           provenance.get("training_step", 0))
        if actual_identity != MODEL_IDENTITIES[label]:
            raise ValueError("Model arguments must match the fixed original/pruned arm labels")
        if index < 2:
            arm = ("original", "calibrated_5888")[index]
            identity = {key: provenance[key] for key in ("checkpoint_kind", "manifest_sha256", "files_sha256")}
            if identity != summary["base_identities"][arm]:
                raise ValueError("Release baseline differs from the reviewed selection baseline")
        if index >= 2:
            expected_arm = ("original", "calibrated_5888")[index - 2]
            selected = read_json(Path(summary["selected"][expected_arm]["checkpoint"]["report_path"]))
            if provenance["merge_manifest"]["source_adapter"]["manifest_sha256"] != selected["provenance"]["manifest_sha256"]:
                raise ValueError("Merged export is not the selected adapter")
        models[label] = {"directory": str(directory.resolve()), "checkpoint_kind": provenance["checkpoint_kind"],
                         "manifest_sha256": provenance["manifest_sha256"], "files_sha256": provenance["files_sha256"]}
    manifest = read_json(args.data_dir / "manifest.json")
    result = {"schema_version": 1, "status": "frozen", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "revision": args.revision, "models": models, "data_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
              "selection_sha256": sha256_file(args.selection), "validation_review_sha256": sha256_file(args.review),
              "validation_review_status": review["status"],
              "precision_policy": PRECISION_POLICY, "acceptance": manifest["protocol"]["acceptance"],
              "final_selection_allowed": False,
              "evaluation_source_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in
                ("run_release_quality.py", "merged_checkpoint.py", "run_quality.py", "run_recovery_quality.py", "cmrc_official.py", "eval_ppl.py")},
              "design": "Evaluate validation then final for the four fixed artifacts. No reselection, more training, or relaxed gates after final.",
              "interpretation": "Final assesses fixed local pipelines, not official hidden-test or universal capability. Report failed gates honestly. Performance checks only for numerically qualifying artifacts.",
              "human_acceptance_performed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
