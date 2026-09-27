"""Bind an operator's own validation-review notes to a numeric summary.

This records a self-declaration; it does not perform or certify human review,
judge factual correctness, approve model quality, or inspect the final split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def record_review(summary_path, notes_path, reviewer=None):
    """Read supplied validation bindings and notes without inventing observations."""
    summary_path, notes_path = Path(summary_path), Path(notes_path)
    summary, summary_sha = _read(summary_path)
    raw_notes = notes_path.read_bytes()
    notes = raw_notes.decode("utf-8-sig")
    if not notes.strip():
        raise ValueError("Review notes must be nonempty operator-written text after actual review")
    if (summary.get("status") != "completed" or summary.get("split") != "validation"
            or summary.get("selection_status") != "numeric_proposals_only"
            or summary.get("qualitative_review_required") is not True
            or set(summary.get("selected", {})) != {"original", "calibrated_5888"}):
        raise ValueError("Require a completed recovery validation summary with both numeric proposals")
    bindings = {}
    for arm, selection in summary["selected"].items():
        chosen = selection.get("checkpoint")
        if not isinstance(chosen, dict) or chosen.get("numeric_quality_gate_passed") is not True:
            raise ValueError(f"{arm}: no numerically passing checkpoint to review for this release")
        path = Path(chosen["report_path"])
        report, digest = _read(path)
        if digest != chosen.get("report_sha256"):
            raise ValueError(f"{arm}: selected validation report changed after summary")
        if (report.get("status") != "completed" or report.get("split") != "validation"
                or report.get("provenance", {}).get("training_step") != chosen.get("training_step")):
            raise ValueError(f"{arm}: selected report is not the completed validation checkpoint")
        bindings[arm] = {"path": str(path.resolve()), "sha256": digest, "training_step": chosen["training_step"]}
    if reviewer is not None and (not isinstance(reviewer, str) or not reviewer.strip()):
        raise ValueError("Reviewer label must be nonblank when supplied")
    return {
        "schema_version": 1, "status": "completed_operator_review",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": {"type": "operator_self_declared", "label": reviewer.strip() if reviewer else "not provided"},
        "scope": "Operator-supplied validation review notes, recorded by explicit command; no automated factual correctness or independent acceptance certification.",
        "bindings": {"numeric_summary": {"path": str(summary_path.resolve()), "sha256": summary_sha},
                     "notes": {"path": str(notes_path.resolve()), "sha256": hashlib.sha256(raw_notes).hexdigest()},
                     "selected_validation_reports": bindings},
        "notes": notes,
        "operator_review_self_declared": True, "actual_reading_verified": False,
        "factual_correctness_automatically_verified": False, "independent_review_performed": False,
        "human_acceptance_performed": False,
        "limitations": [
            "Nonempty notes and this status do not prove that a person read the answers or that the model passed qualitative acceptance.",
            "The script checks file bindings only; it neither generates review observations nor evaluates their truth.",
            "This record cannot override failed numeric gates, justify reselection using final data, or authorize more training.",
            "A reviewer should describe actual repetition, copying, factual errors, and limitations in their own notes; preserve failed cases.",
        ],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True,
                        help="UTF-8 notes written by the operator after actually reading the validation answers")
    parser.add_argument("--reviewer", help="Optional self-declared reviewer label; no identity verification")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite review: {args.output}")
    result = record_review(args.summary, args.notes, args.reviewer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Recorded operator-supplied notes: {args.output.resolve()}")
    print("Self-declared review record only; factual correctness and human acceptance are not verified.")


if __name__ == "__main__":
    main()
