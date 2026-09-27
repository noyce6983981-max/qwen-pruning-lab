"""CPU-only strict summary of eight frozen release quality reports.

Final is a fixed assessment, never a checkpoint-selection or tuning signal.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import math
from pathlib import Path

from compare_quality import compare
from run_release_quality import (
    MODEL_IDS, MODEL_IDENTITIES, PRECISION_POLICY,
    release_provenance, validate_release_policy,
)
from run_quality import read_json, sha256_file


QA_INPUT_FIELDS = ("id", "source", "document_id", "group_id", "paragraph_id", "context_sha256",
                   "normalized_context_sha256", "context", "question", "answers",
                   "prompt_token_ids", "input_tokens", "recovery_tokens_first_answer")
SPLIT_BUDGETS = {"validation": (100, 8192), "final": (200, 16384)}
ACCEPTANCE = {"max_f1_drop_percentage_points": 5, "max_ppl_relative_increase": 0.1,
              "human_review": "No widespread repeated outputs."}


def report_paths(quality_dir: Path) -> list[Path]:
    return [quality_dir / f"{split}_{label}" / "quality.json"
            for split in SPLIT_BUDGETS for label in MODEL_IDS]


def positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"Invalid {name}")


def validate_report(report, policy, policy_sha):
    if report.get("status") != "completed" or report.get("stage") != "completed":
        raise ValueError("Every release report must be completed")
    label, split = report["model_id"], report["split"]
    if label not in MODEL_IDS or split not in SPLIT_BUDGETS:
        raise ValueError("Unknown release model or split")
    if (report["release_policy_sha256"] != policy_sha
            or report["source_model_revision"] != policy["revision"]
            or report["helper_sha256"] != policy["evaluation_source_sha256"]
            or report["runner_sha256"] != policy["evaluation_source_sha256"]["run_release_quality.py"]
            or report["protocol"]["acceptance"] != policy["acceptance"]
            or report["protocol"]["inference_precision"] != PRECISION_POLICY):
        raise ValueError("Release policy/code/precision/gates binding mismatch")
    if datetime.fromisoformat(report["created_at_utc"]) < datetime.fromisoformat(policy["created_at_utc"]):
        raise ValueError("Report predates the frozen release policy")
    expected = policy["models"][label]
    provenance = report["provenance"]
    if (Path(report["model_directory"]).resolve() != Path(expected["directory"]).resolve()
            or (provenance["checkpoint_kind"], provenance["base_checkpoint_kind"], provenance["training_step"])
            != MODEL_IDENTITIES[label]
            or provenance["manifest_sha256"] != expected["manifest_sha256"]
            or provenance["files_sha256"] != expected["files_sha256"]):
        raise ValueError("Release artifact does not match its frozen slot")
    if report.get("adapter_directory") is not None or report["model"]["adapter_parameter_count"] != 0:
        raise ValueError("Release comparison must use standalone models")
    merged = label.endswith("_merged")
    if (report["model"]["adapter_merged"] is not merged or report["model"]["all_parameters_frozen"] is not True
            or report["model"]["parameter_dtypes"] != ["torch.bfloat16"]
            or report["model"]["attention_implementation"] != "sdpa"):
        raise ValueError("Unexpected release model state")
    if merged and (report["reload_validation"].get("status") != "passed"
                   or report["reload_validation"].get("probe_exact") is not True):
        raise ValueError("Merged export probe has not passed")
    model_count = report["model"]["unique_parameter_count"]
    positive_integer(model_count, "parameter count")
    if report["model"]["base_parameter_count"] != model_count:
        raise ValueError("Standalone parameter accounting mismatch")
    qcount, targets = SPLIT_BUDGETS[split]
    if (report["qa_metrics"]["count"] != qcount or report["data"]["qa_count"] != qcount
            or report["qa_metrics"]["skipped"] != 0 or len(report["qa"]) != qcount
            or report["qa_metrics"]["metric"] != "cmrc2018_official"
            or report["data"]["qa_file"] != f"{split}_qa.jsonl"
            or report["data"]["lm_file"] != f"{split}_lm.jsonl"
            or report["data"]["manifest_sha256"] != policy["data_manifest_sha256"]
            or report["data"]["files_sha256"] != report["data_manifest"]["files_sha256"]):
        raise ValueError("Frozen release QA/data budget differs")
    qa_ids = [item["id"] for item in report["qa"]]
    scores = report["qa_metrics"]["per_item"]
    score_ids = [item["id"] for item in scores]
    if len(qa_ids) != len(set(qa_ids)) or len(score_ids) != len(set(score_ids)) or set(qa_ids) != set(score_ids):
        raise ValueError("Duplicate/missing official per-item scores")
    for metric in ("em", "f1"):
        values = [score[metric] for score in scores]
        if (any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 100 for v in values)
                or not math.isclose(math.fsum(values) / qcount, report["qa_metrics"][metric], abs_tol=1e-10)):
            raise ValueError("Official per-item and aggregate scores disagree")
    lm = report["lm"]
    if (type(lm["valid_token_count"]) is not int or lm["valid_token_count"] != targets
            or lm["ppl_overflow"] is not False or not math.isfinite(lm["nll_sum"])
            or lm["nll_sum"] < 0 or not math.isfinite(lm["nll"]) or lm["nll"] < 0
            or not math.isfinite(lm["ppl"]) or lm["ppl"] <= 0
            or not math.isclose(lm["nll_sum"] / targets, lm["nll"], rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(math.exp(lm["nll"]), lm["ppl"], rel_tol=1e-12)):
        raise ValueError("Invalid fixed-denominator LM metrics")
    documents = lm["documents"]
    if (len(documents) != lm["document_count"] or len(documents) != report["data"]["lm_document_count"]
            or len({doc["id"] for doc in documents}) != len(documents)
            or sum(doc["valid_token_count"] for doc in documents) != targets
            or not math.isclose(math.fsum(doc["nll_sum"] for doc in documents), lm["nll_sum"], rel_tol=1e-12)):
        raise ValueError("LM document accounting mismatch")
    diagnostics = report["qa_output_diagnostics"]
    if (diagnostics["empty_answer_count"] != sum(not q["answer"] for q in report["qa"])
            or diagnostics["max_new_tokens_stop_count"] != sum(q["stop_reason"] == "max_new_tokens" for q in report["qa"])):
        raise ValueError("QA output diagnostics disagree with raw answers")


def load_release_reports(policy_path: Path, paths: list[Path], verify_artifacts: bool = True):
    policy = validate_release_policy(policy_path)
    if type(policy["schema_version"]) is not int or policy["acceptance"] != ACCEPTANCE:
        raise ValueError("Fixed release schema/thresholds changed")
    if len(paths) != 8 or len({p.resolve() for p in paths}) != 8:
        raise ValueError("Require all eight distinct fixed release reports")
    policy_sha = sha256_file(policy_path)
    reports = [read_json(path) for path in paths]
    for report in reports:
        validate_report(report, policy, policy_sha)
    keyed = {(r["split"], r["model_id"]): r for r in reports}
    if set(keyed) != {(s, label) for s in SPLIT_BUDGETS for label in MODEL_IDS}:
        raise ValueError("Duplicate or missing model/split slot")
    shared = ("source_model_revision", "protocol", "generation_config", "decode", "runner_sha256",
              "helper_sha256", "scoring_provenance", "canonical_tokenizer_files_sha256", "environment",
              "data_manifest", "tokenizer_compatibility")
    first = reports[0]
    for report in reports[1:]:
        if any(report[k] != first[k] for k in shared):
            raise ValueError("Release reports differ in frozen protocol/environment/tokenizer/source")
    comparisons = {}
    for split in SPLIT_BUDGETS:
        original = keyed[split, "original_step0"]
        original_inputs = [{key: q[key] for key in QA_INPUT_FIELDS} for q in original["qa"]]
        lm_keys = ("id", "token_count", "valid_token_count", "windows")
        for label in MODEL_IDS:
            candidate = keyed[split, label]
            if ([{key: q[key] for key in QA_INPUT_FIELDS} for q in candidate["qa"]] != original_inputs
                    or [{k: d[k] for k in lm_keys} for d in candidate["lm"]["documents"]]
                    != [{k: d[k] for k in lm_keys} for d in original["lm"]["documents"]]
                    or candidate["lm"]["protocol"] != original["lm"]["protocol"]):
                raise ValueError("QA inputs/references/order or LM document windows differ")
            comparisons[split, label] = compare(original, candidate)
    for label in MODEL_IDS:
        before, after = keyed["validation", label], keyed["final", label]
        if any(before[k] != after[k] for k in ("provenance", "model", "model_directory")):
            raise ValueError("Artifact changed between validation and final")
    validation_groups = {q["group_id"] for q in keyed["validation", "original_step0"]["qa"]}
    final_groups = {q["group_id"] for q in keyed["final", "original_step0"]["qa"]}
    if validation_groups & final_groups:
        raise ValueError("Validation and final QA groups overlap")
    if verify_artifacts:
        manifest_path = Path(first["data"]["directory"]) / "manifest.json"
        if (sha256_file(manifest_path) != policy["data_manifest_sha256"]
                or read_json(manifest_path) != first["data_manifest"]):
            raise ValueError("Current data manifest differs from release policy/report")
        for label in MODEL_IDS:
            expected = policy["models"][label]
            actual, _ = release_provenance(Path(expected["directory"]), policy["revision"], policy_path, label)
            for key in ("checkpoint_kind", "base_checkpoint_kind", "training_step", "manifest_sha256", "files_sha256"):
                if actual[key] != keyed["final", label]["provenance"][key]:
                    raise ValueError("Current physical checkpoint differs from evaluated artifact")
    return policy, keyed, comparisons


def summarize(policy_path: Path, paths: list[Path], verify_artifacts: bool = True) -> dict:
    policy, keyed, comparisons = load_release_reports(policy_path, paths, verify_artifacts)
    rows = []
    for split in SPLIT_BUDGETS:
        for label in MODEL_IDS:
            r, c = keyed[split, label], comparisons[split, label]
            rows.append({
                "split": split, "model_id": label,
                "checkpoint_kind": r["provenance"]["checkpoint_kind"],
                "base_checkpoint_kind": r["provenance"]["base_checkpoint_kind"],
                "training_step": r["provenance"]["training_step"],
                "question_count": r["qa_metrics"]["count"], "valid_lm_tokens": r["lm"]["valid_token_count"],
                "parameters": r["model"]["unique_parameter_count"], "em": r["qa_metrics"]["em"],
                "f1": r["qa_metrics"]["f1"], "nll": r["lm"]["nll"], "ppl": r["lm"]["ppl"],
                "cap_count": r["qa_output_diagnostics"]["max_new_tokens_stop_count"],
                "empty_count": r["qa_output_diagnostics"]["empty_answer_count"],
                **{k: c[k] for k in ("parameter_reduction", "f1_drop_percentage_points",
                                    "ppl_relative_increase", "numeric_quality_gate_passed")},
            })
    matched = []
    for split in SPLIT_BUDGETS:
        for step in (0, 64):
            suffix = "0" if step == 0 else "64_merged"
            original = keyed[split, f"original_step{suffix}"]
            pruned = keyed[split, f"calibrated_5888_step{suffix}"]
            matched.append({
                "split": split, "training_step": step,
                "em_difference_pruned_minus_original": pruned["qa_metrics"]["em"] - original["qa_metrics"]["em"],
                "f1_difference_pruned_minus_original": pruned["qa_metrics"]["f1"] - original["qa_metrics"]["f1"],
                "ppl_relative_increase_pruned_vs_original": pruned["lm"]["ppl"] / original["lm"]["ppl"] - 1,
                "parameter_reduction_pruned_vs_original": 1 - pruned["model"]["unique_parameter_count"] / original["model"]["unique_parameter_count"],
            })
    return {
        "schema_version": 1, "status": "completed", "assessment_status": "numeric_only_assistant_review_required",
        "policy": {"path": str(policy_path.resolve()), "sha256": sha256_file(policy_path)},
        "reports": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in paths],
        "physical_artifacts_reverified": verify_artifacts, "rows": rows,
        "matched_budget_differences": matched,
        "comparisons_to_original_step0": {f"{split}_{label}": value for (split, label), value in comparisons.items()},
        "final_numeric_gate_passed_model_ids": [label for label in MODEL_IDS
                                               if comparisons["final", label]["numeric_quality_gate_passed"]],
        "validation_and_final_numeric_gate_passed_model_ids": [label for label in MODEL_IDS
            if all(comparisons[split, label]["numeric_quality_gate_passed"] for split in SPLIT_BUDGETS)],
        "fixed_artifacts": policy["models"], "final_selection_allowed": False, "checkpoint_reselected": False,
        "human_acceptance_performed": False, "human_review_required": True,
        "limits": [
            "No ranking or checkpoint reselection from final; all four artifacts were frozen before final evaluation.",
            "Gate lists are eligibility for a fixed performance comparison, not selection of a new quality winner.",
            "Local length-filtered Chinese reading-comprehension subset; not official hidden-test or universal capability.",
            "A single training seed and finite final set; no statistical significance claim.",
            "Merged BF16 weights can change answers from live FP32 LoRA due to rounding; use these standalone scores for deployment.",
            "Parameter reduction is structural; actual speed/memory must be measured independently.",
        ],
    }


def markdown(result):
    lines = ["# 固定发布产物：验证与最终质量", "",
             "四个产物在最终评测前冻结；保留全部8份结果，不按final重选检查点。这里只判定数值门槛，仍需助手逐题复核与用户验收。", "",
             "| split | 产物 | 参数 | EM | F1 | PPL | cap | 相对本split原0 F1变化pp | PPL变化 | 数值门槛 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in result["rows"]:
        lines.append(f"| {row['split']} | {row['model_id']} | {row['parameters']:,} | {row['em']:.2f} | "
                     f"{row['f1']:.4f} | {row['ppl']:.4f} | {row['cap_count']} | "
                     f"{-row['f1_drop_percentage_points']:+.4f} | {row['ppl_relative_increase']:+.2%} | "
                     f"{'通过' if row['numeric_quality_gate_passed'] else '未通过'} |")
    lines += ["", "validation固定100题/8192 LM目标；final固定200题/16384 LM目标。门槛仍为F1下降≤5pp、PPL增加≤10%，各split对其original_step0。", "",
              "| split | 同训练步数 | 剪后−原模型 EM pp | F1 pp | PPL变化 | 参数减少 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in result["matched_budget_differences"]:
        lines.append(f"| {row['split']} | {row['training_step']} | {row['em_difference_pruned_minus_original']:+.2f} | "
                     f"{row['f1_difference_pruned_minus_original']:+.4f} | {row['ppl_relative_increase_pruned_vs_original']:+.2%} | "
                     f"{row['parameter_reduction_pruned_vs_original']:.2%} |")
    lines += ["", "最终数值合格名单（不代表人工验收）：" + "、".join(result["final_numeric_gate_passed_model_ids"]), ""]
    lines += ["- " + limit for limit in result["limits"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--quality-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.policy, report_paths(args.quality_dir))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "release_quality_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    (args.output_dir / "release_quality_summary.md").write_text(markdown(result), encoding="utf8")


if __name__ == "__main__":
    main()
