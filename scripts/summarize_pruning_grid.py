"""Summarize the fixed, unrecovered 2 x 2 pruning validation controls."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from compare_quality import compare


METHODS = ("joint_l2_squared", "calibrated_rms_mean_abs")
WIDTHS = (5888, 5632)
EXPECTED_PAIRS = {(method, width) for method in METHODS for width in WIDTHS}
METHOD_LABELS = {
    "original": "原模型",
    "joint_l2_squared": "联合平方幅值",
    "calibrated_rms_mean_abs": "校准评分",
}


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"Invalid {name}: expected a positive integer")
    return value


def _metric_row(report, method, width, comparison=None):
    params = _positive_integer(report["model"]["unique_parameter_count"], "parameter count")
    nll = report["lm"]["nll"]
    if isinstance(nll, bool) or not isinstance(nll, (int, float)) or not math.isfinite(nll) or nll < 0:
        raise ValueError("Invalid NLL")
    cap = report["qa_output_diagnostics"]["max_new_tokens_stop_count"]
    if type(cap) is not int or not 0 <= cap <= report["qa_metrics"]["count"]:
        raise ValueError("Invalid generation cap count")
    return {
        "id": "original" if method == "original" else f"{method}_{width}",
        "method": method,
        "width": width,
        "params": params,
        "em": report["qa_metrics"]["em"],
        "f1": report["qa_metrics"]["f1"],
        "nll": nll,
        "ppl": report["lm"]["ppl"],
        "cap": cap,
        "parameter_reduction": comparison["parameter_reduction"] if comparison else 0.0,
        "f1_change_percentage_points": -comparison["f1_drop_percentage_points"] if comparison else 0.0,
        "ppl_change": 0.0,  # Filled against the original row below.
        "ppl_relative_change": comparison["ppl_relative_increase"] if comparison else 0.0,
        "numeric_quality_gate_passed": comparison["numeric_quality_gate_passed"] if comparison else None,
    }


def summarize_grid(original, candidates):
    """Validate all four cells before returning a deterministic five-row summary."""
    if len(candidates) != 4:
        raise ValueError("Expected exactly four candidates covering the 2 x 2 grid")
    provenance = original["provenance"]
    if provenance["checkpoint_kind"] != "original":
        raise ValueError("Original report must describe the original checkpoint")
    download = provenance["download_manifest"]
    model_id, revision = download["model_id"], download["revision"]
    if (not isinstance(model_id, str) or not model_id or not isinstance(revision, str)
            or not revision or revision != original["source_model_revision"]):
        raise ValueError("Invalid original source identity or revision")
    if original["model"]["intermediate_size"] != 6144:
        raise ValueError("This fixed grid requires original width 6144")
    if (original["split"] != "validation" or original["qa_metrics"]["count"] != 100
            or original["data_manifest"]["counts"]["validation_qa"]["groups"] != 99
            or original["lm"]["valid_token_count"] != 8192):
        raise ValueError("This grid requires validation: 100 questions, 99 groups, 8192 LM targets")
    original_params = _positive_integer(original["model"]["unique_parameter_count"], "original parameter count")
    by_pair = {}
    for candidate in candidates:
        comparison = compare(original, candidate)
        p = candidate["provenance"]
        if p["checkpoint_kind"] != "pruned" or p["recovery_applied"] is not False:
            raise ValueError("Each candidate must be pruned with recovery_applied=false")
        pair = (p["method"], p["width_after"])
        if pair not in EXPECTED_PAIRS:
            raise ValueError(f"Unsupported pruning method/width pair: {pair}")
        if pair in by_pair:
            raise ValueError(f"Duplicate pruning method/width pair: {pair}")
        if (p["source"]["model_id"] != model_id or p["source"]["revision"] != revision
                or candidate["source_model_revision"] != revision):
            raise ValueError("Candidate source identity/revision differs from original")
        if (p["width_before"] != 6144 or candidate["model"]["intermediate_size"] != p["width_after"]):
            raise ValueError("Candidate width metadata disagrees with the evaluated model")
        params = _positive_integer(candidate["model"]["unique_parameter_count"], "candidate parameter count")
        if (params >= original_params or p["parameters_before"] != original_params
                or p["parameters_after"] != params):
            raise ValueError("Candidate parameter metadata disagrees with the evaluated model")
        if candidate["data_manifest"]["counts"]["validation_qa"]["groups"] != 99:
            raise ValueError("Candidate validation group count differs")
        by_pair[pair] = _metric_row(candidate, pair[0], pair[1], comparison)
    if set(by_pair) != EXPECTED_PAIRS:
        raise ValueError("Missing pruning method/width pair")

    rows = [_metric_row(original, "original", 6144)]
    same_width = []
    for width in WIDTHS:
        magnitude = by_pair[(METHODS[0], width)]
        calibrated = by_pair[(METHODS[1], width)]
        if magnitude["params"] != calibrated["params"]:
            raise ValueError(f"Different actual parameter counts at the same width {width}")
        rows.extend((magnitude, calibrated))
        same_width.append({
            "width": width,
            "params": magnitude["params"],
            "direction": "calibrated_rms_mean_abs minus joint_l2_squared",
            "magnitude_row": magnitude["id"],
            "calibrated_row": calibrated["id"],
            "em_change_percentage_points": calibrated["em"] - magnitude["em"],
            "f1_change_percentage_points": calibrated["f1"] - magnitude["f1"],
            "nll_change": calibrated["nll"] - magnitude["nll"],
            "ppl_change": calibrated["ppl"] - magnitude["ppl"],
            "ppl_relative_change": calibrated["ppl"] / magnitude["ppl"] - 1,
            "cap_change": calibrated["cap"] - magnitude["cap"],
        })
    for row in rows:
        row["ppl_change"] = row["ppl"] - rows[0]["ppl"]
    eligible = sorted(
        (r for r in rows if r["method"] == METHODS[1] and r["numeric_quality_gate_passed"]),
        key=lambda r: (r["params"], r["width"]),
    )
    return {
        "schema_version": 1,
        "status": "completed",
        "stage": "unrecovered_pruning_control_grid",
        "source": {"model_id": model_id, "revision": revision},
        "evaluation": {"split": "validation", "question_count": 100, "qa_document_groups": 99,
                       "lm_valid_tokens": 8192},
        "protocol": original["protocol"],
        "rows": rows,
        "same_width_comparisons": same_width,
        "qualitative_review_candidates": [
            {"row_id": r["id"], "method": r["method"], "width": r["width"],
             "params": r["params"], "parameter_reduction": r["parameter_reduction"],
             "numeric_quality_gate_passed": True, "human_review_required": True}
            for r in eligible
        ],
        "selection_rule": "Only numerically passing calibrated candidates; prefer greater parameter compression; human review still required",
        "recovery_applied": False,
        "final_evaluation_used_for_selection": False,
        "statistical_significance_tested": False,
    }


def render_markdown(summary):
    lines = ["# 未恢复剪枝：两种评分 × 两种宽度", "",
             "固定验证：100 题（99 个资料组），LM 共 8192 个有效目标 token。以下五行采用相同评测协议。", "",
             "| 方法 | 宽度 | 参数量 | 参数减少 | EM ↑ | F1 ↑ | NLL ↓ | PPL ↓ | cap | ΔF1（百分点） | ΔPPL | 数值门槛 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in summary["rows"]:
        gate = "基准" if row["method"] == "original" else ("通过" if row["numeric_quality_gate_passed"] else "未通过")
        lines.append(
            f"| {METHOD_LABELS[row['method']]} | {row['width']} | {row['params']:,} | {row['parameter_reduction']:.2%} "
            f"| {row['em']:.2f} | {row['f1']:.2f} | {row['nll']:.4f} | {row['ppl']:.4f} | {row['cap']} "
            f"| {row['f1_change_percentage_points']:+.2f} | {row['ppl_relative_change']:+.2%} | {gate} |"
        )
    lines += ["", "ΔF1 与 ΔPPL 均相对原模型；ΔPPL 为相对变化。cap 是达到生成 token 上限的题数。", "",
              "同宽度比较（校准评分减去联合平方幅值；两种方法实际参数量相同）：", "",
              "| 宽度 | ΔEM（百分点） | ΔF1（百分点） | ΔNLL | ΔPPL（绝对值） | Δcap |",
              "|---|---:|---:|---:|---:|---:|"]
    for diff in summary["same_width_comparisons"]:
        lines.append(f"| {diff['width']} | {diff['em_change_percentage_points']:+.2f} | {diff['f1_change_percentage_points']:+.2f} "
                     f"| {diff['nll_change']:+.4f} | {diff['ppl_change']:+.4f} | {diff['cap_change']:+d} |")
    gates = summary["protocol"]["acceptance"]
    lines += ["", f"数值门槛为 F1 下降不超过 {gates['max_f1_drop_percentage_points']:g} 个百分点、PPL 相对增加不超过 {gates['max_ppl_relative_increase']:.0%}。", "",
              "可进入定性复核的校准候选（优先参数压缩更多者）：", ""]
    if summary["qualitative_review_candidates"]:
        for row in summary["qualitative_review_candidates"]:
            lines.append(f"- 宽度 {row['width']}，参数减少 {row['parameter_reduction']:.2%}；仅数值合格，仍需复核原始回答。")
    else:
        lines.append("没有校准候选通过数值门槛。")
    lines += ["", "这些候选均未做恢复训练，最终集未用于本轮筛选。数字门槛不等于人工验收；本汇总不会自动宣布任何候选已通过人工验收。", "",
              "这里只比较固定的短文本验证子集，不代表全量 CMRC 或通用模型能力；同一验证集用于选择配置，差异不构成统计显著性结论。", "",
              "幅值方法是本项目自定义的 gate/up/down 联合权重平方评分基线；结果不能概括所有幅值剪枝。校准方法是 Qwen3 MLP-only 评分适配。", "",
              "本报告不包含恢复效果、最终集成绩或受控速度测试，因此不据此宣称实际部署加速。"]
    return "\n".join(lines) + "\n"


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def _read_report(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8"), parse_constant=_reject_constant), {
        "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
    }


def write_summary(original_path, candidate_paths, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    original, original_source = _read_report(original_path)
    loaded_candidates = [_read_report(path) for path in candidate_paths]
    summary = summarize_grid(original, [report for report, _ in loaded_candidates])
    sources = {"original": original_source}
    for report, source in loaded_candidates:
        p = report["provenance"]
        sources[f"{p['method']}_{p['width_after']}"] = source
    summary["inputs"] = [{"row_id": row["id"], **sources[row["id"]]} for row in summary["rows"]]
    summary["summarizer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    markdown = render_markdown(summary)
    serialized = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "pruning_grid.json").write_text(serialized, encoding="utf-8")
    (output_dir / "pruning_grid.md").write_text(markdown, encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--candidate", required=True, action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = write_summary(args.original, args.candidate, args.output_dir)
    print(json.dumps({"output_directory": str(args.output_dir.resolve()),
                      "rows": len(summary["rows"]),
                      "qualitative_review_candidates": summary["qualitative_review_candidates"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
