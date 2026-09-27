"""Compare two completed quality reports only when the evaluation protocol agrees."""
import argparse
import json
import math
from pathlib import Path


def compare(original, candidate):
    for report in (original, candidate):
        if report["status"] != "completed":
            raise ValueError("Both evaluations must be complete")
        metrics = report["qa_metrics"]
        ids = [row["id"] for row in metrics["per_item"]]
        qa_ids = [row["id"] for row in report["qa"]]
        if (len(ids) != len(set(ids)) or len(qa_ids) != len(set(qa_ids))
                or set(ids) != set(qa_ids) or len(ids) != metrics["count"]
                or metrics["count"] != report["data"]["qa_count"] or metrics["skipped"] != 0):
            raise ValueError("Invalid or incomplete scored question coverage")
        if any(not math.isfinite(metrics[k]) or not 0 <= metrics[k] <= 100 for k in ("em", "f1")):
            raise ValueError("Invalid QA metrics")
        lm = report["lm"]
        if (lm["ppl"] is None or not math.isfinite(lm["ppl"]) or lm["ppl"] <= 0
                or type(lm["valid_token_count"]) is not int or lm["valid_token_count"] <= 0):
            raise ValueError("Invalid PPL or target count")
    if original["provenance"]["checkpoint_kind"] != "original":
        raise ValueError("The original report must describe an unpruned source checkpoint")
    for key in ("source_model_revision", "split", "protocol", "generation_config", "decode", "runner_sha256", "helper_sha256", "scoring_provenance"):
        if original[key] != candidate[key]:
            raise ValueError(f"Incompatible evaluation: {key}")
    for key in ("manifest_sha256", "files_sha256", "qa_file", "lm_file"):
        if original["data"][key] != candidate["data"][key]:
            raise ValueError(f"Different evaluation data: {key}")
    if original["tokenizer_compatibility"]["signature_sha256"] != candidate["tokenizer_compatibility"]["signature_sha256"]:
        raise ValueError("Different tokenizer signatures")
    for key in ("selected_dtype", "attention_implementation", "packages"):
        if original["environment"][key] != candidate["environment"][key]:
            raise ValueError(f"Different evaluation environment: {key}")
    if original["lm"]["valid_token_count"] != candidate["lm"]["valid_token_count"]:
        raise ValueError("Different PPL denominators")
    if original["lm"]["ppl"] is None or candidate["lm"]["ppl"] is None:
        raise ValueError("Cannot compute ratio for overflowing PPL")
    f1_drop = original["qa_metrics"]["f1"] - candidate["qa_metrics"]["f1"]
    ppl_increase = candidate["lm"]["ppl"] / original["lm"]["ppl"] - 1
    gates = original["protocol"]["acceptance"]
    original_scores = {r["id"]: r for r in original["qa_metrics"]["per_item"]}
    candidate_scores = {r["id"]: r for r in candidate["qa_metrics"]["per_item"]}
    if original_scores.keys() != candidate_scores.keys():
        raise ValueError("Different scored question IDs")
    return {
        "split": original["split"], "question_count": original["qa_metrics"]["count"],
        "qa_document_groups": original["data_manifest"]["counts"][original["split"] + "_qa"]["groups"],
        "lm_valid_tokens": original["lm"]["valid_token_count"],
        "parameter_reduction": 1 - candidate["model"]["unique_parameter_count"] / original["model"]["unique_parameter_count"],
        "f1_drop_percentage_points": f1_drop, "ppl_relative_increase": ppl_increase,
        "numeric_quality_gate_passed": f1_drop <= gates["max_f1_drop_percentage_points"] + 1e-12 and ppl_increase <= gates["max_ppl_relative_increase"] + 1e-12,
        "human_review_required": True,
        "per_item": [{"id": qid, "original_em": original_scores[qid]["em"], "candidate_em": candidate_scores[qid]["em"],
                      "original_f1": original_scores[qid]["f1"], "candidate_f1": candidate_scores[qid]["f1"]} for qid in original_scores],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    original, candidate = [json.loads(p.read_text(encoding="utf-8")) for p in (args.original, args.candidate)]
    result = compare(original, candidate)
    gates = original["protocol"]["acceptance"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result["reports"] = {"original": str(args.original.resolve()), "candidate": str(args.candidate.resolve())}
    (args.output_dir / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 固定质量验证：原模型与候选模型", "",
             f"数据：{result['question_count']} 题（{result['qa_document_groups']} 个资料组），独立文本 {result['lm_valid_tokens']:,} 个有效目标 token。",
             "", "| 指标 | 原模型 | 候选模型 |", "|---|---:|---:|"]
    for label, values in [
        ("参数量", [f"{r['model']['unique_parameter_count']:,}" for r in (original, candidate)]),
        ("EM ↑", [f"{r['qa_metrics']['em']:.2f}" for r in (original, candidate)]),
        ("F1 ↑", [f"{r['qa_metrics']['f1']:.2f}" for r in (original, candidate)]),
        ("NLL ↓", [f"{r['lm']['nll']:.4f}" for r in (original, candidate)]),
        ("PPL ↓", [f"{r['lm']['ppl']:.4f}" for r in (original, candidate)]),
        ("达到生成上限的题数", [str(r['qa_output_diagnostics']['max_new_tokens_stop_count']) for r in (original, candidate)]),
    ]:
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    lines += ["", f"参数减少 {result['parameter_reduction']:.2%}；F1 下降 {result['f1_drop_percentage_points']:.2f} 个百分点；PPL 相对增加 {result['ppl_relative_increase']:.2%}。",
              "", f"数值门槛：{'通过' if result['numeric_quality_gate_passed'] else '未通过'}（F1 下降 ≤{gates['max_f1_drop_percentage_points']:g} 个百分点，PPL 增加 ≤{gates['max_ppl_relative_increase']:.0%}）。仍需人工审查原始回答；这是工程筛选规则，不是统计显著性检验。",
              "", "结果只适用于已固定的短文本子集；最终 200 题和最终 LM 语料尚未评测。该对照只判断报告中记录的候选配置，不代表其他剪枝方法的效果。",
              "", "同一 tokenizer、固定提示 token、非思考贪心、BF16、SDPA、batch=1。正式 CMRC EM/F1；PPL 窗口 512、步长 256，按有效 token 加权。耗时不用于声称稳定加速。"]
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("per_item", "reports")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
