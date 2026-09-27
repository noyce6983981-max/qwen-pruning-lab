"""CPU-only post-hoc answer diagnostics bound to all eight frozen release reports.

Uses stored official scores without rescoring, changing answers or choosing a
checkpoint. Generated examples are a review aid, not a completed human review.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import statistics

from summarize_release import MODEL_IDS, SPLIT_BUDGETS, load_release_reports, summarize
from run_quality import read_json, sha256_file


def compact(text):
    return re.sub(r"\s+", "", text)


def strict_repeat(answer):
    """Exact pre-recovery heuristic, not a new acceptance metric."""
    text = compact(answer)
    for start in range(len(text)):
        for width in range(1, 17):
            unit = text[start:start + width]
            if len(unit) != width:
                continue
            count = 1
            while text[start + count * width:start + (count + 1) * width] == unit:
                count += 1
            if (width == 1 and count >= 8) or (width >= 2 and count >= 3 and width * count >= 12):
                return {"start_without_whitespace": start, "unit": unit, "count": count}
    return None


def diagnostic(report):
    qa = report["qa"]
    repeats = [{"position": i, "id": row["id"], **match}
               for i, row in enumerate(qa, 1) if (match := strict_repeat(row["answer"])) is not None]
    echo = [row["id"] for row in qa if compact(row["answer"]) == compact(row["question"])]
    cap = [row["id"] for row in qa if row["stop_reason"] == "max_new_tokens"]
    empty = [row["id"] for row in qa if not row["answer"].strip()]
    return {"count": len(qa), "em": report["qa_metrics"]["em"], "f1": report["qa_metrics"]["f1"],
            "ppl": report["lm"]["ppl"], "strict_continuous_repetition_count": len(repeats), "repetitions": repeats,
            "exact_question_echo_count": len(echo), "exact_question_echo_ids": echo,
            "cap_count": len(cap), "cap_ids": cap, "empty_count": len(empty), "empty_ids": empty,
            "mean_answer_characters": statistics.mean(len(compact(q["answer"])) for q in qa),
            "median_answer_characters": statistics.median(len(compact(q["answer"])) for q in qa),
            "mean_generated_tokens_including_eos": statistics.mean(q["new_tokens"] for q in qa)}


def pair(before, after):
    if ([q["id"] for q in before["qa"]] != [q["id"] for q in after["qa"]]
            or before["split"] != after["split"]):
        raise ValueError("Pair must retain exact split and frozen question order")
    old_scores = {q["id"]: q for q in before["qa_metrics"]["per_item"]}
    new_scores = {q["id"]: q for q in after["qa_metrics"]["per_item"]}
    items = []
    for index, (old, new) in enumerate(zip(before["qa"], after["qa"]), 1):
        a, b = old_scores[old["id"]], new_scores[new["id"]]
        items.append({"position": index, "id": old["id"], "question": old["question"],
            "references": old["answers"], "before_answer": old["answer"], "after_answer": new["answer"],
            "before_em": a["em"], "after_em": b["em"], "before_f1": a["f1"], "after_f1": b["f1"],
            "f1_delta": b["f1"] - a["f1"], "before_stop_reason": old["stop_reason"],
            "after_stop_reason": new["stop_reason"]})
    counts = {"f1_improved": sum(q["f1_delta"] > 0 for q in items),
              "f1_declined": sum(q["f1_delta"] < 0 for q in items),
              "f1_unchanged": sum(q["f1_delta"] == 0 for q in items),
              "em_retained": sum(q["before_em"] == q["after_em"] == 100 for q in items),
              "em_lost": sum(q["before_em"] == 100 and q["after_em"] == 0 for q in items),
              "em_gained": sum(q["before_em"] == 0 and q["after_em"] == 100 for q in items),
              "answers_unchanged": sum(q["before_answer"] == q["after_answer"] for q in items)}
    examples = {
        "first_three_em_gains": [q for q in items if q["after_em"] > q["before_em"]][:3],
        "first_three_em_losses": [q for q in items if q["after_em"] < q["before_em"]][:3],
        "first_three_f1_improvements": [q for q in items if q["f1_delta"] > 0][:3],
        "first_three_f1_declines": [q for q in items if q["f1_delta"] < 0][:3],
        "all_candidate_caps": [q for q in items if q["after_stop_reason"] == "max_new_tokens"],
        "first_three_dates": [q for q in items if re.search(r"何时|什么时候|哪年|哪一年", q["question"])
            and any(re.search(r"\d{3,4}年|(?:公元)?\d+世纪", a) for a in q["references"])][:3],
        "first_three_numbers": [q for q in items if re.search(r"多少|第几", q["question"])
            and any(re.search(r"\d", a) for a in q["references"])][:3],
        "first_two_negations": [q for q in items if re.search(r"是否|没有|不能|不具|并非|未能|不属于", q["question"])
            or any(re.match(r"不|无|未|非", a) for a in q["references"])][:2],
    }
    return {"split": before["split"], "before_model_id": before["model_id"], "after_model_id": after["model_id"],
            "counts": counts, "examples": examples, "per_item": items}


def build(summary_path):
    numeric = read_json(summary_path)
    if (numeric.get("status") != "completed" or numeric.get("physical_artifacts_reverified") is not True
            or numeric.get("checkpoint_reselected") is not False or numeric.get("final_selection_allowed") is not False):
        raise ValueError("Require completed fixed-artifact numeric summary with physical verification")
    policy_path = Path(numeric["policy"]["path"])
    if sha256_file(policy_path) != numeric["policy"]["sha256"]:
        raise ValueError("Bound release policy changed")
    paths = [Path(row["path"]) for row in numeric["reports"]]
    if any(sha256_file(path) != source["sha256"] for path, source in zip(paths, numeric["reports"])):
        raise ValueError("A bound quality report changed")
    # Physical weights were verified by the bound numeric summary. Recheck every
    # report/protocol and numeric result here without rereading ~14GB of weights.
    recomputed = summarize(policy_path, paths, verify_artifacts=False)
    comparable = dict(numeric)
    comparable["physical_artifacts_reverified"] = False
    if recomputed != comparable:
        raise ValueError("Numeric summary is inconsistent with the eight source reports")
    _, reports, _ = load_release_reports(policy_path, paths, verify_artifacts=False)
    pairs = [("original_step0", name) for name in MODEL_IDS if name != "original_step0"]
    pairs += [("original_step64_merged", "calibrated_5888_step64_merged"),
              ("calibrated_5888_step0", "calibrated_5888_step64_merged")]
    paired = {f"{split}:{before}__to__{after}": pair(reports[split, before], reports[split, after])
              for split in SPLIT_BUDGETS for before, after in pairs}
    final_rows = []
    scores = {label: {q["id"]: q for q in reports["final", label]["qa_metrics"]["per_item"]} for label in MODEL_IDS}
    for index in range(200):
        reference = reports["final", "original_step0"]["qa"][index]
        final_rows.append({"position": index + 1, "id": reference["id"], "context": reference["context"],
                          "question": reference["question"], "references": reference["answers"],
                          "models": {label: {"answer": reports["final", label]["qa"][index]["answer"],
                            "stop_reason": reports["final", label]["qa"][index]["stop_reason"],
                            "new_tokens": reports["final", label]["qa"][index]["new_tokens"],
                            "em": scores[label][reference["id"]]["em"], "f1": scores[label][reference["id"]]["f1"]}
                            for label in MODEL_IDS}})
    original_count = reports["final", "original_step64_merged"]["model"]["unique_parameter_count"]
    pruned_count = reports["final", "calibrated_5888_step64_merged"]["model"]["unique_parameter_count"]
    return {"schema_version": 1, "status": "diagnostics_completed_assistant_review_pending",
            "summary": {"path": str(summary_path.resolve()), "sha256": sha256_file(summary_path)},
            "policy": numeric["policy"], "reports": numeric["reports"],
            "helper_sha256": sha256_file(Path(__file__)), "human_acceptance_performed": False,
            "human_review_required": True, "checkpoint_reselected": False, "final_selection_allowed": False,
            "method": {"scores": "Frozen official CMRC per-item scores; no rescoring or answer repair.",
                "strict_repeat": "Remove whitespace; first contiguous unit 1..16 chars; unit=1 repeats>=8; unit>=2 repeats>=3 and total>=12; one hit per answer.",
                "exact_echo": "Answer equals question after whitespace removal, punctuation preserved.",
                "cap": "Stored stop_reason == max_new_tokens; not itself proof of repetition or factual error.",
                "examples": "Post-hoc diagnostic in frozen question order; first 3 EM gains/losses and F1 gains/losses, first 3 dates/numbers, first 2 negations; all caps retained.",
                "scope": "Diagnostics are not new quality gates, statistical tests or user acceptance. Manual semantics must be added separately."},
            "diagnostics": {f"{split}_{label}": diagnostic(report) for (split, label), report in reports.items()},
            "pairs": paired, "final_all_200": final_rows,
            "parameter_efficiency": {"same_training_steps": 64, "original_parameters": original_count,
                "pruned_parameters": pruned_count, "parameters_removed": original_count - pruned_count,
                "parameter_reduction": 1 - pruned_count / original_count,
                "matched_budget_differences": numeric["matched_budget_differences"],
                "interpretation": "Compare quality at equal 64-step training budget and report structural parameter savings; neither F1/parameter ratios nor fewer parameters prove general superiority or stable speed. Wall time/memory require the separate controlled benchmark."}}


def markdown(result):
    lines = ["# 发布答案诊断：待助手逐题复核", "", "本文件由CPU诊断脚本生成，复用冻结官方分数；尚未完成语义逐题复核，也不是用户人工验收。", "",
             "| split/产物 | EM | F1 | 连续重复 | 原样复述问题 | cap | 空答案 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, d in result["diagnostics"].items():
        lines.append(f"| {name} | {d['em']:.2f} | {d['f1']:.4f} | {d['strict_continuous_repetition_count']} | {d['exact_question_echo_count']} | {d['cap_count']} | {d['empty_count']} |")
    lines += ["", "| final配对 | F1改善/下降/相同 | EM保留/丢失/新增 | 答案原文相同 |", "| --- | --- | --- | --- |"]
    for key, value in result["pairs"].items():
        if value["split"] != "final":
            continue
        c = value["counts"]
        lines.append(f"| {value['before_model_id']} → {value['after_model_id']} | {c['f1_improved']}/{c['f1_declined']}/{c['f1_unchanged']} | {c['em_retained']}/{c['em_lost']}/{c['em_gained']} | {c['answers_unchanged']} |")
    selected = result["pairs"]["final:calibrated_5888_step0__to__calibrated_5888_step64_merged"]
    seen = set()
    lines += ["", "以下为剪后恢复前后、按冻结顺序抽取的首3个EM新增/丢失案例；它们只说明分数变化，事实判断须读原段落。所有200题、4产物原文及其他固定类别案例保存在JSON。", ""]
    for category in ("first_three_em_gains", "first_three_em_losses"):
        for q in selected["examples"][category]:
            if q["id"] in seen:
                continue
            seen.add(q["id"])
            lines += [f"## #{q['position']} `{q['id']}`（{category}）", "", q["question"], "",
                      "冻结参考：" + " / ".join(q["references"]), "", "剪后0：" + q["before_answer"], "",
                      "剪后64合并：" + q["after_answer"], "", f"EM {q['before_em']} → {q['after_em']}；F1 {q['before_f1']:.4f} → {q['after_f1']:.4f}。", ""]
    p = result["parameter_efficiency"]
    lines += [f"同为64步，参数 {p['original_parameters']:,} → {p['pruned_parameters']:,}，减少 {p['parameters_removed']:,}（{p['parameter_reduction']:.2%}）。质量须看同预算对照；这不等于已观察到稳定加速。", "",
              "重复启发式只检查去空白后的连续短单元，不能排除长单元或语义重复。cap表示输出达到64 token上限，可能是冗长解释、截断或错误，不能自动归因于循环。", "",
              "数值选型已在validation阶段冻结；final不重选、不改参考，不声称统计显著。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.summary)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "release_answer_diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    (args.output_dir / "release_answer_diagnostics.md").write_text(markdown(result), encoding="utf8")


if __name__ == "__main__":
    main()
