"""Score user-supplied short QA answers, with explicit missing-answer coverage.

Uses only Python's standard library. Character F1 measures textual overlap; it
does not judge factual correctness and is not the official CMRC scoring script.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import unicodedata


def normalize_answer(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("答案必须是字符串")
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))


def answer_scores(prediction: str, answers: list[str]) -> dict:
    if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not normalize_answer(a) for a in answers):
        raise ValueError("answers 必须是非空的参考答案列表，每条答案规范化后也必须非空")
    predicted = normalize_answer(prediction)
    exact_match = 0
    best_f1 = 0.0
    for reference in answers:
        expected = normalize_answer(reference)
        exact_match = max(exact_match, int(predicted == expected))
        overlap = sum((Counter(predicted) & Counter(expected)).values())
        f1 = 2 * overlap / (len(predicted) + len(expected))
        best_f1 = max(best_f1, f1)
    return {"exact_match": exact_match, "char_f1": best_f1}


def _indexed(records: list[dict], gold: bool) -> dict:
    if not isinstance(records, list):
        raise ValueError("记录必须是列表")
    result = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("每条记录必须是 JSON 对象")
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("每条记录需要非空字符串 id")
        if identifier in result:
            raise ValueError(f"重复 id：{identifier}")
        if gold:
            if not isinstance(record.get("question"), str) or not record["question"].strip():
                raise ValueError(f"{identifier} 缺少非空 question")
            answer_scores("", record.get("answers"))
        elif not isinstance(record.get("prediction"), str):
            raise ValueError(f"{identifier} 的 prediction 必须是字符串")
        result[identifier] = record
    return result


def score_records(gold: list[dict], predictions: list[dict]) -> dict:
    references = _indexed(gold, gold=True)
    predicted = _indexed(predictions, gold=False)
    if not references:
        raise ValueError("题库不能为空")
    extra = set(predicted) - set(references)
    if extra:
        raise ValueError("预测包含题库中不存在的 id：" + ", ".join(sorted(extra)))
    rows = []
    for identifier, question in references.items():
        missing = identifier not in predicted
        prediction = "" if missing else predicted[identifier]["prediction"]
        rows.append({
            "id": identifier, "question": question["question"],
            "answers": question["answers"], "prediction": prediction,
            "missing_prediction": missing,
            **answer_scores(prediction, question["answers"]),
        })
    return {
        "schema_version": 1, "count": len(rows), "predicted_count": len(predicted),
        "exact_match": sum(row["exact_match"] for row in rows) / len(rows),
        "char_f1": sum(row["char_f1"] for row in rows) / len(rows),
        "rows": rows,
        "definition": "NFKC + casefold, remove whitespace and Unicode punctuation; maximum normalized exact match and character-multiset F1 over references. Macro mean over ALL questions; absent predictions score zero.",
        "limitation": "Text overlap, not factual correctness. Character F1 is order-insensitive and may be high for a wrong date or a negated answer. This is not the official CMRC metric.",
    }


def read_jsonl(path: Path) -> list[dict]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(f"{path.name} 第 {number} 行不是有效 JSON") from exc
    return records


def _cell(value):
    return html.escape(str(value)).replace("|", "\\|").replace("\n", "<br>")


def report_markdown(report: dict) -> str:
    lines = [
        "# 短答案文本评分", "",
        f"题目数：{report['count']}；收到回答：{report['predicted_count']}。缺失回答按 0 分计入全部题目的平均值。", "",
        f"规范化完全匹配率：{report['exact_match']:.1%}；字符 F1 均值：{report['char_f1']:.1%}。", "",
        "这些指标只衡量与参考答案的文字匹配。错误日期或带否定的错误答案也可能得到较高 F1；长解释、正确的不同表述也可能得分较低。请结合逐题原文检查，不将字符 F1 当成事实正确率。", "",
        "| ID | 问题 | 参考答案 | 实际回答 | 缺失 | 完全匹配 | 字符 F1 |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in report["rows"]:
        values = [row["id"], row["question"], " / ".join(row["answers"]), row["prediction"], "是" if row["missing_prediction"] else "否", row["exact_match"], f"{row['char_f1']:.1%}"]
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    lines += ["", "规范化规则：Unicode NFKC、忽略大小写、去掉空白和 Unicode 标点。每题多个参考答案取最佳匹配。字符 F1 按重复字符数量计算，不考虑字符顺序；这不是 CMRC 官方评测脚本。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="写入新目录，保留旧结果")
    args = parser.parse_args()
    if args.output_dir and args.output_dir.exists():
        parser.error("输出目录已存在，请使用一个新目录名")
    try:
        result = score_records(read_jsonl(args.questions), read_jsonl(args.predictions))
        result["inputs"] = {
            name: {"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in (("questions", args.questions), ("predictions", args.predictions))
        }
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            (args.output_dir / "scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (args.output_dir / "scores.md").write_text(report_markdown(result), encoding="utf-8")
        print(report_markdown(result))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"评分失败：{exc}\n")


if __name__ == "__main__":
    main()
