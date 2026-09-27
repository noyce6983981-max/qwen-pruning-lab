"""Evaluate a local JSONL question set with explicit Qwen3 checkpoint identity.

This independent user-data runner does not read the frozen CMRC evaluation set.
Questions without references retain answers but receive no invented EM/F1 score.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import run_quality as quality


CONTEXT_INSTRUCTION = "仅根据给出的资料回答问题。只输出简短答案，不解释。资料没有提供答案时，回答“资料未提供”。"
QUESTION_INSTRUCTION = "用中文简短回答问题，只输出答案，不解释。不确定时明确回答“不确定”。"
KNOWN_MANIFESTS = {"original": "download_manifest.json", "pruned": "pruning_manifest.json",
                   "merged_lora": "merge_manifest.json"}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_questions(path):
    """Validate the whole file before any model inference; preserve file order."""
    rows, seen = [], set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Question file line {number}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Question file line {number}: expected an object")
        if set(item) - {"id", "context", "question", "answers"}:
            raise ValueError(f"Question file line {number}: unsupported fields {sorted(set(item) - {'id', 'context', 'question', 'answers'})}")
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier != identifier.strip():
            raise ValueError(f"Question file line {number}: id must be a nonblank string without outer whitespace")
        if identifier in seen:
            raise ValueError(f"Duplicate question id: {identifier}")
        seen.add(identifier)
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(f"{identifier}: question must be a nonblank string")
        context = item.get("context", "")
        if not isinstance(context, str):
            raise ValueError(f"{identifier}: context must be a string when present")
        answers = item.get("answers", [])
        if not isinstance(answers, list) or any(not isinstance(answer, str) or not answer.strip() for answer in answers):
            raise ValueError(f"{identifier}: answers must be a list of nonblank strings, or omitted")
        rows.append({"id": identifier, "context": context, "question": item["question"], "answers": answers})
    if not rows:
        raise ValueError("Question file is empty")
    return rows


def generation_protocol(max_input_tokens=448, max_new_tokens=64):
    if (type(max_input_tokens) is not int or type(max_new_tokens) is not int
            or max_input_tokens < 1 or not 1 <= max_new_tokens <= 64
            or max_input_tokens + max_new_tokens > 512):
        raise ValueError("Require positive input budget, 1..64 new tokens, and total budget <=512")
    return {"max_input_tokens": max_input_tokens, "max_new_tokens": max_new_tokens,
            "enable_thinking": False, "do_sample": False, "num_beams": 1, "batch_size": 1,
            "seed": 42, "use_cache": True, "use_model_defaults": False,
            "prompt_truncation": False, "answer_extraction": "strip_only",
            "context_system_prompt": CONTEXT_INSTRUCTION, "no_context_system_prompt": QUESTION_INSTRUCTION,
            "parameter_dtype": "torch.bfloat16", "attention_implementation": "sdpa",
            "autocast": False, "mode": "eval; all parameters frozen; no unmerged adapters"}


def prepare_questions(rows, tokenizer, protocol):
    """Encode the canonical, recorded prompts without truncation or reference leaks."""
    prepared = []
    vocabulary = set(tokenizer.get_vocab().values())
    for item in rows:
        has_context = bool(item["context"].strip())
        messages = [
            {"role": "system", "content": CONTEXT_INSTRUCTION if has_context else QUESTION_INSTRUCTION},
            {"role": "user", "content": f"资料：{item['context']}\n问题：{item['question']}" if has_context else item["question"]},
        ]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if not quality.token_ids_valid(ids) or any(token not in vocabulary for token in ids):
            raise ValueError(f"{item['id']}: tokenizer returned invalid prompt token IDs")
        if len(ids) > protocol["max_input_tokens"]:
            raise ValueError(f"{item['id']}: prompt has {len(ids)} tokens, limit {protocol['max_input_tokens']}; no truncation allowed")
        prepared.append({**item, "messages": messages, "prompt_token_ids": ids,
                         "has_reference": bool(item["answers"]),
                         "context_sha256": hashlib.sha256(item["context"].encode("utf-8")).hexdigest()})
    return prepared


def checkpoint_provenance(directory, revision, checkpoint_kind):
    """Dispatch only an explicitly named identity, never infer a fallback identity."""
    directory = Path(directory)
    if checkpoint_kind not in KNOWN_MANIFESTS:
        raise ValueError("Unsupported checkpoint kind")
    present = {kind for kind, filename in KNOWN_MANIFESTS.items() if (directory / filename).is_file()}
    if present != {checkpoint_kind}:
        raise ValueError(f"Expected only the {checkpoint_kind} manifest; found {sorted(present)}")
    if (directory / "adapter_config.json").exists():
        raise ValueError("This runner accepts a complete base or merged checkpoint, not an unmerged adapter")
    if checkpoint_kind == "merged_lora":
        from merged_checkpoint import merged_model_provenance
        provenance, pruning = merged_model_provenance(directory, revision)
    else:
        provenance, pruning = quality.model_provenance(directory, revision)
    if provenance.get("checkpoint_kind") != checkpoint_kind:
        raise ValueError("Verified checkpoint kind differs from the explicitly requested kind")
    if checkpoint_kind != "merged_lora" and provenance.get("recovery_applied", False):
        raise ValueError("Recovered weights cannot be evaluated as an unrecovered base")
    if checkpoint_kind == "merged_lora" and (provenance.get("adapter_merged") is not True or provenance.get("recovery_applied") is not True):
        raise ValueError("Merged checkpoint requires verified recovery and merge provenance")
    return provenance, pruning


def score_predictions(examples, predictions):
    """Score the exact reference subset, leaving every unreferenced item null."""
    ids = [item["id"] for item in examples]
    prediction_ids = [item.get("id") for item in predictions]
    if (len(set(ids)) != len(ids) or len(set(prediction_ids)) != len(prediction_ids)
            or set(ids) != set(prediction_ids)
            or any(not isinstance(item.get("answer"), str) for item in predictions)):
        raise ValueError("Predictions must cover each question ID exactly once with an answer string")
    with_references = [item for item in examples if item["answers"]]
    answer_map = {item["id"]: item["answer"] for item in predictions}
    scores, provenance = None, None
    if with_references:
        from cmrc_official import score_examples, scoring_provenance
        scores = score_examples(with_references, {item["id"]: answer_map[item["id"]] for item in with_references})
        provenance = scoring_provenance()
    by_id = {item["id"]: item for item in scores["per_item"]} if scores is not None else {}
    return {"status": "completed" if scores is not None else "not_requested_no_references",
            "question_count": len(examples), "scored_count": len(with_references),
            "unscored_count": len(examples) - len(with_references), "metrics": scores,
            "scoring_provenance": provenance,
            "per_item": [{"id": identifier, "scored": identifier in by_id,
                          "em": by_id[identifier]["em"] if identifier in by_id else None,
                          "f1": by_id[identifier]["f1"] if identifier in by_id else None} for identifier in ids]}


def _cell(value):
    return html.escape(str(value), quote=False).replace("|", "&#124;").replace("\r", "").replace("\n", "<br>")


def render_markdown(report):
    lines = ["# 自定义题库评测", "", f"状态：{_cell(report['status'])}。", "",
             "这是用户自定义题库的单次记录，不是固定CMRC验证／最终集成绩，也不包含PPL或稳定测速结论。"]
    if report.get("error"):
        lines += ["", f"失败阶段：{_cell(report.get('stage', 'unknown'))}；{_cell(report['error']['type'])}：{_cell(report['error']['message'])}",
                  "", "本目录保留已生成的逐题答案。重新运行必须使用新输出目录，不能覆盖本次错误记录。"]
    protocol = report["protocol"]
    lines += ["", f"设置：非思考、贪心、BF16、SDPA、batch=1、seed=42，输入上限{protocol['max_input_tokens']} token、输出上限{protocol['max_new_tokens']} token；不截断输入，回答只去首尾空白。"]
    scores = report.get("scoring")
    if scores is not None:
        lines += ["", f"题数{scores['question_count']}，有参考并评分{scores['scored_count']}，无参考不评分{scores['unscored_count']}。"]
        if scores["metrics"] is not None:
            lines += ["", f"有参考子集：EM {scores['metrics']['em']:.2f}，F1 {scores['metrics']['f1']:.2f}（0–100，CMRC官方规则）。",
                      "", "这些分数只覆盖有参考的题目。字面匹配不等于事实正确；开放式问题宜结合原始回答人工检查。"]
        else:
            lines += ["", "本题库没有参考答案，EM/F1为空；不会用0分代替未知分数。"]
    by_id = {item["id"]: item for item in scores["per_item"]} if scores else {}
    lines += ["", "| ID | 问题 | 参考答案 | 模型回答 | EM | F1 | 停止原因 |",
              "|---|---|---|---|---:|---:|---|"]
    for item in report.get("qa", []):
        score = by_id.get(item["id"], {})
        em = f"{score['em']:.2f}" if score.get("em") is not None else "—"
        f1 = f"{score['f1']:.2f}" if score.get("f1") is not None else "—"
        lines.append(f"| {_cell(item['id'])} | {_cell(item['question'])} | {_cell(' / '.join(item['answers'])) if item['answers'] else '未提供'} | {_cell(item['answer'])} | {em} | {f1} | {_cell(item.get('stop_reason', 'unknown'))} |")
    lines += ["", "逐题原始文字、特殊token版本、生成token IDs、提示与参考答案保留在 predictions.jsonl；配置、来源哈希和模型参数量见 questions_report.json。达到输出上限的答案可能不完整。"]
    return "\n".join(lines) + "\n"


def _run_model(args, report, examples, tokenizer, pruning):
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig

    model = None
    device = None
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 is required for real local Qwen3 evaluation")
        device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(device)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        report["environment"].update({"gpu_name": torch.cuda.get_device_name(device),
                                      "selected_dtype": "torch.bfloat16", "attention_implementation": "sdpa",
                                      "autocast_enabled": False})
        config = quality.read_json(args.model_dir / "generation_config.json")
        generation = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                                      use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                      eos_token_id=config["eos_token_id"], bos_token_id=config.get("bos_token_id"))
        if generation.pad_token_id is None:
            generation.pad_token_id = tokenizer.eos_token_id
        report["generation_config"] = generation.to_dict()
        report["stage"] = "model_loading"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", use_safetensors=True,
            low_cpu_mem_usage=True, device_map={"": str(device)}, local_files_only=True, trust_remote_code=False)
        model.requires_grad_(False)
        model.eval()
        report["model"] = quality.inspect_model(model, pruning, str(device))
        if model.config.max_position_embeddings < args.max_input_tokens + args.max_new_tokens:
            raise ValueError("Requested context budget exceeds model capacity")
        if any(".lora_" in name for name, _ in model.named_parameters()):
            raise ValueError("Expected standalone BF16 weights without adapter modules")
        report["stage"] = "generation"
        torch.cuda.reset_peak_memory_stats(device)
        report["qa"] = quality.run_qa(model, tokenizer, examples, generation, device, args.output_dir / "predictions.jsonl")
        report["generation_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["generation_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    finally:
        model = None
        if device is not None:
            gc.collect()
            torch.cuda.empty_cache()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-kind", choices=tuple(KNOWN_MANIFESTS), required=True)
    parser.add_argument("--revision", required=True, help="Pinned original source revision, including for pruned/merged weights")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    protocol = generation_protocol(args.max_input_tokens, args.max_new_tokens)
    if args.output_dir.resolve().is_relative_to(args.model_dir.resolve()):
        raise ValueError("Output directory must be outside the model directory")
    quality.create_output(args.output_dir)
    report = {"schema_version": 1, "status": "starting", "stage": "question_validation",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "split": "custom",
              "scope": "User-supplied questions; not the frozen CMRC evaluation or a benchmark",
              "source_model_revision": args.revision, "model_directory": str(args.model_dir.resolve()),
              "requested_checkpoint_kind": args.checkpoint_kind, "protocol": protocol,
              "runner_sha256": quality.sha256_file(Path(__file__)),
              "helper_sha256": {name: quality.sha256_file(Path(__file__).with_name(name))
                                 for name in ("run_quality.py", "cmrc_official.py")},
              "environment": {"python": platform.python_version(), "packages": quality.package_versions()}, "qa": []}
    started = time.perf_counter()
    try:
        examples = load_questions(args.questions)
        report["data"] = {"questions_path": str(args.questions.resolve()), "questions_sha256": quality.sha256_file(args.questions),
                          "question_count": len(examples), "reference_count": sum(bool(item["answers"]) for item in examples)}
        report["stage"] = "tokenizer_validation"
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
        signature = quality.tokenizer_signature(tokenizer)
        report["tokenizer"] = {"signature_sha256": quality.json_hash(signature), "signature": signature,
                               "files_sha256": {name: quality.sha256_file(args.model_dir / name)
                                                for name in ("tokenizer.json", "tokenizer_config.json")}}
        examples = prepare_questions(examples, tokenizer, protocol)
        report["data"]["encoded_questions_sha256"] = quality.json_hash(examples)
        report["data"]["prompt_token_counts"] = {item["id"]: len(item["prompt_token_ids"]) for item in examples}
        report["stage"] = "checkpoint_validation"
        report["provenance"], pruning = checkpoint_provenance(args.model_dir, args.revision, args.checkpoint_kind)
        if args.checkpoint_kind == "merged_lora":
            report["helper_sha256"]["merged_checkpoint.py"] = quality.sha256_file(Path(__file__).with_name("merged_checkpoint.py"))
        if report["data"]["reference_count"]:
            from cmrc_official import scoring_provenance
            report["scoring_provenance"] = scoring_provenance()
        _run_model(args, report, examples, tokenizer, pruning)
        report["stage"] = "scoring"
        report["scoring"] = score_predictions(examples, report["qa"])
        report["status"] = report["stage"] = "completed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        partial = args.output_dir / "predictions.jsonl"
        if partial.is_file() and partial.stat().st_size:
            report["qa"] = quality.read_jsonl(partial)
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = time.perf_counter() - started
        with (args.output_dir / "questions_report.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        with (args.output_dir / "questions_report.md").open("x", encoding="utf-8") as stream:
            stream.write(render_markdown(report))
        print(f"Report: {args.output_dir / 'questions_report.json'}", flush=True)


if __name__ == "__main__":
    main()
