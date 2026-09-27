"""Freeze a document-disjoint, short-context CMRC2018 local evaluation protocol."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.request import urlopen

REVISION = "c0eb1b6ba219847457e6af3180da722bbeb656af"
RAW_HASHES = {
    "train": "5497aa2f81908e31d6b0e27d99b1f90ab63a8f58fa92fffe5d17cf62eba0c212",
    "dev": "e9ff74231f05c230c6fa88b84441ee334d97234cbb610991cd94b82db00c7f1f",
}
SYSTEM = "请仅根据资料回答问题。只输出资料中的最短答案片段，不要解释，不要重复问题。"
SEED = 20260927


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalized(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def rank(tag: str, identity: str) -> str:
    return digest(f"{SEED}:{tag}:{identity}".encode())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def assign_groups(documents: list[dict]) -> None:
    """Union all matching titles or normalized paragraphs, including cross-source matches."""
    parents = list(range(len(documents)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, doc in enumerate(documents):
        keys = [("context", digest(normalized(p["context"]).encode())) for p in doc["paragraphs"]]
        if normalized(doc["title"]):
            keys.append(("title", normalized(doc["title"])))
        for key in keys:
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, doc in enumerate(documents):
        groups[find(i)].append(doc["source"] + ":" + doc["id"])
    for i, doc in enumerate(documents):
        doc["group_id"] = digest("\n".join(sorted(groups[find(i)])).encode())


def build(raw_dir: Path, output_dir: Path, tokenizer_dir: Path) -> dict:
    from transformers import AutoTokenizer

    if output_dir.exists():
        raise ValueError(f"Refusing to replace frozen data: {output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False)
    documents = []
    sources = {}
    for split, expected in RAW_HASHES.items():
        source_url = f"https://raw.githubusercontent.com/ymcui/cmrc2018/{REVISION}/squad-style-data/cmrc2018_{split}.json"
        path = raw_dir / f"{split}.json"
        if not path.exists():
            raw_dir.mkdir(parents=True, exist_ok=True)
            with urlopen(source_url, timeout=60) as response:
                payload = response.read()
            if digest(payload) != expected:
                raise ValueError(f"Unexpected download hash: {split}")
            path.write_bytes(payload)
        payload = path.read_bytes()
        if digest(payload) != expected:
            raise ValueError(f"Raw source hash mismatch: {path}")
        sources[split] = {"url": source_url, "sha256": expected}
        for doc in json.loads(payload)["data"]:
            documents.append({**doc, "source": split})
    assign_groups(documents)
    dev_groups = {d["group_id"] for d in documents if d["source"] == "dev"}
    paragraphs, questions, exclusions = [], [], []
    seen_contexts, seen_ids = set(), set()
    # Give dev priority; exclude an entire train group matching any dev document.
    documents.sort(key=lambda d: (d["source"] != "dev", d["id"]))
    for doc in documents:
        if doc["source"] == "train" and doc["group_id"] in dev_groups:
            exclusions.append({"id": doc["id"], "kind": "document", "reason": "train_group_overlaps_dev"})
            continue
        for paragraph in doc["paragraphs"]:
            context = paragraph["context"]
            context_hash = digest(context.encode())
            norm_hash = digest(normalized(context).encode())
            if norm_hash in seen_contexts:
                exclusions.append({"id": paragraph["id"], "kind": "paragraph", "reason": "duplicate_normalized_context"})
                continue
            seen_contexts.add(norm_hash)
            meta = {"source": doc["source"], "document_id": doc["id"], "group_id": doc["group_id"],
                    "paragraph_id": paragraph["id"], "context_sha256": context_hash, "normalized_context_sha256": norm_hash}
            token_ids = tokenizer.encode(context, add_special_tokens=False)
            paragraphs.append({**meta, "id": paragraph["id"], "text": context, "token_ids": token_ids})
            for qa in paragraph["qas"]:
                if qa["id"] in seen_ids:
                    raise ValueError("Duplicate source question ID")
                seen_ids.add(qa["id"])
                answers = list(dict.fromkeys(a["text"] for a in qa["answers"]))
                prompt = tokenizer.apply_chat_template([
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"资料：{context}\n问题：{qa['question']}"},
                ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                answer_ids = [tokenizer.encode(a, add_special_tokens=False) for a in answers]
                reasons = []
                if not answers or any(not a or a not in context for a in answers):
                    reasons.append("empty_or_nonextractive_reference")
                if len(prompt_ids) > 448:
                    reasons.append("prompt_over_448_tokens")
                if any(len(a) > 63 for a in answer_ids):
                    reasons.append("reference_over_63_tokens")
                if reasons:
                    exclusions.append({"id": qa["id"], "kind": "question", "reasons": reasons,
                                       "input_tokens": len(prompt_ids)})
                    continue
                questions.append({**meta, "id": qa["id"], "context": context,
                                  "question": qa["question"], "answers": answers,
                                  "prompt_token_ids": prompt_ids, "input_tokens": len(prompt_ids),
                                  "recovery_tokens_first_answer": len(prompt_ids) + len(answer_ids[0]) + 1})

    used_groups = {}
    selections = {}

    def select_qa(name, source, count):
        candidates = [q for q in questions if q["source"] == source and q["group_id"] not in used_groups]
        candidates.sort(key=lambda q: rank(name, q["id"]))
        selected = candidates[:count]
        if len(selected) != count:
            raise ValueError(f"Not enough eligible questions for {name}: {len(selected)}/{count}")
        for q in selected:
            used_groups[q["group_id"]] = name
        selections[name] = selected

    def select_lm(name, source, budget):
        candidates = [p for p in paragraphs if p["source"] == source and p["group_id"] not in used_groups]
        candidates.sort(key=lambda p: rank(name, p["id"]))
        selected, remaining = [], budget
        for p in candidates:
            if not remaining:
                break
            if p["group_id"] in used_groups or len(p["token_ids"]) < 2:
                continue
            ids = p["token_ids"][:remaining + 1]
            selected.append({**p, "original_token_count": len(p["token_ids"]), "token_ids": ids,
                             "text": tokenizer.decode(ids), "prefix_token_count": len(ids)})
            used_groups[p["group_id"]] = name
            remaining -= len(ids) - 1
        if remaining:
            raise ValueError(f"Not enough LM tokens for {name}: missing {remaining}")
        selections[name] = selected

    select_qa("final_qa", "dev", 200)
    select_lm("final_lm", "dev", 16384)
    select_qa("smoke_qa", "train", 10)
    select_lm("smoke_lm", "train", 1024)
    select_qa("validation_qa", "train", 100)
    select_lm("validation_lm", "train", 8192)
    eligible_calibration = [p for p in paragraphs if p["source"] == "train" and len(p["token_ids"]) >= 256]
    eligible_calibration.sort(key=lambda p: rank("calibration", p["id"]))
    calibration = []
    for p in eligible_calibration:
        if p["group_id"] in used_groups:
            continue
        calibration.append({**p, "original_token_count": len(p["token_ids"]), "token_ids": p["token_ids"][:256],
                            "text": tokenizer.decode(p["token_ids"][:256])})
        used_groups[p["group_id"]] = "calibration"
        if len(calibration) == 64:
            break
    if len(calibration) != 64:
        raise ValueError("Not enough calibration documents")
    selections["calibration"] = calibration
    select_qa("recovery", "train", 512)
    if any(q["recovery_tokens_first_answer"] > 512 for q in selections["recovery"]):
        raise AssertionError("Recovery sample exceeds complete sequence budget")
    # The split assignment is authoritative for entire document groups, including unselected questions.
    owners = defaultdict(set)
    for name, rows in selections.items():
        for row in rows:
            owners[row["group_id"]].add(name)
    if any(len(v) != 1 for v in owners.values()):
        raise AssertionError("Document leakage across purposes")
    output_dir.mkdir(parents=True)
    for name, rows in selections.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)
    write_jsonl(output_dir / "excluded.jsonl", exclusions)
    write_json(output_dir / "groups.json", {"assignments": used_groups,
               "documents": [{k: d[k] for k in ("id", "title", "source", "group_id")} for d in documents]})
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "chat_template.jinja"]
    manifest = {
        "schema_version": 1, "seed": SEED, "dataset": "CMRC2018", "revision": REVISION,
        "source": sources, "license": "CC-BY-SA-4.0", "license_url": f"https://github.com/ymcui/cmrc2018/blob/{REVISION}/LICENCE",
        "citation": "Cui et al. (2019), A Span-Extraction Dataset for Chinese Machine Reading Comprehension, EMNLP-IJCNLP, https://aclanthology.org/D19-1600/",
        "builder_sha256": digest(Path(__file__).read_bytes()),
        "tokenizer": {"directory": str(tokenizer_dir.resolve()), "source_model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                      "files_sha256": {name: digest((tokenizer_dir / name).read_bytes()) for name in tokenizer_files if (tokenizer_dir / name).exists()}},
        "protocol": {
            "qa": {"max_new_tokens": 64, "max_input_tokens": 448, "enable_thinking": False, "do_sample": False,
                   "answer_extraction": "strip_only", "system_prompt": SYSTEM,
                   "user_template": "资料：{context}\n问题：{question}", "max_reference_tokens": 63,
                   "length_policy": "Drop overlong complete prompts or references; never truncate QA context/answer."},
            "lm": {"window": 512, "stride": 256, "add_special_tokens": False,
                   "document_boundary": "Reset context and omit first-token target for every document.",
                   "budget_policy": "Use whole documents until budget; last document prefix trimmed in token space to exact target budget."},
            "split_policy": "Union by normalized title or paragraph; NFKC, whitespace removal, casefold. Dev groups exclude matching train groups. Whole group assigned to one purpose. Exact normalized duplicate contexts removed. No semantic near-duplicate guarantee.",
            "sampling": "SHA256(seed:purpose:ID) order, first N eligible questions; reserve their entire groups. Selection order final QA, final LM, smoke QA, smoke LM, validation QA, validation LM, calibration, recovery.",
            "scope": "Length-filtered local development subset, not official hidden-test score; final files locked and not run during development.",
            "acceptance": {"max_f1_drop_percentage_points": 5, "max_ppl_relative_increase": 0.10, "human_review": "No widespread repeated outputs."},
        },
        "counts": {name: {"rows": len(rows), "groups": len({r['group_id'] for r in rows}),
                            **({"valid_lm_targets": sum(len(r['token_ids'])-1 for r in rows)} if name.endswith("_lm") else {})} for name, rows in selections.items()},
        "exclusion_counts": dict(Counter(reason for e in exclusions for reason in e.get("reasons", [e.get("reason")]))),
        "files_sha256": {p.name: digest(p.read_bytes()) for p in sorted(output_dir.iterdir()) if p.is_file()},
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/cmrc2018"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/quality_v1"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("../../work/models/Qwen3-1.7B"))
    args = parser.parse_args()
    manifest = build(args.raw_dir, args.output_dir, args.tokenizer_dir)
    print(json.dumps({"counts": manifest["counts"], "excluded": manifest["exclusion_counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
