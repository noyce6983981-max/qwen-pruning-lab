"""Build auditable 256-token calibration prefixes from a user's local JSONL.

The existing collector uses source='train' as a purpose label. Here the actual
source is explicitly a custom user file, never the CMRC training corpus.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from evaluate_questions import load_questions
from prepare_quality_data import normalized
from run_quality import json_hash, sha256_file, token_ids_valid


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "special_tokens_map.json", "chat_template.jinja")


def _hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_documents(path):
    documents, seen = [], set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line, object_pairs_hook=_unique_object)
        except ValueError as exc:
            raise ValueError(f"Calibration line {line_number}: {exc}") from exc
        if not isinstance(item, dict) or set(item) - {"id", "text", "group_id"}:
            raise ValueError(f"Calibration line {line_number}: expected only id/text/optional group_id")
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier != identifier.strip():
            raise ValueError(f"Calibration line {line_number}: invalid id")
        if identifier in seen:
            raise ValueError(f"Duplicate calibration id: {identifier}")
        seen.add(identifier)
        if not isinstance(item.get("text"), str) or not item["text"].strip():
            raise ValueError(f"{identifier}: text must be nonblank")
        group = item.get("group_id")
        if "group_id" in item and (not isinstance(group, str) or not group.strip() or group != group.strip()):
            raise ValueError(f"{identifier}: group_id must be a nonblank string when supplied")
        documents.append({"id": identifier, "text": item["text"], "input_group_id": group,
                          "input_line": line_number, "context_sha256": _hash(item["text"]),
                          "normalized_context_sha256": _hash(normalized(item["text"]))})
    if not documents:
        raise ValueError("Calibration input is empty")
    return documents


def assign_custom_groups(documents):
    """Union explicit groups and exact normalized duplicates before exclusions."""
    parents = list(range(len(documents)))
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    seen = {}
    for index, row in enumerate(documents):
        keys = [("text", row["normalized_context_sha256"])]
        if row["input_group_id"] is not None:
            keys.append(("group", row["input_group_id"]))
        for key in keys:
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    components = {}
    for index, row in enumerate(documents):
        components.setdefault(find(index), []).append(row["id"])
    for index, row in enumerate(documents):
        row["group_id"] = "custom-" + json_hash(sorted(components[find(index)]))


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def build(input_path, tokenizer_dir, revision, output_dir, max_documents=64, exclude_questions=None, *, tokenizer=None):
    """Create a new collector-compatible directory; no GPU or model is loaded."""
    if type(max_documents) is not int or max_documents < 1:
        raise ValueError("max_documents must be a positive integer")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("Pinned tokenizer source revision is required")
    input_path, tokenizer_dir, output_dir = map(Path, (input_path, tokenizer_dir, output_dir))
    if output_dir.resolve().is_relative_to(tokenizer_dir.resolve()):
        raise ValueError("Output must be outside tokenizer/model directory")
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "source_kind": "custom_user_jsonl", "input_path": str(input_path.resolve()),
              "max_documents": max_documents, "tokens_per_document": 256}
    try:
        documents = load_documents(input_path)
        report["input_sha256"] = sha256_file(input_path)
        assign_custom_groups(documents)
        excluded_hashes = {}
        exclusion_provenance = None
        if exclude_questions is not None:
            questions = load_questions(exclude_questions)
            for question in questions:
                if question["context"].strip():
                    excluded_hashes.setdefault(_hash(normalized(question["context"])), []).append(question["id"])
            exclusion_provenance = {"path": str(Path(exclude_questions).resolve()), "sha256": sha256_file(Path(exclude_questions)),
                                    "question_count": len(questions), "nonempty_unique_contexts": len(excluded_hashes)}
        excluded_groups = {row["group_id"] for row in documents if row["normalized_context_sha256"] in excluded_hashes}
        for required in ("tokenizer.json", "tokenizer_config.json"):
            if not (tokenizer_dir / required).is_file():
                raise ValueError(f"Missing tokenizer binding file: {required}")
        tokenizer_hashes = {name: sha256_file(tokenizer_dir / name) for name in TOKENIZER_FILES if (tokenizer_dir / name).is_file()}
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, trust_remote_code=False)
        vocabulary = set(tokenizer.get_vocab().values())
        selected, excluded, audit = [], [], []
        seen_contexts, selected_groups = set(), set()
        for row in documents:
            ids = tokenizer.encode(row["text"], add_special_tokens=False)
            if not token_ids_valid(ids) or any(token not in vocabulary for token in ids):
                raise ValueError(f"{row['id']}: tokenizer returned invalid source token IDs")
            count = len(ids)
            metadata = {**row, "original_token_count": count}
            audit.append(metadata)
            normalized_hash = row["normalized_context_sha256"]
            reason = None
            if row["group_id"] in excluded_groups:
                reason = "group_overlaps_excluded_question_context"
            elif normalized_hash in seen_contexts:
                reason = "normalized_full_text_duplicate"
            elif count < 256:
                reason = "shorter_than_256_tokens"
            elif row["group_id"] in selected_groups:
                reason = "another_document_in_same_group_selected"
            elif len(selected) >= max_documents:
                reason = "max_documents_reached"
            seen_contexts.add(normalized_hash)
            if reason:
                excluded.append({key: metadata[key] for key in ("id", "input_line", "input_group_id", "group_id",
                                                                 "context_sha256", "normalized_context_sha256", "original_token_count")}
                                | {"reason": reason, "direct_matching_question_ids": excluded_hashes.get(normalized_hash, [])})
                continue
            prefix = ids[:256]
            text = tokenizer.decode(prefix, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{row['id']}: selected prefix decodes to empty text")
            selected.append({"id": row["id"], "source": "train", "source_kind": "custom_user_jsonl",
                             "document_id": row["id"], "paragraph_id": row["id"], "group_id": row["group_id"],
                             "input_group_id": row["input_group_id"], "input_line": row["input_line"],
                             "context_sha256": row["context_sha256"], "normalized_context_sha256": normalized_hash,
                             "original_token_count": count, "text": text, "token_ids": prefix,
                             "prefix_truncated": count > 256})
            selected_groups.add(row["group_id"])
        _write_jsonl(output_dir / "source_documents.jsonl", audit)
        _write_jsonl(output_dir / "excluded.jsonl", excluded)
        report.update(input_documents=len(documents), selected_documents=len(selected), excluded_documents=len(excluded),
                      exclusion_counts=dict(Counter(row["reason"] for row in excluded)))
        if not selected:
            raise ValueError("No eligible calibration documents remain; completed manifest not written")
        from collect_mlp_stats import validate_calibration_tokens
        validate_calibration_tokens(selected, tokenizer, max(vocabulary) + 1)
        groups = {"assignments": {group: "calibration" for group in sorted(selected_groups)},
                  "documents": [{"id": row["id"], "source": "train", "source_kind": "custom_user_jsonl",
                                 "group_id": row["group_id"], "input_group_id": row["input_group_id"]} for row in documents],
                  "grouping": "Union of identical explicit group_id and NFKC/whitespace-removed/casefold complete text; at most one selected document per group.",
                  "excluded_question_groups": sorted(excluded_groups)}
        _write_jsonl(output_dir / "calibration.jsonl", selected)
        _write_json(output_dir / "groups.json", groups)
        manifest = {
            "schema_version": 1, "status": "completed", "dataset": "custom_user_calibration",
            "source": {"kind": "custom_user_jsonl", "path": str(input_path.resolve()), "sha256": report["input_sha256"],
                       "source_train_meaning": "Calibration purpose label required by the existing collector; not CMRC or an assertion about a public dataset split."},
            "builder_sha256": sha256_file(Path(__file__)),
            "tokenizer": {"directory": str(tokenizer_dir.resolve()), "source_model_revision": revision,
                          "files_sha256": tokenizer_hashes},
            "counts": {"calibration": {"rows": len(selected), "groups": len(selected_groups)},
                       "input_documents": len(documents), "excluded_documents": len(excluded)},
            "exclusion_counts": report["exclusion_counts"], "exclude_questions": exclusion_provenance,
            "protocol": {"tokens_per_document": 256, "add_special_tokens": False, "sampling": "first eligible document per group in original file order",
                         "max_documents": max_documents, "prefix_policy": "Explicitly take the first 256 source tokens; skip sources shorter than 256. This truncation applies only to calibration, never QA prompts or answers.",
                         "text": "Decode the exact selected token IDs with special tokens retained and no cleanup.",
                         "duplicate_policy": "Complete-text NFKC, remove whitespace, casefold; no semantic near-duplicate guarantee.",
                         "question_isolation": "If exclusion file supplied, exclude every group containing an exact normalized context match; no semantic, substring, or external-dataset leakage guarantee."},
            "files_sha256": {name: sha256_file(output_dir / name) for name in
                              ("calibration.jsonl", "groups.json", "source_documents.jsonl", "excluded.jsonl")},
        }
        _write_json(output_dir / "manifest.json", manifest)
        report.update(status="completed", manifest_sha256=sha256_file(output_dir / "manifest.json"))
        return manifest
    except BaseException as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        _write_json(output_dir / "build_report.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="UTF-8 JSONL: id/text/optional group_id")
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-documents", type=int, default=64)
    parser.add_argument("--exclude-questions", type=Path, help="Optional custom question JSONL; reject normalized context-overlap groups")
    args = parser.parse_args(argv)
    manifest = build(args.input, args.tokenizer_dir, args.revision, args.output_dir,
                     args.max_documents, args.exclude_questions)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "counts": manifest["counts"],
                      "exclusion_counts": manifest["exclusion_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
