"""Validate frozen recovery-only data and build answer-supervised token rows.

Reads only manifest.json, recovery.jsonl, and groups.json. No model, downloads,
training, truncation, or final-set content access occurs in this module.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from prepare_quality_data import normalized
from run_quality import (json_hash, read_json, read_jsonl, sha256_file,
                         token_ids_valid, validate_protocol, verify_files)


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_recovery_data(data_dir, tokenizer):
    """Return (manifest, encoded rows, provenance), preserving frozen file order.

    Labels are prompt=-100 followed by first answer token IDs and one EOS.
    The model performs the causal label shift; this function does not shift.
    Counts must match the manifest (the frozen project data contains 512 rows
    and 433 groups). Split isolation is checked against the hashed authoritative
    groups mapping; other purposes' question or text files are never opened.
    """
    directory = Path(data_dir)
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path)
    validate_protocol(manifest["protocol"])
    required = ("recovery.jsonl", "groups.json")
    expected = manifest["files_sha256"]
    if any(name not in expected for name in required):
        raise ValueError("Missing frozen recovery/groups SHA256")
    hashes = verify_files(directory, {name: expected[name] for name in required}, required)
    rows = read_jsonl(directory / "recovery.jsonl")
    groups = read_json(directory / "groups.json")
    assignments = groups.get("assignments")
    documents = groups.get("documents")
    if not isinstance(assignments, dict) or not isinstance(documents, list):
        raise ValueError("groups.json requires assignments and document membership")
    for group, purpose in assignments.items():
        if not isinstance(group, str) or not group.strip() or not isinstance(purpose, str) or not purpose.strip():
            raise ValueError("Every group must have exactly one nonempty purpose")
    document_groups, group_sources = {}, {}
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("Malformed group document")
        source, identifier, group = (document.get(key) for key in ("source", "id", "group_id"))
        if source not in ("train", "dev") or not isinstance(identifier, str) or not identifier.strip() or not isinstance(group, str) or not group.strip():
            raise ValueError("Malformed group document membership")
        key = (source, identifier)
        if key in document_groups:
            raise ValueError("Duplicate source document in group membership")
        document_groups[key] = group
        group_sources.setdefault(group, set()).add(source)
    counts = manifest["counts"]["recovery"]
    expected_rows = _positive_int(counts.get("rows"), "recovery row count")
    expected_groups = _positive_int(counts.get("groups"), "recovery group count")
    if len(rows) != expected_rows:
        raise ValueError("Recovery row count disagrees with manifest")
    vocabulary = tokenizer.get_vocab()
    vocab_ids = set(vocabulary.values())
    eos = tokenizer.eos_token_id
    if type(eos) is not int or eos < 0 or eos not in vocab_ids:
        raise ValueError("Tokenizer must supply one valid integer eos_token_id")
    qa = manifest["protocol"]["qa"]
    if not isinstance(qa.get("system_prompt"), str) or not isinstance(qa.get("user_template"), str):
        raise ValueError("Missing frozen prompt templates")
    reference_limit = _positive_int(qa.get("max_reference_tokens"), "reference token limit")
    if reference_limit > 63:
        raise ValueError("Frozen recovery references must not exceed 63 tokens")
    used_groups, context_groups, encoded = set(), {}, []
    for row in rows:
        identifier, group = row["id"], row.get("group_id")
        if any(not isinstance(row.get(key), str) or not row[key].strip()
               for key in ("id", "document_id", "paragraph_id", "group_id")):
            raise ValueError(f"{identifier}: question/document/paragraph/group IDs must be nonempty strings")
        if row.get("source") != "train":
            raise ValueError(f"{identifier}: recovery requires a training document group")
        if assignments.get(group) != "recovery" or group_sources.get(group) != {"train"}:
            raise ValueError(f"{identifier}: group is not exclusively assigned to training recovery")
        if document_groups.get(("train", row.get("document_id"))) != group:
            raise ValueError(f"{identifier}: source document does not belong to its recorded group")
        used_groups.add(group)
        context, question, answers = (row.get(key) for key in ("context", "question", "answers"))
        if not isinstance(context, str) or not context.strip() or not isinstance(question, str) or not question.strip():
            raise ValueError(f"{identifier}: context and question must be nonempty strings")
        if row.get("context_sha256") != hashlib.sha256(context.encode()).hexdigest():
            raise ValueError(f"{identifier}: context SHA256 mismatch")
        normalized_hash = hashlib.sha256(normalized(context).encode()).hexdigest()
        if row.get("normalized_context_sha256") != normalized_hash:
            raise ValueError(f"{identifier}: normalized context SHA256 mismatch")
        if context_groups.setdefault(normalized_hash, group) != group:
            raise ValueError(f"{identifier}: duplicate context assigned to different groups")
        if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a.strip() or a not in context for a in answers):
            raise ValueError(f"{identifier}: references must be nonempty source spans")
        prompt_ids = row.get("prompt_token_ids")
        if not token_ids_valid(prompt_ids) or len(prompt_ids) > qa["max_input_tokens"]:
            raise ValueError(f"{identifier}: invalid or overlong complete prompt; no truncation")
        if type(row.get("input_tokens")) is not int or row["input_tokens"] != len(prompt_ids):
            raise ValueError(f"{identifier}: input_tokens disagrees with frozen prompt")
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": qa["system_prompt"]},
            {"role": "user", "content": qa["user_template"].format(context=context, question=question)},
        ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if tokenizer.encode(prompt, add_special_tokens=False) != prompt_ids:
            raise ValueError(f"{identifier}: canonical prompt re-encoding differs from frozen IDs")
        answer_ids = [tokenizer.encode(answer, add_special_tokens=False) for answer in answers]
        if any(not token_ids_valid(ids) or len(ids) > reference_limit for ids in answer_ids):
            raise ValueError(f"{identifier}: invalid or overlong reference tokens")
        if any(token not in vocab_ids for ids in [prompt_ids, *answer_ids] for token in ids):
            raise ValueError(f"{identifier}: token ID outside tokenizer vocabulary")
        targets = answer_ids[0] + [eos]
        input_ids = list(prompt_ids) + targets
        length = len(input_ids)
        if type(row.get("recovery_tokens_first_answer")) is not int or row["recovery_tokens_first_answer"] != length:
            raise ValueError(f"{identifier}: complete recovery length disagrees with frozen record")
        if length > 512:
            raise ValueError(f"{identifier}: complete sequence exceeds 512; no truncation")
        encoded.append({**copy.deepcopy(row), "input_ids": input_ids,
                        "labels": [-100] * len(prompt_ids) + targets,
                        "attention_mask": [1] * length, "sequence_length": length,
                        "prompt_length": len(prompt_ids), "answer_token_count": len(answer_ids[0]),
                        "supervised_token_count": len(targets), "eos_token_id": eos})
    assigned_recovery = {group for group, purpose in assignments.items() if purpose == "recovery"}
    if len(used_groups) != expected_groups or used_groups != assigned_recovery:
        raise ValueError("Recovery group count or complete assignment disagrees with frozen manifest")
    provenance = {
        "directory": str(directory.resolve()), "manifest_sha256": sha256_file(manifest_path),
        "files_sha256": hashes, "row_count": len(encoded), "group_count": len(used_groups),
        "sample_order": [row["id"] for row in encoded],
        "encoded_rows_sha256": json_hash([{key: row[key] for key in ("id", "input_ids", "labels")} for row in encoded]),
        "supervised_token_count": sum(row["supervised_token_count"] for row in encoded),
        "min_sequence_length": min(row["sequence_length"] for row in encoded),
        "max_sequence_length": max(row["sequence_length"] for row in encoded),
        "eligible_trial_rows_at_256": sum(row["sequence_length"] <= 256 for row in encoded),
        "tokenizer_vocab_sha256": json_hash(vocabulary), "eos_token_id": eos,
        "source_model_revision": manifest["tokenizer"]["source_model_revision"],
        "protocol": {"answer_choice": "first frozen reference", "prompt_labels": -100,
                     "supervise_eos": True, "label_shift": "none here; causal model shifts internally",
                     "max_sequence_length": 512, "truncation": False, "padding": "none; batch size one rows",
                     "order": "frozen recovery.jsonl file order",
                     "isolation": "hashed groups assignments and document memberships; no other split content read"},
    }
    return manifest, encoded, provenance


def select_trial_examples(rows, count=10, max_length=256):
    """Select the first count full rows within the length limit, without shuffle."""
    _positive_int(count, "trial count")
    _positive_int(max_length, "trial max_length")
    if max_length > 512 or not isinstance(rows, list):
        raise ValueError("Trial requires an encoded row list and max_length <= 512")
    eligible, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip() or row["id"] in seen:
            raise ValueError("Encoded trial rows require unique nonempty IDs")
        seen.add(row["id"])
        ids = row.get("input_ids")
        length = row.get("sequence_length")
        if not token_ids_valid(ids) or type(length) is not int or length != len(ids) or length > 512:
            raise ValueError("Trial row has inconsistent or invalid complete sequence length")
        if length <= max_length:
            eligible.append(row)
    if len(eligible) < count:
        raise ValueError(f"Insufficient complete trial examples: {len(eligible)}/{count} at <= {max_length} tokens")
    return copy.deepcopy(eligible[:count])
