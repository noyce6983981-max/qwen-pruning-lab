"""CPU fixture tests of recovery-only loading, labels, limits, and corruption."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from recovery_data import load_recovery_data, select_trial_examples


class FixtureTokenizer:
    eos_token_id = 2

    def get_vocab(self):
        return {str(i): i for i in range(256)}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) + 10 for character in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        return messages[0]["content"] + "|" + messages[1]["content"] + "|A"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def make_row(index, length=100, answer_length=1):
    tokenizer = FixtureTokenizer()
    answer = "X" * answer_length
    context = answer + str(index)
    question = "Q"
    prompt = lambda c: tokenizer.apply_chat_template([{"role": "system", "content": "S"},
                                                     {"role": "user", "content": c + ":" + question}])
    initial = len(tokenizer.encode(prompt(context))) + answer_length + 1
    assert length >= initial
    context += "a" * (length - initial)
    prompt_ids = tokenizer.encode(prompt(context))
    return {"id": f"q{index}", "source": "train", "document_id": f"d{index}",
            "group_id": f"g{index}", "paragraph_id": f"p{index}",
            "context": context, "context_sha256": sha(context),
            "normalized_context_sha256": sha(context.casefold()), "question": question,
            "answers": [answer], "prompt_token_ids": prompt_ids, "input_tokens": len(prompt_ids),
            "recovery_tokens_first_answer": length}


class RecoveryDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="recovery-data-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.tokenizer = FixtureTokenizer()

    def write_fixture(self, rows=None, mutate_groups=None, mutate_manifest=None):
        rows = copy.deepcopy(rows if rows is not None else [make_row(0), make_row(1, 150), make_row(2, 200)])
        groups = {"assignments": {row["group_id"]: "recovery" for row in rows},
                  "documents": [{"id": row["document_id"], "source": row["source"], "group_id": row["group_id"]} for row in rows]}
        if mutate_groups:
            mutate_groups(groups)
        (self.directory / "recovery.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        (self.directory / "groups.json").write_text(json.dumps(groups), encoding="utf-8")
        manifest = {"counts": {"recovery": {"rows": len(rows), "groups": len({row['group_id'] for row in rows})}},
                    "files_sha256": {name: hashlib.sha256((self.directory / name).read_bytes()).hexdigest()
                                     for name in ("recovery.jsonl", "groups.json")},
                    "tokenizer": {"source_model_revision": "fixture-revision"},
                    "protocol": {"qa": {"enable_thinking": False, "do_sample": False, "answer_extraction": "strip_only",
                                        "max_input_tokens": 448, "max_new_tokens": 64, "max_reference_tokens": 63,
                                        "system_prompt": "S", "user_template": "{context}:{question}"},
                                 "lm": {"window": 512, "stride": 256}}}
        # A final filename is deliberately present in the hash manifest but
        # absent on disk: recovery loading must neither open nor hash it.
        manifest["files_sha256"]["final_qa.jsonl"] = "0" * 64
        if mutate_manifest:
            mutate_manifest(manifest)
        (self.directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return rows

    def test_answer_and_eos_only_labels_unshifted_and_no_other_split_access(self):
        original = self.write_fixture()
        original_read_bytes, original_read_text = Path.read_bytes, Path.read_text
        opened = []

        def guarded_bytes(path):
            opened.append(path.name)
            self.assertIn(path.name, {"manifest.json", "recovery.jsonl", "groups.json"})
            return original_read_bytes(path)

        def guarded_text(path, *args, **kwargs):
            opened.append(path.name)
            self.assertIn(path.name, {"manifest.json", "recovery.jsonl", "groups.json"})
            return original_read_text(path, *args, **kwargs)

        with patch.object(Path, "read_bytes", guarded_bytes), patch.object(Path, "read_text", guarded_text):
            manifest, encoded, provenance = load_recovery_data(self.directory, self.tokenizer)
        self.assertEqual([row["id"] for row in encoded], [row["id"] for row in original])
        for source, row in zip(original, encoded):
            answer = self.tokenizer.encode(source["answers"][0])
            expected = source["prompt_token_ids"] + answer + [2]
            self.assertEqual(row["input_ids"], expected)
            self.assertEqual(row["labels"], [-100] * len(source["prompt_token_ids"]) + answer + [2])
            self.assertEqual(row["labels"][-1], row["eos_token_id"])
            self.assertEqual(row["attention_mask"], [1] * len(expected))
            self.assertEqual(sum(label != -100 for label in row["labels"][1:]), len(answer) + 1)
            self.assertEqual(row["supervised_token_count"], len(answer) + 1)
        self.assertEqual(provenance["row_count"], 3)
        self.assertEqual(provenance["group_count"], 3)
        self.assertEqual(set(provenance["files_sha256"]), {"recovery.jsonl", "groups.json"})
        self.assertNotIn("final_qa.jsonl", opened)

    def test_exact_512_boundary_and_first_reference_without_truncation(self):
        row = make_row(0, 512, 63)
        row["answers"].append("X")
        self.write_fixture([row])
        _, encoded, _ = load_recovery_data(self.directory, self.tokenizer)
        self.assertEqual(encoded[0]["prompt_length"], 448)
        self.assertEqual(encoded[0]["sequence_length"], 512)
        self.assertEqual(encoded[0]["supervised_token_count"], 64)
        self.assertEqual(encoded[0]["labels"][448:], [ord("X") + 10] * 63 + [2])
        self.write_fixture([make_row(0, 513, 63)])
        with self.assertRaises(ValueError):
            load_recovery_data(self.directory, self.tokenizer)

    def test_stable_trial_selection_boundary_count_and_copies(self):
        self.write_fixture([make_row(0, 257), make_row(1, 256), make_row(2, 255), make_row(3, 300)])
        _, encoded, _ = load_recovery_data(self.directory, self.tokenizer)
        selected = select_trial_examples(encoded, count=2, max_length=256)
        self.assertEqual([row["id"] for row in selected], ["q1", "q2"])
        selected[0]["input_ids"][0] = -123
        self.assertGreaterEqual(encoded[1]["input_ids"][0], 0)
        with self.assertRaises(ValueError):
            select_trial_examples(encoded, count=3, max_length=256)
        for count, length in [(True, 256), (1.0, 256), (0, 256), (1, True), (1, 0), (1, 513)]:
            with self.subTest(count=count, length=length), self.assertRaises(ValueError):
                select_trial_examples(encoded, count=count, max_length=length)

    def test_checksum_corruption_is_rejected(self):
        for name in ("recovery.jsonl", "groups.json"):
            self.write_fixture()
            with (self.directory / name).open("a", encoding="utf-8") as stream:
                stream.write(" ")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "SHA256"):
                load_recovery_data(self.directory, self.tokenizer)

    def test_manifest_row_and_group_counts_are_enforced(self):
        for field in ("rows", "groups"):
            self.write_fixture(mutate_manifest=lambda m: m["counts"]["recovery"].__setitem__(field, 4))
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_recovery_data(self.directory, self.tokenizer)

    def test_wrong_purpose_membership_or_dev_overlap_is_rejected(self):
        mutations = [lambda g: g["assignments"].__setitem__("g0", "validation_qa"),
                     lambda g: g["documents"][0].__setitem__("group_id", "g1"),
                     lambda g: g["documents"].append({"id": "dev0", "source": "dev", "group_id": "g0"}),
                     lambda g: g["assignments"].__setitem__("unused", "recovery"),
                     lambda g: g["assignments"].__setitem__("g0", ["recovery", "validation_qa"])]
        for index, mutation in enumerate(mutations):
            self.write_fixture(mutate_groups=mutation)
            with self.subTest(case=index), self.assertRaises(ValueError):
                load_recovery_data(self.directory, self.tokenizer)

    def test_corrupted_rows_reject_even_when_file_hash_is_recomputed(self):
        mutations = [lambda r: r.__setitem__("context_sha256", "0" * 64),
                     lambda r: r.__setitem__("normalized_context_sha256", "0" * 64),
                     lambda r: r["prompt_token_ids"].__setitem__(0, 7),
                     lambda r: r.__setitem__("input_tokens", 1),
                     lambda r: r.__setitem__("recovery_tokens_first_answer", 99),
                     lambda r: r.__setitem__("answers", []),
                     lambda r: r.__setitem__("answers", ["not a source span"]),
                     lambda r: r.__setitem__("source", "dev"),
                     lambda r: r.__setitem__("document_id", "nonexistent"),
                     lambda r: r.__setitem__("paragraph_id", " ")]
        for index, mutation in enumerate(mutations):
            row = make_row(0)
            mutation(row)
            self.write_fixture([row], mutate_groups=(lambda g: g["documents"][0].__setitem__("id", "d0")) if index == 8 else None)
            with self.subTest(case=index), self.assertRaises(ValueError):
                load_recovery_data(self.directory, self.tokenizer)

    def test_duplicate_question_ids_are_rejected(self):
        rows = [make_row(0), make_row(1)]
        rows[1]["id"] = rows[0]["id"]
        self.write_fixture(rows)
        with self.assertRaisesRegex(ValueError, "Duplicate id"):
            load_recovery_data(self.directory, self.tokenizer)

    def test_invalid_eos_and_unknown_vocabulary_ids_are_rejected(self):
        self.write_fixture()
        for value in (None, True, -1, 999):
            tokenizer = FixtureTokenizer()
            tokenizer.eos_token_id = value
            with self.subTest(eos=value), self.assertRaises(ValueError):
                load_recovery_data(self.directory, tokenizer)
        tokenizer = FixtureTokenizer()
        vocabulary = tokenizer.get_vocab()
        vocabulary.pop(str(ord("X") + 10))
        with patch.object(tokenizer, "get_vocab", return_value=vocabulary), self.assertRaisesRegex(ValueError, "vocabulary"):
            load_recovery_data(self.directory, tokenizer)


if __name__ == "__main__":
    unittest.main()
