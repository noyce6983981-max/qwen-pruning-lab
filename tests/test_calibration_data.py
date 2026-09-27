"""CPU-only integrity and isolation checks for the calibration input contract.

Fixtures are synthetic and temporary. No model, network, GPU, or real final
evaluation file is needed or read by this module.
"""

from __future__ import annotations

import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_mlp_stats import load_calibration_data, validate_calibration_tokens


class FakeTokenizer:
    """A small vocabulary with observable decode options and removable IDs."""

    def __init__(self, token_ids=range(512)):
        self.vocab = {f"token_{token_id}": token_id for token_id in token_ids}
        self.decode_calls = []

    def get_vocab(self):
        return dict(self.vocab)

    def __len__(self):
        return len(self.vocab)

    def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        self.decode_calls.append((list(token_ids), skip_special_tokens, clean_up_tokenization_spaces))
        return "".join(chr(0x4E00 + token_id) for token_id in token_ids)


def synthetic_row(index=0):
    ids = list(range(256))
    return {
        "id": f"TRAIN_{index}",
        "source": "train",
        "document_id": f"TRAIN_{index}",
        "paragraph_id": f"TRAIN_{index}",
        "group_id": f"group-{index}",
        # This intentionally differs from the hash of the truncated text.
        "context_sha256": hashlib.sha256(f"complete source paragraph {index}".encode()).hexdigest(),
        "text": "".join(chr(0x4E00 + token_id) for token_id in ids),
        "token_ids": ids,
        "original_token_count": 300,
    }


class CalibrationFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.rows = [synthetic_row(0), synthetic_row(1)]
        self.groups = {"assignments": {"group-0": "calibration", "group-1": "calibration", "final-group": "final_qa"}}
        self.manifest = {
            "schema_version": 1,
            "counts": {"calibration": {"rows": 2, "groups": 2}},
            "files_sha256": {},
        }

    def write_fixture(self):
        calibration = self.directory / "calibration.jsonl"
        groups = self.directory / "groups.json"
        calibration.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.rows), encoding="utf-8")
        groups.write_text(json.dumps(self.groups), encoding="utf-8")
        for path in (calibration, groups):
            self.manifest["files_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_manifest()

    def write_manifest(self):
        (self.directory / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def load(self):
        return load_calibration_data(self.directory)


class LoadCalibrationDataTests(CalibrationFixture):
    def test_accepts_small_manifest_defined_count_and_preserves_rows(self):
        self.write_fixture()
        manifest, rows, report = self.load()
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(rows, self.rows)
        self.assertIsInstance(report, dict)

    def test_single_row_manifest_is_valid(self):
        self.rows = self.rows[:1]
        self.groups["assignments"].pop("group-1")
        self.manifest["counts"]["calibration"] = {"rows": 1, "groups": 1}
        self.write_fixture()
        self.assertEqual(self.load()[1], self.rows)

    def test_source_context_hash_is_not_hash_of_truncated_text(self):
        for row in self.rows:
            self.assertNotEqual(row["context_sha256"], hashlib.sha256(row["text"].encode()).hexdigest())
        self.write_fixture()
        self.assertEqual(self.load()[1], self.rows)

    def test_tampered_calibration_and_group_file_are_rejected(self):
        for name in ("calibration.jsonl", "groups.json"):
            with self.subTest(file=name):
                self.write_fixture()
                path = self.directory / name
                path.write_bytes(path.read_bytes() + b" \n")
                with self.assertRaises(ValueError):
                    self.load()

    def test_missing_required_file_hash_is_rejected(self):
        for name in ("calibration.jsonl", "groups.json"):
            with self.subTest(file=name):
                self.write_fixture()
                del self.manifest["files_sha256"][name]
                self.write_manifest()
                with self.assertRaises(ValueError):
                    self.load()

    def test_manifest_counts_must_match(self):
        for counts in ({"rows": 3, "groups": 2}, {"rows": 2, "groups": 1}):
            with self.subTest(counts=counts):
                self.manifest["counts"]["calibration"] = counts
                self.write_fixture()
                with self.assertRaises(ValueError):
                    self.load()

    def test_manifest_count_requires_positive_integer_not_bool(self):
        for field in ("rows", "groups"):
            for value in (0, -1, True, "2", 2.0, None):
                with self.subTest(field=field, value=value):
                    self.manifest["counts"]["calibration"] = {"rows": 2, "groups": 2}
                    self.manifest["counts"]["calibration"][field] = value
                    self.write_fixture()
                    with self.assertRaises(ValueError):
                        self.load()

    def test_boolean_count_cannot_equal_a_single_row_or_group(self):
        self.rows = self.rows[:1]
        self.groups["assignments"].pop("group-1")
        for field in ("rows", "groups"):
            with self.subTest(field=field):
                self.manifest["counts"]["calibration"] = {"rows": 1, "groups": 1}
                self.manifest["counts"]["calibration"][field] = True
                self.write_fixture()
                with self.assertRaises(ValueError):
                    self.load()

    def test_empty_calibration_is_rejected_even_if_zero_count_recorded(self):
        self.rows = []
        self.manifest["counts"]["calibration"] = {"rows": 0, "groups": 0}
        self.write_fixture()
        with self.assertRaises(ValueError):
            self.load()

    def test_only_train_source_is_allowed(self):
        for source in ("dev", "test", None, "TRAIN"):
            with self.subTest(source=source):
                self.rows[0]["source"] = source
                self.write_fixture()
                with self.assertRaises(ValueError):
                    self.load()

    def test_group_assignment_must_be_calibration(self):
        for assignment in ("validation_qa", "final_qa", "recovery", None):
            with self.subTest(assignment=assignment):
                self.groups["assignments"]["group-0"] = assignment
                self.write_fixture()
                with self.assertRaises(ValueError):
                    self.load()

    def test_unknown_group_is_rejected(self):
        self.rows[0]["group_id"] = "unknown-group"
        self.write_fixture()
        with self.assertRaises(ValueError):
            self.load()

    def test_duplicate_group_is_rejected_even_with_matching_unique_count(self):
        self.rows[1]["group_id"] = self.rows[0]["group_id"]
        self.manifest["counts"]["calibration"]["groups"] = 1
        self.write_fixture()
        with self.assertRaises(ValueError):
            self.load()

    def test_identifiers_must_be_nonempty_strings(self):
        for field in ("id", "group_id"):
            for value in (None, 0, True, ""):
                with self.subTest(field=field, value=value):
                    self.rows[0] = synthetic_row(0)
                    self.rows[0][field] = value
                    self.write_fixture()
                    with self.assertRaises(ValueError):
                        self.load()

    def test_token_length_must_be_exactly_256(self):
        for length in (0, 255, 257):
            with self.subTest(length=length):
                self.rows[0]["token_ids"] = list(range(length))
                self.write_fixture()
                with self.assertRaises(ValueError):
                    self.load()

    def test_token_ids_must_be_list_of_nonnegative_ints_without_bools(self):
        for token in (True, False, -1, 1.0, "1", None):
            with self.subTest(token=token):
                self.rows[0] = synthetic_row(0)
                self.rows[0]["token_ids"][1] = token
                self.write_fixture()
                with self.assertRaises(ValueError):
                    self.load()
        self.rows[0]["token_ids"] = "0" * 256
        self.write_fixture()
        with self.assertRaises(ValueError):
            self.load()

    def test_original_token_count_must_be_integer_at_least_256(self):
        for count in (255, -1, True, "300", 300.0, None):
            with self.subTest(count=count):
                self.rows[0]["original_token_count"] = count
                self.write_fixture()
                with self.assertRaises(ValueError):
                    self.load()

    def test_original_token_count_exactly_256_is_allowed(self):
        self.rows[0]["original_token_count"] = 256
        self.write_fixture()
        self.assertEqual(self.load()[1], self.rows)

    def test_text_must_be_nonempty_string(self):
        for text in ("", " \t\n", None, 123):
            with self.subTest(text=text):
                self.rows[0]["text"] = text
                self.write_fixture()
                with self.assertRaises(ValueError):
                    self.load()

    def test_does_not_open_final_or_other_manifest_files(self):
        self.write_fixture()
        for name in ("final_qa.jsonl", "final_lm.jsonl", "validation_qa.jsonl", "recovery.jsonl"):
            (self.directory / name).write_text("This is deliberately invalid JSON; do not read.", encoding="utf-8")
            self.manifest["files_sha256"][name] = "0" * 64
        self.write_manifest()
        allowed = {"manifest.json", "calibration.jsonl", "groups.json"}
        opened = set()
        actual_builtin_open, actual_path_open = builtins.open, Path.open

        def check(path):
            if isinstance(path, (str, os.PathLike)):
                resolved = Path(path).resolve()
                if resolved.parent == self.directory.resolve():
                    self.assertIn(resolved.name, allowed, f"Loader opened a non-calibration file: {resolved.name}")
                    opened.add(resolved.name)

        def guarded_builtin(path, *args, **kwargs):
            check(path)
            return actual_builtin_open(path, *args, **kwargs)

        def guarded_path(path, *args, **kwargs):
            check(path)
            return actual_path_open(path, *args, **kwargs)

        with patch("builtins.open", guarded_builtin), patch.object(Path, "open", guarded_path):
            self.assertEqual(self.load()[1], self.rows)
        self.assertEqual(opened, allowed)


class ValidateCalibrationTokensTests(unittest.TestCase):
    def setUp(self):
        self.rows = [synthetic_row(0), synthetic_row(1)]
        self.tokenizer = FakeTokenizer()

    def test_valid_ids_decode_without_cleanup_or_special_token_removal(self):
        before = copy.deepcopy(self.rows)
        validate_calibration_tokens(self.rows, self.tokenizer, vocab_size=512)
        self.assertEqual(self.rows, before)
        self.assertEqual(len(self.tokenizer.decode_calls), len(self.rows))
        for ids, skip_special_tokens, clean_up in self.tokenizer.decode_calls:
            self.assertEqual(ids, list(range(256)))
            self.assertIs(skip_special_tokens, False)
            self.assertIs(clean_up, False)

    def test_model_vocab_upper_bound_is_exclusive(self):
        # ID 255 exists in the tokenizer, but a model with 255 rows cannot use it.
        with self.assertRaises(ValueError):
            validate_calibration_tokens(self.rows, self.tokenizer, vocab_size=255)

    def test_tokenizer_membership_is_checked_not_just_length(self):
        # ID 17 is a hole below both vocabulary size limits.
        tokenizer = FakeTokenizer(i for i in range(512) if i != 17)
        with self.assertRaises(ValueError):
            validate_calibration_tokens(self.rows, tokenizer, vocab_size=512)

    def test_tokenizer_added_id_within_model_range_is_allowed(self):
        self.rows[0]["token_ids"][-1] = 700
        self.rows[0]["text"] = "".join(chr(0x4E00 + token_id) for token_id in self.rows[0]["token_ids"])
        tokenizer = FakeTokenizer([*range(512), 700])
        validate_calibration_tokens(self.rows, tokenizer, vocab_size=701)

    def test_decode_mismatch_is_rejected(self):
        self.rows[1]["text"] += "different suffix"
        with self.assertRaises(ValueError):
            validate_calibration_tokens(self.rows, self.tokenizer, vocab_size=512)

    def test_decode_text_is_compared_exactly_without_strip(self):
        self.rows[0]["text"] += " "
        with self.assertRaises(ValueError):
            validate_calibration_tokens(self.rows, self.tokenizer, vocab_size=512)

    def test_invalid_token_types_cannot_pass_membership_by_int_equality(self):
        for token in (True, False, 1.0, -1, "1"):
            with self.subTest(token=token):
                self.rows[0] = synthetic_row(0)
                self.rows[0]["token_ids"][1] = token
                with self.assertRaises(ValueError):
                    validate_calibration_tokens(self.rows, self.tokenizer, vocab_size=512)


if __name__ == "__main__":
    unittest.main()
