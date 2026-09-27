"""Synthetic CPU checks for custom calibration prefixes and collector contract."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_custom_calibration as custom
from collect_mlp_stats import load_calibration_data, validate_calibration_tokens


class CharacterTokenizer:
    def get_vocab(self):
        return {chr(i): i for i in range(128)}

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(char) for char in text]

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(chr(value) for value in ids)


class CustomCalibrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.jsonl"
        self.tokenizer_dir = self.root / "tokenizer"
        self.tokenizer_dir.mkdir()
        for name in ("tokenizer.json", "tokenizer_config.json"):
            (self.tokenizer_dir / name).write_text("synthetic-tokenizer")
        self.output = self.root / "calibration"
        self.tokenizer = CharacterTokenizer()

    def write_rows(self, rows, path=None):
        (path or self.source).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def build(self, **kwargs):
        return custom.build(self.source, self.tokenizer_dir, "synthetic-revision", self.output,
                            tokenizer=self.tokenizer, **kwargs)

    def read_rows(self, name):
        return [json.loads(line) for line in (self.output / name).read_text(encoding="utf-8").splitlines()]

    def test_prefix_original_hash_and_arbitrary_positive_count_match_existing_collector(self):
        self.write_rows([{"id": "a", "text": "A" * 256}, {"id": "b", "text": "B" * 300}])
        manifest = self.build(max_documents=2)
        loaded, rows, data = load_calibration_data(self.output)
        validate_calibration_tokens(rows, self.tokenizer, 128)
        self.assertEqual(loaded, manifest)
        self.assertEqual([row["id"] for row in rows], ["a", "b"])
        self.assertEqual(data["valid_tokens_per_layer"], 512)
        self.assertEqual([row["original_token_count"] for row in rows], [256, 300])
        self.assertEqual([row["prefix_truncated"] for row in rows], [False, True])
        self.assertEqual(rows[1]["text"], "B" * 256)
        self.assertEqual(rows[1]["context_sha256"], custom._hash("B" * 300))
        self.assertNotEqual(rows[1]["context_sha256"], custom._hash(rows[1]["text"]))
        self.assertEqual(manifest["source"]["kind"], "custom_user_jsonl")
        self.assertIn("not CMRC", manifest["source"]["source_train_meaning"])
        self.assertEqual(set(manifest["tokenizer"]["files_sha256"]), {"tokenizer.json", "tokenizer_config.json"})

    def test_short_duplicates_same_group_and_limit_are_explicitly_recorded(self):
        self.write_rows([{"id": "short", "text": "a" * 255},
                         {"id": "first", "text": "B" * 300, "group_id": "g"},
                         {"id": "duplicate", "text": "b " * 300},
                         {"id": "same_group", "text": "C" * 300, "group_id": "g"},
                         {"id": "second", "text": "D" * 300},
                         {"id": "over_limit", "text": "E" * 300}])
        manifest = self.build(max_documents=2)
        self.assertEqual([row["id"] for row in self.read_rows("calibration.jsonl")], ["first", "second"])
        excluded = {row["id"]: row["reason"] for row in self.read_rows("excluded.jsonl")}
        self.assertEqual(excluded, {"short": "shorter_than_256_tokens", "duplicate": "normalized_full_text_duplicate",
                                    "same_group": "another_document_in_same_group_selected", "over_limit": "max_documents_reached"})
        self.assertEqual(manifest["counts"]["excluded_documents"], 4)
        self.assertEqual(len(self.read_rows("source_documents.jsonl")), 6)

    def test_context_exclusion_propagates_through_transitive_duplicate_and_supplied_groups(self):
        self.write_rows([{"id": "a", "text": "X" * 300, "group_id": "g1"},
                         {"id": "b", "text": "x " * 300, "group_id": "g2"},
                         {"id": "c", "text": "Y" * 300, "group_id": "g2"},
                         {"id": "retained", "text": "Z" * 300}])
        questions = self.root / "questions.jsonl"
        self.write_rows([{"id": "question-y", "context": "y " * 300, "question": "Q?"}], questions)
        manifest = self.build(exclude_questions=questions)
        self.assertEqual([row["id"] for row in self.read_rows("calibration.jsonl")], ["retained"])
        excluded = self.read_rows("excluded.jsonl")
        self.assertEqual({row["reason"] for row in excluded}, {"group_overlaps_excluded_question_context"})
        self.assertEqual(excluded[2]["direct_matching_question_ids"], ["question-y"])
        self.assertEqual(manifest["exclude_questions"]["nonempty_unique_contexts"], 1)
        self.assertEqual(len({row["group_id"] for row in excluded}), 1)

    def test_no_eligible_sources_preserves_skip_audit_without_completed_manifest(self):
        self.write_rows([{"id": "short", "text": "a" * 255}])
        with self.assertRaisesRegex(ValueError, "No eligible"):
            self.build()
        self.assertFalse((self.output / "manifest.json").exists())
        self.assertEqual(self.read_rows("excluded.jsonl")[0]["reason"], "shorter_than_256_tokens")
        report = json.loads((self.output / "build_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_input_empty_duplicate_ids_bad_schema_and_duplicate_keys_rejected(self):
        texts = ["", '{"id":"a","text":"x"}\n{"id":"a","text":"y"}\n',
                 '{"id":"a","text":"x","group_id":null}\n', '{"id":"a","text":" "}\n',
                 '{"id":"a","text":"x","extra":1}\n', '{"id":"a","id":"b","text":"x"}\n']
        for value in texts:
            self.source.write_text(value, encoding="utf-8")
            with self.subTest(value=value), self.assertRaises(ValueError):
                custom.load_documents(self.source)

    def test_tokenizer_binding_file_required_and_invalid_limit_does_not_create_output(self):
        self.write_rows([{"id": "a", "text": "A" * 300}])
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "positive integer"):
                self.build(max_documents=count)
        self.assertFalse(self.output.exists())
        (self.tokenizer_dir / "tokenizer_config.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing tokenizer binding"):
            self.build()
        self.assertFalse((self.output / "manifest.json").exists())

    def test_mutated_calibration_file_is_rejected_by_unchanged_loader(self):
        self.write_rows([{"id": "a", "text": "A" * 300}])
        self.build(max_documents=1)
        with (self.output / "calibration.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            load_calibration_data(self.output)


if __name__ == "__main__":
    unittest.main()
