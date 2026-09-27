"""CPU checks for frozen evaluation inputs and lossless answer recording."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_quality as quality


PROTOCOL = {
    "qa": {"max_input_tokens": 448, "max_new_tokens": 64, "enable_thinking": False,
           "do_sample": False, "answer_extraction": "strip_only"},
    "lm": {"window": 512, "stride": 256},
}


def qa_item(identifier="q1"):
    context = "北京是中国的首都。"
    return {"id": identifier, "context": context, "question": "中国首都是什么？",
            "answers": ["北京"], "prompt_token_ids": [1, 2, 3],
            "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest()}


class FakeTokenizer:
    bos_token_id = 0
    eos_token_id = 8
    pad_token_id = 0
    special_tokens_map = {"eos_token": "<eos>"}
    all_special_tokens = ["<eos>"]
    all_special_ids = [8]
    chat_template = "frozen"

    def get_vocab(self):
        return {"北京": 1, "<eos>": 8}

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return [5, 6, 7]

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert clean_up_tokenization_spaces is False
        return " \n北京 \n" if skip_special_tokens else " \n北京 \n<eos>"


class FrozenDataTests(unittest.TestCase):
    def make_dataset(self, directory):
        rows = {"smoke_qa.jsonl": [qa_item()],
                "smoke_lm.jsonl": [{"id": "doc1", "token_ids": [1, 2, 3]}]}
        hashes = {}
        for name, items in rows.items():
            path = directory / name
            path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n",
                            encoding="utf-8")
            hashes[name] = quality.sha256_file(path)
        manifest = {"files_sha256": hashes, "protocol": PROTOCOL}
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def test_frozen_manifest_and_lengths(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_dataset(directory)
            manifest, examples, documents, report = quality.load_frozen_data(directory, "smoke")
            self.assertEqual(examples[0]["id"], "q1")
            self.assertEqual(documents[0]["token_ids"], [1, 2, 3])
            self.assertEqual(report["lm_input_token_count"], 3)
            self.assertEqual(manifest["protocol"], PROTOCOL)
            with (directory / "smoke_qa.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                quality.load_frozen_data(directory, "smoke")

    def test_final_split_locked_before_file_access(self):
        with self.assertRaisesRegex(ValueError, "final evaluation remains locked"):
            quality.load_frozen_data(Path("not-present"), "final")
        parser = quality.build_parser()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--model-dir", "m", "--data-dir", "d", "--split", "final",
                               "--output-dir", "o", "--revision", "r"])

    def test_empty_or_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "qa.jsonl"
            path.write_text('{"id":"q"}\n{"id":"q"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate id"):
                quality.read_jsonl(path)
            path.write_text('{"id":""}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonempty string id"):
                quality.read_jsonl(path)

    def test_oversized_input_never_silently_truncated(self):
        item = qa_item()
        item["prompt_token_ids"] = [1] * 449
        with self.assertRaisesRegex(ValueError, "no truncation allowed"):
            quality.validate_examples([item], [{"id": "d", "token_ids": [1, 2]}], PROTOCOL)
        self.assertEqual(len(item["prompt_token_ids"]), 449)

    def test_invalid_tokens_and_answer_data_rejected(self):
        for ids in ([True, 1], [1, -1], [1, 2.0], [], "123"):
            item = qa_item()
            item["prompt_token_ids"] = ids
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "prompt_token_ids"):
                quality.validate_examples([item], [], PROTOCOL)
        for answers in ([], "北京", [None], [" "]):
            item = qa_item()
            item["answers"] = answers
            with self.subTest(answers=answers), self.assertRaisesRegex(ValueError, "answers"):
                quality.validate_examples([item], [], PROTOCOL)

    def test_context_hash_and_short_lm_document_rejected(self):
        item = qa_item()
        item["context"] += "修改"
        with self.assertRaisesRegex(ValueError, "context_sha256 mismatch"):
            quality.validate_examples([item], [], PROTOCOL)
        with self.assertRaisesRegex(ValueError, "at least two"):
            quality.validate_examples([qa_item()], [{"id": "d", "token_ids": [1]}], PROTOCOL)

    def test_manifest_path_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "escapes"):
                quality.verify_files(Path(temporary), {"../outside.json": "hash"})

    def test_missing_hash_is_not_optional(self):
        with self.assertRaisesRegex(ValueError, "Missing frozen hash"):
            quality.verify_files(Path("unused"), {"some.json": "hash"}, ("qa.jsonl",))

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            quality.create_output(output)
            sentinel = output / "quality.json"
            sentinel.write_text("prior result", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                quality.create_output(output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "prior result")

    def test_protocol_rejects_sampling_and_changed_ppl(self):
        protocol = copy.deepcopy(PROTOCOL)
        protocol["qa"]["do_sample"] = True
        with self.assertRaisesRegex(ValueError, "greedy"):
            quality.validate_protocol(protocol)
        protocol = copy.deepcopy(PROTOCOL)
        protocol["lm"]["stride"] = 512
        with self.assertRaisesRegex(ValueError, "stride=256"):
            quality.validate_protocol(protocol)


class TokenizerAndAnswerTests(unittest.TestCase):
    def test_semantic_tokenizer_check(self):
        reference = FakeTokenizer()
        candidate = FakeTokenizer()
        self.assertEqual(quality.check_tokenizers(reference, candidate)["status"], "passed")
        candidate.chat_template = "changed"
        with self.assertRaisesRegex(ValueError, "chat_template"):
            quality.check_tokenizers(reference, candidate)
        candidate = FakeTokenizer()
        candidate.encode = lambda text, **kwargs: [1]
        with self.assertRaisesRegex(ValueError, "fixture_token_ids"):
            quality.check_tokenizers(reference, candidate)

    def test_extraction_does_not_repair_wrong_or_repeated_output(self):
        text = " \n答案：北京。北京。\n "
        self.assertEqual(quality.extract_answer(text), "答案：北京。北京。")
        self.assertEqual(quality.extract_answer(" \n"), "")

    def test_stop_reason_eos_wins_at_length_cap(self):
        self.assertEqual(quality.stop_reason([1, 8], [8, 9], 2), "eos")
        self.assertEqual(quality.stop_reason([1, 2], [8, 9], 2), "max_new_tokens")
        self.assertEqual(quality.stop_reason([1], 8, 2), "other")

    def test_partial_generation_is_durable_before_later_failure(self):
        import torch

        class FailingAfterOne:
            calls = 0

            def generate(self, *, input_ids, generation_config, use_model_defaults, do_sample, **kwargs):
                self.calls += 1
                self.assert_settings = (use_model_defaults, do_sample)
                if self.calls == 2:
                    raise RuntimeError("simulated failure")
                return torch.cat((input_ids, torch.tensor([[1, 8]])), dim=1)

        model = FailingAfterOne()
        config = SimpleNamespace(eos_token_id=[8], max_new_tokens=64)
        with tempfile.TemporaryDirectory() as temporary, patch.object(torch.cuda, "synchronize"):
            path = Path(temporary) / "predictions.jsonl"
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "simulated failure"):
                quality.run_qa(model, FakeTokenizer(), [qa_item("q1"), qa_item("q2")],
                               config, "cpu", path)
            rows = quality.read_jsonl(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["answer"], "北京")
            self.assertEqual(rows[0]["raw_text"], " \n北京 \n")
            self.assertEqual(rows[0]["generated_token_ids"], [1, 8])
            self.assertEqual(rows[0]["prompt_token_ids"], [1, 2, 3])
            self.assertEqual(rows[0]["stop_reason"], "eos")
            self.assertEqual(model.assert_settings, (False, False))


if __name__ == "__main__":
    unittest.main()
