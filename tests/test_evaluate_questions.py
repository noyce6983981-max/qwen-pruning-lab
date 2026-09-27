"""CPU fixtures for custom QA inputs, real official scoring, and durable failures."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_questions as custom


class FakeTokenizer:
    def __init__(self, token_count=4):
        self.token_count = token_count
        self.seen = []

    def get_vocab(self):
        return {"token": 1}

    def apply_chat_template(self, messages, **kwargs):
        if kwargs != {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}:
            raise AssertionError(kwargs)
        self.seen.append(copy.deepcopy(messages))
        return [1] * self.token_count


class CustomQuestionsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.questions = self.root / "questions.jsonl"
        self.rows = [{"id": "one", "context": "虚构资料：研究室的标志颜色是蓝色。", "question": "标志是什么颜色？", "answers": ["蓝色"]},
                     {"id": "two", "question": "说明什么是三角形。"},
                     {"id": "three", "context": "虚构资料：会议在周二举行。", "question": "会议什么时候举行？", "answers": []}]
        self.write_rows(self.rows)

    def write_rows(self, rows):
        self.questions.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def test_mixed_references_optional_context_and_order(self):
        rows = custom.load_questions(self.questions)
        self.assertEqual([row["id"] for row in rows], ["one", "two", "three"])
        self.assertEqual(rows[1]["context"], "")
        self.assertEqual(rows[1]["answers"], [])
        self.assertEqual(rows[2]["answers"], [])
        self.assertEqual(rows[0]["answers"], ["蓝色"])

    def test_empty_duplicate_ids_and_duplicate_json_keys_rejected(self):
        invalid = [("\n\t\n", "empty"),
                   ('{"id":"x","question":"Q"}\n{"id":"x","question":"R"}\n', "Duplicate question id"),
                   ('{"id":"x","id":"y","question":"Q"}\n', "Duplicate JSON key"),
                   ('not-json\n', "line 1"), ('[]\n', "expected an object")]
        for text, error in invalid:
            self.questions.write_text(text, encoding="utf-8")
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, error):
                custom.load_questions(self.questions)

    def test_schema_rejects_bad_fields_without_silently_changing_data(self):
        bad = [{"id": " "}, {"id": " one "}, {"id": 3}, {"question": " "}, {"question": 2},
               {"context": None}, {"answers": None}, {"answers": "蓝色"}, {"answers": [" "]},
               {"answers": [1]}, {"answer": "misspelled field"}]
        for replacement in bad:
            self.write_rows([{**self.rows[0], **replacement}])
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                custom.load_questions(self.questions)
        self.write_rows([{"id": "x"}])
        with self.assertRaisesRegex(ValueError, "question"):
            custom.load_questions(self.questions)

    def test_utf8_bom_and_blank_lines_are_supported(self):
        self.questions.write_text("\n" + json.dumps(self.rows[0], ensure_ascii=False) + "\n\n", encoding="utf-8-sig")
        self.assertEqual(custom.load_questions(self.questions)[0]["context"], self.rows[0]["context"])

    def test_prompts_do_not_contain_reference_and_preserve_input(self):
        rows = custom.load_questions(self.questions)
        rows[0]["answers"] = ["SECRET_REFERENCE_ONLY"]
        before = copy.deepcopy(rows)
        tokenizer = FakeTokenizer()
        prepared = custom.prepare_questions(rows, tokenizer, custom.generation_protocol())
        self.assertEqual(rows, before)
        self.assertNotIn("SECRET_REFERENCE_ONLY", str(tokenizer.seen))
        self.assertIn(rows[0]["context"], tokenizer.seen[0][1]["content"])
        self.assertEqual(tokenizer.seen[1][0]["content"], custom.QUESTION_INSTRUCTION)
        self.assertEqual(prepared[0]["prompt_token_ids"], [1, 1, 1, 1])
        self.assertEqual([item["has_reference"] for item in prepared], [True, False, False])

    def test_exact_budget_passes_one_token_over_fails_without_truncation(self):
        rows = custom.load_questions(self.questions)
        tokenizer = FakeTokenizer(448)
        self.assertEqual(len(custom.prepare_questions(rows, tokenizer, custom.generation_protocol())[0]["prompt_token_ids"]), 448)
        tokenizer.token_count = 449
        with self.assertRaisesRegex(ValueError, "449 tokens.*no truncation"):
            custom.prepare_questions(rows, tokenizer, custom.generation_protocol())
        self.assertEqual(tokenizer.token_count, 449)
        for input_count, output_count in ((449, 64), (448, 65), (0, 64), (448, 0), (True, 64)):
            with self.subTest(input_count=input_count, output_count=output_count), self.assertRaises(ValueError):
                custom.generation_protocol(input_count, output_count)

    def test_real_official_scores_use_only_reference_subset_and_best_reference(self):
        rows = [{"id": "a", "answers": ["北京", "京城"]}, {"id": "b", "answers": []},
                {"id": "c", "answers": ["蓝色"]}]
        predictions = [{"id": "a", "answer": "京城"}, {"id": "b", "answer": "无参考的回答"},
                       {"id": "c", "answer": "红色"}]
        result = custom.score_predictions(rows, predictions)
        self.assertEqual((result["question_count"], result["scored_count"], result["unscored_count"]), (3, 2, 1))
        self.assertEqual(result["metrics"]["em"], 50)
        self.assertEqual(result["metrics"]["f1"], 75)
        self.assertEqual(result["per_item"][0]["f1"], 100)
        self.assertEqual(result["per_item"][1], {"id": "b", "scored": False, "em": None, "f1": None})
        self.assertEqual(result["metrics"]["skipped"], 0)
        self.assertIsNotNone(result["scoring_provenance"])

    def test_no_references_do_not_invoke_scorer_or_invent_zero_scores(self):
        rows = [{"id": "a", "answers": []}]
        with patch("cmrc_official.score_examples", side_effect=AssertionError("must not score")):
            result = custom.score_predictions(rows, [{"id": "a", "answer": "anything"}])
        self.assertEqual(result["status"], "not_requested_no_references")
        self.assertIsNone(result["metrics"])
        self.assertIsNone(result["per_item"][0]["em"])
        self.assertIsNone(result["per_item"][0]["f1"])

    def test_prediction_missing_duplicate_extra_and_nonstring_are_errors(self):
        rows = [{"id": "a", "answers": []}]
        for predictions in ([], [{"id": "other", "answer": ""}], [{"id": "a", "answer": None}],
                            [{"id": "a", "answer": ""}, {"id": "a", "answer": ""}]):
            with self.subTest(predictions=predictions), self.assertRaisesRegex(ValueError, "exactly once"):
                custom.score_predictions(rows, predictions)

    def test_manifest_kind_dispatch_is_explicit_and_ambiguous_identity_rejected(self):
        model = self.root / "model"
        model.mkdir()
        (model / "download_manifest.json").write_text("{}")
        with patch.object(custom.quality, "model_provenance", return_value=({"checkpoint_kind": "original"}, None)) as mocked:
            self.assertEqual(custom.checkpoint_provenance(model, "rev", "original")[0]["checkpoint_kind"], "original")
            mocked.assert_called_once_with(model, "rev")
            with self.assertRaisesRegex(ValueError, "Expected only"):
                custom.checkpoint_provenance(model, "rev", "pruned")
        (model / "pruning_manifest.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Expected only"):
            custom.checkpoint_provenance(model, "rev", "original")

    def test_merged_provenance_uses_new_verified_interface_and_keeps_pruning_manifest(self):
        model = self.root / "merged"
        model.mkdir()
        (model / "merge_manifest.json").write_text("{}")
        pruning = {"width_after": 5888, "parameters_after": 123}
        module = types.ModuleType("merged_checkpoint")
        calls = []
        def verified(directory, revision):
            calls.append((directory, revision))
            return {"checkpoint_kind": "merged_lora", "adapter_merged": True, "recovery_applied": True}, pruning
        module.merged_model_provenance = verified
        with patch.dict(sys.modules, {"merged_checkpoint": module}):
            actual = custom.checkpoint_provenance(model, "pinned", "merged_lora")
        self.assertIs(actual[1], pruning)
        self.assertEqual(calls, [(model, "pinned")])
        (model / "adapter_config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "unmerged adapter"):
            custom.checkpoint_provenance(model, "pinned", "merged_lora")

    def cli_fixture(self):
        model = self.root / "model"
        model.mkdir()
        for name in ("tokenizer.json", "tokenizer_config.json"):
            (model / name).write_text("{}")
        output = self.root / "output"
        argv = ["--model-dir", str(model), "--checkpoint-kind", "original", "--revision", "rev",
                "--questions", str(self.questions), "--output-dir", str(output)]
        return output, argv

    def test_cli_cpu_stub_records_complete_mixed_report_and_escaped_markdown(self):
        self.rows[0]["question"] = "<script> | 问题\n第二行"
        self.write_rows(self.rows)
        output, argv = self.cli_fixture()
        def fake_model(args, report, rows, tokenizer, pruning):
            report["qa"] = [{**row, "answer": "蓝色" if row["id"] == "one" else "未评分回答", "stop_reason": "eos"} for row in rows]
            with (args.output_dir / "predictions.jsonl").open("x", encoding="utf-8") as stream:
                for row in report["qa"]:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()), \
                patch.object(custom.quality, "tokenizer_signature", return_value={"test": "cpu"}), \
                patch.object(custom, "checkpoint_provenance", return_value=({"checkpoint_kind": "original"}, None)), \
                patch.object(custom, "_run_model", side_effect=fake_model):
            custom.main(argv)
        report = json.loads((output / "questions_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["scoring"]["metrics"]["f1"], 100)
        self.assertEqual(report["scoring"]["unscored_count"], 2)
        rendered = (output / "questions_report.md").read_text(encoding="utf-8")
        self.assertIn("&lt;script&gt; &#124; 问题<br>第二行", rendered)
        self.assertNotIn("<script>", rendered)
        self.assertIn("不是固定CMRC", rendered)

    def test_cli_partial_failure_and_rerun_cannot_overwrite_error_outputs(self):
        output, argv = self.cli_fixture()
        def fail_after_one(args, report, rows, tokenizer, pruning):
            report["stage"] = "generation"
            row = {**rows[0], "answer": "已完成的回答", "stop_reason": "eos"}
            (args.output_dir / "predictions.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            raise RuntimeError("simulated failure after one answer")
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()), \
                patch.object(custom.quality, "tokenizer_signature", return_value={"test": "cpu"}), \
                patch.object(custom, "checkpoint_provenance", return_value=({"checkpoint_kind": "original"}, None)), \
                patch.object(custom, "_run_model", side_effect=fail_after_one), \
                self.assertRaisesRegex(RuntimeError, "simulated failure"):
            custom.main(argv)
        report = json.loads((output / "questions_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["stage"], "generation")
        self.assertEqual([item["id"] for item in report["qa"]], ["one"])
        self.assertNotIn("scoring", report)
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        with self.assertRaises(FileExistsError):
            custom.main(argv)
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})

    def test_invalid_question_file_leaves_failed_report_before_tokenizer_or_gpu(self):
        output, argv = self.cli_fixture()
        self.questions.write_text("", encoding="utf-8")
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=AssertionError("no tokenizer")), \
                patch.object(custom, "_run_model", side_effect=AssertionError("no model")), \
                self.assertRaisesRegex(ValueError, "empty"):
            custom.main(argv)
        report = json.loads((output / "questions_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["stage"], "question_validation")
        self.assertFalse((output / "predictions.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
