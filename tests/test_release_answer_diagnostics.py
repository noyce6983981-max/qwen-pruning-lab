"""Post-hoc diagnostics reuse frozen scores; synthetic fixtures only."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import review_release_answers as review
import summarize_release as summary
import test_release_summary as summary_fixtures


class AnswerDiagnosticTests(unittest.TestCase):
    setUp = summary_fixtures.ReleaseSummaryTests.setUp
    write = summary_fixtures.ReleaseSummaryTests.write
    flush = summary_fixtures.ReleaseSummaryTests.flush

    def test_repeat_thresholds_preserve_previous_definition(self):
        for text in ("a" * 7, "abcd" * 2, "ab" * 5, "abcdefghijklmnopq" * 3, "有用的正常回答"):
            with self.subTest(text=text):
                self.assertIsNone(review.strict_repeat(text))
        for text in ("a" * 8, "abcd" * 3, "ab" * 6, "前缀" + "测 试文本" * 3):
            with self.subTest(text=text):
                self.assertIsNotNone(review.strict_repeat(text))

    def test_echo_keeps_punctuation_and_cap_does_not_mean_repetition(self):
        report = copy.deepcopy(self.reports[0])
        report["qa"][0].update(answer="测 试？", stop_reason="max_new_tokens")
        report["qa"][1].update(answer="测试?", stop_reason="eos")
        report["qa"][2].update(answer=" \n", stop_reason="eos")
        d = review.diagnostic(report)
        self.assertEqual(d["exact_question_echo_count"], 1)
        self.assertEqual(d["cap_count"], 1)
        self.assertEqual(d["strict_continuous_repetition_count"], 0)
        self.assertEqual(d["empty_count"], 1)

    def test_pair_counts_and_fixed_first_examples_use_stored_scores(self):
        before, after = copy.deepcopy(self.reports[4]), copy.deepcopy(self.reports[7])
        before["qa_metrics"]["per_item"][0].update(em=0, f1=20)
        after["qa_metrics"]["per_item"][1].update(em=0, f1=50)
        result = review.pair(before, after)
        self.assertEqual(result["counts"]["f1_improved"], 1)
        self.assertEqual(result["counts"]["f1_declined"], 1)
        self.assertEqual(result["counts"]["f1_unchanged"], 198)
        self.assertEqual(result["counts"]["em_gained"], 1)
        self.assertEqual(result["counts"]["em_lost"], 1)
        self.assertEqual(result["examples"]["first_three_em_gains"][0]["position"], 1)
        self.assertEqual(result["examples"]["first_three_em_losses"][0]["position"], 2)

    def make_summary(self):
        def provenance(directory, revision, policy_path, label):
            return next(r["provenance"] for r in self.reports if r["model_id"] == label), None
        with patch.object(summary, "release_provenance", side_effect=provenance):
            result = summary.summarize(self.policy_path, self.paths)
        path = self.root / "summary.json"
        self.write(path, result)
        return path, result

    def test_all_200_four_model_answers_retained_and_review_stays_pending(self):
        path, _ = self.make_summary()
        result = review.build(path)
        self.assertEqual(len(result["final_all_200"]), 200)
        self.assertEqual(len(result["final_all_200"][0]["models"]), 4)
        self.assertEqual(len(result["pairs"]), 10)
        self.assertEqual(result["status"], "diagnostics_completed_assistant_review_pending")
        self.assertFalse(result["human_acceptance_performed"])
        self.assertFalse(result["checkpoint_reselected"])
        self.assertEqual(result["parameter_efficiency"]["parameters_removed"], 100)
        self.assertIn("待助手逐题复核", review.markdown(result))

    def test_changed_report_hash_or_numeric_claim_is_rejected(self):
        path, numeric = self.make_summary()
        numeric["rows"][0]["f1"] = 12
        self.write(path, numeric)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            review.build(path)
        path, _ = self.make_summary()
        with self.paths[-1].open("a", encoding="utf8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "report changed"):
            review.build(path)


if __name__ == "__main__":
    unittest.main()
