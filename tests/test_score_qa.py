"""CPU-only behavioral tests for transparent short-answer scoring.

Character F1 is a normalized multiset overlap, not a factual-correctness score.
No model, tokenizer, network access, or GPU is used by this test module.
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from score_qa import answer_scores, normalize_answer, score_records


class NormalizeAnswerTests(unittest.TestCase):
    def test_nfkc_casefold_whitespace_and_unicode_punctuation(self):
        text = " \tＡＢＣ，Straße\u3000①【测试】—_…\n！؟ "
        self.assertEqual(normalize_answer(text), "abcstrasse1测试")

    def test_punctuation_categories_are_removed_but_symbols_remain(self):
        self.assertEqual(normalize_answer("Ａ＋Ｂ＝２ 😀￥"), "a+b=2😀¥")
        self.assertEqual(normalize_answer("‘甲’（乙）/丙：丁；戊。"), "甲乙丙丁戊")

    def test_normalization_is_idempotent(self):
        once = normalize_answer("【ＮＦＫＣ】 Straße\n①—Ｂ")
        self.assertEqual(normalize_answer(once), once)


class AnswerScoreTests(unittest.TestCase):
    def test_normalized_equivalence_counts_as_exact_match(self):
        result = answer_scores(" ＡＢＣ，Straße！ ", ["abc STRASSE"])
        self.assertEqual(result["exact_match"], 1)
        self.assertEqual(result["char_f1"], 1.0)

    def test_repeated_characters_use_multiset_overlap(self):
        # Intersection is two characters, not the two entire sets of characters:
        # precision=2/3, recall=2/4, F1=4/7.
        result = answer_scores("甲甲乙", ["甲乙乙乙"])
        self.assertEqual(result["exact_match"], 0)
        self.assertAlmostEqual(result["char_f1"], 4 / 7)

    def test_best_reference_is_used_for_partial_match(self):
        references = ["丙丁", "甲乙乙乙", "甲丙"]
        result = answer_scores("甲甲乙", references)
        self.assertEqual(result["exact_match"], 0)
        self.assertAlmostEqual(result["char_f1"], 4 / 7)
        self.assertEqual(result, answer_scores("甲甲乙", list(reversed(references))))

    def test_exact_reference_need_not_be_first(self):
        result = answer_scores("周五下午五点", ["星期四", "周五下午五点", "周五"])
        self.assertEqual(result["exact_match"], 1)
        self.assertEqual(result["char_f1"], 1.0)

    def test_empty_prediction_and_no_overlap_score_zero(self):
        for prediction in ("", " \t\n", "【】！？", "丙丁"):
            with self.subTest(prediction=prediction):
                result = answer_scores(prediction, ["甲乙"])
                self.assertEqual(result["exact_match"], 0)
                self.assertEqual(result["char_f1"], 0.0)

    def test_negation_can_have_high_f1_despite_opposite_meaning(self):
        # These answers contradict each other. The scorer deliberately reports
        # lexical overlap only; a high F1 must not be advertised as correctness.
        result = answer_scores("不允许延期提交报告", ["允许延期提交报告"])
        self.assertEqual(result["exact_match"], 0)
        self.assertAlmostEqual(result["char_f1"], 16 / 17)
        self.assertGreater(result["char_f1"], 0.9)


class ScoreRecordTests(unittest.TestCase):
    @staticmethod
    def gold():
        return [
            {"id": "a", "question": "第一个答案？", "answers": ["甲"]},
            {"id": "b", "question": "第二个答案？", "answers": ["乙"]},
            {"id": "c", "question": "第三个答案？", "answers": ["丙"]},
        ]

    def test_missing_predictions_stay_in_the_denominator(self):
        gold = self.gold()
        predictions = [{"id": "b", "prediction": "丁"}, {"id": "a", "prediction": "甲"}]
        original_gold = copy.deepcopy(gold)
        original_predictions = copy.deepcopy(predictions)
        report = score_records(gold, predictions)
        self.assertEqual(report["count"], 3)
        self.assertEqual(report["predicted_count"], 2)
        self.assertAlmostEqual(report["exact_match"], 1 / 3)
        self.assertAlmostEqual(report["char_f1"], 1 / 3)
        self.assertEqual([row["id"] for row in report["rows"]], ["a", "b", "c"])
        for row, reference in zip(report["rows"], gold):
            self.assertTrue({"id", "question", "answers", "prediction", "missing_prediction", "exact_match", "char_f1"}.issubset(row))
            self.assertEqual(row["question"], reference["question"])
            self.assertEqual(row["answers"], reference["answers"])
        missing = report["rows"][2]
        self.assertTrue(missing["missing_prediction"])
        self.assertEqual(missing["exact_match"], 0)
        self.assertEqual(missing["char_f1"], 0.0)
        self.assertFalse(report["rows"][0]["missing_prediction"])
        self.assertEqual(gold, original_gold)
        self.assertEqual(predictions, original_predictions)

    def test_present_but_empty_prediction_is_not_missing(self):
        report = score_records(self.gold()[:1], [{"id": "a", "prediction": ""}])
        self.assertEqual(report["predicted_count"], 1)
        self.assertFalse(report["rows"][0]["missing_prediction"])
        self.assertEqual(report["rows"][0]["prediction"], "")
        self.assertEqual(report["exact_match"], 0.0)
        self.assertEqual(report["char_f1"], 0.0)

    def test_all_missing_predictions_score_zero_with_zero_coverage(self):
        report = score_records(self.gold(), [])
        self.assertEqual(report["count"], 3)
        self.assertEqual(report["predicted_count"], 0)
        self.assertEqual(report["exact_match"], 0.0)
        self.assertEqual(report["char_f1"], 0.0)
        self.assertTrue(all(row["missing_prediction"] for row in report["rows"]))

    def test_dataset_char_f1_is_mean_over_questions(self):
        gold = [
            {"id": "short", "question": "短答案", "answers": ["甲甲"]},
            {"id": "long", "question": "长答案", "answers": ["乙" * 20]},
        ]
        report = score_records(gold, [
            {"id": "short", "prediction": "甲"},
            {"id": "long", "prediction": "丙" * 20},
        ])
        self.assertEqual(report["exact_match"], 0.0)
        self.assertAlmostEqual(report["rows"][0]["char_f1"], 2 / 3)
        self.assertAlmostEqual(report["char_f1"], 1 / 3)

    def test_extra_prediction_id_is_rejected(self):
        with self.assertRaises(ValueError):
            score_records(self.gold(), [{"id": "outside", "prediction": "甲"}])

    def test_duplicate_gold_id_is_rejected(self):
        with self.assertRaises(ValueError):
            score_records(self.gold() + [{"id": "a", "question": "另一题", "answers": ["丁"]}], [])

    def test_duplicate_prediction_id_is_rejected_even_if_one_is_empty(self):
        with self.assertRaises(ValueError):
            score_records(self.gold(), [{"id": "a", "prediction": ""}, {"id": "a", "prediction": "甲"}])

    def test_invalid_gold_schema_is_rejected(self):
        invalid_rows = [
            None,
            "not a record",
            {"question": "题目", "answers": ["甲"]},
            {"id": "a", "answers": ["甲"]},
            {"id": "a", "question": "题目"},
            {"id": "", "question": "题目", "answers": ["甲"]},
            {"id": 1, "question": "题目", "answers": ["甲"]},
            {"id": "a", "question": None, "answers": ["甲"]},
            {"id": "a", "question": "题目", "answers": "甲"},
            {"id": "a", "question": "题目", "answers": []},
            {"id": "a", "question": "题目", "answers": [""]},
            {"id": "a", "question": "题目", "answers": ["甲", None]},
            {"id": "a", "question": "题目", "answers": [1]},
        ]
        for row in invalid_rows:
            with self.subTest(row=row), self.assertRaises((TypeError, ValueError)):
                score_records([row], [])

    def test_invalid_prediction_schema_is_rejected(self):
        invalid_rows = [
            None,
            "not a record",
            {"prediction": "甲"},
            {"id": "a"},
            {"id": "", "prediction": "甲"},
            {"id": 1, "prediction": "甲"},
            {"id": "a", "prediction": None},
            {"id": "a", "prediction": 1},
            {"id": "a", "prediction": ["甲"]},
        ]
        for row in invalid_rows:
            with self.subTest(row=row), self.assertRaises((TypeError, ValueError)):
                score_records(self.gold(), [row])

    def test_invalid_top_level_schema_is_rejected(self):
        for value in (None, "not a list", {"id": "a"}):
            with self.subTest(gold=value), self.assertRaises((TypeError, ValueError)):
                score_records(value, [])
            with self.subTest(predictions=value), self.assertRaises((TypeError, ValueError)):
                score_records(self.gold(), value)


if __name__ == "__main__":
    unittest.main()
