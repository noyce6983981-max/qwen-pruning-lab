"""CPU-only regression checks for the pinned official metric and strict adapter."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cmrc_official as cmrc


class OfficialMetricTests(unittest.TestCase):
    def score(self, prediction, answers):
        return cmrc.score_examples([{"id": "q", "answers": answers}], {"q": prediction})

    def test_chinese_punctuation_case_and_multiple_references(self):
        result = self.score("《中国ABC》，", ["无关", "中国abc"])
        self.assertEqual((result["em"], result["f1"]), (100.0, 100.0))

    def test_f1_is_longest_contiguous_match_not_bag_overlap(self):
        result = self.score("甲丁丙", ["甲乙丙"])
        self.assertEqual(result["em"], 0)
        self.assertAlmostEqual(result["f1"], 100 / 3)

    def test_repeated_tokens_do_not_become_set_overlap(self):
        self.assertAlmostEqual(self.score("甲甲乙", ["甲乙乙乙"])["f1"], 400 / 7)

    def test_english_uses_words_and_numbers_remain_words(self):
        result = self.score("Model 2020", ["Model 2020 version"])
        self.assertEqual(result["em"], 0)
        self.assertAlmostEqual(result["f1"], 80)

    def test_internal_spaces_are_not_removed_by_em(self):
        result = self.score("new  york", ["New York"])
        self.assertEqual((result["em"], result["f1"]), (0.0, 100.0))

    def test_no_nfkc_normalization(self):
        result = self.score("ABC", ["ＡＢＣ"])
        self.assertEqual((result["em"], result["f1"]), (0.0, 0.0))

    def test_empty_and_no_overlap_predictions(self):
        for pred in ("", "乙", "，。"):
            with self.subTest(pred=pred):
                result = self.score(pred, ["甲"])
                self.assertEqual((result["em"], result["f1"]), (0.0, 0.0))

    def test_official_empty_normalization_quirk_is_preserved(self):
        # Upstream EM accepts two punctuation-only answers, but F1 has no LCS.
        result = self.score("，", ["。"])
        self.assertEqual((result["em"], result["f1"]), (100.0, 0.0))

    def test_aggregate_and_each_item_match_official_evaluate(self):
        examples = [{"id": "a", "answers": ["甲乙丙", "甲乙"]},
                    {"id": "b", "answers": ["New York 2018"]},
                    {"id": "c", "answers": ["答案"]}]
        predictions = {"a": "甲乙", "b": "New York", "c": ""}
        result = cmrc.score_examples(examples, predictions)
        official = cmrc.load_official_scorer()
        dataset = {"data": [{"paragraphs": [{"qas": [
            {"id": e["id"], "question": "?", "answers": [{"text": a} for a in e["answers"]]}
            for e in examples]}]}]}
        f1, em, total, skipped = official.evaluate(dataset, predictions)
        self.assertEqual((result["f1"], result["em"], result["count"], result["skipped"]), (f1, em, total, skipped))
        for example, row in zip(examples, result["per_item"]):
            dataset["data"][0]["paragraphs"][0]["qas"] = [{"id": example["id"], "question": "?", "answers": [{"text": a} for a in example["answers"]]}]
            item_f1, item_em, _, _ = official.evaluate(dataset, predictions)
            self.assertEqual((row["f1"], row["em"]), (item_f1, item_em))

    def test_provenance_and_unmodified_source(self):
        provenance = cmrc.scoring_provenance()
        self.assertEqual(provenance["upstream_sha256"], cmrc.OFFICIAL_SHA256)
        self.assertEqual(provenance["nltk_version"], "3.9.2")
        self.assertIn("str(in_str).decode('utf-8')", cmrc.OFFICIAL_PATH.read_text(encoding="utf-8"))


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.example = {"id": "a", "answers": ["甲"]}

    def test_empty_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonempty list"):
            cmrc.score_examples([], {})

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate example"):
            cmrc.score_examples([self.example, self.example], {"a": "甲"})

    def test_missing_or_extra_predictions_are_rejected(self):
        for predictions in ({}, {"a": "甲", "extra": "甲"}, {"extra": "甲"}):
            with self.subTest(predictions=predictions), self.assertRaisesRegex(ValueError, "id mismatch"):
                cmrc.score_examples([self.example], predictions)

    def test_nonstring_prediction_and_nonmapping_rejected(self):
        for predictions in ({"a": 123}, [("a", "甲")], {1: "甲"}):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                cmrc.score_examples([self.example], predictions)

    def test_invalid_answers_rejected(self):
        for answers in ([], "甲", [123], [""], ["  "], None):
            with self.subTest(answers=answers), self.assertRaisesRegex(ValueError, "answers"):
                cmrc.score_examples([dict(self.example, answers=answers)], {"a": "甲"})

    def test_invalid_id_rejected_before_upstream_strip(self):
        for qid in ("", " a ", 1, None):
            with self.subTest(qid=qid), self.assertRaisesRegex(ValueError, "string id"):
                cmrc.score_examples([dict(self.example, id=qid)], {"a": "甲"})

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.json"
            path.write_text('{"a":"first","a":"second"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                cmrc.load_json_strict(path)


if __name__ == "__main__":
    unittest.main()
