"""Comparison integrity and engineering-gate boundary checks; CPU only."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_quality import compare


def report_fixture():
    return {
        "status": "completed", "source_model_revision": "pinned-source", "split": "validation",
        "protocol": {"qa": {"max_new_tokens": 64}, "lm": {"window": 512, "stride": 256},
                     "acceptance": {"max_f1_drop_percentage_points": 5, "max_ppl_relative_increase": 0.1}},
        "generation_config": {"do_sample": False, "max_new_tokens": 64},
        "decode": {"seed": 42, "batch_size": 1}, "runner_sha256": "runner-v1",
        "helper_sha256": {"cmrc_official.py": "qa-helper-v1", "eval_ppl.py": "ppl-helper-v1"},
        "scoring_provenance": {"metric": "cmrc2018_official", "upstream_sha256": "vendor-v1",
                               "punkt": {"archive_sha256": "punkt-v1"}},
        "data": {"manifest_sha256": "manifest-v1", "files_sha256": {"qa.jsonl": "qa-v1", "lm.jsonl": "lm-v1"},
                 "qa_file": "validation_qa.jsonl", "lm_file": "validation_lm.jsonl", "qa_count": 2},
        "data_manifest": {"counts": {"validation_qa": {"groups": 2}}},
        "tokenizer_compatibility": {"status": "passed", "signature_sha256": "tokenizer-v1"},
        "environment": {"selected_dtype": "torch.bfloat16", "attention_implementation": "sdpa",
                        "packages": {"torch": "2.9.1", "transformers": "4.57.6"}},
        "lm": {"ppl": 10.0, "nll": 2.302585092994046, "valid_token_count": 8192},
        "qa_metrics": {"count": 2, "skipped": 0, "em": 50.0, "f1": 80.0,
                       "per_item": [{"id": "q1", "em": 0.0, "f1": 60.0},
                                    {"id": "q2", "em": 100.0, "f1": 100.0}]},
        "qa": [{"id": "q1", "answer": "北京"}, {"id": "q2", "answer": "1903年"}],
        "provenance": {"checkpoint_kind": "original"},
        "model": {"unique_parameter_count": 1000},
    }


class CompareQualityTests(unittest.TestCase):
    def setUp(self):
        self.original = report_fixture()
        self.candidate = copy.deepcopy(self.original)
        self.candidate["provenance"]["checkpoint_kind"] = "pruned"
        self.candidate["model"]["unique_parameter_count"] = 950

    def test_same_protocol_uses_percentage_points_and_weighted_ppl_ratio(self):
        self.candidate["qa_metrics"]["f1"] = 78.0
        self.candidate["qa_metrics"]["per_item"][0]["f1"] = 56.0
        self.candidate["lm"]["ppl"] = 10.5
        result = compare(self.original, self.candidate)
        self.assertEqual(result["f1_drop_percentage_points"], 2.0)
        self.assertAlmostEqual(result["ppl_relative_increase"], 0.05)
        self.assertAlmostEqual(result["parameter_reduction"], 0.05)
        self.assertTrue(result["numeric_quality_gate_passed"])
        self.assertTrue(result["human_review_required"])
        self.assertEqual(result["question_count"], 2)

    def test_exact_ten_percent_ppl_and_five_point_f1_boundary_passes(self):
        self.candidate["lm"]["ppl"] = 11.0
        self.candidate["qa_metrics"]["f1"] = 75.0
        self.candidate["qa_metrics"]["per_item"][0]["f1"] = 50.0
        self.assertTrue(compare(self.original, self.candidate)["numeric_quality_gate_passed"])

    def test_either_gate_exceeded_fails(self):
        self.candidate["lm"]["ppl"] = 11.000001
        self.assertFalse(compare(self.original, self.candidate)["numeric_quality_gate_passed"])
        self.candidate["lm"]["ppl"] = 10.0
        self.candidate["qa_metrics"]["f1"] = 74.999999
        self.assertFalse(compare(self.original, self.candidate)["numeric_quality_gate_passed"])

    def test_improvements_are_not_misread_as_degradation(self):
        self.candidate["qa_metrics"]["f1"] = 90
        self.candidate["lm"]["ppl"] = 8
        result = compare(self.original, self.candidate)
        self.assertTrue(result["numeric_quality_gate_passed"])
        self.assertEqual(result["f1_drop_percentage_points"], -10)
        self.assertAlmostEqual(result["ppl_relative_increase"], -0.2)

    def test_custom_thresholds_drive_gate(self):
        for report in (self.original, self.candidate):
            report["protocol"]["acceptance"]["max_f1_drop_percentage_points"] = 2
        self.candidate["qa_metrics"]["f1"] = 77.9
        self.assertFalse(compare(self.original, self.candidate)["numeric_quality_gate_passed"])

    def test_failed_report_and_reversed_baseline_rejected(self):
        self.candidate["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "complete"):
            compare(self.original, self.candidate)
        self.candidate["status"] = "completed"
        with self.assertRaisesRegex(ValueError, "original"):
            compare(self.candidate, self.original)

    def test_original_candidate_is_allowed_for_same_budget_controls(self):
        self.candidate["provenance"]["checkpoint_kind"] = "original"
        result = compare(self.original, self.candidate)
        self.assertTrue(result["numeric_quality_gate_passed"])

    def test_changed_revision_protocol_and_code_rejected(self):
        replacements = {
            "source_model_revision": "another-source", "split": "smoke", "protocol": {},
            "generation_config": {}, "decode": {}, "runner_sha256": "runner-v2",
            "helper_sha256": {}, "scoring_provenance": {},
        }
        for key, changed in replacements.items():
            candidate = copy.deepcopy(self.candidate)
            candidate[key] = changed
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare(self.original, candidate)

    def test_changed_vendor_scorer_and_punkt_assets_rejected(self):
        for key in ("upstream_sha256", "punkt"):
            candidate = copy.deepcopy(self.candidate)
            candidate["scoring_provenance"][key] = "changed"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "scoring_provenance"):
                compare(self.original, candidate)

    def test_changed_data_hashes_or_tokenizer_rejected(self):
        for key in ("manifest_sha256", "files_sha256", "qa_file", "lm_file"):
            candidate = copy.deepcopy(self.candidate)
            candidate["data"][key] = "different"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "data"):
                compare(self.original, candidate)
        self.candidate["tokenizer_compatibility"]["signature_sha256"] = "tokenizer-v2"
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            compare(self.original, self.candidate)

    def test_changed_precision_backend_or_library_rejected(self):
        for key, value in (("selected_dtype", "torch.float16"), ("attention_implementation", "eager"),
                           ("packages", {"torch": "2.8.0"})):
            candidate = copy.deepcopy(self.candidate)
            candidate["environment"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "environment"):
                compare(self.original, candidate)

    def test_different_ppl_denominator_rejected(self):
        self.candidate["lm"]["valid_token_count"] -= 1
        with self.assertRaisesRegex(ValueError, "denominator"):
            compare(self.original, self.candidate)

    def test_invalid_ppl_values_rejected(self):
        for value in (None, float("nan"), float("inf"), 0, -1):
            candidate = copy.deepcopy(self.candidate)
            candidate["lm"]["ppl"] = value
            with self.subTest(ppl=value), self.assertRaises(ValueError):
                compare(self.original, candidate)
        for report in (self.original, self.candidate):
            report["lm"]["valid_token_count"] = 0
        with self.assertRaises(ValueError):
            compare(self.original, self.candidate)

    def test_duplicate_omitted_and_unscored_questions_rejected(self):
        mutations = [
            lambda r: r["qa_metrics"]["per_item"].append(copy.deepcopy(r["qa_metrics"]["per_item"][0])),
            lambda r: r["qa"].append(copy.deepcopy(r["qa"][0])),
            lambda r: r["qa_metrics"].update(count=1),
            lambda r: r["data"].update(qa_count=3),
            lambda r: r["qa_metrics"].update(skipped=1),
            lambda r: r["qa"].pop(),
            lambda r: r["qa_metrics"]["per_item"][0].update(id="different-id"),
        ]
        for mutation in mutations:
            candidate = copy.deepcopy(self.candidate)
            mutation(candidate)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "coverage"):
                compare(self.original, candidate)

    def test_coherent_but_different_question_ids_rejected(self):
        self.candidate["qa"][0]["id"] = "other"
        self.candidate["qa_metrics"]["per_item"][0]["id"] = "other"
        with self.assertRaisesRegex(ValueError, "question IDs"):
            compare(self.original, self.candidate)

    def test_invalid_qa_aggregate_rejected(self):
        for value in (-1, 101, float("nan"), float("inf")):
            candidate = copy.deepcopy(self.candidate)
            candidate["qa_metrics"]["f1"] = value
            with self.subTest(f1=value), self.assertRaisesRegex(ValueError, "QA metrics"):
                compare(self.original, candidate)


if __name__ == "__main__":
    unittest.main()
