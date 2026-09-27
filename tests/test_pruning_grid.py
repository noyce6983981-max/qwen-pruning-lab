"""CPU-only integrity and arithmetic checks for the fixed pruning control grid."""

import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_pruning_grid import METHODS, render_markdown, summarize_grid, write_summary


def grid_fixture():
    original = {
        "status": "completed", "source_model_revision": "pinned-source", "split": "validation",
        "protocol": {"qa": {"max_new_tokens": 64}, "lm": {"window": 512, "stride": 256},
                     "acceptance": {"max_f1_drop_percentage_points": 5, "max_ppl_relative_increase": 0.1}},
        "generation_config": {"do_sample": False}, "decode": {"seed": 42},
        "runner_sha256": "runner-v1", "helper_sha256": {"eval_ppl.py": "ppl-v1"},
        "scoring_provenance": {"metric": "cmrc2018_official", "source_sha256": "scorer-v1"},
        "data": {"manifest_sha256": "manifest-v1", "files_sha256": {"qa.jsonl": "qa-v1"},
                 "qa_file": "validation_qa.jsonl", "lm_file": "validation_lm.jsonl", "qa_count": 100},
        "data_manifest": {"counts": {"validation_qa": {"rows": 100, "groups": 99}}},
        "tokenizer_compatibility": {"signature_sha256": "tokenizer-v1"},
        "environment": {"selected_dtype": "torch.bfloat16", "attention_implementation": "sdpa",
                        "packages": {"torch": "2.9.1", "transformers": "4.57.6"}},
        "lm": {"ppl": 10.0, "nll": math.log(10), "valid_token_count": 8192},
        "qa_metrics": {"count": 100, "skipped": 0, "em": 50.0, "f1": 80.0,
                       "per_item": [{"id": f"q{i}", "em": 100.0 if i < 50 else 0.0, "f1": 80.0}
                                    for i in range(100)]},
        "qa": [{"id": f"q{i}", "answer": "答案"} for i in range(100)],
        "qa_output_diagnostics": {"max_new_tokens_stop_count": 0},
        "provenance": {"checkpoint_kind": "original", "download_manifest": {
            "model_id": "Qwen/Qwen3-1.7B", "revision": "pinned-source"}},
        "model": {"unique_parameter_count": 1000, "intermediate_size": 6144},
    }
    candidates = []
    for method, width, f1, ppl, cap in [
        (METHODS[0], 5888, 60.0, 20.0, 20), (METHODS[1], 5888, 78.0, 10.5, 2),
        (METHODS[0], 5632, 10.0, 100.0, 90), (METHODS[1], 5632, 76.0, 11.0, 3),
    ]:
        report = copy.deepcopy(original)
        params = 960 if width == 5888 else 920
        report["provenance"] = {
            "checkpoint_kind": "pruned", "method": method, "width_before": 6144,
            "width_after": width, "parameters_before": 1000, "parameters_after": params,
            "recovery_applied": False,
            "source": {"model_id": "Qwen/Qwen3-1.7B", "revision": "pinned-source"},
        }
        report["model"] = {"unique_parameter_count": params, "intermediate_size": width}
        report["qa_metrics"]["f1"] = f1
        for item in report["qa_metrics"]["per_item"]:
            item["f1"] = f1
        report["lm"].update({"ppl": ppl, "nll": math.log(ppl)})
        report["qa_output_diagnostics"]["max_new_tokens_stop_count"] = cap
        candidates.append(report)
    return original, candidates


class PruningGridTests(unittest.TestCase):
    def setUp(self):
        self.original, self.candidates = grid_fixture()

    def test_five_rows_differences_and_compression_priority(self):
        summary = summarize_grid(self.original, list(reversed(self.candidates)))
        rows = summary["rows"]
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["id"], "original")
        self.assertIsNone(rows[0]["numeric_quality_gate_passed"])
        self.assertEqual(rows[2]["f1_change_percentage_points"], -2)
        self.assertEqual(rows[2]["ppl_change"], 0.5)
        self.assertAlmostEqual(rows[2]["ppl_relative_change"], 0.05)
        self.assertAlmostEqual(rows[2]["parameter_reduction"], 0.04)
        self.assertFalse(rows[1]["numeric_quality_gate_passed"])
        self.assertTrue(rows[4]["numeric_quality_gate_passed"])
        diff = summary["same_width_comparisons"][0]
        self.assertEqual(diff["f1_change_percentage_points"], 18)
        self.assertEqual(diff["ppl_change"], -9.5)
        self.assertAlmostEqual(diff["nll_change"], math.log(10.5) - math.log(20))
        self.assertEqual(diff["cap_change"], -18)
        self.assertEqual([r["width"] for r in summary["qualitative_review_candidates"]], [5632, 5888])
        self.assertTrue(all(r["human_review_required"] for r in summary["qualitative_review_candidates"]))
        markdown = render_markdown(summary)
        self.assertIn("数字门槛不等于人工验收", markdown)
        self.assertIn("不能概括所有幅值剪枝", markdown)

    def test_missing_duplicate_and_unsupported_pairs_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly four"):
            summarize_grid(self.original, self.candidates[:3])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_grid(self.original, self.candidates[:3] + [copy.deepcopy(self.candidates[0])])
        self.candidates[0]["provenance"]["width_after"] = 5376
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            summarize_grid(self.original, self.candidates)

    def test_compare_rejects_protocol_mismatch_and_incomplete_report(self):
        self.candidates[0]["decode"]["seed"] = 43
        with self.assertRaisesRegex(ValueError, "Incompatible evaluation: decode"):
            summarize_grid(self.original, self.candidates)
        self.candidates[0]["decode"]["seed"] = 42
        self.candidates[1]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "complete"):
            summarize_grid(self.original, self.candidates)

    def test_same_width_requires_equal_actual_parameter_counts(self):
        self.candidates[1]["model"]["unique_parameter_count"] = 959
        self.candidates[1]["provenance"]["parameters_after"] = 959
        with self.assertRaisesRegex(ValueError, "same width 5888"):
            summarize_grid(self.original, self.candidates)

    def test_candidate_identity_recovery_and_width_checked(self):
        for path, value, pattern in [
            (("provenance", "source", "model_id"), "another/model", "source identity"),
            (("provenance", "source", "revision"), "another-revision", "source identity"),
            (("provenance", "recovery_applied"), True, "recovery_applied"),
            (("provenance", "checkpoint_kind"), "original", "must be pruned"),
            (("model", "intermediate_size"), 5632, "width metadata"),
        ]:
            candidates = copy.deepcopy(self.candidates)
            target = candidates[0]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, pattern):
                summarize_grid(self.original, candidates)

    def test_smoke_cannot_be_mislabeled_as_fixed_validation(self):
        self.original["split"] = "smoke"
        with self.assertRaisesRegex(ValueError, "requires validation"):
            summarize_grid(self.original, self.candidates)

    def test_only_numerically_passing_calibrated_candidates_are_eligible(self):
        self.candidates[3]["lm"]["ppl"] = 11.1
        summary = summarize_grid(self.original, self.candidates)
        self.assertEqual([r["width"] for r in summary["qualitative_review_candidates"]], [5888])
        self.candidates[1]["qa_metrics"]["f1"] = 74.9
        summary = summarize_grid(self.original, self.candidates)
        self.assertEqual(summary["qualitative_review_candidates"], [])
        self.assertIn("没有校准候选通过数值门槛", render_markdown(summary))

    def test_reports_hashed_and_outputs_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for i, report in enumerate([self.original, *self.candidates]):
                path = root / f"report-{i}.json"
                path.write_text(json.dumps(report), encoding="utf-8")
                paths.append(path)
            out = root / "summary"
            summary = write_summary(paths[0], paths[1:], out)
            self.assertEqual(len(summary["inputs"]), 5)
            for item in summary["inputs"]:
                self.assertEqual(hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest(), item["sha256"])
            saved = json.loads((out / "pruning_grid.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, summary)
            before = (out / "pruning_grid.json").read_bytes()
            with self.assertRaises(FileExistsError):
                write_summary(paths[0], paths[1:], out)
            self.assertEqual((out / "pruning_grid.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
