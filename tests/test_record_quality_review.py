"""CPU-only review-note binding, with no claims of automated human acceptance."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import record_quality_review as review


class RecordQualityReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.summary = self.root / "summary.json"
        self.notes = self.root / "notes.md"
        self.notes.write_text("这是合成测试中的操作员笔记，不是真实模型验收。\n保留观察到的失败案例。\n", encoding="utf-8")
        self.value = {"status": "completed", "split": "validation", "selection_status": "numeric_proposals_only",
                      "qualitative_review_required": True, "selected": {}}
        for arm in ("original", "calibrated_5888"):
            path = self.root / f"{arm}.json"
            self.write(path, {"status": "completed", "split": "validation", "provenance": {"training_step": 64}})
            self.value["selected"][arm] = {"checkpoint": {"training_step": 64, "numeric_quality_gate_passed": True,
                                                           "report_path": str(path), "report_sha256": self.digest(path)}}
        self.write(self.summary, self.value)

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def digest(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_preserves_notes_hashes_and_self_declaration_without_acceptance(self):
        result = review.record_review(self.summary, self.notes, "Operator A")
        self.assertEqual(result["status"], "completed_operator_review")
        self.assertEqual(result["notes"], self.notes.read_bytes().decode("utf-8"))
        self.assertEqual(result["bindings"]["numeric_summary"]["sha256"], self.digest(self.summary))
        self.assertEqual(result["bindings"]["notes"]["sha256"], self.digest(self.notes))
        self.assertEqual(len(result["bindings"]["selected_validation_reports"]), 2)
        self.assertTrue(result["operator_review_self_declared"])
        for field in ("actual_reading_verified", "factual_correctness_automatically_verified",
                      "independent_review_performed", "human_acceptance_performed"):
            self.assertIs(result[field], False)

    def test_empty_notes_do_not_create_review(self):
        self.notes.write_text(" \n\t ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "nonempty operator-written"):
            review.record_review(self.summary, self.notes)

    def test_changed_selected_report_is_rejected(self):
        path = Path(self.value["selected"]["original"]["checkpoint"]["report_path"])
        with path.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "changed after summary"):
            review.record_review(self.summary, self.notes)

    def test_final_or_incomplete_summary_and_failing_selection_rejected(self):
        for key, value in (("status", "failed"), ("split", "final"), ("qualitative_review_required", False)):
            old = self.value[key]
            self.value[key] = value
            self.write(self.summary, self.value)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "completed recovery validation"):
                review.record_review(self.summary, self.notes)
            self.value[key] = old
        self.value["selected"]["original"]["checkpoint"]["numeric_quality_gate_passed"] = False
        self.write(self.summary, self.value)
        with self.assertRaisesRegex(ValueError, "no numerically passing"):
            review.record_review(self.summary, self.notes)

    def test_bound_final_report_or_wrong_step_cannot_be_passed_as_validation(self):
        chosen = self.value["selected"]["original"]["checkpoint"]
        path = Path(chosen["report_path"])
        for split, step in (("final", 64), ("validation", 32)):
            self.write(path, {"status": "completed", "split": split, "provenance": {"training_step": step}})
            chosen["report_sha256"] = self.digest(path)
            self.write(self.summary, self.value)
            with self.subTest(split=split, step=step), self.assertRaisesRegex(ValueError, "completed validation checkpoint"):
                review.record_review(self.summary, self.notes)

    def test_cli_refuses_overwrite_before_work_and_records_new_file(self):
        output = self.root / "review.json"
        arguments = ["--summary", str(self.summary), "--notes", str(self.notes), "--output", str(output)]
        review.main(arguments)
        before = output.read_bytes()
        with patch.object(review, "record_review", side_effect=AssertionError("no input work")), self.assertRaises(FileExistsError):
            review.main(arguments)
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
