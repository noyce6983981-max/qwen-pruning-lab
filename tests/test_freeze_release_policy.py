"""Release-policy binding tests using real operator-note recording and synthetic artifacts."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import freeze_release_policy as freeze
import record_quality_review as record


class FreezeReleasePolicyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.selection = self.root / "summary.json"
        self.review = self.root / "review.json"
        self.output = self.root / "policy.json"
        self.notes = self.root / "notes.md"
        self.notes.write_text("合成测试笔记：阅读后仍应保留事实错误与长度上限案例。", encoding="utf-8")
        self.data = self.root / "data"
        self.data.mkdir()
        self.acceptance = {"max_f1_drop_points": 5, "max_ppl_increase_fraction": 0.1,
                           "human_review": "No widespread repeated outputs."}
        self.write(self.data / "manifest.json", {"protocol": {"acceptance": self.acceptance}})
        self.directories = [self.root / label for label in freeze.MODEL_IDS]
        for directory in self.directories:
            directory.mkdir()
        self.provenances = {}
        self.summary = {"status": "completed", "split": "validation",
                        "selection_status": "numeric_proposals_only", "qualitative_review_required": True,
                        "selected": {}, "base_identities": {}}
        for index, arm in enumerate(("original", "calibrated_5888")):
            kind = ("original", "pruned")[index]
            base = {"checkpoint_kind": kind, "manifest_sha256": f"base-manifest-{index}",
                    "files_sha256": {"model.safetensors": f"base-weights-{index}"}}
            self.provenances[self.directories[index]] = base
            self.summary["base_identities"][arm] = copy.deepcopy(base)
            adapter_sha = f"adapter-manifest-{index}"
            report = self.root / f"{arm}-validation.json"
            self.write(report, {"status": "completed", "split": "validation",
                                "provenance": {"training_step": 64, "manifest_sha256": adapter_sha}})
            self.summary["selected"][arm] = {"checkpoint": {
                "training_step": 64, "numeric_quality_gate_passed": True,
                "report_path": str(report), "report_sha256": self.digest(report)}}
            self.provenances[self.directories[index + 2]] = {
                "checkpoint_kind": "merged_lora", "base_checkpoint_kind": kind, "training_step": 64,
                "manifest_sha256": f"merged-manifest-{index}",
                "files_sha256": {"model.safetensors": f"merged-weights-{index}"},
                "merge_manifest": {"source_adapter": {"manifest_sha256": adapter_sha}}}
        self.write(self.selection, self.summary)
        self.operator_review = record.record_review(self.selection, self.notes, "Fixture operator")
        self.write(self.review, self.operator_review)
        self.arguments = []
        for flag, value in zip(("original-model", "pruned-model", "original-merged", "pruned-merged"),
                               self.directories):
            self.arguments.extend(["--" + flag, str(value)])
        for flag, value in (("data-dir", self.data), ("selection", self.selection),
                            ("review", self.review), ("output", self.output), ("revision", "fixture-revision")):
            self.arguments.extend(["--" + flag, str(value)])

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def run_builder(self):
        def provenance(directory, revision):
            self.assertEqual(revision, "fixture-revision")
            return copy.deepcopy(self.provenances[directory]), None
        with ExitStack() as stack:
            stack.enter_context(patch.object(freeze, "model_provenance", side_effect=provenance))
            stack.enter_context(patch.object(freeze, "merged_model_provenance", side_effect=provenance))
            freeze.main(self.arguments)

    def assert_refused(self, message):
        with self.assertRaisesRegex(ValueError, message):
            self.run_builder()
        self.assertFalse(self.output.exists())

    def test_actual_operator_record_is_accepted_without_claiming_human_acceptance(self):
        self.run_builder()
        result = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "frozen")
        self.assertEqual(result["validation_review_status"], "completed_operator_review")
        self.assertEqual(result["validation_review_sha256"], self.digest(self.review))
        self.assertEqual(result["selection_sha256"], self.digest(self.selection))
        self.assertEqual(result["acceptance"], self.acceptance)
        self.assertEqual(list(result["models"]), list(freeze.MODEL_IDS))
        self.assertIs(result["human_acceptance_performed"], False)
        self.assertIs(result["final_selection_allowed"], False)
        self.assertEqual(len(result["evaluation_source_sha256"]), 6)

    def test_summary_byte_change_after_operator_record_is_rejected(self):
        with self.selection.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with patch.object(freeze, "model_provenance", side_effect=AssertionError("must fail before models")):
            with self.assertRaisesRegex(ValueError, "completed, bound"):
                freeze.main(self.arguments)
        self.assertFalse(self.output.exists())

    def test_selected_report_byte_change_after_operator_record_is_rejected(self):
        report = Path(self.summary["selected"]["calibrated_5888"]["checkpoint"]["report_path"])
        with report.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        self.assert_refused("Selected validation report changed")

    def test_operator_notes_cannot_claim_automated_or_independent_acceptance(self):
        for field in ("actual_reading_verified", "factual_correctness_automatically_verified",
                      "independent_review_performed", "human_acceptance_performed"):
            with self.subTest(field=field):
                value = copy.deepcopy(self.operator_review)
                value[field] = True
                self.write(self.review, value)
                self.assert_refused("unverified self-declaration")

    def test_operator_schema_rejects_missing_self_declaration_and_malformed_notes(self):
        changes = [("operator_review_self_declared", False), ("notes", " \n "), ("notes", 5),
                   ("notes", None), ("reviewer", None), ("reviewer", {"type": "independent_certifier"})]
        for field, wrong in changes:
            with self.subTest(field=field, wrong=wrong):
                value = copy.deepcopy(self.operator_review)
                value[field] = wrong
                self.write(self.review, value)
                self.assert_refused("unverified self-declaration")

    def test_changed_base_file_hash_is_rejected_even_with_correct_arm(self):
        self.provenances[self.directories[1]]["files_sha256"]["model.safetensors"] = "other-weights"
        self.assert_refused("baseline differs")

    def test_merged_export_from_unselected_adapter_is_rejected(self):
        self.provenances[self.directories[2]]["merge_manifest"]["source_adapter"]["manifest_sha256"] = "another-adapter"
        self.assert_refused("not the selected adapter")

    def test_swapped_merged_arms_are_rejected(self):
        a, b = self.directories[2:]
        self.provenances[a], self.provenances[b] = self.provenances[b], self.provenances[a]
        self.assert_refused("fixed original/pruned arm labels")

    def test_non64_or_failed_numeric_selection_cannot_be_overridden_by_notes(self):
        for step, passed in ((32, True), (64, False)):
            with self.subTest(step=step, passed=passed):
                chosen = self.summary["selected"]["original"]["checkpoint"]
                chosen["training_step"] = step
                chosen["numeric_quality_gate_passed"] = passed
                self.write(self.selection, self.summary)
                value = copy.deepcopy(self.operator_review)
                value["bindings"]["numeric_summary"]["sha256"] = self.digest(self.selection)
                self.write(self.review, value)
                self.assert_refused("requires the two selected step64")

    def test_prior_assistant_review_schema_remains_accepted(self):
        self.write(self.review, {"status": "completed_assistant_review",
                                 "bindings": {"numeric_summary": {"sha256": self.digest(self.selection)}}})
        self.run_builder()
        result = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(result["validation_review_status"], "completed_assistant_review")

    def test_existing_output_is_not_overwritten(self):
        self.output.write_text("existing output", encoding="utf-8")
        with patch.object(freeze, "read_json", side_effect=AssertionError("no work before refusal")):
            with self.assertRaises(FileExistsError):
                freeze.main(self.arguments)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "existing output")


if __name__ == "__main__":
    unittest.main()
