"""CPU-only report fixtures for frozen recovery selection and provenance."""
import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_recovery import ORDERING, PRECISION_POLICY, main, markdown, summarize


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def base(kind):
    return {"checkpoint_kind": kind, "recovery_applied": False,
            "manifest_sha256": digest(kind + "manifest"),
            "files_sha256": {"config.json": digest(kind + "config"),
                             "model.safetensors": digest(kind + "weights")}}


class RecoverySummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.policy_path = self.directory / "policy.json"
        self.policy = {
            "schema_version": 1, "revision": "pinned-source",
            "settings": {"rank": 8, "checkpoint_steps": [16, 32, 64], "optimizer_steps": 64},
            "training_data": {"row_count": 512, "supervised_token_count": 6762,
                              "manifest_sha256": digest("training data")},
            "arms": {"original": {"base_provenance": base("original"), "expected_base_width": 12},
                     "calibrated_5888": {"base_provenance": base("pruned"), "expected_base_width": 10}},
            "selection": {"candidate_steps": [0, 16, 32, 64], "ordering": ORDERING,
                          "apply_separately_to_each_arm": True, "step0_is_no_training_fallback": True,
                          "numeric_gates_relative_to_new_original_step0": {
                              "max_f1_drop_percentage_points": 5, "max_ppl_relative_increase": 0.1}},
        }
        self.write(self.policy_path, self.policy)
        self.policy_sha = hashlib.sha256(self.policy_path.read_bytes()).hexdigest()
        self.reports = {}
        for kind in ("original", "pruned"):
            for step in (0, 16, 32, 64):
                self.reports[kind, step] = self.report(kind, step)
        self.original = self.path("original", 0)
        self.pruned = self.path("pruned", 0)
        self.candidates = [self.path(kind, step) for kind in ("original", "pruned") for step in (16, 32, 64)]
        self.save_reports()

    def write(self, path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def path(self, kind, step):
        return self.directory / f"{kind}-{step}.json"

    def save_reports(self):
        for (kind, step), report in self.reports.items():
            self.write(self.path(kind, step), report)

    def report(self, kind, step):
        arm = self.policy["arms"]["original" if kind == "original" else "calibrated_5888"]
        base_info = copy.deepcopy(arm["base_provenance"])
        provenance = {**base_info, "base_checkpoint_kind": kind, "training_step": step,
                      "recovery_applied": bool(step), "adapter_present": bool(step)}
        if step:
            manifest = {"status": "completed", "checkpoint_kind": "recovered_lora", "training_step": step,
                        "source_model_revision": self.policy["revision"], "training_policy_sha256": self.policy_sha,
                        "base_provenance": copy.deepcopy(base_info), "settings": copy.deepcopy(self.policy["settings"]),
                        "data": copy.deepcopy(self.policy["training_data"]),
                        "training_source_sha256": {"train_recovery.py": digest("trainer")}}
            provenance = {"checkpoint_kind": "recovered_lora", "base_checkpoint_kind": kind, "training_step": step,
                          "recovery_applied": True, "adapter_present": True, "base_provenance": base_info,
                          "manifest_sha256": digest(kind + str(step) + "adapter manifest"),
                          "recovery_manifest": manifest,
                          "formal_training_run": {"status": "completed", "optimizer_steps": 64,
                                                  "sample_count": 512, "supervised_token_count": 6762,
                                                  "base_parameters_unchanged": True,
                                                  "path": f"{kind}/run.json", "sha256": digest(kind + "run")}}
        base_count = 1000 if kind == "original" else 900
        adapter_count = (20 if kind == "original" else 18) if step else 0
        return {
            "status": "completed", "source_model_revision": self.policy["revision"], "split": "validation",
            "protocol": {"acceptance": {**copy.deepcopy(self.policy["selection"]["numeric_gates_relative_to_new_original_step0"]),
                                        "human_review": "No widespread repeated outputs."},
                         "inference_precision": copy.deepcopy(PRECISION_POLICY)},
            "generation_config": {"do_sample": False}, "decode": {"seed": 42},
            "runner_sha256": digest("new runner"),
            "helper_sha256": {"run_recovery_quality.py": digest("new runner"), "run_quality.py": digest("qa"),
                              "cmrc_official.py": digest("official"), "eval_ppl.py": digest("ppl")},
            "scoring_provenance": {"metric": "official"},
            "data": {"manifest_sha256": digest("data"), "files_sha256": {"validation_qa.jsonl": digest("qa data")},
                     "qa_file": "validation_qa.jsonl", "lm_file": "validation_lm.jsonl", "qa_count": 2},
            "data_manifest": {"counts": {"validation_qa": {"groups": 2}}},
            "tokenizer_compatibility": {"signature_sha256": digest("tokenizer")},
            "environment": {"selected_dtype": "torch.bfloat16", "attention_implementation": "sdpa", "packages": {}},
            "lm": {"ppl": 10.0, "nll": math.log(10), "valid_token_count": 8192},
            "qa_metrics": {"count": 2, "skipped": 0, "em": 50.0, "f1": 80.0,
                           "per_item": [{"id": "q1", "em": 0, "f1": 60}, {"id": "q2", "em": 100, "f1": 100}]},
            "qa": [{"id": "q1", "answer": "答案一"}, {"id": "q2", "answer": "答案二"}],
            "provenance": provenance,
            "model": {"unique_parameter_count": base_count + adapter_count, "base_parameter_count": base_count,
                      "adapter_parameter_count": adapter_count, "adapter_merged": False, "all_parameters_frozen": True,
                      "intermediate_size": arm["expected_base_width"]},
            "reload_validation": {"status": "passed", "probe_exact": True} if step else {"status": "not_applicable"},
        }

    def scores(self, kind, step, f1, ppl):
        report = self.reports[kind, step]
        report["qa_metrics"]["f1"] = f1
        report["qa_metrics"]["per_item"][0]["f1"] = 2 * f1 - 100
        report["lm"].update(ppl=ppl, nll=math.log(ppl))

    def run_summary(self, candidates=None):
        self.save_reports()
        return summarize(self.policy_path, self.original, self.pruned,
                         self.candidates if candidates is None else candidates)

    def test_all_ties_select_step0_and_include_all_rows_and_true_total_counts(self):
        result = self.run_summary()
        self.assertEqual(len(result["rows"]), 8)
        self.assertEqual([row["training_step"] for row in result["rows"]], [0, 16, 32, 64] * 2)
        for choice in result["selected"].values():
            self.assertEqual(choice["checkpoint"]["training_step"], 0)
            self.assertTrue(choice["step0_fallback_selected"])
        original16, pruned16 = result["rows"][1], result["rows"][5]
        self.assertEqual((original16["total_parameter_count"], original16["base_parameter_count"],
                          original16["parameter_reduction_count_vs_original_step0"]), (1020, 1000, -20))
        self.assertEqual(pruned16["total_parameter_count"], 918)
        self.assertAlmostEqual(pruned16["parameter_reduction"], 0.082)
        self.assertEqual(result["matched_step_differences"][1]["parameter_difference_count"], -102)
        self.assertEqual(result["rows"][0]["report_sha256"], hashlib.sha256(self.original.read_bytes()).hexdigest())

    def test_f1_before_ppl_then_exact_tie_earliest_step_independently(self):
        self.scores("original", 16, 85, 10.9)
        self.scores("original", 32, 84, 8)
        self.scores("original", 64, 85, 10.5)
        self.scores("pruned", 16, 83, 9)
        self.scores("pruned", 32, 83, 8)
        self.scores("pruned", 64, 83, 8)
        result = self.run_summary(candidates=list(reversed(self.candidates)))
        self.assertEqual(result["selected"]["original"]["checkpoint"]["training_step"], 64)
        self.assertEqual(result["selected"]["calibrated_5888"]["checkpoint"]["training_step"], 32)
        self.assertAlmostEqual(result["matched_step_differences"][2]["f1_difference_percentage_points"], -1)

    def test_real_acceptance_schema_preserves_human_review_beside_numeric_policy(self):
        numeric = self.policy["selection"]["numeric_gates_relative_to_new_original_step0"]
        acceptance = self.reports["original", 0]["protocol"]["acceptance"]
        self.assertEqual(set(acceptance), set(numeric) | {"human_review"})
        result = self.run_summary()
        self.assertEqual(result["evaluation"]["protocol"]["acceptance"], acceptance)
        self.assertTrue(result["qualitative_review_required"])
        self.assertTrue(all(choice["qualitative_review_status"] == "pending"
                            for choice in result["selected"].values()))

    def test_numeric_acceptance_mismatch_or_omission_is_rejected_with_human_review_present(self):
        for field in ("max_f1_drop_percentage_points", "max_ppl_relative_increase"):
            for omit in (False, True):
                old = copy.deepcopy(self.reports)
                for report in self.reports.values():
                    acceptance = report["protocol"]["acceptance"]
                    if omit:
                        acceptance.pop(field)
                    else:
                        acceptance[field] += 1
                with self.subTest(field=field, omit=omit), self.assertRaisesRegex(ValueError, "acceptance gates"):
                    self.run_summary()
                self.reports = old

    def test_human_review_cannot_be_removed_weakened_or_changed_in_one_report(self):
        for replacement in (None, "", "Review optional."):
            old = copy.deepcopy(self.reports)
            for report in self.reports.values():
                if replacement is None:
                    report["protocol"]["acceptance"].pop("human_review")
                else:
                    report["protocol"]["acceptance"]["human_review"] = replacement
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, "human_review"):
                self.run_summary()
            self.reports = old
        self.reports["pruned", 64]["protocol"]["acceptance"]["human_review"] = "Review optional."
        with self.assertRaisesRegex(ValueError, "Incompatible evaluation: protocol"):
            self.run_summary()

    def test_gate_is_relative_to_original_step0_not_each_arms_step0(self):
        self.scores("pruned", 0, 76, 10.9)
        self.scores("pruned", 16, 90, 11.000001)
        self.scores("pruned", 32, 74.999999, 9)
        self.scores("pruned", 64, 75, 11)
        result = self.run_summary()
        arm = result["rows"][4:]
        self.assertEqual([row["numeric_quality_gate_passed"] for row in arm], [True, False, False, True])
        self.assertEqual(result["selected"]["calibrated_5888"]["checkpoint"]["training_step"], 0)

    def test_failed_trained_candidates_fall_back_only_if_step0_passes(self):
        for step in (16, 32, 64):
            self.scores("pruned", step, 70, 10)
        result = self.run_summary()
        choice = result["selected"]["calibrated_5888"]
        self.assertFalse(choice["trained_candidate_passed"])
        self.assertTrue(choice["step0_fallback_selected"])
        self.scores("pruned", 0, 70, 10)
        self.assertIsNone(self.run_summary()["selected"]["calibrated_5888"]["checkpoint"])

    def test_candidate_missing_duplicate_and_wrong_step_rejected(self):
        with self.assertRaisesRegex(ValueError, "eight distinct"):
            self.run_summary(self.candidates[:-1])
        with self.assertRaisesRegex(ValueError, "eight distinct"):
            self.run_summary(self.candidates[:-1] + [self.candidates[0]])
        self.reports["pruned", 64]["provenance"]["training_step"] = 32
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_summary()
        self.reports["pruned", 64]["provenance"]["training_step"] = 8
        with self.assertRaisesRegex(ValueError, "scheduled"):
            self.run_summary()

    def test_base_hash_mismatch_in_candidate_baseline_or_manifest_rejected(self):
        for location in ("candidate", "baseline", "manifest"):
            old = copy.deepcopy(self.reports)
            target = self.reports["pruned", 16]["provenance"]["base_provenance"]
            if location == "baseline":
                target = self.reports["pruned", 0]["provenance"]
            if location == "manifest":
                target = self.reports["pruned", 16]["provenance"]["recovery_manifest"]["base_provenance"]
            target["files_sha256"]["model.safetensors"] = digest("wrong actual weight")
            with self.subTest(location=location), self.assertRaisesRegex(ValueError, "base"):
                self.run_summary()
            self.reports = old

    def test_policy_hash_settings_and_data_manifest_binding_rejected(self):
        for field, value in (("training_policy_sha256", digest("other policy")), ("status", "failed"),
                             ("settings", {}), ("data", {})):
            old = copy.deepcopy(self.reports)
            self.reports["original", 16]["provenance"]["recovery_manifest"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_summary()
            self.reports = old

    def test_uncompleted_or_different_formal_runs_and_training_code_rejected(self):
        for field, value in (("status", "running"), ("base_parameters_unchanged", False),
                             ("optimizer_steps", 16), ("sha256", digest("different completed run"))):
            old = copy.deepcopy(self.reports)
            self.reports["original", 32]["provenance"]["formal_training_run"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "run"):
                self.run_summary()
            self.reports = old
        self.reports["pruned", 32]["provenance"]["recovery_manifest"]["training_source_sha256"] = {"train.py": digest("different")}
        with self.assertRaisesRegex(ValueError, "Training source"):
            self.run_summary()

    def test_version_data_precision_and_final_split_mismatch_rejected(self):
        changes = [lambda r: r.update(runner_sha256=digest("old runner")),
                   lambda r: r["helper_sha256"].update({"eval_ppl.py": digest("changed helper")}),
                   lambda r: r["data"].update(manifest_sha256=digest("changed data")),
                   lambda r: r["protocol"].update(inference_precision={}),
                   lambda r: r.update(split="final")]
        for change in changes:
            old = copy.deepcopy(self.reports)
            change(self.reports["pruned", 32])
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_summary()
            self.reports = old

    def test_all_eight_using_legacy_eval_still_rejected(self):
        for report in self.reports.values():
            report["helper_sha256"].pop("run_recovery_quality.py")
            report["protocol"].pop("inference_precision")
        with self.assertRaisesRegex(ValueError, "new common recovery"):
            self.run_summary()

    def test_wrong_provenance_flags_reload_and_parameter_counts_rejected(self):
        changes = [lambda r: r["provenance"].update(checkpoint_kind="pruned"),
                   lambda r: r["provenance"].update(adapter_present=False),
                   lambda r: r["model"].update(unique_parameter_count=900),
                   lambda r: r["model"].update(adapter_merged=True),
                   lambda r: r["model"].update(base_parameter_count=901, unique_parameter_count=919),
                   lambda r: r["reload_validation"].update(status="failed"),
                   lambda r: r["lm"].update(nll=float("nan"))]
        for change in changes:
            old = copy.deepcopy(self.reports)
            change(self.reports["pruned", 32])
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_summary()
            self.reports = old

    def test_changed_selection_order_or_gate_cannot_silently_reuse_algorithm(self):
        self.policy["selection"]["ordering"] = list(reversed(ORDERING))
        self.write(self.policy_path, self.policy)
        with self.assertRaisesRegex(ValueError, "selection policy"):
            self.run_summary()
        self.policy["selection"]["ordering"] = ORDERING
        self.policy["selection"]["numeric_gates_relative_to_new_original_step0"]["max_ppl_relative_increase"] = 0.2
        self.write(self.policy_path, self.policy)
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.run_summary()

    def test_cli_writes_new_directory_only_and_marks_review_and_limits(self):
        output = self.directory / "summary"
        arguments = ["--policy", str(self.policy_path), "--original-base", str(self.original),
                     "--pruned-base", str(self.pruned), "--output-dir", str(output)]
        for path in self.candidates:
            arguments += ["--candidate", str(path)]
        main(arguments)
        result = json.loads((output / "recovery_summary.json").read_text(encoding="utf-8"))
        self.assertTrue(result["qualitative_review_required"])
        self.assertEqual(result["selection_status"], "numeric_proposals_only")
        rendered = (output / "recovery_summary.md").read_text(encoding="utf-8")
        self.assertEqual(rendered, markdown(result))
        for phrase in ("没有使用 final", "定性回答复核仍待完成", "不追更长训练", "同一验证集用于检查点选优", "总参数包含未合并"):
            self.assertIn(phrase, rendered)
        with self.assertRaises(FileExistsError):
            main(arguments)


if __name__ == "__main__":
    unittest.main()
