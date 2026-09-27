"""Eight-slot release arithmetic/protocol tests using synthetic questions only."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import summarize_release as summary


class ReleaseSummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.policy_path = self.root / "policy.json"
        names = ("run_release_quality.py", "merged_checkpoint.py", "run_quality.py",
                 "run_recovery_quality.py", "cmrc_official.py", "eval_ppl.py")
        self.manifest = {"files_sha256": {"synthetic.jsonl": "synthetic-sha"},
                         "counts": {"validation_qa": {"groups": 100}, "final_qa": {"groups": 200}}}
        self.write(self.root / "manifest.json", self.manifest)
        self.policy = {"schema_version": 1, "status": "frozen", "revision": "synthetic-revision",
                       "created_at_utc": "2026-09-27T00:00:00+00:00", "final_selection_allowed": False,
                       "precision_policy": summary.PRECISION_POLICY, "acceptance": summary.ACCEPTANCE,
                       "data_manifest_sha256": summary.sha256_file(self.root / "manifest.json"),
                       "evaluation_source_sha256": {n: summary.sha256_file(Path(summary.__file__).with_name(n)) for n in names},
                       "models": {}}
        for label, identity in summary.MODEL_IDENTITIES.items():
            self.policy["models"][label] = {"directory": str(self.root / label), "checkpoint_kind": identity[0],
                "manifest_sha256": label + "-manifest", "files_sha256": {"model.safetensors": label + "-weights"}}
        self.write(self.policy_path, self.policy)
        self.paths, self.reports = [], []
        for split, (count, targets) in summary.SPLIT_BUDGETS.items():
            for label in summary.MODEL_IDS:
                kind, basekind, step = summary.MODEL_IDENTITIES[label]
                slot = self.policy["models"][label]
                qa = [{"id": f"synthetic-{split}-{i}", "source": "dev", "document_id": str(i),
                    "group_id": f"synthetic-{split}-{i}", "paragraph_id": i, "context_sha256": "context-sha",
                    "normalized_context_sha256": "normalized-sha", "context": "合成资料", "question": "测试？",
                    "answers": ["合成"], "prompt_token_ids": [1, 2], "input_tokens": 2,
                    "recovery_tokens_first_answer": 1, "answer": "合成", "stop_reason": "eos", "new_tokens": 2} for i in range(count)]
                parameters = 900 if basekind == "pruned" else 1000
                r = {"schema_version": 1, "status": "completed", "stage": "completed", "split": split,
                    "model_id": label, "created_at_utc": "2026-09-27T01:00:00+00:00",
                    "release_policy_sha256": summary.sha256_file(self.policy_path),
                    "source_model_revision": self.policy["revision"], "helper_sha256": self.policy["evaluation_source_sha256"],
                    "runner_sha256": self.policy["evaluation_source_sha256"]["run_release_quality.py"],
                    "protocol": {"acceptance": summary.ACCEPTANCE, "inference_precision": summary.PRECISION_POLICY},
                    "generation_config": {"max_new_tokens": 64}, "decode": {"strip_only": True},
                    "scoring_provenance": {"official": "pinned"}, "canonical_tokenizer_files_sha256": {"tokenizer.json": "same"},
                    "tokenizer_compatibility": {"status": "passed", "signature_sha256": "same"},
                    "environment": {"selected_dtype": "torch.bfloat16", "attention_implementation": "sdpa", "packages": {}},
                    "model_directory": slot["directory"], "adapter_directory": None,
                    "provenance": {"checkpoint_kind": kind, "base_checkpoint_kind": basekind, "training_step": step,
                        "manifest_sha256": slot["manifest_sha256"], "files_sha256": slot["files_sha256"]},
                    "model": {"unique_parameter_count": parameters, "base_parameter_count": parameters,
                        "adapter_parameter_count": 0, "adapter_merged": bool(step), "all_parameters_frozen": True,
                        "parameter_dtypes": ["torch.bfloat16"], "attention_implementation": "sdpa"},
                    "reload_validation": {"status": "passed", "probe_exact": True},
                    "data": {"directory": str(self.root), "manifest_sha256": self.policy["data_manifest_sha256"],
                        "files_sha256": self.manifest["files_sha256"], "qa_count": count, "lm_document_count": 1,
                        "qa_file": f"{split}_qa.jsonl", "lm_file": f"{split}_lm.jsonl"},
                    "data_manifest": self.manifest, "qa": qa,
                    "qa_metrics": {"count": count, "skipped": 0, "metric": "cmrc2018_official", "em": 100, "f1": 100,
                        "per_item": [{"id": q["id"], "em": 100, "f1": 100} for q in qa]},
                    "lm": {"valid_token_count": targets, "nll_sum": targets, "nll": 1, "ppl": math.e,
                        "ppl_overflow": False, "document_count": 1, "protocol": {"window": 512, "stride": 256},
                        "documents": [{"id": f"lm-{split}", "token_count": targets + 1, "valid_token_count": targets,
                            "nll_sum": targets, "windows": [{"synthetic": True}]}]},
                    "qa_output_diagnostics": {"empty_answer_count": 0, "max_new_tokens_stop_count": 0}}
                self.reports.append(copy.deepcopy(r))
                self.paths.append(self.root / f"{split}_{label}.json")
        self.flush()

    def write(self, path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf8")

    def flush(self):
        for path, report in zip(self.paths, self.reports):
            self.write(path, report)

    def run_summary(self):
        self.flush()
        return summary.summarize(self.policy_path, self.paths, verify_artifacts=False)

    def test_all_fixed_slots_and_matched_budget_are_retained_without_selection(self):
        result = self.run_summary()
        self.assertEqual(len(result["rows"]), 8)
        self.assertEqual(len(result["matched_budget_differences"]), 4)
        self.assertEqual(result["final_numeric_gate_passed_model_ids"], list(summary.MODEL_IDS))
        self.assertFalse(result["final_selection_allowed"])
        self.assertFalse(result["checkpoint_reselected"])
        self.assertFalse(result["human_acceptance_performed"])
        self.assertIn("最终质量", summary.markdown(result))
        self.assertEqual(result["matched_budget_differences"][0]["parameter_reduction_pruned_vs_original"], 1 - .9)

    def test_failed_final_stays_in_table_and_does_not_pick_another_checkpoint(self):
        report = self.reports[-1]
        for item in report["qa_metrics"]["per_item"]:
            item["f1"] = 94
        report["qa_metrics"]["f1"] = 94
        result = self.run_summary()
        self.assertEqual(len(result["rows"]), 8)
        self.assertNotIn(report["model_id"], result["final_numeric_gate_passed_model_ids"])
        self.assertEqual(result["rows"][-1]["training_step"], 64)

    def test_physical_provider_and_manifest_are_reverified_for_all_four_artifacts(self):
        def provenance(directory, revision, policy_path, label):
            return next(r["provenance"] for r in self.reports if r["model_id"] == label), None
        with patch.object(summary, "release_provenance", side_effect=provenance) as provider:
            result = summary.summarize(self.policy_path, self.paths)
        self.assertTrue(result["physical_artifacts_reverified"])
        self.assertEqual(provider.call_count, 4)
        with patch.object(summary, "release_provenance", side_effect=ValueError("changed weights")), self.assertRaises(ValueError):
            summary.summarize(self.policy_path, self.paths)

    def test_missing_duplicate_or_wrong_slot_rejected(self):
        for paths in (self.paths[:-1], self.paths[:-1] + [self.paths[0]]):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                summary.summarize(self.policy_path, paths, False)
        self.reports[-1] = copy.deepcopy(self.reports[-2])
        with self.assertRaisesRegex(ValueError, "slot"):
            self.run_summary()

    def test_malformed_identity_protocol_completion_and_budget_rejected(self):
        for path, value in ((["status"], "running"), (["stage"], "lm"), (["release_policy_sha256"], "changed"),
                (["provenance", "training_step"], 32), (["provenance", "files_sha256"], {}),
                (["created_at_utc"], "2026-09-26T00:00:00+00:00"), (["model", "all_parameters_frozen"], False),
                (["model", "adapter_parameter_count"], 12), (["reload_validation", "probe_exact"], False),
                (["qa_metrics", "count"], 199), (["lm", "valid_token_count"], 16383),
                (["qa_output_diagnostics", "max_new_tokens_stop_count"], 1)):
            original = copy.deepcopy(self.reports[-1])
            node = self.reports[-1]
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.run_summary()
            self.reports[-1] = original

    def test_question_order_references_windows_environment_mismatch_rejected(self):
        changes = [lambda r: r["qa"].reverse(), lambda r: r["qa"][0]["answers"].append("篡改"),
                   lambda r: r["lm"]["documents"][0]["windows"].append({"different": True}),
                   lambda r: r["environment"].update(selected_dtype="float32")]
        for change in changes:
            original = copy.deepcopy(self.reports[-1])
            change(self.reports[-1])
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_summary()
            self.reports[-1] = original

    def test_duplicate_scores_bad_mean_bad_nll_and_nonfinite_rejected(self):
        changes = [lambda r: r["qa_metrics"]["per_item"].append(r["qa_metrics"]["per_item"][0]),
                   lambda r: r["qa_metrics"].update(f1=99), lambda r: r["lm"].update(nll_sum=1),
                   lambda r: r["lm"].update(ppl=float("nan")),
                   lambda r: r["lm"]["documents"][0].update(nll_sum=2)]
        for change in changes:
            original = copy.deepcopy(self.reports[-1])
            change(self.reports[-1])
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_summary()
            self.reports[-1] = original

    def test_final_group_leakage_rejected_even_if_all_models_share_it(self):
        leaked = self.reports[0]["qa"][0]["group_id"]
        for report in self.reports[4:]:
            report["qa"][0]["group_id"] = leaked
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.run_summary()

    def test_changed_policy_threshold_cannot_relax_final_gate(self):
        self.policy["acceptance"] = {**self.policy["acceptance"], "max_f1_drop_percentage_points": 6}
        self.write(self.policy_path, self.policy)
        with self.assertRaisesRegex(ValueError, "threshold"):
            self.run_summary()


if __name__ == "__main__":
    unittest.main()
