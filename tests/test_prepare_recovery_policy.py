"""Fresh policy preparation with CPU tokenizer/loader mocks and real file hashes."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_recovery_policy as prepare


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class PrepareRecoveryPolicyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original, self.pruned, self.data = [self.root / name for name in ("my-original", "my-pruned", "my-data")]
        for directory in (self.original, self.pruned, self.data):
            directory.mkdir()
        (self.original / "download_manifest.json").write_text("{}")
        (self.pruned / "pruning_manifest.json").write_text("{}")
        for directory, width in ((self.original, 6144), (self.pruned, 5888)):
            self.write(directory / "config.json", {"model_type": "qwen3", "num_hidden_layers": 28,
                                                    "hidden_size": 2048, "intermediate_size": width})
        tokenizer_hashes = {}
        for name in ("tokenizer.json", "tokenizer_config.json"):
            (self.original / name).write_text("synthetic tokenizer")
            tokenizer_hashes[name] = prepare.sha256_file(self.original / name)
        self.revision = "pinned-fixture-source"
        self.write(self.data / "manifest.json", {"tokenizer": {"source_model_revision": self.revision,
                                                               "files_sha256": tokenizer_hashes}})
        source = {"model_id": "Qwen/Qwen3-1.7B", "revision": self.revision}
        original_hashes = {"config.json": digest("original config"), "model.safetensors": digest("original weights")}
        self.provenances = {
            self.original: {"checkpoint_kind": "original", "manifest_sha256": digest("download manifest"),
                            "files_sha256": original_hashes, "download_manifest": source},
            self.pruned: {"checkpoint_kind": "pruned", "recovery_applied": False,
                          "method": "calibrated_rms_mean_abs", "width_before": 6144, "width_after": 5888,
                          "manifest_sha256": digest("pruning manifest"),
                          "files_sha256": {"config.json": digest("pruned config"), "model.safetensors": digest("pruned weights")},
                          "source": {**source, "download_manifest_sha256": digest("download manifest"),
                                     "files": [{"name": name, "sha256": sha} for name, sha in original_hashes.items()]}}
        }
        self.rows = [{"id": f"r-{i}", "supervised_token_count": 13 + (i < 106)} for i in range(512)]
        self.data_provenance = {"directory": str(self.data.resolve()), "row_count": 512, "group_count": 433,
                                "supervised_token_count": 6762, "source_model_revision": self.revision,
                                "sample_order": [row["id"] for row in self.rows], "encoded_rows_sha256": digest("encoded"),
                                "manifest_sha256": digest("recovery manifest")}
        self.tokenizer = object()
        self.checks = {"status": "passed", "signature_sha256": digest("tokenizer signature")}

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def run_prepare(self):
        with patch.object(prepare, "model_provenance", side_effect=lambda directory, revision: (copy.deepcopy(self.provenances[directory]), None)), \
                patch("transformers.AutoTokenizer.from_pretrained", return_value=self.tokenizer) as tokenizer_loader, \
                patch.object(prepare, "check_tokenizers", return_value=self.checks) as tokenizer_check, \
                patch.object(prepare, "load_recovery_data", return_value=({}, self.rows, self.data_provenance)) as data_loader:
            result = prepare.prepare_policy(self.original, self.pruned, self.data, self.revision)
        self.assertEqual(tokenizer_loader.call_count, 2)
        tokenizer_check.assert_called_once_with(self.tokenizer, self.tokenizer)
        data_loader.assert_called_once_with(self.data, self.tokenizer)
        return result

    def test_new_policy_uses_current_paths_and_matches_existing_trainer_and_summary(self):
        from train_recovery import validate_policy
        from summarize_recovery import _policy_arms
        result = self.run_prepare()
        self.assertEqual(validate_policy(result)["optimizer_steps"], 64)
        self.assertEqual(_policy_arms(result)[0], {"original": "original", "pruned": "calibrated_5888"})
        self.assertEqual(result["arms"]["original"]["model_directory"], str(self.original.resolve()))
        self.assertEqual(result["arms"]["calibrated_5888"]["model_directory"], str(self.pruned.resolve()))
        self.assertEqual(result["training_data"], self.data_provenance)
        self.assertEqual(result["tokenizer_compatibility"], self.checks)
        self.assertIn("not the historical experiment policy", result["recording_stage"])
        self.assertIn("+00:00", result["created_at_utc"])
        self.assertEqual(result["settings"]["checkpoint_steps"], [16, 32, 64])
        self.assertEqual(result["settings"]["accumulation_steps"], 8)
        self.assertEqual(result["settings"]["max_length"], 512)

    def test_formal_settings_are_fresh_and_do_not_mutate_trial_defaults(self):
        from trial_lora import SETTINGS
        before = copy.deepcopy(SETTINGS)
        first = prepare.formal_settings()
        first["target_modules"].append("wrong")
        first["checkpoint_steps"].append(128)
        self.assertEqual(SETTINGS, before)
        self.assertEqual(prepare.formal_settings()["checkpoint_steps"], [16, 32, 64])
        self.assertNotIn("wrong", prepare.formal_settings()["target_modules"])
        self.assertEqual(SETTINGS["optimizer_steps"], 10)

    def test_wrong_width_method_or_recovered_base_rejected(self):
        for field, value in (("method", "magnitude"), ("width_after", 5632), ("recovery_applied", True)):
            before = copy.deepcopy(self.provenances)
            self.provenances[self.pruned][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_prepare()
            self.provenances = before
        self.write(self.pruned / "config.json", {"model_type": "qwen3", "num_hidden_layers": 28,
                                                 "hidden_size": 2048, "intermediate_size": 5632})
        with self.assertRaisesRegex(ValueError, "width 5888"):
            self.run_prepare()

    def test_source_revision_manifest_or_actual_weight_inventory_mismatch_rejected(self):
        for field, value in (("revision", "other"), ("download_manifest_sha256", digest("other manifest")),
                             ("files", [{"name": "model.safetensors", "sha256": digest("other weights")}] )):
            before = copy.deepcopy(self.provenances)
            self.provenances[self.pruned]["source"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_prepare()
            self.provenances = before

    def test_tokenizer_files_checked_against_frozen_data_before_loading(self):
        (self.original / "tokenizer.json").write_text("changed file")
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=AssertionError("must not load")), \
                self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.run_prepare()

    def test_semantic_tokenizer_difference_is_not_silently_accepted(self):
        with patch.object(prepare, "model_provenance", side_effect=lambda directory, revision: (self.provenances[directory], None)), \
                patch("transformers.AutoTokenizer.from_pretrained", return_value=self.tokenizer), \
                patch.object(prepare, "check_tokenizers", side_effect=ValueError("chat_template differs")), \
                patch.object(prepare, "load_recovery_data", side_effect=AssertionError("do not encode with mismatched tokenizer")), \
                self.assertRaisesRegex(ValueError, "chat_template differs"):
            prepare.prepare_policy(self.original, self.pruned, self.data, self.revision)

    def test_wrong_sample_or_supervision_budget_is_rejected(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError, "512-row"):
            self.run_prepare()
        self.rows.append({"id": "last", "supervised_token_count": 12})
        with self.assertRaisesRegex(ValueError, "6762-target"):
            self.run_prepare()

    def test_existing_output_refused_before_reading_inputs(self):
        output = self.root / "existing.json"
        output.write_text("keep old policy")
        with patch.object(prepare, "prepare_policy", side_effect=AssertionError("no work")), self.assertRaises(FileExistsError):
            prepare.main(self.arguments(output))
        self.assertEqual(output.read_text(), "keep old policy")

    def arguments(self, output):
        return ["--original-model", str(self.original), "--pruned-model", str(self.pruned), "--data-dir", str(self.data),
                "--revision", self.revision, "--output", str(output)]

    def test_cli_writes_new_file_only_after_validation_and_never_inside_inputs(self):
        output = self.root / "new-results" / "policy.json"
        with patch.object(prepare, "prepare_policy", return_value={"new": "policy"}):
            prepare.main(self.arguments(output))
        self.assertEqual(json.loads(output.read_text()), {"new": "policy"})
        for directory in (self.original, self.pruned, self.data):
            with self.subTest(directory=directory), self.assertRaisesRegex(ValueError, "outside"):
                prepare.main(self.arguments(directory / "new-policy.json"))
        failed = self.root / "failed.json"
        with patch.object(prepare, "prepare_policy", side_effect=ValueError("input failure")), self.assertRaisesRegex(ValueError, "input failure"):
            prepare.main(self.arguments(failed))
        self.assertFalse(failed.exists())


if __name__ == "__main__":
    unittest.main()
