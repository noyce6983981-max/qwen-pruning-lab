"""Synthetic CPU-only release policy/identity/budget tests; no real final data."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_release_quality as release


class ReleaseQualityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.policy_path = self.root / "policy.json"
        self.acceptance = {"max_f1_drop_percentage_points": 5, "max_ppl_relative_increase": 0.1,
                           "human_review": "No widespread repeated outputs."}
        names = ("run_release_quality.py", "merged_checkpoint.py", "run_quality.py",
                 "run_recovery_quality.py", "cmrc_official.py", "eval_ppl.py")
        self.policy = {"schema_version": 1, "status": "frozen", "revision": "synthetic-revision",
                       "final_selection_allowed": False, "precision_policy": release.PRECISION_POLICY,
                       "evaluation_source_sha256": {name: release.quality.sha256_file(Path(release.__file__).with_name(name)) for name in names},
                       "acceptance": self.acceptance, "models": {}}
        for label, identity in release.MODEL_IDENTITIES.items():
            self.policy["models"][label] = {"directory": str((self.root / label).resolve()),
                                           "checkpoint_kind": identity[0], "manifest_sha256": label + "-manifest",
                                           "files_sha256": {"model.safetensors": label + "-weights"}}
        self.write_policy()

    def write_json(self, path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def write_policy(self):
        self.write_json(self.policy_path, self.policy)

    def actual(self, label):
        kind, basekind, step = release.MODEL_IDENTITIES[label]
        entry = self.policy["models"][label]
        result = {"checkpoint_kind": kind, "base_checkpoint_kind": basekind, "training_step": step,
                  "recovery_applied": bool(step), "files_sha256": copy.deepcopy(entry["files_sha256"]),
                  "manifest_sha256": entry["manifest_sha256"]}
        source = {"model_id": "synthetic-model", "revision": self.policy["revision"]}
        if step == 0:
            result["download_manifest" if kind == "original" else "source"] = source
        return result

    def create_data(self, split="final", qa_count=200, target_count=16384):
        directory = self.root / "synthetic-data"
        directory.mkdir(exist_ok=True)
        context = "合成测试资料，不能当真实最终题目。"
        rows = [{"id": f"synthetic-{i}", "context": context, "question": "测试？", "answers": ["合成"],
                 "prompt_token_ids": [1, 2, 3], "context_sha256": hashlib.sha256(context.encode()).hexdigest()}
                for i in range(qa_count)]
        docs = [{"id": "synthetic-document", "token_ids": [1] * (target_count + 1)}]
        hashes = {}
        for name, values in ((f"{split}_qa.jsonl", rows), (f"{split}_lm.jsonl", docs)):
            path = directory / name
            path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")
            hashes[name] = release.quality.sha256_file(path)
        manifest = {"files_sha256": hashes, "protocol": {
            "qa": {"max_input_tokens": 448, "max_new_tokens": 64, "enable_thinking": False,
                   "do_sample": False, "answer_extraction": "strip_only"},
            "lm": {"window": 512, "stride": 256}, "acceptance": copy.deepcopy(self.acceptance)}}
        self.write_json(directory / "manifest.json", manifest)
        self.policy["data_manifest_sha256"] = release.quality.sha256_file(directory / "manifest.json")
        self.write_policy()
        return directory

    def test_policy_accepts_current_sources_but_rejects_missing_four_arm_freeze(self):
        self.assertEqual(release.validate_release_policy(self.policy_path)["status"], "frozen")
        for key, value in (("status", "draft"), ("final_selection_allowed", True), ("precision_policy", {})):
            original = copy.deepcopy(self.policy)
            self.policy[key] = value
            self.write_policy()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "four-artifact"):
                release.validate_release_policy(self.policy_path)
            self.policy = original
        self.policy["models"].pop("original_step0")
        self.write_policy()
        with self.assertRaisesRegex(ValueError, "four-artifact"):
            release.validate_release_policy(self.policy_path)

    def test_changed_source_hash_or_source_file_is_rejected(self):
        self.policy["evaluation_source_sha256"]["eval_ppl.py"] = "changed"
        self.write_policy()
        with self.assertRaisesRegex(ValueError, "source changed"):
            release.validate_release_policy(self.policy_path)
        self.policy["evaluation_source_sha256"]["eval_ppl.py"] = release.quality.sha256_file(Path(release.__file__).with_name("eval_ppl.py"))
        self.write_policy()
        real_hash = release.quality.sha256_file
        def altered_file(path):
            return "changed-on-disk" if Path(path).name == "cmrc_official.py" else real_hash(path)
        with patch.object(release.quality, "sha256_file", side_effect=altered_file), self.assertRaisesRegex(ValueError, "source changed"):
            release.validate_release_policy(self.policy_path)

    def test_four_actual_identities_match_their_bound_paths_and_hashes(self):
        for label in release.MODEL_IDS:
            provider = "merged_model_provenance" if label.endswith("_merged") else "model_provenance"
            owner = release if label.endswith("_merged") else release.quality
            with self.subTest(label=label), patch.object(owner, provider, return_value=(self.actual(label), None)):
                provenance, _ = release.release_provenance(Path(self.policy["models"][label]["directory"]),
                                                           self.policy["revision"], self.policy_path, label)
                self.assertEqual((provenance["checkpoint_kind"], provenance["base_checkpoint_kind"], provenance["training_step"]),
                                 release.MODEL_IDENTITIES[label])

    def test_same_kind_merged_arm_swap_cannot_mislabel_original_and_pruned(self):
        label = "original_step64_merged"
        swapped = self.actual("calibrated_5888_step64_merged")
        entry = self.policy["models"]["calibrated_5888_step64_merged"]
        self.policy["models"][label] = copy.deepcopy(entry)
        self.write_policy()
        with patch.object(release, "merged_model_provenance", return_value=(swapped, None)), \
                self.assertRaisesRegex(ValueError, "release label/base kind/training step"):
            release.release_provenance(Path(entry["directory"]), self.policy["revision"], self.policy_path, label)

    def test_wrong_step0_kind_and_merged_step_are_rejected(self):
        self.policy["models"]["original_step0"]["checkpoint_kind"] = "pruned"
        self.write_policy()
        with self.assertRaisesRegex(ValueError, "wrong checkpoint kind"):
            release.validate_release_policy(self.policy_path)
        self.policy["models"]["original_step0"]["checkpoint_kind"] = "original"
        self.write_policy()
        label = "original_step64_merged"
        wrong = self.actual(label)
        wrong["training_step"] = 32
        with patch.object(release, "merged_model_provenance", return_value=(wrong, None)), self.assertRaisesRegex(ValueError, "training step"):
            release.release_provenance(Path(self.policy["models"][label]["directory"]), self.policy["revision"], self.policy_path, label)

    def test_wrong_model_path_revision_manifest_or_weight_hash_rejected(self):
        label = "original_step64_merged"
        directory = Path(self.policy["models"][label]["directory"])
        for path, revision in ((directory / "other", self.policy["revision"]), (directory, "wrong-revision")):
            with self.subTest(path=path, revision=revision), self.assertRaisesRegex(ValueError, "directory or revision"):
                release.release_provenance(path, revision, self.policy_path, label)
        for field, value in (("manifest_sha256", "other-manifest"), ("files_sha256", {"model.safetensors": "other-weights"})):
            actual = self.actual(label)
            actual[field] = value
            with self.subTest(field=field), patch.object(release, "merged_model_provenance", return_value=(actual, None)), \
                    self.assertRaisesRegex(ValueError, "identity or files changed"):
                release.release_provenance(directory, self.policy["revision"], self.policy_path, label)

    def test_final_accepts_exact_200_questions_and_16384_targets(self):
        directory = self.create_data()
        _, rows, docs, data = release.load_release_data(directory, "final", self.policy_path)
        self.assertEqual(len(rows), 200)
        self.assertEqual(sum(len(row["token_ids"]) - 1 for row in docs), 16384)
        self.assertEqual(data["qa_file"], "final_qa.jsonl")
        self.assertEqual(data["lm_input_token_count"], 16385)

    def test_final_rejects_any_missing_or_extra_question_or_target(self):
        for qa_count, target_count in ((199, 16384), (201, 16384), (200, 16383), (200, 16385)):
            directory = self.create_data(qa_count=qa_count, target_count=target_count)
            with self.subTest(qa_count=qa_count, target_count=target_count), self.assertRaisesRegex(ValueError, "token budget differs"):
                release.load_release_data(directory, "final", self.policy_path)

    def test_validation_has_its_own_fixed_100_and_8192_budget(self):
        directory = self.create_data("validation", 100, 8192)
        self.assertEqual(release.load_release_data(directory, "validation", self.policy_path)[3]["qa_count"], 100)
        directory = self.create_data("validation", 200, 16384)
        with self.assertRaisesRegex(ValueError, "token budget differs"):
            release.load_release_data(directory, "validation", self.policy_path)

    def test_dataset_manifest_changed_bytes_rejected_before_reading_examples(self):
        directory = self.create_data()
        with (directory / "manifest.json").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with patch.object(release.quality, "read_jsonl", side_effect=AssertionError("do not read examples")), \
                self.assertRaisesRegex(ValueError, "manifest changed"):
            release.load_release_data(directory, "final", self.policy_path)

    def test_changed_data_bytes_and_thresholds_are_rejected(self):
        directory = self.create_data()
        with (directory / "final_qa.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            release.load_release_data(directory, "final", self.policy_path)
        directory = self.create_data()
        self.policy["acceptance"] = {**self.acceptance, "max_f1_drop_percentage_points": 6}
        self.write_policy()
        with self.assertRaisesRegex(ValueError, "thresholds changed"):
            release.load_release_data(directory, "final", self.policy_path)

    def test_unknown_split_is_rejected_before_any_dataset_read(self):
        with patch.object(release.quality, "read_jsonl", side_effect=AssertionError("do not read examples")), \
                self.assertRaisesRegex(ValueError, "validation or final"):
            release.load_release_data(self.root / "not-created", "smoke", self.policy_path)


if __name__ == "__main__":
    unittest.main()
