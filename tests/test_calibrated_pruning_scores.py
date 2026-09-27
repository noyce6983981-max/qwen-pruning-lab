"""CPU-only checks before a saved calibration ranking can drive pruning."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibrated_pruning_scores as loader


def digest(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def make_report():
    files = {name: digest(name) for name in (
        "config.json", "generation_config.json", "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors", "model.safetensors.index.json", "tokenizer.json")}
    source = {"revision": "pinned-original", "download_manifest_sha256": digest("download"),
              "files": [{"name": name, "sha256": value, "bytes": 123} for name, value in files.items()]}
    report = {
        "schema_version": 1, "status": "completed", "stage": "completed",
        "weights_modified": False, "pruning_applied": False, "model_forward_calls": 0,
        "runner_sha256": digest("scorer"), "helpers_sha256": {"run_quality.py": digest("helper")},
        "method": {
            "name": "Qwen3 MLP-only calibration score adaptation",
            "formula": "mean_i(abs(W_down[i,j])) * sqrt(sum_t(z[t,j]^2) / valid_token_count)",
            "weight_abs_dtype": "float32", "weight_reduction_and_score_dtype": "float64",
            "ranking": "ascending within each layer; stable ties use original channel index",
            "activation_source": "all layers of the unchanged original model",
        },
        "model": {"hidden_size": 2048, "intermediate_size": 6144, "num_hidden_layers": 28},
        "per_layer": [{"layer": index, "width": 6144} for index in range(28)],
        "source": {
            "revision": "pinned-original",
            "provenance": {
                "checkpoint_kind": "original", "manifest_sha256": digest("download"),
                "download_manifest": {"revision": "pinned-original"},
                "files_sha256": {name: value for name, value in files.items()
                                 if name in ("config.json", "generation_config.json") or name.endswith(".safetensors")},
            },
            "weight_index_sha256": files["model.safetensors.index.json"],
            "collection_report_sha256": digest("collection"), "activation_stats_sha256": digest("activations"),
            "calibration_data": {
                "manifest_sha256": digest("data manifest"), "token_ids_sha256": digest("tokens"),
                "files_sha256": {"calibration.jsonl": digest("calibration"), "groups.json": digest("groups")},
                "sample_order": [{"id": "train1", "group_id": "group1"}],
            },
        },
        "output": {"file": "channel_scores.safetensors", "reload_exact": True},
    }
    return report, source


class CalibratedPruningScoresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_temporary = tempfile.TemporaryDirectory()
        cls.fixture_path = Path(cls.fixture_temporary.name) / "scores.safetensors"
        tensors = {}
        for index in range(28):
            # Deliberate equal-score pairs exercise stable original-index tie handling.
            score = torch.arange(6144, dtype=torch.float64).roll(index).floor_divide(2)
            tensors[f"layer.{index}.mean_abs_weight"] = torch.ones(6144, dtype=torch.float64)
            tensors[f"layer.{index}.score"] = score
            tensors[f"layer.{index}.ascending_indices"] = torch.argsort(score, stable=True)
        save_file(tensors, str(cls.fixture_path))

    @classmethod
    def tearDownClass(cls):
        cls.fixture_temporary.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.tensor_path = self.directory / "channel_scores.safetensors"
        shutil.copyfile(self.fixture_path, self.tensor_path)
        self.report, self.source = make_report()
        self.update_tensor_record()

    def write_report(self):
        (self.directory / "scoring.json").write_text(json.dumps(self.report), encoding="utf-8")

    def update_tensor_record(self):
        self.report["output"].update(sha256=loader.sha256_file(self.tensor_path), bytes=self.tensor_path.stat().st_size)
        self.write_report()

    def load(self):
        return loader.load_calibrated_scores(self.directory, self.source)

    def reject_report_mutation(self, mutate, error=None):
        report = copy.deepcopy(self.report)
        mutate(report)
        with self.assertRaisesRegex(ValueError, error or ".+"):
            loader.validate_scoring_report(report, self.source)

    def test_loads_fixed_shape_cpu_scores_and_audit_chain(self):
        scores, provenance = self.load()
        self.assertEqual(len(scores), 28)
        self.assertTrue(all(score.shape == (6144,) and score.dtype == torch.float64
                            and score.device.type == "cpu" for score in scores))
        self.assertEqual(scores[0][:6].tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(provenance["scoring_report_sha256"], loader.sha256_file(self.directory / "scoring.json"))
        self.assertEqual(provenance["score_tensor_sha256"], self.report["output"]["sha256"])
        self.assertEqual(provenance["source"]["calibration_data"], self.report["source"]["calibration_data"])
        self.assertTrue(provenance["validation"]["stable_rankings_match_scores"])
        self.assertFalse(provenance["validation"]["linked_calibration_artifacts_reread"])

    def test_returns_detached_copies_of_loaded_scores(self):
        tensors = load_file(str(self.tensor_path), device="cpu")
        with patch.object(loader, "load_file", return_value=tensors):
            scores, provenance = self.load()
        scores[0][0] = 999
        self.assertEqual(tensors["layer.0.score"][0].item(), 0)
        provenance["source"]["calibration_data"]["sample_order"][0]["id"] = "modified"
        self.assertEqual(self.report["source"]["calibration_data"]["sample_order"][0]["id"], "train1")

    def test_rejects_incomplete_or_mutating_scoring_runs(self):
        for key, value in (("status", "failed"), ("stage", "scoring"), ("schema_version", True),
                           ("weights_modified", True), ("weights_modified", 0),
                           ("pruning_applied", True), ("model_forward_calls", 1),
                           ("model_forward_calls", False)):
            with self.subTest(key=key, value=value):
                self.reject_report_mutation(lambda report: report.update({key: value}))

    def test_rejects_other_formula_or_activation_origin(self):
        for key, value in (("formula", "joint_l2_squared"), ("activation_source", "pruned model"),
                           ("ranking", "descending"), ("weight_reduction_and_score_dtype", "float32")):
            with self.subTest(key=key):
                self.reject_report_mutation(lambda report: report["method"].update({key: value}), "method")

    def test_rejects_wrong_model_shape_and_per_layer_metadata(self):
        mutations = [
            lambda r: r["model"].update(hidden_size=1024),
            lambda r: r["model"].update(intermediate_size=5632),
            lambda r: r["model"].update(num_hidden_layers=27),
            lambda r: r["per_layer"].pop(),
            lambda r: r["per_layer"][3].update(layer=2),
            lambda r: r["per_layer"][0].update(layer=False),
            lambda r: r["per_layer"][0].update(width=5632),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.reject_report_mutation(mutation)

    def test_rejects_wrong_revision_checkpoint_and_index(self):
        mutations = [
            lambda r: r["source"].update(revision="other"),
            lambda r: r["source"]["provenance"].update(checkpoint_kind="pruned"),
            lambda r: r["source"].update(weight_index_sha256=digest("another index")),
            lambda r: r["source"]["provenance"].update(manifest_sha256=digest("another manifest")),
            lambda r: r["source"]["provenance"]["download_manifest"].update(revision="other"),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.reject_report_mutation(mutation)

    def test_every_verified_weight_shard_and_config_must_be_bound(self):
        for name in ("config.json", "generation_config.json", "model-00002-of-00002.safetensors"):
            with self.subTest(missing=name):
                self.reject_report_mutation(lambda r: r["source"]["provenance"]["files_sha256"].pop(name))
            with self.subTest(changed=name):
                self.reject_report_mutation(lambda r: r["source"]["provenance"]["files_sha256"].update({name: digest("changed")}))
        self.reject_report_mutation(lambda r: r["source"]["provenance"].update(files_sha256={}))
        self.reject_report_mutation(lambda r: r["source"]["provenance"]["files_sha256"].update({"other.safetensors": digest("other")}))

    def test_rejects_ambiguous_or_incomplete_verified_source_inventory(self):
        invalid_sources = []
        duplicate = copy.deepcopy(self.source)
        duplicate["files"].append(copy.deepcopy(duplicate["files"][0]))
        invalid_sources.append(duplicate)
        no_index = copy.deepcopy(self.source)
        no_index["files"] = [row for row in no_index["files"] if row["name"] != "model.safetensors.index.json"]
        invalid_sources.append(no_index)
        bad_hash = copy.deepcopy(self.source)
        bad_hash["files"][0]["sha256"] = "not-a-digest"
        invalid_sources.append(bad_hash)
        for source in invalid_sources:
            with self.subTest(source=source), self.assertRaises(ValueError):
                loader.validate_scoring_report(self.report, source)

    def test_hash_chain_and_reload_validation_are_required(self):
        mutations = [
            lambda r: r["source"].pop("collection_report_sha256"),
            lambda r: r["source"].update(activation_stats_sha256="bad"),
            lambda r: r["source"]["calibration_data"].pop("manifest_sha256"),
            lambda r: r["source"]["calibration_data"]["files_sha256"].pop("groups.json"),
            lambda r: r["output"].update(reload_exact=False),
            lambda r: r["output"].update(reload_exact=1),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.reject_report_mutation(mutation)

    def test_modified_tensor_bytes_rejected_before_loading(self):
        raw = bytearray(self.tensor_path.read_bytes())
        raw[-1] ^= 1
        self.tensor_path.write_bytes(raw)
        with patch.object(loader, "load_file") as load_tensors, self.assertRaisesRegex(ValueError, "SHA256"):
            self.load()
        load_tensors.assert_not_called()

    def test_tensor_output_cannot_escape_scores_directory(self):
        for path in ("../outside.safetensors", str(self.directory.parent / "outside.safetensors")):
            self.report["output"]["file"] = path
            self.write_report()
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "escapes"):
                self.load()

    def test_missing_extra_or_malformed_tensor_rejected(self):
        fixture = load_file(str(self.fixture_path), device="cpu")
        mutations = [
            lambda t: t.pop("layer.27.score"),
            lambda t: t.update(extra=torch.tensor([1])),
            lambda t: t.update({"layer.0.score": torch.zeros(6143, dtype=torch.float64)}),
            lambda t: t.update({"layer.0.score": torch.zeros(6144, dtype=torch.float32)}),
            lambda t: t.update({"layer.0.mean_abs_weight": torch.full((6144,), -1.0, dtype=torch.float64)}),
            lambda t: t.update({"layer.0.score": torch.full((6144,), float("nan"), dtype=torch.float64)}),
            lambda t: t.update({"layer.0.mean_abs_weight": torch.full((6144,), float("inf"), dtype=torch.float64)}),
        ]
        for mutation in mutations:
            tensors = dict(fixture)
            mutation(tensors)
            save_file(tensors, str(self.tensor_path))
            self.update_tensor_record()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.load()

    def test_ranking_must_match_values_and_stable_tie_order(self):
        for failure in ("tie_order", "different_scores", "dtype", "duplicate"):
            tensors = load_file(str(self.fixture_path), device="cpu")
            ranking = tensors["layer.0.ascending_indices"].clone()
            if failure == "tie_order":
                ranking[0], ranking[1] = 1, 0
            elif failure == "different_scores":
                ranking = ranking.flip(0)
            elif failure == "dtype":
                ranking = ranking.to(dtype=torch.float64)
            else:
                ranking[1] = ranking[0]
            tensors["layer.0.ascending_indices"] = ranking
            save_file(tensors, str(self.tensor_path))
            self.update_tensor_record()
            with self.subTest(failure=failure), self.assertRaisesRegex(ValueError, "stable ascending"):
                self.load()


if __name__ == "__main__":
    unittest.main()
