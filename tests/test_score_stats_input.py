"""CPU contract tests for consuming saved calibration activation statistics.

The fixture mirrors collection.json's protocol but uses two width-three layers.
It is self-contained: no real checkpoint, calibration data, or GPU is loaded.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from score_mlp_channels import load_activation_statistics


def fixture_report():
    return {
        "schema_version": 1,
        "status": "completed",
        "stage": "completed",
        "provenance": {"checkpoint_kind": "original"},
        "protocol": {
            "activation": "z = SiLU(gate_proj(x)) * up_proj(x); down_proj input",
            "batch_size": 1,
            "model_dtype": "bfloat16",
            "attention": "sdpa",
            "square_dtype": "float32",
            "reduction_dtype": "float64",
            "accumulator_device": "cpu",
            "accumulator_dtype": "float64",
            "rms_definition": "sqrt(sum_squares / valid_token_count)",
            "all_valid_positions_counted_including_first": True,
            "use_cache": False,
            "gradients": False,
            "weights_updated": False,
            "full_activations_saved": False,
            "chat_template": False,
            "document_context": "reset; frozen raw token IDs, no added special tokens",
        },
        "model": {
            "class": "Qwen3ForCausalLM",
            "num_hidden_layers": 2,
            "hidden_size": 2,
            "intermediate_size": 3,
            "mlps": [
                {"layer": i, "gate_proj": [3, 2], "up_proj": [3, 2], "down_proj": [2, 3]}
                for i in range(2)
            ],
        },
        "data": {"document_count": 2, "tokens_per_document": 256, "valid_tokens_per_layer": 512},
        "completed_documents": 2,
        "statistics": {
            "file": "activation_stats.safetensors",
            "reload_exact": True,
            "layers": [
                {"layer": i, "width": 3, "token_count": 512, "forward_calls": 2}
                for i in range(2)
            ],
        },
    }


def fixture_tensors():
    result = {}
    for i, values in enumerate(([0.0, 128.0, 512.0], [128.0, 512.0, 2048.0])):
        sums = torch.tensor(values, dtype=torch.float64)
        result[f"layer.{i}.sum_squares"] = sums
        result[f"layer.{i}.rms"] = (sums / 512).sqrt()
    return result


class SavedStatisticsInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "collection"
        self.directory.mkdir()
        self.report = fixture_report()
        self.tensors = fixture_tensors()

    def save_fixture(self):
        tensor_path = self.directory / self.report["statistics"]["file"]
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(self.tensors, str(tensor_path), metadata={"schema_version": "1", "kind": "qwen3_mlp_activation_stats"})
        self.report["statistics"]["sha256"] = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
        self.report["statistics"]["bytes"] = tensor_path.stat().st_size
        self.save_report()
        return tensor_path

    def save_report(self):
        (self.directory / "collection.json").write_text(json.dumps(self.report), encoding="utf-8")

    def load(self):
        return load_activation_statistics(self.directory)

    def test_valid_statistics_reload_exactly_as_cpu_float64(self):
        self.save_fixture()
        report, tensors = self.load()
        self.assertEqual(report, self.report)
        self.assertEqual(set(tensors), set(self.tensors))
        for name, expected in self.tensors.items():
            self.assertEqual(tensors[name].device.type, "cpu")
            self.assertEqual(tensors[name].dtype, torch.float64)
            self.assertTrue(torch.equal(tensors[name], expected))

    def test_unfinished_or_wrong_schema_reports_are_rejected(self):
        for field, value in (("status", "failed"), ("status", "starting"), ("stage", "saving_statistics"),
                             ("schema_version", 2), ("schema_version", True), ("schema_version", 1.0)):
            with self.subTest(field=field, value=value):
                self.report = fixture_report()
                self.report[field] = value
                self.save_fixture()
                with self.assertRaises(ValueError):
                    self.load()

    def test_nonoriginal_or_different_activation_protocol_is_rejected(self):
        changes = [
            ("provenance", "checkpoint_kind", "pruned"),
            ("protocol", "activation", "up_proj output"),
            ("protocol", "model_dtype", "float32"),
            ("protocol", "rms_definition", "sqrt(sum_squares)"),
            ("protocol", "weights_updated", True),
            ("statistics", "reload_exact", False),
        ]
        for section, field, value in changes:
            with self.subTest(section=section, field=field):
                self.report = fixture_report()
                self.report[section][field] = value
                self.save_fixture()
                with self.assertRaises(ValueError):
                    self.load()

    def test_statistics_file_tamper_fails_even_if_it_remains_valid_safetensors(self):
        tensor_path = self.save_fixture()
        altered = fixture_tensors()
        altered["layer.0.sum_squares"][1] = 512.0
        altered["layer.0.rms"] = (altered["layer.0.sum_squares"] / 512).sqrt()
        save_file(altered, str(tensor_path))
        with self.assertRaises(ValueError):
            self.load()

    def test_statistics_path_cannot_escape_collection_directory(self):
        for path in ("../outside.safetensors", str(Path(self.temp.name) / "absolute_outside.safetensors")):
            with self.subTest(path=path):
                self.report = fixture_report()
                self.report["statistics"]["file"] = path
                self.save_fixture()  # A valid external file proves containment is checked.
                with self.assertRaises(ValueError):
                    self.load()

    def test_nested_statistics_file_within_directory_is_allowed(self):
        self.report["statistics"]["file"] = "stats/activation.safetensors"
        self.save_fixture()
        self.assertEqual(set(self.load()[1]), set(self.tensors))

    def test_model_dimensions_are_positive_integers_not_bools(self):
        for field in ("num_hidden_layers", "hidden_size", "intermediate_size"):
            for value in (0, -1, True, 2.0):
                with self.subTest(field=field, value=value):
                    self.report = fixture_report()
                    self.report["model"][field] = value
                    self.save_fixture()
                    with self.assertRaises(ValueError):
                        self.load()

    def test_budget_counts_and_completed_documents_must_agree(self):
        mutations = [
            ("data", "document_count", 3),
            ("data", "tokens_per_document", 255),
            ("data", "valid_tokens_per_layer", 511),
            ("data", "document_count", 2.0),
            ("data", "tokens_per_document", 256.0),
            ("data", "valid_tokens_per_layer", 512.0),
            (None, "completed_documents", 1),
            (None, "completed_documents", 2.0),
        ]
        for section, field, value in mutations:
            with self.subTest(section=section, field=field, value=value):
                self.report = fixture_report()
                target = self.report if section is None else self.report[section]
                target[field] = value
                self.save_fixture()
                with self.assertRaises(ValueError):
                    self.load()

    def test_layer_summaries_match_order_width_and_budget(self):
        original_layers = fixture_report()["statistics"]["layers"]
        variants = [list(reversed(original_layers)), original_layers[:1], original_layers + [copy.deepcopy(original_layers[-1])]]
        for field, value in (("layer", 1), ("width", 4), ("token_count", 511), ("forward_calls", 1),
                             ("layer", False), ("width", 3.0), ("token_count", 512.0), ("forward_calls", 2.0)):
            layers = copy.deepcopy(original_layers)
            layers[0][field] = value
            variants.append(layers)
        for index, layers in enumerate(variants):
            with self.subTest(variant=index):
                self.report = fixture_report()
                self.report["statistics"]["layers"] = layers
                self.save_fixture()
                with self.assertRaises(ValueError):
                    self.load()

    def test_tensor_keys_must_be_exact_no_missing_or_extra_layers(self):
        for action in ("missing_rms", "extra_layer", "unrelated_tensor"):
            with self.subTest(action=action):
                self.tensors = fixture_tensors()
                if action == "missing_rms":
                    del self.tensors["layer.1.rms"]
                elif action == "extra_layer":
                    self.tensors["layer.2.rms"] = torch.ones(3, dtype=torch.float64)
                else:
                    self.tensors["extra"] = torch.ones(3, dtype=torch.float64)
                self.save_fixture()
                with self.assertRaises(ValueError):
                    self.load()

    def test_tensor_shapes_and_dtypes_are_checked(self):
        for suffix in ("sum_squares", "rms"):
            for replacement in (torch.ones(3, dtype=torch.float32), torch.ones(3, dtype=torch.int64),
                                torch.ones(1, 3, dtype=torch.float64), torch.ones(2, dtype=torch.float64)):
                with self.subTest(suffix=suffix, dtype=replacement.dtype, shape=list(replacement.shape)):
                    self.tensors = fixture_tensors()
                    self.tensors[f"layer.0.{suffix}"] = replacement
                    self.save_fixture()
                    with self.assertRaises(ValueError):
                        self.load()

    def test_negative_and_nonfinite_statistics_are_rejected(self):
        for suffix in ("sum_squares", "rms"):
            for value in (-1.0, float("nan"), float("inf")):
                with self.subTest(suffix=suffix, value=value):
                    self.tensors = fixture_tensors()
                    self.tensors[f"layer.0.{suffix}"][1] = value
                    self.save_fixture()
                    with self.assertRaises(ValueError):
                        self.load()

    def test_rms_recalculation_requires_exact_equality_not_allclose(self):
        original_rms = self.tensors["layer.0.rms"].clone()
        self.tensors["layer.0.rms"][1] = torch.nextafter(original_rms[1], torch.tensor(float("inf"), dtype=torch.float64))
        self.assertTrue(torch.allclose(self.tensors["layer.0.rms"], original_rms))
        self.assertFalse(torch.equal(self.tensors["layer.0.rms"], original_rms))
        self.save_fixture()
        with self.assertRaises(ValueError):
            self.load()


if __name__ == "__main__":
    unittest.main()
