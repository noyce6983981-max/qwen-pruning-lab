"""CPU structural and behavioral tests for externally scored Qwen3 MLP pruning.

Tiny models are initialized locally; no tokenizer, network, or GPU is used.
The masked reference checks the pruning operation, not language-model quality.
"""

from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prune_qwen_mlp import load_model, parameter_count, probe_logits, prune_model, verify_reload


def tiny_model():
    torch.manual_seed(801)
    config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=8,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=4,
                        max_position_embeddings=64, attention_dropout=0.0,
                        pad_token_id=0, bos_token_id=1, eos_token_id=2,
                        tie_word_embeddings=True, use_cache=False)
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).cpu().float().eval()


def selected_scores():
    # Hand-written rankings deliberately differ by layer and are not in index
    # order. Remove three lowest: layer 0 -> [1,3,6], layer 1 -> [0,2,4].
    return [torch.tensor([5, 1, 7, 0, 6, 3, 2, 4], dtype=torch.float64),
            torch.tensor([1, 6, 0, 4, 2, 7, 5, 3], dtype=torch.float32)]


class CalibratedPruningTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.redirect = redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def assert_model_unchanged(self, model, original_state, original_parameters, original_config):
        self.assertEqual(model.config.to_dict(), original_config)
        self.assertEqual(set(model.state_dict()), set(original_state))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original_state[name]), msg=name)
        for name, parameter in model.named_parameters():
            original, requires_grad = original_parameters[name]
            self.assertIs(parameter, original, msg=name)
            self.assertEqual(parameter.requires_grad, requires_grad, msg=name)

    def test_per_layer_rankings_keep_exact_original_ordered_weight_slices(self):
        model = tiny_model()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        scores = selected_scores()
        scores_before = [value.clone() for value in scores]
        before_count = parameter_count(model)
        report = prune_model(model, 5, channel_scores=scores)
        kept_by_layer = [[0, 2, 4, 5, 7], [1, 3, 5, 6, 7]]
        removed_by_layer = [[1, 3, 6], [0, 2, 4]]
        touched = set()
        for index, (layer, kept, removed) in enumerate(zip(model.model.layers, kept_by_layer, removed_by_layer)):
            self.assertEqual(report["per_layer"][index]["kept_indices"], kept)
            self.assertEqual(report["per_layer"][index]["removed_indices"], removed)
            for name in ("gate_proj", "up_proj"):
                key = f"model.layers.{index}.mlp.{name}.weight"
                touched.add(key)
                linear = getattr(layer.mlp, name)
                self.assertTrue(torch.equal(linear.weight, before[key][kept, :]), msg=key)
                self.assertEqual((linear.in_features, linear.out_features), (16, 5))
            key = f"model.layers.{index}.mlp.down_proj.weight"
            touched.add(key)
            self.assertTrue(torch.equal(layer.mlp.down_proj.weight, before[key][:, kept]), msg=key)
            self.assertEqual((layer.mlp.down_proj.in_features, layer.mlp.down_proj.out_features), (5, 16))
            self.assertEqual(layer.mlp.intermediate_size, 5)
        for name, value in model.state_dict().items():
            if name not in touched:
                self.assertTrue(torch.equal(value, before[name]), msg=name)
        self.assertEqual(model.config.intermediate_size, 5)
        self.assertEqual(model.config.hidden_size, 16)
        self.assertEqual(model.config.num_hidden_layers, 2)
        self.assertEqual(before_count - parameter_count(model), 3 * 16 * 3 * 2)
        self.assertEqual(report["parameters_removed"], 288)
        self.assertEqual(report["parameters_before"], before_count)
        self.assertEqual(report["parameters_after"], parameter_count(model))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        for original, actual in zip(scores_before, scores):
            self.assertTrue(torch.equal(original, actual))

    def test_full_model_logits_match_original_with_same_down_columns_zeroed(self):
        pruned = tiny_model()
        masked = copy.deepcopy(pruned)
        # Independent expected deletion sets; do not infer from prune report.
        with torch.no_grad():
            masked.model.layers[0].mlp.down_proj.weight[:, [1, 3, 6]] = 0
            masked.model.layers[1].mlp.down_proj.weight[:, [0, 2, 4]] = 0
        prune_model(pruned, 5, channel_scores=selected_scores())
        cpu = torch.device("cpu")
        actual, expected = probe_logits(pruned, cpu), probe_logits(masked, cpu)
        self.assertEqual(tuple(actual.shape), (1, 16, 64))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_pruned_model_save_reload_has_exact_weights_config_and_probe_logits(self):
        model = tiny_model()
        report = prune_model(model, 5, channel_scores=selected_scores())
        cpu = torch.device("cpu")
        expected = probe_logits(model, cpu)
        with tempfile.TemporaryDirectory(prefix="tiny-calibrated-pruning-") as temporary:
            directory = Path(temporary)
            model.save_pretrained(directory, safe_serialization=True)
            verification = verify_reload(directory, cpu, torch.float32, "eager", expected,
                                         report["parameters_after"], 5)
            self.assertEqual(verification["status"], "passed")
            self.assertEqual(verification["max_absolute_error"], 0)
            reloaded = load_model(directory, cpu, torch.float32, "eager")
            self.assertEqual(reloaded.config.intermediate_size, 5)
            self.assertEqual(parameter_count(reloaded), parameter_count(model))
            self.assertEqual(set(reloaded.state_dict()), set(model.state_dict()))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, reloaded.state_dict()[name]), msg=name)
            torch.testing.assert_close(probe_logits(reloaded, cpu), expected, rtol=0, atol=0)

    def test_same_score_ties_delete_lower_original_indices_first(self):
        model = tiny_model()
        scores = [torch.ones(8), torch.tensor([4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 5.0, 1.0])]
        report = prune_model(model, 5, channel_scores=scores)
        self.assertEqual(report["per_layer"][0]["removed_indices"], [0, 1, 2])
        self.assertEqual(report["per_layer"][0]["kept_indices"], [3, 4, 5, 6, 7])
        # Channel 2 has zero; the first two tied unit-score channels are 1,3.
        self.assertEqual(report["per_layer"][1]["removed_indices"], [1, 2, 3])
        self.assertEqual(report["per_layer"][1]["kept_indices"], [0, 4, 5, 6, 7])

    def test_bad_last_layer_scores_fail_before_any_layer_or_grad_flag_changes(self):
        invalid_last = [torch.ones(7), torch.ones(1, 8), torch.empty(0),
                        torch.ones(8, dtype=torch.int64), torch.ones(8, dtype=torch.bool),
                        torch.ones(8, dtype=torch.complex64), torch.empty(8, device="meta")]
        for invalid_value in (float("nan"), float("inf"), -float("inf"), -0.01):
            item = torch.ones(8)
            item[-1] = invalid_value
            invalid_last.append(item)
        for index, bad_score in enumerate(invalid_last):
            with self.subTest(case=index):
                model = tiny_model()
                before = {name: value.clone() for name, value in model.state_dict().items()}
                parameters = {name: (value, value.requires_grad) for name, value in model.named_parameters()}
                config = copy.deepcopy(model.config.to_dict())
                with self.assertRaises(ValueError):
                    prune_model(model, 5, channel_scores=[torch.arange(8, dtype=torch.float64), bad_score])
                self.assert_model_unchanged(model, before, parameters, config)

    def test_score_layer_count_is_validated_before_mutation(self):
        for scores in ([], [torch.ones(8)], [torch.ones(8)] * 3):
            with self.subTest(layer_count=len(scores)):
                model = tiny_model()
                before = {name: value.clone() for name, value in model.state_dict().items()}
                parameters = {name: (value, value.requires_grad) for name, value in model.named_parameters()}
                config = copy.deepcopy(model.config.to_dict())
                with self.assertRaises(ValueError):
                    prune_model(model, 5, channel_scores=scores)
                self.assert_model_unchanged(model, before, parameters, config)

    def test_invalid_keep_width_types_and_bounds_fail_before_mutation(self):
        for keep_width in (True, False, 5.0, 0, -1, 8, 9):
            with self.subTest(keep_width=keep_width):
                model = tiny_model()
                before = {name: value.clone() for name, value in model.state_dict().items()}
                parameters = {name: (value, value.requires_grad) for name, value in model.named_parameters()}
                config = copy.deepcopy(model.config.to_dict())
                with self.assertRaises(ValueError):
                    prune_model(model, keep_width, channel_scores=selected_scores())
                self.assert_model_unchanged(model, before, parameters, config)

    def test_default_magnitude_path_retains_joint_three_projection_behavior(self):
        model = tiny_model()
        # Each projection carries different information. Ranking any projection
        # alone would choose different channels from the joint squared score.
        # Sum of the three amplitude squares per channel:
        # L0 [25,4,1,0,1,4,4,3]; L1 [1,3,0,1,4,25,4,4].
        amplitudes = [
            ([0, 0, 1, 0, 0, 0, 2, 1], [0, 2, 0, 0, 1, 0, 0, 1], [5, 0, 0, 0, 0, 2, 0, 1]),
            ([0, 1, 0, 1, 2, 0, 0, 0], [1, 1, 0, 0, 0, 0, 2, 0], [0, 1, 0, 0, 0, 5, 0, 2]),
        ]
        with torch.no_grad():
            for layer, (gate, up, down) in zip(model.model.layers, amplitudes):
                for channel in range(8):
                    layer.mlp.gate_proj.weight[channel].fill_(gate[channel])
                    layer.mlp.up_proj.weight[channel].fill_(-up[channel])
                    layer.mlp.down_proj.weight[:, channel].fill_(down[channel])
        explicit_none = copy.deepcopy(model)
        report = prune_model(model, 5)
        none_report = prune_model(explicit_none, 5, channel_scores=None)
        self.assertEqual(report["per_layer"][0]["removed_indices"], [2, 3, 4])
        self.assertEqual(report["per_layer"][1]["removed_indices"], [0, 2, 3])
        self.assertEqual(report["per_layer"], none_report["per_layer"])
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, explicit_none.state_dict()[name]), msg=name)


if __name__ == "__main__":
    unittest.main()
