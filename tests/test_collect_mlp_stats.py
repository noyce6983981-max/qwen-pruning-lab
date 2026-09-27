"""Independent CPU checks of Qwen3 gated-MLP activation collection.

All models are tiny and randomly initialized locally. No model download,
tokenizer, GPU, or calibration corpus is required.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_mlp_stats import MLPActivationCollector


def tiny_model():
    torch.manual_seed(173)
    config = Qwen3Config(vocab_size=97, hidden_size=24, intermediate_size=40,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=6,
                        max_position_embeddings=64, attention_dropout=0.0,
                        pad_token_id=0, bos_token_id=1, eos_token_id=2,
                        tie_word_embeddings=True)
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()


def hook_counts(model):
    return [(len(module._forward_pre_hooks), len(module._forward_hooks)) for module in model.modules()]


def cloned_stats(rows):
    return [{key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in row.items()}
            for row in rows]


class MLPCollectorTests(unittest.TestCase):
    def assert_statistics_close(self, first, second, *, rtol=1e-5, atol=1e-10):
        self.assertEqual(len(first), len(second))
        for actual, expected in zip(first, second):
            for key in ("layer", "width", "token_count"):
                self.assertEqual(actual[key], expected[key])
            for key in ("sum_squares", "rms"):
                self.assertEqual(actual[key].dtype, torch.float64)
                self.assertEqual(actual[key].device.type, "cpu")
                self.assertFalse(actual[key].requires_grad)
                torch.testing.assert_close(actual[key], expected[key], rtol=rtol, atol=atol)

    def test_statistics_match_independent_gated_activation_formula_with_mask(self):
        model = tiny_model()
        ids = torch.tensor([[1, 7, 8, 9, 0], [1, 4, 6, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
        references = {}
        handles = []
        for index, layer in enumerate(model.model.layers):
            def capture(mlp, inputs, layer_index=index):
                # Independent location and computation: recompute the gated
                # product from the MLP input, not down_proj's hooked tensor.
                hidden = inputs[0]
                z = mlp.act_fn(mlp.gate_proj(hidden)) * mlp.up_proj(hidden)
                squares = z.float().square()[mask.bool()].double().sum(dim=0).cpu()
                references[layer_index] = {"layer": layer_index, "width": 40, "token_count": 7,
                                           "sum_squares": squares, "rms": (squares / 7).sqrt()}
            handles.append(layer.mlp.register_forward_pre_hook(capture))
        try:
            with MLPActivationCollector(model) as collector:
                collector.forward(ids, attention_mask=mask)
                actual = collector.statistics()
        finally:
            for handle in handles:
                handle.remove()
        self.assert_statistics_close(actual, [references[i] for i in range(2)], rtol=0, atol=0)

    def test_right_padded_batch_matches_separate_unpadded_sequences(self):
        model = tiny_model()
        ids = torch.tensor([[1, 7, 8, 9, 10], [1, 4, 6, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        with MLPActivationCollector(model) as batched:
            batched_hidden = batched.forward(ids, attention_mask=mask)
            batch_stats = cloned_stats(batched.statistics())
        with MLPActivationCollector(model) as separate:
            for index, length in enumerate((5, 3)):
                separate_hidden = separate.forward(ids[index:index + 1, :length])
                torch.testing.assert_close(batched_hidden[index:index + 1, :length], separate_hidden,
                                           rtol=1e-5, atol=1e-6)
            separate_stats = separate.statistics()
        self.assert_statistics_close(batch_stats, separate_stats)
        self.assertTrue(all(row["token_count"] == 8 for row in batch_stats))

    def test_collection_does_not_change_backbone_outputs_weights_or_gradients(self):
        model = tiny_model()
        ids = torch.tensor([[1, 12, 15, 21], [1, 4, 2, 0]])
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
        before = {name: value.clone() for name, value in model.state_dict().items()}
        with torch.no_grad():
            # Compare identical input semantics: each nonpadding sequence starts
            # at position zero; the collector assigns zero to padded positions.
            positions = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]])
            expected = model.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                                   use_cache=False).last_hidden_state
        original_hooks = hook_counts(model)
        with MLPActivationCollector(model) as collector:
            actual = collector.forward(ids, attention_mask=mask)
            self.assertIsInstance(actual, torch.Tensor)
            self.assertEqual(tuple(actual.shape), (2, 4, 24))
            self.assertFalse(actual.requires_grad)
            self.assertIsNone(actual.grad_fn)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(hook_counts(model), original_hooks)
        self.assertFalse(model.training)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), msg=name)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_left_padding_resets_valid_positions_like_separate_unpadded_sequences(self):
        model = tiny_model()
        ids = torch.tensor([[1, 7, 8, 9, 10], [0, 0, 1, 4, 6]])
        mask = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]])
        with MLPActivationCollector(model) as batched:
            batch_hidden = batched.forward(ids, attention_mask=mask)
            batch_stats = cloned_stats(batched.statistics())
        with MLPActivationCollector(model) as separate:
            for index, start in enumerate((0, 2)):
                expected_hidden = separate.forward(ids[index:index + 1, start:])
                torch.testing.assert_close(batch_hidden[index:index + 1, start:], expected_hidden,
                                           rtol=1e-5, atol=1e-6)
            separate_stats = separate.statistics()
        self.assert_statistics_close(batch_stats, separate_stats)

    def test_nonfinite_padding_is_excluded_before_square_and_reduction(self):
        model = tiny_model()
        ids = torch.tensor([[1, 7, 8, 0]])
        mask = torch.tensor([[1, 1, 1, 0]])
        with MLPActivationCollector(model) as clean:
            clean.forward(ids, attention_mask=mask)
            expected = cloned_stats(clean.statistics())

        def poison_padding(module, inputs):
            altered = inputs[0].clone()
            altered[:, -1, :] = float("nan")
            return (altered,)

        # Use the last MLP so padding NaNs cannot enter a later attention block;
        # only the collector's handling of excluded positions is under test.
        poison_hook = model.model.layers[-1].mlp.down_proj.register_forward_pre_hook(poison_padding)
        try:
            with MLPActivationCollector(model) as collector:
                collector.forward(ids, attention_mask=mask)
                actual = collector.statistics()
        finally:
            poison_hook.remove()
        self.assert_statistics_close(actual, expected, rtol=0, atol=0)

    def test_partitioned_batches_preserve_counts_sums_and_rms(self):
        model = tiny_model()
        ids = torch.tensor([[1, 12, 15, 21], [1, 4, 2, 5], [1, 3, 6, 7]])
        with MLPActivationCollector(model) as one_batch:
            one_batch.forward(ids)
            expected = cloned_stats(one_batch.statistics())
        with MLPActivationCollector(model) as partitioned:
            partitioned.forward(ids[:1])
            partitioned.forward(ids[1:])
            actual = partitioned.statistics()
        self.assert_statistics_close(actual, expected)
        self.assertTrue(all(row["token_count"] == 12 for row in actual))

    def test_failure_after_first_layer_does_not_commit_partial_statistics(self):
        model = tiny_model()
        ids = torch.tensor([[1, 9, 3]])
        original_hooks = hook_counts(model)

        def fail_later(module, inputs):
            raise RuntimeError("deliberate second-layer failure")

        with MLPActivationCollector(model) as collector:
            collector.forward(ids)
            expected = cloned_stats(collector.statistics())
            failure_hook = model.model.layers[1].mlp.register_forward_pre_hook(fail_later)
            try:
                with self.assertRaisesRegex(RuntimeError, "deliberate"):
                    collector.forward(ids)
            finally:
                failure_hook.remove()
            self.assert_statistics_close(collector.statistics(), expected, rtol=0, atol=0)
            collector.forward(ids)
            after = collector.statistics()
            for actual, initial in zip(after, expected):
                self.assertEqual(actual["token_count"], 2 * initial["token_count"])
                torch.testing.assert_close(actual["sum_squares"], 2 * initial["sum_squares"], rtol=0, atol=0)
                torch.testing.assert_close(actual["rms"], initial["rms"], rtol=0, atol=0)
        self.assertEqual(hook_counts(model), original_hooks)

    def test_context_manager_removes_its_hooks_when_forward_exception_escapes(self):
        model = tiny_model()
        ids = torch.tensor([[1, 3]])

        def fail_later(module, inputs):
            raise RuntimeError("escaping forward failure")

        failure_hook = model.model.layers[1].mlp.register_forward_pre_hook(fail_later)
        existing_hooks = hook_counts(model)
        try:
            with self.assertRaisesRegex(RuntimeError, "escaping"):
                with MLPActivationCollector(model) as collector:
                    collector.forward(ids)
            self.assertEqual(hook_counts(model), existing_hooks)
            with self.assertRaises(ValueError):
                collector.statistics()
        finally:
            failure_hook.remove()

    def test_statistics_require_successfully_collected_tokens(self):
        with MLPActivationCollector(tiny_model()) as collector:
            with self.assertRaises(ValueError):
                collector.statistics()

    def test_invalid_masks_reject_without_changing_previous_statistics(self):
        model = tiny_model()
        ids = torch.tensor([[1, 3, 4], [1, 5, 6]])
        malformed = [torch.ones(2, 2), torch.ones(2, 3, 1), torch.ones(3),
                     torch.tensor([[1, 1, 2], [1, 1, 1]]),
                     torch.tensor([[1, 1, -1], [1, 1, 1]]),
                     torch.tensor([[1, 1, 0.5], [1, 1, 1]]),
                     torch.tensor([[1, 1, float("nan")], [1, 1, 1]]),
                     torch.zeros(2, 3), torch.tensor([[1, 1, 1], [0, 0, 0]])]
        with MLPActivationCollector(model) as collector:
            collector.forward(ids)
            expected = cloned_stats(collector.statistics())
            for mask in malformed:
                with self.subTest(shape=tuple(mask.shape), mask=mask.tolist()):
                    with self.assertRaises(ValueError):
                        collector.forward(ids, attention_mask=mask)
                    self.assert_statistics_close(collector.statistics(), expected, rtol=0, atol=0)

    def test_nonfinite_activation_or_float32_square_overflow_rejects_transaction(self):
        for poison in (float("nan"), float("inf"), 1e20):
            with self.subTest(poison=poison):
                model = tiny_model()
                ids = torch.tensor([[1, 3, 4]])
                # Register this before the collector, so the collector sees
                # a nonfinite activation, or a finite activation whose required
                # float32 square overflows. Both must prevent metric corruption.
                def inject(module, inputs):
                    return (torch.full_like(inputs[0], poison),)

                poison_hook = model.model.layers[1].mlp.down_proj.register_forward_pre_hook(inject)
                existing_hooks = hook_counts(model)
                try:
                    with self.assertRaises(ValueError):
                        with MLPActivationCollector(model) as collector:
                            collector.forward(ids)
                    with self.assertRaises(ValueError):
                        collector.statistics()
                    self.assertEqual(hook_counts(model), existing_hooks)
                finally:
                    poison_hook.remove()


if __name__ == "__main__":
    unittest.main()
