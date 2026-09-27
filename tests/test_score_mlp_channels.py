"""Independent CPU mathematics checks for the local MLP channel score.

The ablation test concerns only one channel's contribution to a fixed linear
down projection, not the effect of deleting a channel from the whole model.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from score_mlp_channels import compute_channel_scores


class ChannelScoreMathTests(unittest.TestCase):
    def assert_result(self, result, means, scores, order):
        self.assertEqual(result["mean_abs_weight"].dtype, torch.float64)
        self.assertEqual(result["score"].dtype, torch.float64)
        self.assertEqual(result["ascending_indices"].dtype, torch.int64)
        for key in ("mean_abs_weight", "score", "ascending_indices"):
            self.assertEqual(result[key].device.type, "cpu")
        torch.testing.assert_close(result["mean_abs_weight"], torch.tensor(means, dtype=torch.float64), rtol=0, atol=0)
        torch.testing.assert_close(result["score"], torch.tensor(scores, dtype=torch.float64), rtol=0, atol=0)
        self.assertEqual(result["ascending_indices"].tolist(), order)

    def test_hand_calculated_nonsquare_matrix_uses_channel_columns(self):
        # Three output coordinates, four channels. Every column's absolute
        # sum can be checked by hand: [6, 6, 6, 12], then divide by three.
        weights = torch.tensor([[3, -2, 0, 8], [-1, -4, 5, -2], [2, 0, -1, 2]], dtype=torch.float32)
        rms = torch.tensor([0.5, 2, 0, 0.75], dtype=torch.float64)
        result = compute_channel_scores(weights, rms)
        self.assert_result(result, [2, 2, 2, 4], [1, 4, 0, 3], [2, 0, 3, 1])

    def test_opposite_weight_signs_do_not_cancel_before_magnitude_average(self):
        weights = torch.tensor([[1.0, -3.0], [-1.0, 3.0]])
        result = compute_channel_scores(weights, torch.ones(2))
        self.assert_result(result, [1, 3], [1, 3], [0, 1])

    def test_score_equals_average_coordinate_rms_of_local_channel_ablation(self):
        # Compute actual local output differences, independently of the score
        # formula. Delete a channel by setting only its observed activation to 0.
        weights = torch.tensor([[1.0, -4.0], [-2.0, 1.0], [3.0, 0.0]], dtype=torch.float64)
        activations = torch.tensor([[1.0, 2.0], [-1.0, 4.0], [2.0, -2.0], [-2.0, -4.0]], dtype=torch.float64)
        rms = torch.tensor([math.sqrt(2.5), math.sqrt(10.0)], dtype=torch.float64)
        result = compute_channel_scores(weights, rms)
        original_output = activations @ weights.T
        for channel in range(2):
            ablated = activations.clone()
            ablated[:, channel] = 0
            changed_output = ablated @ weights.T
            differences = original_output - changed_output
            coordinate_rms = differences.square().mean(dim=0).sqrt()
            observed_effect = coordinate_rms.mean()
            torch.testing.assert_close(result["score"][channel], observed_effect, rtol=1e-14, atol=1e-14)

    def test_rms_and_l2_rank_identically_when_observation_count_is_shared(self):
        weights = torch.tensor([[2.0, -1.0, 3.0, 0.0], [-4.0, 3.0, 1.0, 0.0]])
        # Nine observations: L2 = 3 * RMS for every channel, including zeros.
        activations = torch.tensor([[1.0, 2.0, 4.0, 0.0]] * 9, dtype=torch.float64)
        rms = activations.square().mean(dim=0).sqrt()
        l2 = activations.square().sum(dim=0).sqrt()
        rms_result = compute_channel_scores(weights, rms)
        l2_result = compute_channel_scores(weights, l2)
        self.assertEqual(rms_result["ascending_indices"].tolist(), [3, 0, 1, 2])
        self.assertTrue(torch.equal(rms_result["ascending_indices"], l2_result["ascending_indices"]))
        torch.testing.assert_close(l2_result["score"], rms_result["score"] * 3, rtol=0, atol=0)

    def test_equal_scores_preserve_original_channel_indices(self):
        weights = torch.tensor([[2.0, -1.0, 0.0, -4.0, 8.0], [2.0, 1.0, 0.0, 4.0, -8.0]])
        rms = torch.tensor([1.0, 2.0, 7.0, 0.5, 0.25])
        result = compute_channel_scores(weights, rms)
        self.assert_result(result, [2, 1, 0, 4, 8], [2, 2, 0, 2, 2], [2, 0, 1, 3, 4])

    def test_bfloat16_weights_and_rms_produce_float64_hand_calculated_scores(self):
        # Each input value is exactly representable as BF16, so the expected
        # fractions are independent of a float32 reference implementation.
        weights = torch.tensor([[-1.5, 0.25, 3.125], [2.25, -2.5, -1.25]], dtype=torch.bfloat16)
        rms = torch.tensor([0.5, 1.5, 2.0], dtype=torch.bfloat16)
        result = compute_channel_scores(weights, rms)
        self.assert_result(result, [1.875, 1.375, 2.1875], [0.9375, 2.0625, 4.375], [0, 1, 2])

    def test_float64_weights_follow_prescribed_float32_conversion_before_reduction(self):
        # Both small increments round away in float32. Reducing the original
        # float64 inputs directly would retain them and violate the protocol.
        weights = torch.tensor([[1 + 2**-30], [-(1 + 2**-29)]], dtype=torch.float64)
        result = compute_channel_scores(weights, torch.tensor([2.0], dtype=torch.float64))
        self.assert_result(result, [1], [2], [0])

    def test_reduction_preserves_small_float32_values_using_float64_sum(self):
        # Both values are exact float32 inputs. Their sum needs more precision:
        # a float32 reduction loses the +1, but float64 gives mean 8,388,608.5.
        weights = torch.tensor([[16777216.0], [-1.0]], dtype=torch.float32)
        result = compute_channel_scores(weights, torch.tensor([1.0]))
        self.assert_result(result, [8388608.5], [8388608.5], [0])

    def test_strided_inputs_are_supported_without_inplace_mutation(self):
        weight_storage = torch.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0], [-10.0, 11.0, -12.0]])
        rms_storage = torch.tensor([1.0, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0, 99.0])
        weights, rms = weight_storage.T, rms_storage[::2]
        self.assertFalse(weights.is_contiguous())
        self.assertFalse(rms.is_contiguous())
        before_weights, before_rms = weight_storage.clone(), rms_storage.clone()
        result = compute_channel_scores(weights, rms)
        self.assert_result(result, [2, 5, 8, 11], [2, 10, 24, 44], [0, 1, 2, 3])
        self.assertTrue(torch.equal(weight_storage, before_weights))
        self.assertTrue(torch.equal(rms_storage, before_rms))

    def test_invalid_inputs_reject_instead_of_emitting_unusable_rankings(self):
        weights = torch.ones(2, 3)
        rms = torch.ones(3)
        invalid = [
            (torch.ones(2, 3, dtype=torch.int64), rms),
            (weights, torch.ones(3, dtype=torch.int64)),
            (torch.ones(2, 3, dtype=torch.bool), rms),
            (weights, torch.ones(3, dtype=torch.bool)),
            (torch.ones(2, 3, dtype=torch.complex64), rms),
            (weights, torch.ones(3, dtype=torch.complex64)),
            (torch.empty(0, 3), rms),
            (torch.empty(2, 0), torch.empty(0)),
            (torch.ones(3), rms),
            (torch.ones(1, 2, 3), rms),
            (weights, torch.ones(1, 3)),
            (weights, torch.ones(2)),
            (weights, torch.empty(0)),
            (weights, torch.tensor([1.0, -0.01, 1.0])),
            (weights, torch.tensor([1.0, float("nan"), 1.0])),
            (weights, torch.tensor([1.0, float("inf"), 1.0])),
            (weights, torch.tensor([1.0, -float("inf"), 1.0])),
        ]
        for value in (float("nan"), float("inf"), -float("inf")):
            corrupted = weights.clone()
            corrupted[0, 1] = value
            invalid.append((corrupted, rms))
        for index, (bad_weights, bad_rms) in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                compute_channel_scores(bad_weights, bad_rms)


if __name__ == "__main__":
    unittest.main()
