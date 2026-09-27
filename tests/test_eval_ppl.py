"""Small deterministic CPU tests of target coverage and language-model loss."""

from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_ppl import evaluate_ppl


VOCAB = 7


def manual_logits(prefix):
    """Exactly representable half-precision scores sensitive to full context."""
    center = sum(prefix) % VOCAB
    return [-abs(token - center) * 0.5 + ((token + len(prefix)) % 3) * 0.25 for token in range(VOCAB)]


def manual_loss(prefix, target):
    logits = manual_logits(prefix)
    maximum = max(logits)
    return maximum + math.log(sum(math.exp(value - maximum) for value in logits)) - logits[target]


class ToyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids, use_cache):
        assert not use_cache
        assert not self.training
        assert torch.is_inference_mode_enabled()
        tokens = input_ids[0].tolist()
        self.calls.append(tokens)
        logits = [manual_logits(tokens[:index + 1]) for index in range(len(tokens))]
        return SimpleNamespace(logits=torch.tensor([logits], dtype=torch.float16))


class PPLTests(unittest.TestCase):
    def setUp(self):
        self.progress = io.StringIO()
        self.redirect = redirect_stdout(self.progress)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def test_short_document_matches_independent_manual_next_token_reference(self):
        tokens = [2, 4, 3, 0, 6]
        model = ToyLM()
        result = evaluate_ppl(model, [{"id": "short", "token_ids": tokens}], "cpu", window=8, stride=3)
        expected = math.fsum(manual_loss(tokens[:position], tokens[position]) for position in range(1, len(tokens)))
        self.assertEqual(result["valid_token_count"], 4)
        self.assertAlmostEqual(result["nll_sum"], expected, places=6)
        self.assertAlmostEqual(result["nll"], expected / 4, places=6)
        self.assertAlmostEqual(result["ppl"], math.exp(expected / 4), places=5)
        self.assertTrue(model.training)

    def test_overlap_and_tail_use_every_target_once_without_extra_subtraction(self):
        tokens = [0, 3, 1, 6, 2, 5, 4, 1, 2, 0]
        model = ToyLM()
        result = evaluate_ppl(model, [{"id": "tail", "token_ids": tokens}], "cpu", window=5, stride=2)
        # Explicit independent expected context for each target (window=5).
        contexts = [(0, p) for p in range(1, 5)] + [(2, 5), (2, 6), (4, 7), (4, 8), (6, 9)]
        expected = math.fsum(manual_loss(tokens[start:position], tokens[position]) for start, position in contexts)
        self.assertEqual(result["valid_token_count"], 9)
        self.assertAlmostEqual(result["nll_sum"], expected, places=6)
        self.assertEqual(model.calls, [tokens[0:5], tokens[2:7], tokens[4:9], tokens[6:10]])
        windows = result["documents"][0]["windows"]
        self.assertEqual([row["target_count"] for row in windows], [4, 2, 2, 1])
        scored_positions = [position for row in windows for position in range(row["first_target"], row["end"])]
        self.assertEqual(scored_positions, list(range(1, 10)))

    def test_many_lengths_and_strides_have_no_missing_or_duplicate_targets(self):
        for window in range(2, 7):
            for stride in range(1, window):
                for length in range(2, 15):
                    with self.subTest(window=window, stride=stride, length=length):
                        tokens = [position % VOCAB for position in range(length)]
                        result = evaluate_ppl(ToyLM(), [{"id": "coverage", "token_ids": tokens}], "cpu", window, stride)
                        windows = result["documents"][0]["windows"]
                        positions = [p for row in windows for p in range(row["first_target"], row["end"])]
                        self.assertEqual(positions, list(range(1, length)))
                        self.assertEqual(result["valid_token_count"], length - 1)
                        self.assertEqual(windows[-1]["end"], length)

    def test_documents_reset_context_and_pool_losses_by_target_count(self):
        documents = [{"id": "a", "token_ids": [6, 0]}, {"id": "b", "token_ids": [0, 0, 0, 0, 0]}]
        original = copy.deepcopy(documents)
        model = ToyLM().eval()
        result = evaluate_ppl(model, documents, "cpu", window=8, stride=3)
        first = manual_loss([6], 0)
        second = math.fsum(manual_loss([0] * position, 0) for position in range(1, 5))
        self.assertEqual(result["valid_token_count"], 5)
        self.assertEqual(result["document_count"], 2)
        self.assertAlmostEqual(result["nll"], (first + second) / 5, places=6)
        self.assertNotAlmostEqual(result["nll"], (first + second / 4) / 2, places=3)
        self.assertNotAlmostEqual(result["ppl"], sum(row["ppl"] for row in result["documents"]) / 2, places=3)
        self.assertEqual(model.calls, [document["token_ids"] for document in documents])
        self.assertEqual(documents, original)
        self.assertFalse(model.training)
        json.dumps(result, allow_nan=False)

    def test_invalid_inputs_raise_before_any_forward_instead_of_skipping(self):
        document = {"id": "valid", "token_ids": [0, 1]}
        invalid_documents = [[], None, [None], [{"id": "x", "token_ids": []}],
                             [{"id": "x", "token_ids": [0]}], [document, document],
                             [{"id": "", "token_ids": [0, 1]}],
                             [{"id": "x", "token_ids": [True, 1]}],
                             [{"id": "x", "token_ids": [-1, 1]}],
                             [{"id": "x", "token_ids": [0.0, 1]}],
                             [document, {"id": "later-invalid", "token_ids": [1]}]]
        for documents in invalid_documents:
            model = ToyLM()
            with self.subTest(documents=documents), self.assertRaises(ValueError):
                evaluate_ppl(model, documents, "cpu")
            self.assertEqual(model.calls, [])
        for window, stride in [(1, 1), (5, 0), (5, -1), (5, 5), (5, 6), (5.0, 2), (5, True)]:
            with self.subTest(window=window, stride=stride), self.assertRaises(ValueError):
                evaluate_ppl(ToyLM(), [document], "cpu", window, stride)

    def test_nonfinite_loss_fails_and_restores_model_training_mode(self):
        class BrokenLM(ToyLM):
            def forward(self, input_ids, use_cache):
                output = super().forward(input_ids, use_cache)
                output.logits.fill_(float("nan"))
                return output

        model = BrokenLM()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            evaluate_ppl(model, [{"id": "broken", "token_ids": [0, 1]}], "cpu")
        self.assertTrue(model.training)

    def test_ppl_overflow_is_explicit_and_json_safe(self):
        class ExtremeLM(ToyLM):
            def forward(self, input_ids, use_cache):
                output = super().forward(input_ids, use_cache)
                output.logits.zero_()
                output.logits[:, :, 0] = 1000
                return output

        result = evaluate_ppl(ExtremeLM(), [{"id": "extreme", "token_ids": [0, 1]}], "cpu")
        self.assertEqual(result["nll"], 1000)
        self.assertIsNone(result["ppl"])
        self.assertTrue(result["ppl_overflow"])
        json.dumps(result, allow_nan=False)

    def test_chunked_ce_matches_previous_full_boolean_reference(self):
        # Deliberately exceed the 64-target loss chunk boundary and include
        # masked overlap, unequal documents, and a short final window.
        documents = [{"id": "long", "token_ids": [(i * 3 + 2) % VOCAB for i in range(307)]},
                     {"id": "unequal", "token_ids": [(i * 5) % VOCAB for i in range(77)]}]
        window, stride = 145, 71
        old_loss_parts = []
        old_count = 0
        reference_model = ToyLM().eval()
        with torch.inference_mode():
            for document in documents:
                tokens = document["token_ids"]
                previous_end = 0
                for start in range(0, len(tokens), stride):
                    end = min(start + window, len(tokens))
                    inputs = torch.tensor([tokens[start:end]], dtype=torch.long)
                    labels = inputs.clone()
                    labels[:, :max(1, previous_end) - start] = -100
                    shifted = labels[:, 1:]
                    valid = shifted != -100
                    logits = reference_model(inputs, use_cache=False).logits
                    losses = torch.nn.functional.cross_entropy(logits[:, :-1, :][valid].float(),
                                                              shifted[valid], reduction="none")
                    old_loss_parts.append(float(losses.double().sum().item()))
                    old_count += int(valid.sum().item())
                    previous_end = end
                    if end == len(tokens):
                        break
        result = evaluate_ppl(ToyLM(), documents, "cpu", window, stride)
        self.assertEqual(result["valid_token_count"], old_count)
        self.assertEqual(result["nll_sum"], math.fsum(old_loss_parts))
        self.assertEqual(result["nll"], math.fsum(old_loss_parts) / old_count)
        self.assertEqual(result["protocol"]["loss_chunk_size"], 64)

    def test_loss_chunk_limit_progress_and_no_cuda_calls_on_cpu(self):
        original_ce = torch.nn.functional.cross_entropy
        shapes = []

        def recorded_ce(logits, targets, **kwargs):
            shapes.append(tuple(logits.shape))
            return original_ce(logits, targets, **kwargs)

        with patch("eval_ppl.F.cross_entropy", side_effect=recorded_ce), patch("torch.cuda.empty_cache") as empty_cache:
            evaluate_ppl(ToyLM(), [{"id": "chunk-check", "token_ids": [i % VOCAB for i in range(150)]}],
                         "cpu", window=145, stride=71)
        self.assertEqual(shapes, [(64, VOCAB), (64, VOCAB), (16, VOCAB), (5, VOCAB)])
        empty_cache.assert_not_called()
        progress = self.progress.getvalue()
        self.assertIn("window 1 [0, 145), targets 144/149", progress)
        self.assertIn("window 2 [71, 150), targets 149/149", progress)
        self.assertIn("completed: 149 targets", progress)


if __name__ == "__main__":
    unittest.main()
