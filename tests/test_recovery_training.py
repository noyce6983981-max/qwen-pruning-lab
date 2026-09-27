"""CPU checks that accumulated updates normalize by supervised tokens."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from recovery_training import batch_groups, train_accumulated_group


class Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(5, 7, bias=False).double().requires_grad_(False)
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(5, 2, bias=False).double()})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 7, bias=False).double()})

    def forward(self, hidden):
        return self.base(hidden) + self.lora_B["default"](self.lora_A["default"](hidden))


class TinyLM(torch.nn.Module):
    """Deterministic causal next-token loss, with frozen base and no dropout."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(93)
        self.embedding = torch.nn.Embedding(7, 5).double().requires_grad_(False)
        self.projection = Projection()
        self.calls = 0
        self.fail_on = None
        self.nan_on = None

    def forward(self, input_ids, labels, attention_mask, use_cache):
        assert use_cache is False
        assert attention_mask.eq(1).all()
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("intentional forward failure")
        logits = self.projection(self.embedding(input_ids))
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 7), labels[:, 1:].reshape(-1), ignore_index=-100)
        if self.calls == self.nan_on:
            loss = loss * float("nan")
        return SimpleNamespace(loss=loss, logits=logits)


class RecordingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.03)
        self.step_calls = 0
        self.captured_gradients = None

    def step(self, closure=None):
        self.step_calls += 1
        self.captured_gradients = [p.grad.clone() if p.grad is not None else None
                                   for group in self.param_groups for p in group["params"]]
        return super().step(closure)


def rows():
    return [
        {"id": "short", "input_ids": [1, 3, 4, 2], "labels": [-100, -100, 4, 2],
         "attention_mask": [1] * 4, "sequence_length": 4, "prompt_length": 2,
         "supervised_token_count": 2, "eos_token_id": 2},
        {"id": "long", "input_ids": [1, 5, 1, 4, 5, 6, 2], "labels": [-100, -100, 1, 4, 5, 6, 2],
         "attention_mask": [1] * 7, "sequence_length": 7, "prompt_length": 2,
         "supervised_token_count": 5, "eos_token_id": 2},
    ]


def trainable(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def reference_token_losses(model, items):
    """Independent pooled-token reference; never uses helper's row weighting."""
    collected = []
    for row in items:
        inputs = torch.tensor([row["input_ids"]])
        labels = torch.tensor(row["labels"][1:])
        logits = model.projection(model.embedding(inputs))[0, :-1]
        per_token = F.cross_entropy(logits, labels, reduction="none", ignore_index=-100)
        collected.append(per_token[labels.ne(-100)])
    return collected


class AccumulationTests(unittest.TestCase):
    def test_weighted_gradients_and_update_match_one_pooled_token_reference(self):
        actual = TinyLM()
        expected = copy.deepcopy(actual)
        before = {name: parameter.clone() for name, parameter in actual.named_parameters() if not parameter.requires_grad}
        optimizer = RecordingSGD(trainable(actual))
        reference_optimizer = RecordingSGD(trainable(expected))
        pooled = torch.cat(reference_token_losses(expected, rows()))
        self.assertEqual(pooled.numel(), 7)  # includes both EOS labels
        expected_loss = pooled.mean()
        expected_loss.backward()
        expected_norm = torch.nn.utils.clip_grad_norm_(trainable(expected), 0.001, error_if_nonfinite=True)
        reference_optimizer.step()
        metrics = train_accumulated_group(actual, rows(), optimizer, "cpu", max_grad_norm=0.001)
        self.assertEqual(optimizer.step_calls, 1)
        self.assertAlmostEqual(metrics["weighted_loss"], expected_loss.item(), places=13)
        self.assertAlmostEqual(metrics["gradient_norm_before_clip"], expected_norm.item(), places=13)
        self.assertEqual(metrics["supervised_token_count"], 7)
        self.assertEqual(metrics["sample_ids"], ["short", "long"])
        self.assertEqual(metrics["sequence_lengths"], [4, 7])
        self.assertEqual([item["backward_weight"] for item in metrics["per_item"]], [2/7, 5/7])
        for actual_grad, reference_grad in zip(optimizer.captured_gradients, reference_optimizer.captured_gradients):
            torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-12, atol=1e-14)
        for actual_parameter, expected_parameter in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(actual_parameter, expected_parameter, rtol=1e-12, atol=1e-14)
        self.assertTrue(all(parameter.grad is None for parameter in actual.parameters()))
        for name, parameter in actual.named_parameters():
            if name in before:
                self.assertTrue(torch.equal(parameter, before[name]))
        self.assertEqual(metrics["A_nonzero_tensor_count"], 1)
        self.assertEqual(metrics["B_nonzero_tensor_count"], 1)

    def test_per_question_average_would_give_a_different_update(self):
        actual = TinyLM()
        wrong = copy.deepcopy(actual)
        optimizer = RecordingSGD(trainable(actual))
        wrong_optimizer = RecordingSGD(trainable(wrong))
        losses = reference_token_losses(wrong, rows())
        wrong_loss = torch.stack([loss.mean() for loss in losses]).mean()
        wrong_loss.backward()
        wrong_optimizer.step()
        metrics = train_accumulated_group(actual, rows(), optimizer, "cpu", max_grad_norm=1000)
        self.assertGreater(abs(metrics["weighted_loss"] - wrong_loss.item()), 0.01)
        self.assertTrue(any(not torch.allclose(a, b, rtol=1e-6, atol=1e-8)
                            for a, b in zip(trainable(actual), trainable(wrong))))

    def test_later_forward_failure_or_nan_does_not_step_and_clears_earlier_gradients(self):
        for failure in ("fail_on", "nan_on"):
            with self.subTest(failure=failure):
                model = TinyLM()
                setattr(model, failure, 2)
                before = {name: value.clone() for name, value in model.state_dict().items()}
                optimizer = RecordingSGD(trainable(model))
                with self.assertRaises((RuntimeError, ValueError)):
                    train_accumulated_group(model, rows(), optimizer, "cpu")
                self.assertEqual(optimizer.step_calls, 0)
                self.assertEqual(model.calls, 2)
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
                self.assertTrue(all(torch.equal(value, before[name]) for name, value in model.state_dict().items()))

    def test_nonfinite_accumulated_gradient_is_rejected_before_step(self):
        model = TinyLM()
        optimizer = RecordingSGD(trainable(model))
        parameter = model.projection.lora_A["default"].weight
        handle = parameter.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        try:
            with self.assertRaisesRegex(ValueError, "Nonfinite accumulated gradient"):
                train_accumulated_group(model, rows(), optimizer, "cpu")
        finally:
            handle.remove()
        self.assertEqual(optimizer.step_calls, 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_frozen_base_gradient_and_unexpected_optimizer_parameters_are_rejected(self):
        for failure in ("base_gradient", "base_in_optimizer", "missing_adapter", "trainable_base"):
            with self.subTest(failure=failure):
                model = TinyLM()
                if failure == "trainable_base":
                    model.embedding.weight.requires_grad_(True)
                parameters = trainable(model)
                if failure == "base_in_optimizer":
                    parameters.append(model.embedding.weight)
                if failure == "missing_adapter":
                    parameters = parameters[:-1]
                optimizer = RecordingSGD(parameters)
                if failure == "base_gradient":
                    model.embedding.weight.grad = torch.ones_like(model.embedding.weight)
                with self.assertRaises(ValueError):
                    train_accumulated_group(model, rows(), optimizer, "cpu")
                self.assertEqual(model.calls, 0)
                self.assertEqual(optimizer.step_calls, 0)
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_invalid_counts_mask_eos_and_empty_groups_never_start_forward(self):
        mutations = [lambda r: r[1].__setitem__("supervised_token_count", 4),
                     lambda r: r[1].__setitem__("eos_token_id", 3),
                     lambda r: r[0]["labels"].__setitem__(0, 1),
                     lambda r: r[0]["labels"].__setitem__(2, -100),
                     lambda r: r[0]["attention_mask"].__setitem__(0, 0),
                     lambda r: r[1].__setitem__("id", "short"),
                     lambda r: r.clear()]
        for index, mutation in enumerate(mutations):
            with self.subTest(case=index):
                items = rows()
                mutation(items)
                model = TinyLM()
                optimizer = RecordingSGD(trainable(model))
                with self.assertRaises(ValueError):
                    train_accumulated_group(model, items, optimizer, "cpu")
                self.assertEqual(model.calls, 0)
                self.assertEqual(optimizer.step_calls, 0)

    def test_nonfinite_adapter_is_rejected_before_forward(self):
        model = TinyLM()
        with torch.no_grad():
            model.projection.lora_B["default"].weight[0, 0] = float("nan")
        optimizer = RecordingSGD(trainable(model))
        with self.assertRaisesRegex(ValueError, "Nonfinite adapter parameter"):
            train_accumulated_group(model, rows(), optimizer, "cpu")
        self.assertEqual(model.calls, 0)
        self.assertEqual(optimizer.step_calls, 0)

    def test_zero_initialized_B_allows_first_A_gradient_to_be_zero(self):
        model = TinyLM()
        with torch.no_grad():
            model.projection.lora_B["default"].weight.zero_()
        optimizer = RecordingSGD(trainable(model))
        metrics = train_accumulated_group(model, rows(), optimizer, "cpu")
        self.assertEqual(metrics["A_nonzero_tensor_count"], 0)
        self.assertEqual(metrics["B_nonzero_tensor_count"], 1)
        self.assertEqual(optimizer.step_calls, 1)

    def test_batch_groups_preserve_order_and_reject_incomplete_tail(self):
        items = [{"id": str(index)} for index in range(16)]
        groups = batch_groups(items)
        self.assertEqual([len(group) for group in groups], [8, 8])
        self.assertEqual([row["id"] for group in groups for row in group], [str(index) for index in range(16)])
        for invalid_items, size in [(items[:-1], 8), ([], 8), (items, True), (items, 8.0), (items, 0)]:
            with self.subTest(size=size, length=len(invalid_items)), self.assertRaises(ValueError):
                batch_groups(invalid_items, size)


if __name__ == "__main__":
    unittest.main()
