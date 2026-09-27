"""CPU-only LoRA trial checks with a tiny locally initialized Qwen3 model."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from peft import PeftModel
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from trial_lora import (SETTINGS, attach_adapter, batch_tensors, is_adapter,
                        parameter_fingerprints, probe_logits, train_step)


DEVICE = torch.device("cpu")


def tiny_base(dtype=torch.float32):
    torch.manual_seed(1741)
    config = Qwen3Config(
        vocab_size=53, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=32, attention_dropout=0.0,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).to(device=DEVICE, dtype=dtype).eval()


def answer_row():
    # Three prompt tokens, two answer tokens, then EOS. Labels are NOT shifted
    # here: the model predicts token 13 from position 2, 17 from 3, EOS from 4.
    return {"id": "tiny-answer", "input_ids": [1, 7, 9, 13, 17, 2],
            "labels": [-100, -100, -100, 13, 17, 2], "attention_mask": [1] * 6,
            "sequence_length": 6, "prompt_length": 3, "supervised_token_count": 3}


def optimizer_for(model):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=SETTINGS["learning_rate"], weight_decay=0.0)


class TrialLoRATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_exact_mlp_inventory_shapes_count_and_frozen_bf16_base(self):
        model, metadata = attach_adapter(tiny_base(dtype=torch.bfloat16))
        expected_modules = {
            f"base_model.model.model.layers.{i}.mlp.{projection}"
            for i in range(2) for projection in ("gate_proj", "up_proj", "down_proj")
        }
        actual_modules = {name for name, module in model.named_modules()
                          if hasattr(module, "lora_A") and hasattr(module, "lora_B")}
        self.assertEqual(actual_modules, expected_modules)
        rank = SETTINGS["rank"]
        for name, module in model.named_modules():
            if name not in actual_modules:
                continue
            inputs, outputs = (24, 16) if name.endswith("down_proj") else (16, 24)
            self.assertEqual(tuple(module.lora_A["default"].weight.shape), (rank, inputs))
            self.assertEqual(tuple(module.lora_B["default"].weight.shape), (outputs, rank))
            self.assertEqual(module.scaling["default"], SETTINGS["alpha"] / rank)
        self.assertEqual(metadata["target_module_count"], 6)
        self.assertEqual(metadata["trainable_tensor_count"], 12)
        self.assertEqual(metadata["trainable_parameter_count"], 2 * 3 * rank * (16 + 24))
        trainable_count = 0
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.device.type, "cpu")
            self.assertEqual(parameter.requires_grad, is_adapter(name))
            self.assertEqual(parameter.dtype, torch.float32 if is_adapter(name) else torch.bfloat16)
            if parameter.requires_grad:
                trainable_count += parameter.numel()
        self.assertEqual(trainable_count, metadata["trainable_parameter_count"])
        self.assertTrue(model.is_gradient_checkpointing)
        self.assertFalse(model.config.use_cache)

    def test_zero_b_initialization_preserves_probe_exactly(self):
        base = tiny_base()
        before = probe_logits(base, answer_row()["input_ids"], DEVICE)
        model, _ = attach_adapter(base)
        after = probe_logits(model, answer_row()["input_ids"], DEVICE)
        self.assertTrue(torch.equal(before, after))
        for name, parameter in model.named_parameters():
            if ".lora_B." in name:
                self.assertEqual(int(torch.count_nonzero(parameter)), 0)
            elif ".lora_A." in name:
                self.assertGreater(int(torch.count_nonzero(parameter)), 0)

    def test_first_a_gradient_zero_then_both_learn_without_base_changes(self):
        model, _ = attach_adapter(tiny_base())
        optimizer = optimizer_for(model)
        base_before = parameter_fingerprints(model)
        adapter_before = parameter_fingerprints(model, adapter=True)
        first = train_step(model, answer_row(), optimizer, DEVICE)
        after_first = parameter_fingerprints(model, adapter=True)
        self.assertTrue(math.isfinite(first["loss"]))
        self.assertGreater(first["gradient_norm_before_clip"], 0)
        self.assertEqual(first["A_nonzero_tensor_count"], 0)
        self.assertEqual(first["B_nonzero_tensor_count"], 6)
        for name, before in adapter_before.items():
            if ".lora_A." in name:
                self.assertEqual(before, after_first[name])
            else:
                self.assertNotEqual(before, after_first[name])
        second = train_step(model, answer_row(), optimizer, DEVICE)
        after_second = parameter_fingerprints(model, adapter=True)
        self.assertEqual(second["A_nonzero_tensor_count"], 6)
        self.assertEqual(second["B_nonzero_tensor_count"], 6)
        self.assertTrue(all(after_second[name] != after_first[name] for name in adapter_before))
        self.assertEqual(parameter_fingerprints(model), base_before)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_adapter_save_reload_preserves_hashes_and_probe_on_same_base(self):
        base = tiny_base()
        base_state = {name: tensor.detach().clone() for name, tensor in base.state_dict().items()}
        model, _ = attach_adapter(base)
        optimizer = optimizer_for(model)
        for _ in range(2):
            train_step(model, answer_row(), optimizer, DEVICE)
        expected_hashes = parameter_fingerprints(model, adapter=True)
        expected_base = parameter_fingerprints(model)
        expected_probe = probe_logits(model, answer_row()["input_ids"], DEVICE)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True, save_embedding_layers=False)
            reloaded_base = tiny_base()
            reloaded_base.load_state_dict(base_state, strict=True)
            reloaded = PeftModel.from_pretrained(
                reloaded_base, directory, is_trainable=False,
                autocast_adapter_dtype=True, local_files_only=True,
            )
            self.assertEqual(parameter_fingerprints(reloaded, adapter=True), expected_hashes)
            self.assertEqual(parameter_fingerprints(reloaded), expected_base)
            self.assertTrue(torch.equal(probe_logits(reloaded, answer_row()["input_ids"], DEVICE), expected_probe))

    def test_only_answer_and_eos_supervised_with_correct_causal_shift(self):
        model = tiny_base()
        batch = batch_tensors(answer_row(), DEVICE)
        self.assertTrue(torch.equal(batch["labels"][0, :3], torch.full((3,), -100)))
        self.assertEqual(batch["labels"][0, 3:].tolist(), [13, 17, 2])
        with torch.no_grad():
            result = model(**batch)
        # Choose the three predictive positions explicitly, including EOS.
        target_logits = result.logits[0, [2, 3, 4], :].float()
        targets = torch.tensor([13, 17, 2], dtype=torch.long)
        expected = -torch.log_softmax(target_logits, dim=-1)[torch.arange(3), targets].mean()
        torch.testing.assert_close(result.loss, expected, rtol=1e-6, atol=1e-7)
        all_tokens = -torch.log_softmax(result.logits[0, :5, :].float(), dim=-1)[
            torch.arange(5), torch.tensor([7, 9, 13, 17, 2])].mean()
        self.assertGreater(float((all_tokens - expected).abs()), 1e-6)

    def test_wrong_supervised_count_and_padding_rejected(self):
        row = answer_row()
        row["supervised_token_count"] = 2
        with self.assertRaisesRegex(ValueError, "Supervised token count"):
            batch_tensors(row, DEVICE)
        row = answer_row()
        row["attention_mask"][-1] = 0
        with self.assertRaisesRegex(ValueError, "unpadded complete"):
            batch_tensors(row, DEVICE)

    def test_unexpected_or_missing_target_module_rejected(self):
        extra = tiny_base()
        extra.extra_gate_proj = torch.nn.Linear(16, 24, bias=False)
        missing = tiny_base()
        del missing.model.layers[0].mlp.down_proj
        for base in (extra, missing):
            with self.subTest(case=type(base).__name__), self.assertRaisesRegex(ValueError, "module inventory"):
                attach_adapter(base)


if __name__ == "__main__":
    unittest.main()
