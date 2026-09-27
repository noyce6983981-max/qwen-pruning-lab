import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM
from export_lora import tensor_hash, collect_probes, probe_difference
from merged_checkpoint import merged_model_provenance
from run_quality import sha256_file


class MergeTests(unittest.TestCase):
    def test_real_peft_bf16_arithmetic_non_mlp_and_reload(self):
        torch.manual_seed(12)
        config = Qwen3Config(vocab_size=40, hidden_size=16, intermediate_size=24,
                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                            head_dim=8, max_position_embeddings=64, attention_dropout=0)
        config._attn_implementation = "sdpa"
        previous_dtype = torch.get_default_dtype()
        try:
            # Match from_pretrained(dtype=BF16): parameters BF16, rotary buffer FP32.
            torch.set_default_dtype(torch.bfloat16)
            base = Qwen3ForCausalLM(config).eval()
        finally:
            torch.set_default_dtype(previous_dtype)
        before = {name: tensor_hash(value) for name, value in base.named_parameters()}
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=.05,
                              target_modules=["gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
        expected = dict(before)
        for name, module in model.get_base_model().named_modules():
            if hasattr(module, "lora_A"):
                module.lora_A.default.to(torch.float32)
                module.lora_B.default.to(torch.float32)
                with torch.no_grad():
                    module.lora_B.default.weight.normal_(std=.08)
                    # Independent explicit formula, not get_delta_weight itself.
                    delta = 2 * (module.lora_B.default.weight @ module.lora_A.default.weight)
                    weight = module.get_base_layer().weight.detach().clone()
                    weight += delta.to(torch.bfloat16)
                    expected[name + ".weight"] = tensor_hash(weight)
        probes = [[1, 2, 3], [2, 9, 5, 7]]
        pre = collect_probes(model, probes, torch.device("cpu"))
        merged = model.merge_and_unload(safe_merge=True)
        self.assertEqual(expected, {name: tensor_hash(value) for name, value in merged.named_parameters()})
        self.assertFalse(any("lora" in name for name, _ in merged.named_parameters()))
        post = collect_probes(merged, probes, torch.device("cpu"))
        difference = probe_difference(pre, post)
        self.assertEqual(difference["probe_count"], 2)
        self.assertGreaterEqual(difference["max_abs_logit_difference"], 0)
        with tempfile.TemporaryDirectory() as temporary:
            merged.save_pretrained(temporary, safe_serialization=True)
            reloaded = Qwen3ForCausalLM.from_pretrained(temporary, dtype=torch.bfloat16, attn_implementation="sdpa")
            self.assertEqual(merged.config._attn_implementation, reloaded.config._attn_implementation)
            self.assertEqual({n: tensor_hash(p) for n, p in merged.named_parameters()},
                             {n: tensor_hash(p) for n, p in reloaded.named_parameters()})
            again = collect_probes(reloaded, probes, torch.device("cpu"))
            self.assertTrue(torch.equal(post, again), str({"max_error": float((post-again).abs().max()),
                "buffers_before": [(n, str(b.dtype)) for n,b in merged.named_buffers()],
                "buffers_after": [(n, str(b.dtype)) for n,b in reloaded.named_buffers()]}))

    def make_manifest(self, directory):
        config = {"model_type": "qwen3", "num_hidden_layers": 28, "hidden_size": 2048, "intermediate_size": 6144}
        (directory / "config.json").write_text(json.dumps(config))
        for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "merge_probes.safetensors", "model.safetensors"):
            (directory / name).write_text("fixture")
        manifest = {"schema_version": 1, "status": "completed", "checkpoint_kind": "merged_lora",
                    "source_model_revision": "rev", "parameter_dtype": "torch.bfloat16", "training_step": 64,
                    "base_checkpoint_kind": "original", "intermediate_size": 6144,
                    "reload_validation": {"status": "passed", "probe_exact": True},
                    "weight_audit": {"all_expected_hashes_match": True},
                    "source_adapter": {"checkpoint_kind": "recovered_lora", "training_step": 64,
                        "source_model_revision": "rev", "base_checkpoint_kind": "original", "source_model_id": "Qwen/Qwen3-1.7B"},
                    "files_sha256": {p.name: sha256_file(p) for p in directory.iterdir()}}
        (directory / "merge_manifest.json").write_text(json.dumps(manifest))
        return manifest

    def test_standalone_manifest_files_and_ambiguous_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_manifest(directory)
            provenance, pruning = merged_model_provenance(directory, "rev")
            self.assertTrue(provenance["adapter_merged"])
            self.assertIsNone(pruning)
            (directory / "pruning_manifest.json").write_text("{}")
            with self.assertRaises(ValueError):
                merged_model_provenance(directory, "rev")

    def test_tampered_export_and_wrong_revision_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.make_manifest(directory)
            with self.assertRaises(ValueError):
                merged_model_provenance(directory, "other")
            (directory / "model.safetensors").write_text("changed")
            with self.assertRaises(ValueError):
                merged_model_provenance(directory, "rev")


if __name__ == "__main__":
    unittest.main()
