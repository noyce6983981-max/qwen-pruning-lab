"""CPU tiny-checkpoint tests of recovery provenance and unmerged evaluation."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_recovery_quality as recovery
import run_quality as quality


REVISION = "tiny-pinned-revision"
DEVICE = torch.device("cpu")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def tiny_base():
    torch.manual_seed(2761)
    config = Qwen3Config(vocab_size=53, hidden_size=16, intermediate_size=24,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=8, max_position_embeddings=64, attention_dropout=0.0,
                        pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=True)
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config).to(device=DEVICE, dtype=torch.bfloat16).eval()


class RecoveryQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base_dir = self.root / "base"
        self.run_dir = self.root / "formal-run"
        self.adapter_dir = self.run_dir / "step-0016"
        self.adapter_dir.mkdir(parents=True)
        base = tiny_base()
        self.base_report = quality.inspect_model(base, None, "cpu")
        base.save_pretrained(self.base_dir, safe_serialization=True)
        write_json(self.base_dir / "download_manifest.json", {
            "model_id": "unit-test/tiny-qwen3", "revision": REVISION,
            "files": [{"name": p.name, "bytes": p.stat().st_size} for p in self.base_dir.iterdir() if p.is_file()],
        })
        self.base_provenance, _ = quality.model_provenance(self.base_dir, REVISION)
        self.config = quality.read_json(self.base_dir / "config.json")
        adapter_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                                    lora_dropout=0.05, target_modules=list(recovery.TARGETS), bias="none")
        self.model = get_peft_model(base, adapter_config, autocast_adapter_dtype=True)
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if ".lora_B." in name:
                    parameter.fill_(0.03125)
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.save_pretrained(self.adapter_dir, safe_serialization=True, save_embedding_layers=False)
        self.probe_ids = [1, 7, 9, 11, 12, 13]
        with torch.inference_mode():
            ids = torch.tensor([self.probe_ids])
            logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                logits_to_keep=1).logits.float().cpu().clone()
        save_file({"logits": logits}, self.adapter_dir / "probe.safetensors")
        self.data = {"manifest_sha256": "1" * 64, "files_sha256": {
            "recovery.jsonl": "2" * 64, "groups.json": "3" * 64}}
        self.manifest = {
            "schema_version": 1, "status": "completed", "checkpoint_kind": "recovered_lora",
            "training_step": 16, "source_model_revision": REVISION,
            "base_provenance": copy.deepcopy(self.base_provenance),
            "settings": {"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": list(recovery.TARGETS)},
            "data": copy.deepcopy(self.data), "training_policy_sha256": "4" * 64,
            "training_source_sha256": {"train_recovery.py": "5" * 64},
            "formal_training_run": str(self.run_dir.resolve()),
            "samples_seen": 128, "supervised_tokens_seen": 16 * 104, "checkpoint_rng_preserved": True,
            "probe": {"file": "probe.safetensors", "input_ids": self.probe_ids},
            "adapter_files_sha256": {name: quality.sha256_file(self.adapter_dir / name) for name in
                                     ("adapter_config.json", "adapter_model.safetensors", "probe.safetensors")},
        }
        self.run = {
            "status": "completed", "mode": "train", "base_parameters_unchanged": True,
            "sample_count": 512, "supervised_token_count": 6762,
            **{key: copy.deepcopy(self.manifest[key]) for key in
               ("base_provenance", "settings", "data", "training_policy_sha256", "source_model_revision")},
            "source_sha256": copy.deepcopy(self.manifest["training_source_sha256"]),
            "steps": [{"step": i + 1, "sample_ids": [f"row-{j}" for j in range(i * 8, (i + 1) * 8)],
                       "supervised_token_count": 104 if i < 63 else 210} for i in range(64)],
            "checkpoints": [{"step": step, "directory": str(self.run_dir / f"step-{step:04d}"),
                             "manifest_sha256": "9" * 64, "samples_seen": step * 8,
                             "supervised_tokens_seen": step * 104 if step < 64 else 6762,
                             "rng_preserved": True} for step in (16, 32, 64)],
        }
        self.write_manifest_and_bind()

    def write_manifest_and_bind(self):
        path = self.adapter_dir / "checkpoint_manifest.json"
        write_json(path, self.manifest)
        self.run["checkpoints"][0]["manifest_sha256"] = quality.sha256_file(path)
        write_json(self.run_dir / "run.json", self.run)

    def validate(self, base=None, data=None):
        return recovery.validate_adapter(self.adapter_dir, base or self.base_provenance, REVISION,
                                          self.config, data or self.data)

    def test_actual_peft_checkpoint_bound_to_complete_formal_run_and_base(self):
        manifest, provenance = self.validate()
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(provenance["checkpoint_kind"], "recovered_lora")
        self.assertEqual(provenance["base_checkpoint_kind"], "original")
        self.assertTrue(provenance["recovery_applied"])
        self.assertEqual(provenance["training_step"], 16)
        self.assertEqual(provenance["formal_training_run"]["sha256"], quality.sha256_file(self.run_dir / "run.json"))
        self.assertEqual(provenance["adapter_files_sha256"], self.manifest["adapter_files_sha256"])

    def test_wrong_base_and_adapter_file_tampering_rejected(self):
        wrong_base = copy.deepcopy(self.base_provenance)
        wrong_base["files_sha256"]["model.safetensors"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "different base files"):
            self.validate(base=wrong_base)
        with (self.adapter_dir / "adapter_model.safetensors").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.validate()

    def test_manifest_relabeling_or_unbound_manifest_rejected(self):
        self.manifest["training_step"] = 32
        write_json(self.adapter_dir / "checkpoint_manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "manifest hash/path"):
            self.validate()
        self.manifest["training_step"] = 16
        self.manifest["checkpoint_kind"] = "pruned"
        self.write_manifest_and_bind()
        with self.assertRaisesRegex(ValueError, "completed recovered_lora"):
            self.validate()

    def test_failed_partial_or_memory_run_cannot_supply_formal_candidate(self):
        initial = copy.deepcopy(self.run)
        for key, value in (("status", "failed"), ("mode", "memory"), ("base_parameters_unchanged", False),
                           ("supervised_token_count", 6761), ("steps", initial["steps"][:16])):
            run = copy.deepcopy(initial)
            run[key] = value
            write_json(self.run_dir / "run.json", run)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "completed formal run"):
                self.validate()

    def test_data_policy_and_checkpoint_hash_chain_rejected_when_changed(self):
        wrong_data = copy.deepcopy(self.data)
        wrong_data["files_sha256"]["groups.json"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "data provenance"):
            self.validate(data=wrong_data)
        self.run["training_policy_sha256"] = "0" * 64
        write_json(self.run_dir / "run.json", self.run)
        with self.assertRaisesRegex(ValueError, "training_policy_sha256"):
            self.validate()

    def test_rehashed_invalid_adapter_tensor_dtype_rejected(self):
        path = self.adapter_dir / "adapter_model.safetensors"
        # Clone to release Windows' mapped source file before replacing it.
        tensors = {name: tensor.clone() for name, tensor in load_file(str(path)).items()}
        first = next(iter(tensors))
        tensors[first] = tensors[first].to(torch.bfloat16)
        save_file(tensors, path)
        self.manifest["adapter_files_sha256"][path.name] = quality.sha256_file(path)
        self.write_manifest_and_bind()
        with self.assertRaisesRegex(ValueError, "Invalid adapter shape, dtype"):
            self.validate()

    def test_rehashed_nonstandard_adapter_config_cannot_change_base_or_structure(self):
        path = self.adapter_dir / "adapter_config.json"
        original = quality.read_json(path)
        for key, value in (("init_lora_weights", "olora"), ("init_lora_weights", "pissa"),
                           ("layer_replication", [[0, 1]]), ("trainable_token_indices", [1]),
                           ("loftq_config", {"loftq_bits": 4})):
            config = copy.deepcopy(original)
            config[key] = value
            write_json(path, config)
            self.manifest["adapter_files_sha256"][path.name] = quality.sha256_file(path)
            self.write_manifest_and_bind()
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "configuration"):
                self.validate()

    def test_real_total_storage_and_loaded_probe_are_verified(self):
        manifest, _ = self.validate()
        base = tiny_base()
        base.load_state_dict(load_file(str(self.base_dir / "model.safetensors")), strict=False)
        loaded = PeftModel.from_pretrained(base, self.adapter_dir, is_trainable=False,
                                           autocast_adapter_dtype=True, local_files_only=True)
        loaded.requires_grad_(False)
        loaded.eval()
        report = recovery.inspect_evaluation_model(loaded, self.base_report, manifest, DEVICE)
        self.assertEqual(report["adapter_module_count"], 6)
        self.assertEqual(report["adapter_tensor_count"], 12)
        self.assertEqual(report["adapter_parameter_count"], 2 * 3 * 8 * (16 + 24))
        self.assertEqual(report["adapter_parameter_storage_bytes"], report["adapter_parameter_count"] * 4)
        self.assertEqual(report["base_parameter_storage_bytes"], report["base_parameter_count"] * 2)
        self.assertEqual(report["unique_parameter_count"], sum(p.numel() for p in loaded.parameters()))
        self.assertEqual(report["unique_parameter_storage_bytes"], sum(p.numel() * p.element_size() for p in loaded.parameters()))
        self.assertEqual(recovery.check_saved_probe(loaded, self.adapter_dir, manifest, DEVICE)["probe_max_abs_error"], 0)
        probe_path = self.adapter_dir / "probe.safetensors"
        logits = load_file(str(probe_path))["logits"] + 0.25
        save_file({"logits": logits}, probe_path)
        with self.assertRaisesRegex(ValueError, "reload probe differs"):
            recovery.check_saved_probe(loaded, self.adapter_dir, manifest, DEVICE)

    def test_unadapted_original_remains_original_with_actual_base_count(self):
        provenance = recovery.baseline_provenance(self.base_provenance)
        self.assertEqual(provenance["checkpoint_kind"], "original")
        self.assertEqual(provenance["training_step"], 0)
        self.assertFalse(provenance["adapter_present"])
        self.assertFalse(provenance["recovery_applied"])
        base = tiny_base().requires_grad_(False).eval()
        report = recovery.inspect_evaluation_model(base, self.base_report, None, DEVICE)
        self.assertEqual(report["unique_parameter_count"], self.base_report["unique_parameter_count"])
        self.assertEqual(report["adapter_parameter_count"], 0)
        self.assertEqual(report["parameter_dtypes"], ["torch.bfloat16"])

    def test_manifest_parent_escape_and_duplicate_manifests_rejected(self):
        self.manifest["formal_training_run"] = str(self.root)
        self.write_manifest_and_bind()
        with self.assertRaisesRegex(ValueError, "actual parent"):
            self.validate()
        self.manifest["formal_training_run"] = str(self.run_dir)
        self.write_manifest_and_bind()
        write_json(self.adapter_dir / "recovery_manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
