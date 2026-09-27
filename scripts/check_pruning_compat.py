"""CPU-only structural-pruning check on a tiny RANDOM Qwen3, not trained weights.

The masked reference removes contributions of the same FFN channels. This checks
index coupling and serialization, not the quality of a pruning criterion.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_pruning as tp
from transformers import Qwen3Config, Qwen3ForCausalLM


def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=128,
        tie_word_embeddings=False, use_cache=False,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    original = Qwen3ForCausalLM(config).cpu().float().eval()
    masked = copy.deepcopy(original)
    pruned = copy.deepcopy(original)
    removed = list(range(0, 128, 4))
    kept = [index for index in range(128) if index not in removed]
    assert len(kept) == 96
    for layer in masked.model.layers:
        # Zeroing the down-projection columns removes the complete contribution
        # of each selected gated-MLP intermediate channel.
        with torch.no_grad():
            layer.mlp.down_proj.weight[:, removed] = 0
    for layer in pruned.model.layers:
        mlp = layer.mlp
        # Explicit low-level pruning: no whole-model dependency graph and no
        # changes to attention heads, Q/K normalization or residual width.
        tp.prune_linear_out_channels(mlp.gate_proj, idxs=removed)
        tp.prune_linear_out_channels(mlp.up_proj, idxs=removed)
        tp.prune_linear_in_channels(mlp.down_proj, idxs=removed)
        mlp.intermediate_size = len(kept)
        assert tuple(mlp.gate_proj.weight.shape) == (96, 64)
        assert tuple(mlp.up_proj.weight.shape) == (96, 64)
        assert tuple(mlp.down_proj.weight.shape) == (64, 96)
    pruned.config.intermediate_size = len(kept)
    input_ids = torch.randint(3, 256, (2, 12), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        masked_logits = masked(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        pruned_logits = pruned(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    torch.testing.assert_close(pruned_logits, masked_logits, rtol=1e-5, atol=1e-6)
    before = parameter_count(original)
    after = parameter_count(pruned)
    expected_removed_parameters = 2 * 3 * 64 * 32
    assert before - after == expected_removed_parameters
    with tempfile.TemporaryDirectory(prefix="tiny-qwen3-compat-") as temporary:
        pruned.save_pretrained(temporary, safe_serialization=True)
        reloaded = Qwen3ForCausalLM.from_pretrained(
            temporary, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=torch.float32,
            attn_implementation="eager",
        ).cpu().eval()
        assert reloaded.config.intermediate_size == 96
        assert parameter_count(reloaded) == after
        with torch.inference_mode():
            reloaded_logits = reloaded(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        torch.testing.assert_close(reloaded_logits, pruned_logits, rtol=1e-5, atol=1e-6)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "scope": "CPU-only tiny randomly initialized Qwen3. Proves only matched MLP channel removal and save/reload consistency for this configuration; does not prove pretrained Qwen3-1.7B quality, GPU fit or speedup.",
        "device": "cpu",
        "dtype": "float32",
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "torch-pruning", "safetensors")},
        "random_seed": 42,
        "config": {"layers": 2, "hidden_size": 64, "heads": 4, "kv_heads": 2, "head_dim": 16, "vocab_size": 256},
        "intermediate_size_before": 128,
        "intermediate_size_after": 96,
        "removed_channel_indices": removed,
        "kept_channel_indices": kept,
        "parameters_before": before,
        "parameters_after": after,
        "removed_parameters": before - after,
        "masked_vs_pruned_max_absolute_error": (masked_logits - pruned_logits).abs().max().item(),
        "pruned_vs_reloaded_max_absolute_error": (pruned_logits - reloaded_logits).abs().max().item(),
        "assert_close_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "checkpoint_retained": False,
    }
    report_path = args.output_dir / "pruning_compat.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: parameters {before} -> {after}; masked/pruned and save/reload outputs match.")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
