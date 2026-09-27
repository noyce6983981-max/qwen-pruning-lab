"""Run one local CUDA model: transparent demos and a fixed-length microbenchmark.

This is a first-run baseline, not a task-accuracy evaluation. No model or tokenizer
files are downloaded, and no remote model code is executed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


DEMOS = [
    {
        "id": "demo_1",
        "document": "以下为虚构的课程资料：机器人课程报告于周五下午五点截止，只需提交PDF文件。",
        "question": "报告什么时候截止，需要提交什么格式？",
        "expected_information": ["周五下午五点", "PDF"],
    },
    {
        "id": "demo_2",
        "document": "以下为虚构的设备说明：蓝色指示灯表示待机，绿色表示运行。清洁设备前，需要先断开电源。",
        "question": "清洁设备之前应该做什么？",
        "expected_information": ["断开电源"],
    },
    {
        "id": "demo_3",
        "document": "以下为虚构的图书室通知：图书室周一至周五上午九点开放，下午六点关闭。通知未提供周末安排。",
        "question": "图书室周日几点开放？",
        "expected_information": ["资料未提供周日开放时间，不应编造"],
    },
]


def host_memory() -> dict:
    info = psutil.Process().memory_info()
    system = psutil.virtual_memory()
    return {
        "process_resident_bytes": info.rss,
        "process_peak_working_set_bytes": getattr(info, "peak_wset", None),
        "system_total_bytes": system.total,
        "system_available_bytes": system.available,
        "definition": "OS process working set / RAM; peak is since process start on Windows. Excludes GPU memory and is not a minimum system RAM requirement.",
    }


def package_versions() -> dict[str, str | None]:
    result = {}
    for name in ("torch", "transformers", "accelerate", "safetensors", "torch-pruning"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def model_inventory(model) -> dict:
    mlps = []
    for index, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        mlps.append({
            "layer": index,
            "gate_proj_weight_shape": list(mlp.gate_proj.weight.shape),
            "up_proj_weight_shape": list(mlp.up_proj.weight.shape),
            "down_proj_weight_shape": list(mlp.down_proj.weight.shape),
        })
    parameters = list(model.parameters())
    return {
        "class": type(model).__name__,
        "unique_parameter_count": sum(p.numel() for p in parameters),
        "unique_parameter_storage_bytes": sum(p.numel() * p.element_size() for p in parameters),
        "parameter_devices": sorted({str(p.device) for p in parameters}),
        "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "head_dim": getattr(model.config, "head_dim", None),
        "mlps": mlps,
    }


@torch.inference_mode()
def run_demos(model, tokenizer, device, max_new_tokens: int) -> list[dict]:
    results = []
    for demo in DEMOS:
        messages = [
            {"role": "system", "content": "仅根据给出的资料，用一到两句中文回答问题。资料中没有答案时，明确说资料未提供。"},
            {"role": "user", "content": f"资料：{demo['document']}\n问题：{demo['question']}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_count = inputs["input_ids"].shape[1]
        if input_count + max_new_tokens > 512:
            raise ValueError(f"{demo['id']} exceeds the 512-token context budget")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        generated = model.generate(
            **inputs,
            use_model_defaults=False,
            do_sample=False,
            generation_config=GenerationConfig(
                max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                pad_token_id=pad_token_id,
                eos_token_id=model.generation_config.eos_token_id,
                bos_token_id=model.generation_config.bos_token_id,
            ),
        )
        new_ids = generated[0, input_count:].cpu().tolist()
        results.append({
            **demo,
            "answer": tokenizer.decode(new_ids, skip_special_tokens=True),
            "generated_token_ids": new_ids,
            "input_tokens": input_count,
            "new_tokens": len(new_ids),
            "max_new_tokens": max_new_tokens,
            "enable_thinking": False,
            "do_sample": False,
            "use_model_defaults": False,
            "automatically_scored": False,
        })
        print(f"{demo['id']}: {results[-1]['answer']}", flush=True)
        del inputs, generated
    return results


@torch.inference_mode()
def timed_greedy_run(model, input_ids, output_tokens: int, device) -> dict:
    """Generate an exact count, intentionally ignoring EOS for timing only."""
    attention_mask = torch.ones_like(input_ids)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    past = output.past_key_values
    next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    del output
    torch.cuda.synchronize(device)
    first_token_end = time.perf_counter()
    for _ in range(output_tokens - 1):
        attention_mask = torch.cat((attention_mask, attention_mask.new_ones((1, 1))), dim=1)
        output = model(
            input_ids=next_token, attention_mask=attention_mask,
            past_key_values=past, use_cache=True,
        )
        past = output.past_key_values
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        del output
    torch.cuda.synchronize(device)
    end = time.perf_counter()
    # Read peaks before transferring generated IDs to CPU. Peaks include model
    # weights and input tensors resident before this function was called.
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    decode_seconds = end - first_token_end
    return {
        "first_token_seconds": first_token_end - start,
        "decode_seconds": decode_seconds,
        "decode_token_count": output_tokens - 1,
        "decode_tokens_per_second": (output_tokens - 1) / decode_seconds,
        "total_seconds": end - start,
        "generated_token_count": output_tokens,
        "generated_token_ids": torch.cat(generated, dim=1)[0].cpu().tolist(),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }


def benchmark(model, tokenizer, device, input_tokens: int, output_tokens: int,
              warmups: int = 1, repeats: int = 3) -> dict:
    fixture_text = "这是一段用于固定长度性能测试的中文资料。请根据资料给出简短回答。"
    seed_ids = tokenizer.encode(fixture_text, add_special_tokens=False)
    if not seed_ids:
        raise ValueError("Tokenizer produced an empty benchmark fixture")
    fixture_ids = (seed_ids * ((input_tokens + len(seed_ids) - 1) // len(seed_ids)))[:input_tokens]
    input_ids = torch.tensor([fixture_ids], dtype=torch.long, device=device)
    print(f"Benchmark: {input_tokens} input / {output_tokens} output tokens; {warmups} warmups, {repeats} repeats", flush=True)
    for _ in range(warmups):
        timed_greedy_run(model, input_ids, output_tokens, device)
    runs = []
    for index in range(repeats):
        # Release idle allocator blocks before each measurement while keeping the
        # single model and benchmark input resident on the GPU.
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        start_allocated = torch.cuda.memory_allocated(device)
        start_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        run = timed_greedy_run(model, input_ids, output_tokens, device)
        run.update({
            "repeat": index + 1,
            "resident_allocated_before_run_bytes": start_allocated,
            "resident_reserved_before_run_bytes": start_reserved,
            "incremental_peak_allocated_bytes": max(0, run["peak_allocated_bytes"] - start_allocated),
        })
        runs.append(run)
        print(f"Repeat {index + 1}: first token {run['first_token_seconds']:.4f}s, decode {run['decode_tokens_per_second']:.2f} tokens/s", flush=True)
    return {
        "kind": "manual_greedy_kv_cache_microbenchmark_not_task_evaluation",
        "batch_size": 1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "warmups": warmups,
        "repeats": repeats,
        "eos_policy": "Ignore EOS and generate exactly output_tokens; used only for performance timing.",
        "first_token_definition": "Synchronized wall time for prefill forward plus greedy selection of the first new token; excludes tokenization, model loading and CPU transfer.",
        "decode_definition": "The remaining output_tokens - 1 tokens, using KV cache; includes Python loop, CUDA launches, growing attention masks and final synchronization.",
        "memory_definition": "PyTorch CUDA allocator peaks reset after warmup and empty_cache before each repeat. Absolute peaks include resident model weights and inputs; incremental allocated subtracts memory_allocated immediately before the run. Reserved is allocator reservation, not tensor use. Neither is whole-device memory shown by nvidia-smi.",
        "fixture_text": fixture_text,
        "fixture_token_ids": fixture_ids,
        "fixture_sha256": hashlib.sha256(json.dumps(fixture_ids).encode()).hexdigest(),
        "runs": runs,
        "median_first_token_seconds": statistics.median(r["first_token_seconds"] for r in runs),
        "median_decode_tokens_per_second": statistics.median(r["decode_tokens_per_second"] for r in runs),
        "max_peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in runs),
        "max_peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="Pinned source-model revision for provenance; loading is local only.")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--bench-input-tokens", type=int, default=128)
    parser.add_argument("--bench-output-tokens", type=int, default=32)
    parser.add_argument("--demo-max-new-tokens", type=int, default=64)
    parser.add_argument("--bench-warmups", type=int, default=1)
    parser.add_argument("--bench-repeats", type=int, default=3)
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        parser.error("--model-dir must be an existing local snapshot directory")
    if args.bench_input_tokens < 1 or args.bench_output_tokens < 2 or args.bench_input_tokens + args.bench_output_tokens > 512:
        parser.error("Benchmark requires input >= 1, output >= 2, and total <= 512 tokens")
    if not 1 <= args.demo_max_new_tokens <= 64:
        parser.error("--demo-max-new-tokens must be between 1 and 64")
    if args.bench_warmups < 1 or not 1 <= args.bench_repeats <= 50:
        parser.error("Require at least one warmup and 1-50 benchmark repeats")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. No CPU fallback is used for this baseline.")
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    bf16_available = torch.cuda.is_bf16_supported()
    if args.dtype == "bfloat16" and not bf16_available:
        raise RuntimeError("Requested bfloat16 is not supported on the selected CUDA device")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" or (args.dtype == "auto" and bf16_available) else torch.float16
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    properties = torch.cuda.get_device_properties(device)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_model_revision": args.revision,
        "model_directory": str(args.model_dir.resolve()),
        "status": "starting",
        "scope": "Evaluation only: three manually authored fictional Chinese document-QA demos and a performance microbenchmark. This script does not modify the checkpoint, train, or report formal accuracy. Check checkpoint_kind for original vs pruned provenance.",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
            "torch_cuda_build": torch.version.cuda,
            "cuda_device": str(device),
            "gpu_name": properties.name,
            "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
            "gpu_total_memory_bytes": properties.total_memory,
            "bf16_supported": bf16_available,
            "selected_dtype": str(dtype),
            "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
        },
        "loading": {"local_files_only": True, "trust_remote_code": False, "use_safetensors": True, "low_cpu_mem_usage": True, "device_map": {"": str(device)}},
        "host_memory_before_load": host_memory(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "baseline.json"
    try:
        pruning_path = args.model_dir / "pruning_manifest.json"
        pruning_manifest = None
        if pruning_path.is_file():
            pruning_manifest = json.loads(pruning_path.read_text(encoding="utf-8"))
            if pruning_manifest["source"]["revision"] != args.revision:
                raise ValueError("Requested revision differs from the pruned checkpoint's source")
            if pruning_manifest["status"] != "completed" or pruning_manifest["checkpoint_kind"] != "pruned":
                raise ValueError("Pruned checkpoint has not completed its save/reload validation")
            report["checkpoint_kind"] = "pruned"
            report["pruning"] = {
                "manifest_path": str(pruning_path.resolve()),
                "manifest_sha256": hashlib.sha256(pruning_path.read_bytes()).hexdigest(),
                **{key: pruning_manifest[key] for key in (
                    "source", "method", "width_before", "width_after",
                    "parameters_before", "parameters_after", "recovery_applied",
                )},
            }
        else:
            manifest_path = args.model_dir / "download_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["revision"] != args.revision:
                raise ValueError("Requested revision differs from the local download manifest")
            report["checkpoint_kind"] = "original"
            report["download_manifest"] = manifest
        print(f"Loading one model directly on {device} using {dtype}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
        torch.cuda.reset_peak_memory_stats(device)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=dtype, use_safetensors=True,
            low_cpu_mem_usage=True, device_map={"": str(device)},
            local_files_only=True, trust_remote_code=False,
        ).eval()
        report["load_peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["host_memory_after_load"] = host_memory()
        inventory = model_inventory(model)
        if inventory["parameter_devices"] != [str(device)]:
            raise RuntimeError(f"Unexpected parameter placement: {inventory['parameter_devices']}")
        if pruning_manifest is not None:
            if inventory["unique_parameter_count"] != pruning_manifest["parameters_after"]:
                raise ValueError("Pruned parameter count disagrees with the manifest")
            if inventory["intermediate_size"] != pruning_manifest["width_after"]:
                raise ValueError("Pruned MLP width disagrees with the manifest")
        report["model"] = inventory
        report["demos"] = run_demos(model, tokenizer, device, args.demo_max_new_tokens)
        report["benchmark"] = benchmark(model, tokenizer, device, args.bench_input_tokens,
                                        args.bench_output_tokens, args.bench_warmups, args.bench_repeats)
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["host_memory_at_exit"] = host_memory()
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
