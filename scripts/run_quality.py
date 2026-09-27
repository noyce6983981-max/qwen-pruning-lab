"""Evaluate one local Qwen3 checkpoint on a frozen smoke/validation protocol.

No downloads, remote code, training, prompt truncation, or final-set evaluation.
Every generated answer is flushed to disk before the official scorer is called.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def within(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(directory.resolve()):
        raise ValueError(f"Manifest file escapes its directory: {relative}")
    return path


def verify_files(directory: Path, expected: dict, required=()) -> dict:
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Expected a nonempty files_sha256 mapping")
    for name in required:
        if name not in expected:
            raise ValueError(f"Missing frozen hash: {name}")
    actual = {}
    for name, digest in expected.items():
        path = within(directory, name)
        actual[name] = sha256_file(path)
        if actual[name] != digest:
            raise ValueError(f"SHA256 mismatch: {path}")
    return actual


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path.name}:{line_number}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError(f"Expected an object with nonempty string id at {path.name}:{line_number}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate id in {path.name}: {row['id']}")
        seen.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError(f"No examples in {path}")
    return rows


def token_ids_valid(ids, minimum: int = 1) -> bool:
    return (isinstance(ids, list) and len(ids) >= minimum
            and all(type(token) is int and token >= 0 for token in ids))


def validate_protocol(protocol: dict) -> None:
    qa = protocol["qa"]
    lm = protocol["lm"]
    if (qa.get("enable_thinking") is not False or qa.get("do_sample") is not False
            or qa.get("answer_extraction") != "strip_only"):
        raise ValueError("QA requires non-thinking greedy decoding and strip_only extraction")
    for key in ("max_input_tokens", "max_new_tokens"):
        if type(qa.get(key)) is not int or qa[key] < 1:
            raise ValueError(f"Invalid QA {key}")
    if qa["max_new_tokens"] > 64 or qa["max_input_tokens"] + qa["max_new_tokens"] > 512:
        raise ValueError("QA input plus output budget must be <= 512; output <= 64")
    if lm.get("window") != 512 or lm.get("stride") != 256:
        raise ValueError("This protocol requires LM window=512 and stride=256")


def validate_examples(examples: list[dict], documents: list[dict], protocol: dict) -> None:
    validate_protocol(protocol)
    for item in examples:
        for field in ("context", "question"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"{item['id']}: invalid {field}")
        answers = item.get("answers")
        if (not isinstance(answers, list) or not answers
                or not all(isinstance(answer, str) and answer.strip() for answer in answers)):
            raise ValueError(f"{item['id']}: expected nonempty string answers")
        ids = item.get("prompt_token_ids")
        if not token_ids_valid(ids):
            raise ValueError(f"{item['id']}: invalid prompt_token_ids")
        if len(ids) > protocol["qa"]["max_input_tokens"]:
            raise ValueError(f"{item['id']}: prompt exceeds frozen input budget; no truncation allowed")
        expected = hashlib.sha256(item["context"].encode("utf-8")).hexdigest()
        if item.get("context_sha256") != expected:
            raise ValueError(f"{item['id']}: context_sha256 mismatch")
    for item in documents:
        if not token_ids_valid(item.get("token_ids"), minimum=2):
            raise ValueError(f"{item['id']}: LM document requires at least two valid token IDs")


def load_frozen_data(directory: Path, split: str):
    if split not in ("smoke", "validation"):
        raise ValueError("Only smoke and validation are enabled; final evaluation remains locked")
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path)
    qa_file, lm_file = f"{split}_qa.jsonl", f"{split}_lm.jsonl"
    hashes = verify_files(directory, manifest["files_sha256"], required=(qa_file, lm_file))
    examples = read_jsonl(directory / qa_file)
    documents = read_jsonl(directory / lm_file)
    validate_examples(examples, documents, manifest["protocol"])
    return manifest, examples, documents, {
        "directory": str(directory.resolve()), "manifest_sha256": sha256_file(manifest_path),
        "files_sha256": hashes, "qa_file": qa_file, "lm_file": lm_file,
        "qa_count": len(examples), "lm_document_count": len(documents),
        "lm_input_token_count": sum(len(item["token_ids"]) for item in documents),
    }


TOKENIZER_FIXTURES = [
    "中国北京。2026年9月27日，3.14公里，否定词：不是。",
    " Qwen3 isn't a Mistral tokenizer.\n中文 English 12345\t0.5%",
    "ＡＢＣ，é café\u00a0空格🙂 <|im_start|>assistant\n",
]


def tokenizer_signature(tokenizer) -> dict:
    messages = [{"role": "system", "content": "只输出答案。"},
                {"role": "user", "content": "资料：北京是中国首都。\n问题：中国首都是什么？"}]
    return {
        "vocab_sha256": json_hash(tokenizer.get_vocab()),
        "special_tokens_map": tokenizer.special_tokens_map,
        "all_special_tokens": tokenizer.all_special_tokens,
        "all_special_ids": tokenizer.all_special_ids,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "chat_template": tokenizer.chat_template,
        "fixture_token_ids": [tokenizer.encode(text, add_special_tokens=False)
                              for text in TOKENIZER_FIXTURES],
        "chat_fixture_token_ids": tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False),
    }


def check_tokenizers(canonical, candidate) -> dict:
    reference = tokenizer_signature(canonical)
    other = tokenizer_signature(candidate)
    mismatches = [key for key in reference if reference[key] != other[key]]
    if mismatches:
        raise ValueError(f"Checkpoint tokenizer differs from frozen tokenizer: {mismatches}")
    return {"status": "passed", "signature_sha256": json_hash(reference),
            "fixtures": TOKENIZER_FIXTURES, "signature": reference,
            "fix_mistral_regex": "not enabled", "decode_tokenizer": "frozen canonical tokenizer"}


def model_provenance(directory: Path, revision: str) -> tuple[dict, dict | None]:
    pruning_path = directory / "pruning_manifest.json"
    if pruning_path.is_file():
        manifest = read_json(pruning_path)
        if manifest["source"]["revision"] != revision:
            raise ValueError("Requested revision differs from pruned checkpoint source")
        if (manifest.get("status") != "completed" or manifest.get("checkpoint_kind") != "pruned"
                or manifest.get("reload_validation", {}).get("status") != "passed"):
            raise ValueError("Pruned checkpoint did not complete save/reload validation")
        report = {"checkpoint_kind": "pruned", "manifest_sha256": sha256_file(pruning_path),
                  "manifest_path": str(pruning_path.resolve()), **{key: manifest[key] for key in (
                      "source", "method", "width_before", "width_after", "parameters_before",
                      "parameters_after", "recovery_applied", "reload_validation")}}
        listed_files = manifest["saved_files"]
    else:
        download_path = directory / "download_manifest.json"
        manifest = read_json(download_path)
        if manifest["revision"] != revision:
            raise ValueError("Requested revision differs from local download manifest")
        report = {"checkpoint_kind": "original", "manifest_sha256": sha256_file(download_path),
                  "manifest_path": str(download_path.resolve()), "download_manifest": manifest}
        listed_files = manifest["files"]
    for item in listed_files:
        path = within(directory, item["name"])
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"Checkpoint file size disagrees with manifest: {path.name}")
    paths = sorted(set(directory.glob("*.safetensors")) | {directory / "config.json",
                                                              directory / "generation_config.json"})
    if not any(path.suffix == ".safetensors" for path in paths):
        raise ValueError("Local checkpoint contains no safetensors weights")
    report["files_sha256"] = {path.name: sha256_file(path) for path in paths}
    report["hash_definition"] = "Hashes of actual local evaluation files; not independent upstream attestation."
    return report, manifest if pruning_path.is_file() else None


def inspect_model(model, pruning_manifest: dict | None, device: str) -> dict:
    parameters = list(model.parameters())
    config = model.config
    if config.model_type != "qwen3":
        raise ValueError("This runner currently supports Qwen3 checkpoints only")
    shapes = []
    for index, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        item = {"layer": index, **{name: list(getattr(mlp, name).weight.shape)
                                 for name in ("gate_proj", "up_proj", "down_proj")}}
        if (item["gate_proj"] != [config.intermediate_size, config.hidden_size]
                or item["up_proj"] != item["gate_proj"]
                or item["down_proj"] != [config.hidden_size, config.intermediate_size]):
            raise ValueError(f"MLP shape mismatch at layer {index}")
        shapes.append(item)
    report = {"class": type(model).__name__,
              "unique_parameter_count": sum(parameter.numel() for parameter in parameters),
              "unique_parameter_storage_bytes": sum(p.numel() * p.element_size() for p in parameters),
              "parameter_devices": sorted({str(parameter.device) for parameter in parameters}),
              "parameter_dtypes": sorted({str(parameter.dtype) for parameter in parameters}),
              "attention_implementation": getattr(config, "_attn_implementation", None),
              "hidden_size": config.hidden_size, "intermediate_size": config.intermediate_size,
              "num_hidden_layers": config.num_hidden_layers, "mlps": shapes}
    if report["parameter_devices"] != [device] or report["parameter_dtypes"] != ["torch.bfloat16"]:
        raise ValueError("All model parameters must be resident on the selected CUDA device in bfloat16")
    if len(shapes) != config.num_hidden_layers or report["attention_implementation"] != "sdpa":
        raise ValueError("Unexpected layer count or attention backend")
    if pruning_manifest is not None:
        if (report["unique_parameter_count"] != pruning_manifest["parameters_after"]
                or config.intermediate_size != pruning_manifest["width_after"]):
            raise ValueError("Pruned parameter count or MLP width disagrees with manifest")
    return report


def extract_answer(raw_text: str) -> str:
    # No regex answer repair, prefix removal, or guessing a span after generation.
    return raw_text.strip()


def stop_reason(new_ids: list[int], eos_ids, budget: int) -> str:
    eos_ids = [eos_ids] if type(eos_ids) is int else (eos_ids or [])
    if new_ids and new_ids[-1] in eos_ids:
        return "eos"
    if len(new_ids) == budget:
        return "max_new_tokens"
    return "other"


def run_qa(model, tokenizer, examples: list[dict], generation_config, device,
           predictions_path: Path) -> list[dict]:
    import torch

    results = []
    with predictions_path.open("x", encoding="utf-8", newline="\n") as stream:
        with torch.inference_mode():
            for index, item in enumerate(examples):
                input_ids = torch.tensor([item["prompt_token_ids"]], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                generated = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    generation_config=generation_config, use_model_defaults=False,
                    do_sample=False,
                )
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                new_ids = generated[0, input_ids.shape[1]:].cpu().tolist()
                raw_text = tokenizer.decode(new_ids, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
                row = {**item, "raw_text": raw_text,
                       "raw_text_with_special_tokens": tokenizer.decode(
                           new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                       "answer": extract_answer(raw_text), "generated_token_ids": new_ids,
                       "input_tokens": len(item["prompt_token_ids"]), "new_tokens": len(new_ids),
                       "stop_reason": stop_reason(new_ids, generation_config.eos_token_id,
                                                  generation_config.max_new_tokens),
                       "generation_seconds": elapsed}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                results.append(row)
                print(f"QA {index + 1}/{len(examples)} {item['id']}: {row['answer']}", flush=True)
                del input_ids, attention_mask, generated
    return results


def host_memory() -> dict:
    import psutil
    process = psutil.Process().memory_info()
    system = psutil.virtual_memory()
    return {"process_resident_bytes": process.rss,
            "process_peak_working_set_bytes": getattr(process, "peak_wset", None),
            "system_available_bytes": system.available, "system_total_bytes": system.total}


def package_versions() -> dict:
    versions = {}
    for name in ("torch", "transformers", "tokenizers", "accelerate", "safetensors", "nltk", "psutil"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def create_output(directory: Path) -> None:
    # Refuse even an empty existing directory, so no earlier partial run is overwritten.
    directory.mkdir(parents=True, exist_ok=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("smoke", "validation"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="Pinned original source revision")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    create_output(args.output_dir)
    report_path = args.output_dir / "quality.json"
    report = {"status": "starting", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "split": args.split, "source_model_revision": args.revision,
              "model_directory": str(args.model_dir.resolve()),
              "runner_sha256": sha256_file(Path(__file__)),
              "scope": "Frozen local CMRC2018 subset and Chinese language-model evaluation; not official hidden-test results.",
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "packages": package_versions()}}
    started = time.perf_counter()
    model = None
    torch = None
    device = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        import torch
        from cmrc_official import score_examples, scoring_provenance
        from eval_ppl import evaluate_ppl

        report["stage"] = "data_validation"
        manifest, examples, documents, data_report = load_frozen_data(args.data_dir, args.split)
        report["data"] = data_report
        report["data_manifest"] = manifest
        report["protocol"] = manifest["protocol"]
        report["helper_sha256"] = {name: sha256_file(Path(__file__).with_name(name))
                                   for name in ("cmrc_official.py", "eval_ppl.py")}
        report["scoring_provenance"] = scoring_provenance()
        canonical_dir = Path(manifest["tokenizer"]["directory"])
        report["canonical_tokenizer_files_sha256"] = verify_files(
            canonical_dir, manifest["tokenizer"]["files_sha256"],
            required=("tokenizer.json", "tokenizer_config.json"))
        canonical = AutoTokenizer.from_pretrained(canonical_dir, local_files_only=True, trust_remote_code=False)
        candidate = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
        report["tokenizer_compatibility"] = check_tokenizers(canonical, candidate)
        del candidate
        vocab_ids = set(canonical.get_vocab().values())
        for item in examples + documents:
            ids = item.get("prompt_token_ids", item.get("token_ids"))
            if any(token not in vocab_ids for token in ids):
                raise ValueError(f"{item['id']}: token ID outside frozen tokenizer vocabulary")
        report["stage"] = "checkpoint_validation"
        provenance, pruning_manifest = model_provenance(args.model_dir, args.revision)
        report["provenance"] = provenance
        canonical_generation = read_json(canonical_dir / "generation_config.json")
        generation_config = GenerationConfig(
            max_new_tokens=manifest["protocol"]["qa"]["max_new_tokens"],
            do_sample=False, use_cache=True, pad_token_id=canonical.pad_token_id,
            eos_token_id=canonical_generation["eos_token_id"],
            bos_token_id=canonical_generation.get("bos_token_id"),
        )
        report["generation_config"] = generation_config.to_dict()
        report["decode"] = {"use_model_defaults": False, "batch_size": 1,
                            "skip_special_tokens": True, "clean_up_tokenization_spaces": False,
                            "answer_extraction": "strip_only", "prompt_truncation": False,
                            "seed": 42}
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; this evaluation has no CPU fallback")
        torch.cuda.set_device(args.device)
        device = torch.device(f"cuda:{args.device}")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This frozen evaluation requires CUDA bfloat16 support")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        properties = torch.cuda.get_device_properties(device)
        report["environment"].update({
            "torch_cuda_build": torch.version.cuda, "cuda_device": str(device),
            "gpu_name": properties.name, "gpu_total_memory_bytes": properties.total_memory,
            "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
            "selected_dtype": "torch.bfloat16", "attention_implementation": "sdpa",
            "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
        })
        report["host_memory_before_load"] = host_memory()
        torch.cuda.reset_peak_memory_stats(device)
        report["stage"] = "model_loading"
        print(f"Loading one model on {device} in bfloat16 with sdpa", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", use_safetensors=True,
            low_cpu_mem_usage=True, device_map={"": str(device)},
            local_files_only=True, trust_remote_code=False,
        ).eval()
        report["loading"] = {"local_files_only": True, "trust_remote_code": False,
                             "use_safetensors": True, "low_cpu_mem_usage": True,
                             "device_map": {"": str(device)}}
        report["load_peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["load_peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["host_memory_after_load"] = host_memory()
        report["model"] = inspect_model(model, pruning_manifest, str(device))
        for key in ("eos_token_id", "bos_token_id"):
            if getattr(model.generation_config, key) != getattr(generation_config, key):
                raise ValueError(f"Checkpoint generation {key} differs from frozen original")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        report["stage"] = "qa_generation"
        report["qa"] = run_qa(model, canonical, examples, generation_config, device,
                              args.output_dir / "qa_predictions.jsonl")
        report["qa_peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["qa_peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["stage"] = "qa_scoring"
        report["qa_metrics"] = score_examples(examples, {row["id"]: row["answer"] for row in report["qa"]})
        report["qa_output_diagnostics"] = {
            "empty_answer_count": sum(not row["answer"] for row in report["qa"]),
            "max_new_tokens_stop_count": sum(row["stop_reason"] == "max_new_tokens" for row in report["qa"]),
        }
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        report["stage"] = "language_model_evaluation"
        print(f"LM: {len(documents)} independent documents", flush=True)
        report["lm"] = evaluate_ppl(model, documents, device, window=512, stride=256)
        report["lm_peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["lm_peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["status"] = "completed"
        report["stage"] = "completed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        partial = args.output_dir / "qa_predictions.jsonl"
        if partial.is_file():
            report["qa"] = read_jsonl(partial) if partial.stat().st_size else []
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = time.perf_counter() - started
        report["host_memory_at_exit"] = host_memory()
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
        print(f"Report: {report_path}", flush=True)
        if model is not None:
            del model
        if torch is not None and device is not None:
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
