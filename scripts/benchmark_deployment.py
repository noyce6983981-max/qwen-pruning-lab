"""Controlled two-session deployment timing; no training or quality scoring.

The run subcommand loads exactly one local BF16 model. The summarize subcommand
is CPU-only and requires all four model identities in forward/reverse order.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time
import uuid


MODEL_ORDER = (
    "original_step0", "calibrated_5888_step0",
    "original_step64_merged", "calibrated_5888_step64_merged",
)
FIXTURE_TEXT = "这是一段用于固定长度性能测试的中文资料。请根据资料给出简短回答。"
SETTINGS = {
    "input_lengths": [128, 512], "output_tokens": 32, "warmups": 2, "repeats": 7,
    "batch_size": 1, "dtype": "torch.bfloat16", "attention_implementation": "sdpa",
    "autocast": False, "use_cache": True, "do_sample": False,
    "seed": 42, "tf32_matmul": False, "tf32_cudnn": False,
    "eos_policy": "ignore EOS; exactly 32 output tokens for timing only",
    "length_order": [128, 512],
    "allocator_policy": "gc + empty_cache + synchronize + reset peaks before each measured repeat",
    "first_token_definition": "Synchronized prefill plus first greedy selection; excludes tokenizer, loading and CPU transfer.",
    "decode_definition": "Next 31 tokens with KV cache; Python loop, launches, growing masks and final synchronization included.",
    "precision_scope": "All four checkpoints are pure BF16 without live LoRA modules; no autocast.",
    "telemetry_policy": "nvidia-smi and Windows power before/after each length, outside timed sections; not continuous sampling",
}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
SOURCE_FILES = ("benchmark_deployment.py", "run_baseline.py", "run_quality.py", "merged_checkpoint.py")
SMI_FIELDS = (
    "index", "uuid", "name", "driver_version", "temperature.gpu", "power.draw",
    "clocks.current.sm", "clocks.current.memory", "memory.used", "memory.total",
    "utilization.gpu", "pstate",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf8")).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf8"))


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                         + "\n", encoding="utf8")
    temporary.replace(path)


def expected_order(session: int) -> tuple[str, ...]:
    if type(session) is not int or session not in (1, 2):
        raise ValueError("session must be 1 or 2")
    return MODEL_ORDER if session == 1 else MODEL_ORDER[::-1]


def fixture_ids(seed_ids: list[int], count: int) -> list[int]:
    if (not seed_ids or not all(type(token) is int and token >= 0 for token in seed_ids)
            or type(count) is not int or count not in SETTINGS["input_lengths"]):
        raise ValueError("Invalid benchmark seed IDs or input length")
    return (seed_ids * ((count + len(seed_ids) - 1) // len(seed_ids)))[:count]


def fixture_hash(ids: list[int]) -> str:
    # Identical serialization to the frozen run_baseline.benchmark.
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def finite_positive(value, label: str):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def nonnegative_int(value, label: str):
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def distribution(values) -> dict:
    values = list(values)
    if not values:
        raise ValueError("Empty distribution")
    for value in values:
        finite_positive(value, "measurement")
    return {"count": len(values), "median": statistics.median(values),
            "minimum": min(values), "maximum": max(values)}


def _command(args: list[str]) -> dict:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15,
                                errors="replace", check=False,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        return {"available": result.returncode == 0, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def nvidia_snapshot() -> dict:
    result = _command(["nvidia-smi", "--query-gpu=" + ",".join(SMI_FIELDS),
                       "--format=csv,noheader,nounits"])
    result["requested_fields"] = list(SMI_FIELDS)
    result["units"] = {"temperature.gpu": "C", "power.draw": "W",
                       "clocks.current.sm": "MHz", "clocks.current.memory": "MHz",
                       "memory.used": "MiB", "memory.total": "MiB", "utilization.gpu": "%"}
    if result["available"]:
        rows = list(csv.reader(result["stdout"].splitlines()))
        if not rows or any(len(row) != len(SMI_FIELDS) for row in rows):
            result.update(available=False, parse_error="unexpected nvidia-smi columns")
        else:
            result["gpus"] = [dict(zip(SMI_FIELDS, (cell.strip() for cell in row))) for row in rows]
    return result


def windows_power_snapshot() -> dict:
    if platform.system() != "Windows":
        return {"available": False, "reason": "not Windows"}
    result = {"available": True}
    scheme = _command(["powercfg", "/getactivescheme"])
    if scheme["available"]:
        match = re.search(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
                          scheme["stdout"])
        scheme["scheme_guid"] = match.group(0).lower() if match else None
    result["active_scheme"] = scheme
    try:
        import ctypes
        from ctypes import wintypes

        class PowerStatus(ctypes.Structure):
            _fields_ = [("ACLineStatus", wintypes.BYTE), ("BatteryFlag", wintypes.BYTE),
                        ("BatteryLifePercent", wintypes.BYTE), ("SystemStatusFlag", wintypes.BYTE),
                        ("BatteryLifeTime", wintypes.DWORD), ("BatteryFullLifeTime", wintypes.DWORD)]

        status = PowerStatus()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            raise OSError("GetSystemPowerStatus returned zero")
        result["system_power"] = {name: int(getattr(status, name)) for name, _ in status._fields_}
        result["system_power_available"] = True
    except (OSError, AttributeError) as exc:
        result.update(system_power_available=False, error=str(exc))
    result["settings_modified"] = False
    return result


def telemetry() -> dict:
    return {"at_utc": datetime.now(timezone.utc).isoformat(),
            "nvidia_smi": nvidia_snapshot(), "windows_power": windows_power_snapshot()}


def hardware_identity(snapshot: dict) -> dict:
    smi = snapshot["nvidia_smi"]
    return {"available": smi["available"], "gpus": [
        {key: gpu[key] for key in ("index", "uuid", "name", "driver_version", "memory.total")}
        for gpu in smi.get("gpus", [])]}


def power_identity(snapshot: dict) -> dict:
    power = snapshot["windows_power"]
    status = power.get("system_power", {})
    return {"available": power.get("available"),
            "scheme_guid": power.get("active_scheme", {}).get("scheme_guid"),
            "ACLineStatus": status.get("ACLineStatus"),
            "SystemStatusFlag": status.get("SystemStatusFlag")}


def checkpoint_provenance(directory: Path, revision: str, model_id: str) -> tuple[dict, dict | None]:
    if model_id.endswith("_merged"):
        from merged_checkpoint import merged_model_provenance
        provenance, pruning = merged_model_provenance(directory, revision)
        if provenance.get("checkpoint_kind") != "merged_lora":
            raise ValueError("Merged identity requires a validated merged_lora checkpoint")
    else:
        from run_quality import model_provenance
        provenance, pruning = model_provenance(directory, revision)
        expected = "pruned" if model_id.startswith("calibrated") else "original"
        if provenance.get("checkpoint_kind") != expected:
            raise ValueError("Requested model identity and checkpoint kind disagree")
        if pruning is not None:
            for item in pruning["saved_files"]:
                if item.get("sha256") is not None:
                    path = (directory / item["name"]).resolve()
                    if not path.is_relative_to(directory.resolve()) or sha256_file(path) != item["sha256"]:
                        raise ValueError("Pruned checkpoint frozen file SHA256 mismatch")
    return provenance, pruning


def is_model_weight(name: str) -> bool:
    return name == "model.safetensors" or re.fullmatch(r"model-\d+-of-\d+\.safetensors", name) is not None


def weight_files(directory: Path, hashes: dict) -> dict:
    weights = {name: (directory / name).stat().st_size
               for name in hashes if is_model_weight(name)}
    if not weights:
        raise ValueError("No full-model safetensors weights")
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        referenced = set(read_json(index)["weight_map"].values())
        if referenced != set(weights):
            raise ValueError("Safetensors index and hashed weight files differ")
    actual = {path.name for path in directory.glob("*.safetensors")}
    if actual - {"merge_probes.safetensors"} != set(weights):
        raise ValueError("Full-model weight inventory mismatch or unmerged adapter files")
    # Count each file once. Source snapshots can serialize tied weights twice,
    # whereas save_pretrained may omit the alias; disk delta is not pruning alone.
    return {"files_bytes": weights, "total_bytes": sum(weights.values()),
            "definition": "Actual full-model safetensors file bytes; each indexed shard once, includes headers and any serialized alias copies; excludes config/tokenizer/adapter/source artifacts. Serialization of tied weights can differ from unique in-memory parameter bytes."}


def loader_file_hashes(directory: Path, provenance: dict) -> dict:
    # The legacy quality provenance omits the shard index. Bind it separately
    # without changing the frozen base-provenance identity used by recovery.
    names = {"config.json", "generation_config.json"} | {
        name for name in provenance["files_sha256"] if is_model_weight(name)}
    if (directory / "model.safetensors.index.json").is_file():
        names.add("model.safetensors.index.json")
    return {name: (provenance["files_sha256"][name] if name in provenance["files_sha256"]
                   else sha256_file(directory / name)) for name in sorted(names)}


def validate_model_label(report: dict):
    label = report["model_id"]
    p = report["provenance"]
    expected_base = "pruned" if label.startswith("calibrated") else "original"
    merged = label.endswith("_merged")
    revision = (p.get("source_model_revision") if merged else
                p.get("source", {}).get("revision") if expected_base == "pruned" else
                p.get("download_manifest", {}).get("revision"))
    if revision != report["source_model_revision"]:
        raise ValueError("Benchmark revision and checkpoint provenance disagree")
    if merged:
        if (p.get("checkpoint_kind") != "merged_lora"
                or p.get("base_checkpoint_kind") != expected_base
                or type(p.get("training_step")) is not int or p["training_step"] != 64
                or p.get("merge_manifest", {}).get("reload_validation", {}).get("status") != "passed"
                or p["merge_manifest"]["reload_validation"].get("probe_exact") is not True
                or p["merge_manifest"].get("parameter_count") != report["model"]["unique_parameter_count"]):
            raise ValueError("Merged report is not the named base / selected step64")
    elif p.get("checkpoint_kind") != expected_base or p.get("recovery_applied", False):
        raise ValueError("Unrecovered identity mismatch")
    if not merged and expected_base == "pruned" and p.get("method") != "calibrated_rms_mean_abs":
        raise ValueError("Named calibrated model requires calibrated_rms_mean_abs provenance")
    model = report["model"]
    if (model["parameter_dtypes"] != ["torch.bfloat16"]
            or model["attention_implementation"] != "sdpa"
            or model["live_lora_parameter_count"] != 0
            or model["intermediate_size"] != (5888 if expected_base == "pruned" else 6144)):
        raise ValueError("Unexpected model precision, live adapter, backend or width")


def run(args):
    from run_quality import check_tokenizers, inspect_model
    from run_baseline import timed_greedy_run, host_memory
    import torch
    import psutil
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if expected_order(args.session)[args.position - 1] != args.model_id:
        raise ValueError("Position does not follow the fixed forward/reverse session order")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device with BF16 support is required")
    torch.cuda.set_device(args.device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected CUDA device does not support BF16")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir / "benchmark.json"
    started = datetime.now(timezone.utc)
    report = {
        "schema_version": 1, "status": "starting", "model_id": args.model_id,
        "session": args.session, "position": args.position,
        "session_order": list(expected_order(args.session)),
        "process_instance_id": str(uuid.uuid4()), "process_id": os.getpid(),
        "process_create_time": psutil.Process().create_time(),
        "started_at_utc": started.isoformat(), "source_model_revision": args.revision,
        "model_directory": str(args.model_dir.resolve()), "settings": SETTINGS.copy(),
        "source_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in SOURCE_FILES},
        "scope": "Performance only; fixed synthetic repeated-token fixture; no quality metric or EOS-respecting task output.",
        "benchmarks": [], "host_memory_before_load": host_memory(),
    }
    write_json(path, report)
    try:
        provenance, pruning = checkpoint_provenance(args.model_dir, args.revision, args.model_id)
        report["provenance"] = provenance
        report["weights"] = weight_files(args.model_dir, provenance["files_sha256"])
        report["model_loader_files_sha256"] = loader_file_hashes(args.model_dir, provenance)
        canonical = AutoTokenizer.from_pretrained(args.canonical_tokenizer_dir, local_files_only=True,
                                                 trust_remote_code=False)
        candidate = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
        report["tokenizer_compatibility"] = check_tokenizers(canonical, candidate)
        report["canonical_tokenizer_files_sha256"] = {
            name: sha256_file(args.canonical_tokenizer_dir / name) for name in TOKENIZER_FILES}
        seed_ids = canonical.encode(FIXTURE_TEXT, add_special_tokens=False)
        del candidate
        torch.cuda.set_device(args.device)
        device = torch.device(f"cuda:{args.device}")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        props = torch.cuda.get_device_properties(device)
        report["telemetry_before_load"] = telemetry()
        report["environment"] = {
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "accelerate", "safetensors", "psutil")},
            "torch_cuda_build": torch.version.cuda, "cuda_device": str(device),
            "gpu_name": props.name, "gpu_total_memory_bytes": props.total_memory,
            "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF"),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "hardware_identity": hardware_identity(report["telemetry_before_load"]),
            "power_identity": power_identity(report["telemetry_before_load"]),
        }
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        load_start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, attn_implementation="sdpa",
            use_safetensors=True, low_cpu_mem_usage=True, device_map={"": str(device)},
            local_files_only=True, trust_remote_code=False,
        ).eval().requires_grad_(False)
        torch.cuda.synchronize(device)
        report["load"] = {
            "seconds": time.perf_counter() - load_start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "resident_allocated_bytes": torch.cuda.memory_allocated(device),
            "resident_reserved_bytes": torch.cuda.memory_reserved(device),
        }
        inventory = inspect_model(model, pruning, str(device))
        parameters = list(model.parameters())  # PyTorch removes tied Parameter aliases.
        stores = {(str(p.device), p.untyped_storage().data_ptr()): p.untyped_storage().nbytes()
                  for p in parameters}
        inventory.update(unique_storage_bytes=sum(stores.values()),
                         live_lora_parameter_count=sum(p.numel() for name, p in model.named_parameters()
                                                       if "lora_" in name),
                         parameter_count_definition="Each distinct Parameter once; PyTorch named_parameters removes aliases.",
                         storage_definition="Each distinct device/storage pointer once; includes full underlying storage.")
        report["model"] = inventory
        validate_model_label(report)
        if model.config.max_position_embeddings < 544:
            raise ValueError("512 input plus 32 output exceeds model context capacity")
        report["host_memory_after_load"] = host_memory()
        for length in SETTINGS["input_lengths"]:
            ids = fixture_ids(seed_ids, length)
            if any(token >= model.config.vocab_size for token in ids):
                raise ValueError("Fixture token lies outside the model vocabulary")
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            bench = {"input_tokens": length, "fixture_text": FIXTURE_TEXT,
                     "fixture_seed_token_ids": seed_ids, "fixture_token_ids": ids,
                     "fixture_sha256": fixture_hash(ids), "warmups_completed": 0, "runs": [],
                     "telemetry_before": telemetry()}
            report["benchmarks"].append(bench)
            for _ in range(2):
                timed_greedy_run(model, input_ids, 32, device)
                bench["warmups_completed"] += 1
            for repeat in range(1, 8):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                torch.cuda.reset_peak_memory_stats(device)
                measurement = timed_greedy_run(model, input_ids, 32, device)
                measurement.update(repeat=repeat, resident_allocated_before_run_bytes=allocated,
                                   resident_reserved_before_run_bytes=reserved,
                                   incremental_peak_allocated_bytes=max(0, measurement["peak_allocated_bytes"] - allocated))
                bench["runs"].append(measurement)
                write_json(path, report)
                print(f"{args.model_id} session {args.session} input {length} repeat {repeat}: "
                      f"TTFT {measurement['first_token_seconds']:.4f}s; "
                      f"decode {measurement['decode_tokens_per_second']:.2f} tok/s", flush=True)
            bench["telemetry_after"] = telemetry()
            del input_ids
        report["status"] = "completed"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        validate_report(report)
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        raise
    finally:
        report["host_memory_at_exit"] = host_memory()
        write_json(path, report)
        print(f"Report: {path}", flush=True)


def validate_report(report: dict):
    if (type(report.get("schema_version")) is not int or report["schema_version"] != 1
            or report.get("status") != "completed"):
        raise ValueError("Require a completed schema_version=1 benchmark")
    label, session = report["model_id"], report["session"]
    if label not in MODEL_ORDER or type(report["position"]) is not int:
        raise ValueError("Unknown model or position")
    order = expected_order(session)
    if (report["position"] not in range(1, 5)
            or order[report["position"] - 1] != label or report["session_order"] != list(order)):
        raise ValueError("Session order mismatch")
    if json_hash(report["settings"]) != json_hash(SETTINGS):
        raise ValueError("Benchmark settings differ from frozen protocol")
    if (set(report["source_sha256"]) != set(SOURCE_FILES)
            or set(report["canonical_tokenizer_files_sha256"]) != set(TOKENIZER_FILES)):
        raise ValueError("Missing source/tokenizer hashes")
    hash_values = list(report["source_sha256"].values())
    hash_values += list(report["canonical_tokenizer_files_sha256"].values())
    hash_values += list(report["provenance"]["files_sha256"].values())
    hash_values += list(report["model_loader_files_sha256"].values())
    hash_values += [report["provenance"]["manifest_sha256"]]
    if not all(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) for value in hash_values):
        raise ValueError("Invalid source/tokenizer/model hash")
    if not {"config.json", "generation_config.json"}.issubset(report["provenance"]["files_sha256"]):
        raise ValueError("Model configuration hashes are missing")
    loader = report["model_loader_files_sha256"]
    expected = {"config.json", "generation_config.json"} | {
        name for name in report["provenance"]["files_sha256"] if is_model_weight(name)}
    if set(loader) not in (expected, expected | {"model.safetensors.index.json"}):
        raise ValueError("Loader file inventory mismatch")
    if any(loader[name] != report["provenance"]["files_sha256"][name] for name in expected):
        raise ValueError("Loader hashes and model provenance disagree")
    if report["tokenizer_compatibility"].get("status") != "passed":
        raise ValueError("Tokenizer semantic equivalence not checked")
    validate_model_label(report)
    for key in ("unique_parameter_count", "unique_parameter_storage_bytes", "unique_storage_bytes"):
        if type(report["model"][key]) is not int or report["model"][key] <= 0:
            raise ValueError("Invalid model size")
    weights = report["weights"]["files_bytes"]
    if (not weights or set(weights) != {n for n in report["provenance"]["files_sha256"] if is_model_weight(n)}
            or any(type(v) is not int or v <= 0 for v in weights.values())
            or report["weights"]["total_bytes"] != sum(weights.values())):
        raise ValueError("Weight byte inventory disagrees with hashed files")
    begin, end = (datetime.fromisoformat(report[key]) for key in ("started_at_utc", "finished_at_utc"))
    if begin.tzinfo is None or end.tzinfo is None or end <= begin:
        raise ValueError("Invalid run timestamps")
    uuid.UUID(report["process_instance_id"])
    finite_positive(report["load"]["seconds"], "load.seconds")
    for key in ("peak_allocated_bytes", "peak_reserved_bytes", "resident_allocated_bytes", "resident_reserved_bytes"):
        nonnegative_int(report["load"][key], "load." + key)
    if (report["load"]["peak_allocated_bytes"] > report["load"]["peak_reserved_bytes"]
            or report["load"]["resident_allocated_bytes"] > report["load"]["resident_reserved_bytes"]):
        raise ValueError("Inconsistent allocator load memory")
    if [b["input_tokens"] for b in report["benchmarks"]] != [128, 512]:
        raise ValueError("Missing or reordered input lengths")
    for bench in report["benchmarks"]:
        ids = fixture_ids(bench["fixture_seed_token_ids"], bench["input_tokens"])
        if (bench["fixture_text"] != FIXTURE_TEXT or bench["fixture_token_ids"] != ids
                or bench["fixture_sha256"] != fixture_hash(ids)
                or type(bench["warmups_completed"]) is not int or bench["warmups_completed"] != 2
                or len(bench["runs"]) != 7):
            raise ValueError("Fixture, warmup or repeat count mismatch")
        outputs = []
        for repeat, sample in enumerate(bench["runs"], 1):
            if (type(sample["repeat"]) is not int or sample["repeat"] != repeat
                    or type(sample["generated_token_count"]) is not int or sample["generated_token_count"] != 32
                    or type(sample["decode_token_count"]) is not int or sample["decode_token_count"] != 31
                    or len(sample["generated_token_ids"]) != 32
                    or not all(type(t) is int and t >= 0 for t in sample["generated_token_ids"])):
                raise ValueError("Repeat numbering or output token count mismatch")
            outputs.append(sample["generated_token_ids"])
            for key in ("first_token_seconds", "decode_seconds", "total_seconds", "decode_tokens_per_second"):
                finite_positive(sample[key], key)
            if (not math.isclose(sample["total_seconds"], sample["first_token_seconds"] + sample["decode_seconds"], rel_tol=1e-10, abs_tol=1e-10)
                    or not math.isclose(sample["decode_tokens_per_second"], 31 / sample["decode_seconds"], rel_tol=1e-10)):
                raise ValueError("Timing arithmetic mismatch")
            for key in ("peak_allocated_bytes", "peak_reserved_bytes", "resident_allocated_before_run_bytes",
                        "resident_reserved_before_run_bytes", "incremental_peak_allocated_bytes"):
                nonnegative_int(sample[key], key)
            if (sample["peak_allocated_bytes"] > sample["peak_reserved_bytes"]
                    or sample["resident_allocated_before_run_bytes"] > sample["resident_reserved_before_run_bytes"]
                    or sample["peak_allocated_bytes"] < sample["resident_allocated_before_run_bytes"]
                    or sample["peak_reserved_bytes"] < sample["resident_reserved_before_run_bytes"]
                    or sample["incremental_peak_allocated_bytes"] != sample["peak_allocated_bytes"] - sample["resident_allocated_before_run_bytes"]):
                raise ValueError("Inconsistent generation allocator memory")
        if any(output != outputs[0] for output in outputs[1:]):
            raise ValueError("Greedy outputs changed across measured repeats")
        for point in ("telemetry_before", "telemetry_after"):
            if (hardware_identity(bench[point]) != report["environment"]["hardware_identity"]
                    or power_identity(bench[point]) != report["environment"]["power_identity"]):
                raise ValueError("GPU/driver inventory or power mode changed during run")
    if (hardware_identity(report["telemetry_before_load"]) != report["environment"]["hardware_identity"]
            or power_identity(report["telemetry_before_load"]) != report["environment"]["power_identity"]):
        raise ValueError("Load telemetry identity mismatch")


def compare_distributions(baseline: list[dict], candidate: list[dict], higher_is_better: bool) -> dict:
    if len(baseline) != 2 or len(candidate) != 2:
        raise ValueError("Require two independent session distributions")
    change = [100 * (c["median"] / b["median"] - 1) for b, c in zip(baseline, candidate)]
    improve = [delta > 0 if higher_is_better else delta < 0 for delta in change]
    regress = [delta < 0 if higher_is_better else delta > 0 for delta in change]
    separate_improve = [(c["minimum"] > b["maximum"] if higher_is_better else c["maximum"] < b["minimum"])
                        for b, c in zip(baseline, candidate)]
    separate_regress = [(c["maximum"] < b["minimum"] if higher_is_better else c["minimum"] > b["maximum"])
                        for b, c in zip(baseline, candidate)]
    if all(separate_improve):
        outcome = "stable_improvement_observed_in_this_microbenchmark"
    elif all(separate_regress):
        outcome = "stable_regression_observed_in_this_microbenchmark"
    else:
        outcome = "no_stable_speedup_observed"
    return {"session_median_relative_changes_percent": change,
            "median_direction_consistent": all(improve) or all(regress) or all(d == 0 for d in change),
            "candidate_faster_median_both_sessions": all(improve),
            "candidate_faster_ranges_separated_both_sessions": all(separate_improve),
            "session_ranges_overlap": [not (c["maximum"] < b["minimum"] or c["minimum"] > b["maximum"])
                                       for b, c in zip(baseline, candidate)],
            "outcome": outcome,
            "interpretation": "未观察到稳定加速" if outcome == "no_stable_speedup_observed" else
                              ("此固定微基准中观察到两轮方向一致且范围分离的加速" if all(separate_improve)
                               else "此固定微基准中观察到两轮方向一致且范围分离的变慢"),
            "limit": "Descriptive conservative range rule, not a confidence interval or significance test."}


def summarize(report_paths: list[Path], verify_model_files: bool = True) -> dict:
    if len(report_paths) != 8 or len({p.resolve() for p in report_paths}) != 8:
        raise ValueError("Require eight distinct benchmark reports")
    reports = [read_json(path) for path in report_paths]
    for report in reports:
        validate_report(report)
    keyed = {(r["session"], r["model_id"]): r for r in reports}
    if len(keyed) != 8 or set(keyed) != {(s, m) for s in (1, 2) for m in MODEL_ORDER}:
        raise ValueError("Missing or duplicate session/model identity")
    if len({r["process_instance_id"] for r in reports}) != 8:
        raise ValueError("Each benchmark must use an independent process")
    ordered = [keyed[s, model] for s in (1, 2) for model in expected_order(s)]
    if any(datetime.fromisoformat(a["finished_at_utc"]) > datetime.fromisoformat(b["started_at_utc"])
           for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Benchmark runs overlap or do not follow forward/reverse wall-time order")
    first = ordered[0]
    shared = ("source_model_revision", "settings", "source_sha256", "environment",
              "canonical_tokenizer_files_sha256", "tokenizer_compatibility")
    for report in ordered[1:]:
        if any(report[key] != first[key] for key in shared):
            raise ValueError("Environment/settings/source/tokenizer/revision differ across reports")
        for a, b in zip(first["benchmarks"], report["benchmarks"]):
            if any(a[key] != b[key] for key in ("fixture_text", "fixture_seed_token_ids", "fixture_token_ids", "fixture_sha256")):
                raise ValueError("Fixed fixture differs across checkpoints or sessions")
    for model_id in MODEL_ORDER:
        a, b = keyed[1, model_id], keyed[2, model_id]
        if any(a[key] != b[key] for key in ("provenance", "model", "weights", "model_directory", "model_loader_files_sha256")):
            raise ValueError("Same model identity/files/inventory changed across sessions")
        for old, new in zip(a["benchmarks"], b["benchmarks"]):
            if old["runs"][0]["generated_token_ids"] != new["runs"][0]["generated_token_ids"]:
                raise ValueError("Same model greedy output changed across sessions")
        if verify_model_files:
            directory = Path(a["model_directory"])
            manifest_name = ("merge_manifest.json" if model_id.endswith("_merged") else
                             "pruning_manifest.json" if model_id.startswith("calibrated") else "download_manifest.json")
            manifest_path = Path(a["provenance"]["manifest_path"])
            if (manifest_path.resolve() != (directory / manifest_name).resolve()
                    or sha256_file(manifest_path) != a["provenance"]["manifest_sha256"]):
                raise ValueError("Current benchmark manifest SHA256/path mismatch")
            for name, digest in a["provenance"]["files_sha256"].items():
                target = (directory / name).resolve()
                if (Path(name).is_absolute() or not target.is_relative_to(directory.resolve())
                        or sha256_file(target) != digest):
                    raise ValueError("Current benchmark checkpoint file SHA256 mismatch")
            if weight_files(directory, a["provenance"]["files_sha256"]) != a["weights"]:
                raise ValueError("Current checkpoint disk byte inventory changed")
            if loader_file_hashes(directory, a["provenance"]) != a["model_loader_files_sha256"]:
                raise ValueError("Current checkpoint loader index/inventory changed")
            for name, digest in a["model_loader_files_sha256"].items():
                if name not in a["provenance"]["files_sha256"] and sha256_file(directory / name) != digest:
                    raise ValueError("Current checkpoint loader file SHA256 mismatch")
    for before, merged in (("original_step0", "original_step64_merged"),
                           ("calibrated_5888_step0", "calibrated_5888_step64_merged")):
        if keyed[1, before]["model"]["unique_parameter_count"] != keyed[1, merged]["model"]["unique_parameter_count"]:
            raise ValueError("Merge changed base parameter count")
        if keyed[1, before]["provenance"]["files_sha256"] == keyed[1, merged]["provenance"]["files_sha256"]:
            raise ValueError("Recovered merge has identical whole-model hashes to untrained base")
        bound = keyed[1, merged]["provenance"]["merge_manifest"]["source_adapter"]["base_provenance"]
        source = keyed[1, before]["provenance"]
        if any(bound.get(key) != source[key] for key in ("checkpoint_kind", "files_sha256", "manifest_sha256")):
            raise ValueError("Merged adapter's source base is not the compared step0 checkpoint")
    original_files = keyed[1, "original_step0"]["provenance"]["files_sha256"]
    pruning_source = keyed[1, "calibrated_5888_step0"]["provenance"]["source"]["files"]
    pruning_source_hashes = {entry["name"]: entry["sha256"] for entry in pruning_source}
    if any(pruning_source_hashes.get(name) != digest for name, digest in original_files.items()):
        raise ValueError("Calibrated checkpoint source is not the compared original checkpoint")
    models = {}
    for label in MODEL_ORDER:
        source = keyed[1, label]
        lengths = {}
        for index, length in enumerate((128, 512)):
            session_stats = {}
            all_runs = []
            for session in (1, 2):
                runs = keyed[session, label]["benchmarks"][index]["runs"]
                all_runs.extend(runs)
                session_stats[str(session)] = {
                    metric: distribution(run[metric] for run in runs)
                    for metric in ("first_token_seconds", "decode_tokens_per_second", "total_seconds")}
            lengths[str(length)] = {
                "sessions": session_stats,
                "pooled_14_repeats": {metric: distribution(r[metric] for r in all_runs)
                                     for metric in ("first_token_seconds", "decode_tokens_per_second", "total_seconds")},
                "max_generation_allocated_bytes": max(r["peak_allocated_bytes"] for r in all_runs),
                "max_generation_reserved_bytes": max(r["peak_reserved_bytes"] for r in all_runs),
                "max_incremental_generation_allocated_bytes": max(r["incremental_peak_allocated_bytes"] for r in all_runs),
            }
        models[label] = {
            "model": source["model"], "weights": source["weights"], "provenance": source["provenance"],
            "load_by_session": {str(s): keyed[s, label]["load"] for s in (1, 2)}, "lengths": lengths,
        }
    comparisons = []
    pairs = [(MODEL_ORDER[0], label) for label in MODEL_ORDER[1:]]
    pairs += [("original_step64_merged", "calibrated_5888_step64_merged"),
              ("calibrated_5888_step0", "calibrated_5888_step64_merged")]
    for baseline, candidate in pairs:
        for length in ("128", "512"):
            metrics = {}
            for metric, higher in (("first_token_seconds", False), ("decode_tokens_per_second", True),
                                   ("total_seconds", False)):
                stats = [[models[label]["lengths"][length]["sessions"][str(s)][metric] for s in (1, 2)]
                         for label in (baseline, candidate)]
                metrics[metric] = compare_distributions(*stats, higher)
            comparisons.append({"baseline": baseline, "candidate": candidate, "input_tokens": int(length), "metrics": metrics})
    return {
        "schema_version": 1, "status": "completed", "scope": first["scope"],
        "reports": [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in report_paths],
        "settings": SETTINGS, "environment": first["environment"], "source_sha256": first["source_sha256"],
        "protocol_checks": "All eight complete; fixed order, unique processes, nonoverlapping timestamps, same fixture/settings/environment/source/tokenizer; exact per-model hashes/inventory/greedy output across sessions.",
        "model_files_reverified_at_summary": verify_model_files,
        "models": models, "comparisons": comparisons,
        "telemetry": [{key: r[key] for key in ("model_id", "session", "telemetry_before_load")} |
                      {"lengths": [{k: b[k] for k in ("input_tokens", "telemetry_before", "telemetry_after")}
                                   for b in r["benchmarks"]]} for r in ordered],
        "limitations": [
            "Single machine and GPU, two independent process rounds; synthetic repeated Chinese token fixture.",
            "No live LoRA; merged BF16 checkpoints only. 512 input plus 32 output is a timing workload, not the 512-token QA protocol.",
            "TTFT excludes tokenization, disk loading and CPU output transfer; later-token rate includes Python/control overhead.",
            "Allocator allocated/reserved are not whole-device memory. Disk weight bytes include safetensors headers.",
            "Original snapshot and save_pretrained exports may serialize tied weights differently; disk-byte savings must not be attributed entirely to pruning.",
            "Telemetry is sampled outside timing, not continuous; unavailable fields remain unavailable. No automatic power-setting changes.",
            "Nonoverlapping ranges in both sessions are a conservative descriptive condition, not statistical significance or guaranteed deployment speed.",
            "Quality acceptance is a separate prerequisite enforced by the experiment orchestrator, not inferred from speed.",
        ],
    }


def markdown(summary: dict) -> str:
    lines = ["# 受控部署微基准", "", "两轮独立进程，四模型正序/反序；batch=1，固定32输出，预热2次、计时7次。每格为该轮中位数 [最小, 最大]。", "",
             "| 模型 | 输入 | 轮次 | 首token 秒 | 后31token tok/s | 分配峰值 GiB | 保留峰值 GiB |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for label, model in summary["models"].items():
        for length, entry in model["lengths"].items():
            for session, stats in entry["sessions"].items():
                cells = []
                for metric in ("first_token_seconds", "decode_tokens_per_second"):
                    d = stats[metric]
                    cells.append(f"{d['median']:.4f} [{d['minimum']:.4f}, {d['maximum']:.4f}]")
                lines.append(f"| {label} | {length} | {session} | {' | '.join(cells)} | "
                             f"{entry['max_generation_allocated_bytes']/2**30:.4f} | {entry['max_generation_reserved_bytes']/2**30:.4f} |")
    lines += ["", "显存列为该输入长度两轮的最大值；加载峰值和每次计时原始记录在JSON。", "",
              "| 对比 | 输入 | 首token判断 | 后31token判断 |", "| --- | --- | --- | --- |"]
    for comparison in summary["comparisons"]:
        metrics = comparison["metrics"]
        lines.append(f"| {comparison['baseline']} → {comparison['candidate']} | {comparison['input_tokens']} | "
                     f"{metrics['first_token_seconds']['interpretation']} | {metrics['decode_tokens_per_second']['interpretation']} |")
    lines += ["", "只要两轮方向不一致，或任一轮范围重叠，就不把中位数差称为稳定加速。这里的判断不是显著性检验。", "",
              "| 模型 | 去重参数 | 去重权重张量字节 | 权重文件字节 |", "| --- | --- | --- | --- |"]
    for label, model in summary["models"].items():
        lines.append(f"| {label} | {model['model']['unique_parameter_count']} | "
                     f"{model['model']['unique_storage_bytes']} | {model['weights']['total_bytes']} |")
    lines += ["", "温度、功耗、SM/显存频率、整卡占用、驱动与Windows电源状态的可得性和原始值保留在JSON；未修改电源设置。", ""]
    lines += ["- " + limit for limit in summary["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--model-dir", type=Path, required=True)
    runner.add_argument("--canonical-tokenizer-dir", type=Path, required=True)
    runner.add_argument("--revision", required=True)
    runner.add_argument("--model-id", choices=MODEL_ORDER, required=True)
    runner.add_argument("--session", type=int, choices=(1, 2), required=True)
    runner.add_argument("--position", type=int, choices=(1, 2, 3, 4), required=True)
    runner.add_argument("--device", type=int, default=0)
    runner.add_argument("--output-dir", type=Path, required=True)
    summ = sub.add_parser("summarize")
    summ.add_argument("--reports", nargs=8, type=Path, required=True)
    summ.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        result = summarize(args.reports)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(args.output_dir / "deployment_summary.json", result)
        (args.output_dir / "deployment_summary.md").write_text(markdown(result), encoding="utf8")


if __name__ == "__main__":
    main()
