"""Create an explicit presentation export; never rewrite frozen source reports.

The output omits local paths/device UUIDs and is not a strict-runner input.
Every source is identified by its full-file SHA256, not a sanitized-file hash.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "completed":
        raise ValueError(f"Source must be completed: {path.name}")
    return data


def selected(data, names):
    return {name: data[name] for name in names}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def telemetry(value):
    smi = value.get("nvidia_smi", {})
    power = value.get("windows_power", {})
    return {"at_utc": value.get("at_utc"),
            "nvidia_smi": {"available": smi.get("available"), "units": smi.get("units"),
                "gpus": [{key: item for key, item in row.items() if key != "uuid"}
                         for row in smi.get("gpus", [])]},
            "power": {"available": power.get("available"),
                "system_power": power.get("system_power"), "settings_modified": power.get("settings_modified")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("quality", "resources", "recovery", "pruning-grid", "original-merge", "pruned-merge", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sources = {name: getattr(args, name) for name in
               ("quality", "resources", "recovery", "pruning_grid", "original_merge", "pruned_merge")}
    raw = {name: read(path) for name, path in sources.items()}
    resource = raw["resources"]
    result = {
        "schema_version": 1, "status": "completed_presentation_export",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "notice": "Derived public presentation data, with local paths omitted. Not original reports; not valid input for strict provenance comparison. Original sources remain immutable and are identified by SHA256 below.",
        "source_report_sha256": {name: sha(path) for name, path in sources.items()},
        "exporter_sha256": sha(Path(__file__)),
        "source_model": {"model_id": "Qwen/Qwen3-1.7B", "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"},
        "quality": selected(raw["quality"], ("rows", "matched_budget_differences", "final_selection_allowed", "checkpoint_reselected", "human_acceptance_performed", "limits")),
        "pruning_validation": raw["pruning_grid"]["rows"],
        "recovery_validation": [{k: v for k, v in row.items() if k != "report_path"} for row in raw["recovery"]["rows"]],
        "resources": {"settings": resource["settings"], "comparisons": resource["comparisons"],
                      "limitations": resource["limitations"], "models": {}, "raw_sessions": [],
                      "environment": selected(resource["environment"], ("python", "platform", "packages",
                          "torch_cuda_build", "gpu_name", "gpu_total_memory_bytes", "gpu_compute_capability"))},
        "merges": {},
    }
    for label, model in resource["models"].items():
        result["resources"]["models"][label] = {
            "model": model["model"], "weight_file_bytes": model["weights"]["total_bytes"],
            "load_by_session": model["load_by_session"], "lengths": model["lengths"],
        }
    for binding in resource["reports"]:
        path = Path(binding["path"])
        if sha(path) != binding["sha256"]:
            raise ValueError("Raw benchmark changed after strict summary")
        report = read(path)
        result["resources"]["raw_sessions"].append({
            **selected(report, ("model_id", "session", "position", "started_at_utc", "finished_at_utc", "source_sha256")),
            "original_report_sha256": binding["sha256"],
            "telemetry_before_load": telemetry(report["telemetry_before_load"]),
            "benchmarks": [{**selected(item, ("input_tokens", "fixture_sha256", "warmups_completed", "runs")),
                            "telemetry_before": telemetry(item["telemetry_before"]),
                            "telemetry_after": telemetry(item["telemetry_after"])} for item in report["benchmarks"]],
        })
    for name in ("original_merge", "pruned_merge"):
        result["merges"][name] = selected(raw[name], ("checkpoint_kind", "parameter_dtype", "training_step",
            "base_checkpoint_kind", "merge_method", "weight_audit", "merge_difference", "parameter_count",
            "intermediate_size", "reload_validation", "source_sha256"))
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    # Reject unexpected private path metadata instead of silently sanitizing new fields.
    if any(marker in payload.lower() for marker in ("c:\\\\users\\\\", "c:/users/", "/home/", "gpu-")):
        raise ValueError("Unexpected machine-specific field in presentation whitelist; inspect before exporting")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
