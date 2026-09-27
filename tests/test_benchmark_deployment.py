"""CPU-only protocol and arithmetic checks; no CUDA calls or model downloads."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_deployment as bench


def snapshot():
    return {"at_utc": "2026-09-27T10:00:00+00:00",
            "nvidia_smi": {"available": True, "gpus": [{
                "index": "0", "uuid": "GPU-test", "name": "test-gpu", "driver_version": "test",
                "memory.total": "6144", "memory.used": "4000", "temperature.gpu": "55",
                "power.draw": "40", "clocks.current.sm": "1200", "clocks.current.memory": "6000"}]},
            "windows_power": {"available": True, "settings_modified": False,
                              "active_scheme": {"scheme_guid": "fixed-guid"},
                              "system_power": {"ACLineStatus": 1, "SystemStatusFlag": 0}}}


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.provenance = {}
        self.inventories = {}
        for label in bench.MODEL_ORDER:
            directory = self.root / label
            directory.mkdir()
            names = ["model.safetensors", "config.json", "generation_config.json"]
            if label.endswith("_merged"):
                names.append("merge_probes.safetensors")
            for name in names:
                (directory / name).write_text(label + "/" + name, encoding="utf8")
            hashes = {name: bench.sha256_file(directory / name) for name in names}
            kind = "pruned" if label.startswith("calibrated") else "original"
            merged = label.endswith("_merged")
            manifest = directory / ("merge_manifest.json" if merged else
                                    "pruning_manifest.json" if kind == "pruned" else "download_manifest.json")
            manifest.write_text(label, encoding="utf8")
            p = {"checkpoint_kind": "merged_lora" if merged else kind,
                 "manifest_path": str(manifest), "manifest_sha256": bench.sha256_file(manifest),
                 "files_sha256": hashes}
            if merged:
                p["source_model_revision"] = "a" * 40
            else:
                p["source" if kind == "pruned" else "download_manifest"] = {"revision": "a" * 40}
                if kind == "pruned":
                    p["method"] = "calibrated_rms_mean_abs"
                    p["source"]["files"] = [{"name": name, "sha256": digest} for name, digest in
                                           self.provenance["original_step0"]["files_sha256"].items()]
            count = 900 if kind == "pruned" else 1000
            if merged:
                before = "calibrated_5888_step0" if kind == "pruned" else "original_step0"
                p.update(base_checkpoint_kind=kind, training_step=64, merge_manifest={
                    "parameter_count": count, "reload_validation": {"status": "passed", "probe_exact": True},
                    "source_adapter": {"base_provenance": copy.deepcopy(self.provenance[before])}})
            self.provenance[label] = p
            self.inventories[label] = {
                "unique_parameter_count": count, "unique_parameter_storage_bytes": count * 2,
                "unique_storage_bytes": count * 2, "parameter_dtypes": ["torch.bfloat16"],
                "attention_implementation": "sdpa", "live_lora_parameter_count": 0,
                "intermediate_size": 5888 if kind == "pruned" else 6144,
            }
        self.reports, self.paths = {}, []
        ordered = [(s, model) for s in (1, 2) for model in bench.expected_order(s)]
        for order_index, (session, label) in enumerate(ordered):
            start = datetime(2026, 9, 27, 10, tzinfo=timezone.utc) + timedelta(minutes=order_index * 2)
            report = {
                "schema_version": 1, "status": "completed", "model_id": label, "session": session,
                "position": bench.expected_order(session).index(label) + 1,
                "session_order": list(bench.expected_order(session)),
                "process_instance_id": str(uuid.uuid4()), "process_id": 1000 + order_index,
                "started_at_utc": start.isoformat(), "finished_at_utc": (start + timedelta(minutes=1)).isoformat(),
                "source_model_revision": "a" * 40, "scope": "test fixed benchmark",
                "settings": copy.deepcopy(bench.SETTINGS),
                "source_sha256": {name: "a" * 64 for name in bench.SOURCE_FILES},
                "canonical_tokenizer_files_sha256": {name: "b" * 64 for name in bench.TOKENIZER_FILES},
                "tokenizer_compatibility": {"status": "passed", "signature_sha256": "c" * 64},
                "environment": {"python": "test", "hardware_identity": bench.hardware_identity(snapshot()),
                                "power_identity": bench.power_identity(snapshot())},
                "telemetry_before_load": snapshot(),
                "model_directory": str(self.root / label), "provenance": copy.deepcopy(self.provenance[label]),
                "model": copy.deepcopy(self.inventories[label]),
                "model_loader_files_sha256": bench.loader_file_hashes(self.root / label, self.provenance[label]),
                "weights": bench.weight_files(self.root / label, self.provenance[label]["files_sha256"]),
                "load": {"seconds": 1, "peak_allocated_bytes": 3000, "peak_reserved_bytes": 4000,
                         "resident_allocated_bytes": 2000, "resident_reserved_bytes": 4000},
                "benchmarks": [],
            }
            for length in (128, 512):
                ids = bench.fixture_ids([12, 34, 56], length)
                rates = [31 + i for i in range(7)]
                runs = []
                for repeat, rate in enumerate(rates, 1):
                    first, decode = .1, 31 / rate
                    runs.append({"repeat": repeat, "first_token_seconds": first, "decode_seconds": decode,
                                 "total_seconds": first + decode, "decode_token_count": 31,
                                 "decode_tokens_per_second": rate, "generated_token_count": 32,
                                 "generated_token_ids": [1] * 32, "peak_allocated_bytes": 3000,
                                 "peak_reserved_bytes": 4000, "resident_allocated_before_run_bytes": 2000,
                                 "resident_reserved_before_run_bytes": 4000, "incremental_peak_allocated_bytes": 1000})
                report["benchmarks"].append({
                    "input_tokens": length, "fixture_text": bench.FIXTURE_TEXT,
                    "fixture_seed_token_ids": [12, 34, 56], "fixture_token_ids": ids,
                    "fixture_sha256": bench.fixture_hash(ids), "warmups_completed": 2, "runs": runs,
                    "telemetry_before": snapshot(), "telemetry_after": snapshot(),
                })
            self.reports[session, label] = report
            self.paths.append(self.root / f"session{session}-{label}.json")
        self.save()

    def save(self):
        ordered = [self.reports[s, m] for s in (1, 2) for m in bench.expected_order(s)]
        for path, report in zip(self.paths, ordered):
            bench.write_json(path, report)

    def result(self):
        self.save()
        return bench.summarize(self.paths)

    def target(self):
        return self.reports[2, "calibrated_5888_step64_merged"]

    def assert_rejected(self, mutation):
        mutation(self.target())
        with self.assertRaises((ValueError, KeyError)):
            self.result()

    def test_full_protocol_and_merged_probe_excluded(self):
        result = self.result()
        self.assertEqual(len(result["models"]), 4)
        self.assertEqual(len(result["comparisons"]), 10)
        self.assertTrue(result["model_files_reverified_at_summary"])
        for model in result["models"].values():
            self.assertEqual(set(model["weights"]["files_bytes"]), {"model.safetensors"})
            self.assertEqual(model["lengths"]["128"]["pooled_14_repeats"]["decode_tokens_per_second"],
                             {"count": 14, "median": 34.0, "minimum": 31, "maximum": 37})
        self.assertIn("未观察到稳定加速", bench.markdown(result))

    def test_order_is_pairwise_reversed(self):
        self.assertEqual(bench.expected_order(2), bench.MODEL_ORDER[::-1])
        with self.assertRaises(ValueError):
            bench.expected_order(True)

    def test_fixture_exact_repeat_and_validation(self):
        ids = bench.fixture_ids([7, 8, 9], 128)
        self.assertEqual(len(ids), 128)
        self.assertEqual(ids[:6], [7, 8, 9, 7, 8, 9])
        self.assertEqual(ids[-2:], [7, 8])
        for bad in ([True], [], [-1]):
            with self.assertRaises(ValueError):
                bench.fixture_ids(bad, 128)

    def test_distribution_rejects_nonfinite(self):
        self.assertEqual(bench.distribution([4, 1, 2]), {"count": 3, "median": 2, "minimum": 1, "maximum": 4})
        for bad in ([float("nan")], [float("inf")], [0], [True], []):
            with self.assertRaises(ValueError):
                bench.distribution(bad)

    def test_direction_consistent_but_overlapping_is_not_speedup(self):
        base = [bench.distribution([10, 11, 12])] * 2
        candidate = [bench.distribution([11, 12, 13])] * 2
        result = bench.compare_distributions(base, candidate, True)
        self.assertTrue(result["candidate_faster_median_both_sessions"])
        self.assertEqual(result["interpretation"], "未观察到稳定加速")

    def test_reversed_session_direction_not_speedup(self):
        base = [bench.distribution([10, 11, 12])] * 2
        candidate = [bench.distribution([13, 14, 15]), bench.distribution([7, 8, 9])]
        result = bench.compare_distributions(base, candidate, True)
        self.assertFalse(result["median_direction_consistent"])
        self.assertEqual(result["outcome"], "no_stable_speedup_observed")

    def test_strict_separation_respects_metric_direction(self):
        base = [bench.distribution([10, 11, 12])] * 2
        candidate = [bench.distribution([7, 8, 9])] * 2
        self.assertIn("improvement", bench.compare_distributions(base, candidate, False)["outcome"])
        self.assertIn("regression", bench.compare_distributions(base, candidate, True)["outcome"])

    def test_incomplete_or_duplicate_reports_rejected(self):
        for paths in (self.paths[:-1], self.paths[:-1] + [self.paths[0]]):
            with self.assertRaises(ValueError):
                bench.summarize(paths)
        self.assert_rejected(lambda r: r.update(status="running"))

    def test_wrong_session_order_rejected(self):
        self.assert_rejected(lambda r: r.update(position=4))

    def test_provenance_revision_rejected(self):
        self.assert_rejected(lambda r: r["provenance"].update(source_model_revision="b" * 40))

    def test_reused_process_rejected(self):
        self.assert_rejected(lambda r: r.update(process_instance_id=self.reports[1, "original_step0"]["process_instance_id"]))

    def test_overlapping_timestamps_rejected(self):
        self.assert_rejected(lambda r: r.update(started_at_utc=self.reports[1, "original_step0"]["started_at_utc"]))

    def test_changed_fixture_with_valid_hash_rejected(self):
        r = self.target()
        for b in r["benchmarks"]:
            b["fixture_seed_token_ids"] = [42]
            b["fixture_token_ids"] = bench.fixture_ids([42], b["input_tokens"])
            b["fixture_sha256"] = bench.fixture_hash(b["fixture_token_ids"])
        with self.assertRaisesRegex(ValueError, "fixture differs"):
            self.result()

    def test_settings_types_and_environment_rejected(self):
        original = copy.deepcopy(self.target())
        self.target()["settings"]["batch_size"] = True
        with self.assertRaises(ValueError):
            self.result()
        self.reports[2, "calibrated_5888_step64_merged"] = original
        self.assert_rejected(lambda r: r["environment"].update(python="other"))

    def test_source_tokenizer_or_model_hash_changes_rejected(self):
        for key in ("source_sha256", "canonical_tokenizer_files_sha256"):
            old = copy.deepcopy(self.target())
            self.target()[key][next(iter(self.target()[key]))] = "d" * 64
            with self.assertRaises(ValueError):
                self.result()
            self.reports[2, "calibrated_5888_step64_merged"] = old
        self.assert_rejected(lambda r: r["provenance"]["files_sha256"].update({"model.safetensors": "d" * 64}))

    def test_current_checkpoint_tampering_rejected(self):
        (self.root / "original_step0" / "model.safetensors").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.result()

    def test_manifest_tampering_rejected(self):
        (self.root / "original_step0" / "download_manifest.json").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.result()

    def test_shard_index_is_bound_separately_from_base_provenance(self):
        label = "original_step0"
        index = self.root / label / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"weight": "model.safetensors"}}), encoding="utf8")
        for session in (1, 2):
            self.reports[session, label]["model_loader_files_sha256"] = bench.loader_file_hashes(
                self.root / label, self.provenance[label])
        self.result()
        index.write_text(json.dumps({"weight_map": {"other_weight": "model.safetensors"}}), encoding="utf8")
        with self.assertRaisesRegex(ValueError, "loader"):
            self.result()

    def test_timer_wrong_denominator_and_total_rejected(self):
        for field, value in (("decode_tokens_per_second", 32), ("total_seconds", 9)):
            old = copy.deepcopy(self.target())
            self.target()["benchmarks"][0]["runs"][0][field] = value
            with self.assertRaisesRegex(ValueError, "arithmetic"):
                self.result()
            self.reports[2, "calibrated_5888_step64_merged"] = old

    def test_wrong_repeat_output_and_memory_rejected(self):
        for field, value in (("repeat", 2), ("generated_token_count", 31),
                             ("decode_token_count", 32), ("incremental_peak_allocated_bytes", 2)):
            old = copy.deepcopy(self.target())
            self.target()["benchmarks"][0]["runs"][0][field] = value
            with self.assertRaises(ValueError):
                self.result()
            self.reports[2, "calibrated_5888_step64_merged"] = old

    def test_changed_greedy_output_rejected(self):
        self.target()["benchmarks"][0]["runs"][0]["generated_token_ids"][0] = 2
        with self.assertRaisesRegex(ValueError, "Greedy outputs"):
            self.result()

    def test_power_change_rejected_temperature_variation_allowed(self):
        for r in self.reports.values():
            r["benchmarks"][0]["telemetry_after"]["nvidia_smi"]["gpus"][0]["temperature.gpu"] = "62"
        self.result()
        self.target()["benchmarks"][0]["telemetry_after"]["windows_power"]["system_power"]["ACLineStatus"] = 0
        with self.assertRaisesRegex(ValueError, "power mode"):
            self.result()

    def test_wrong_merge_source_base_rejected(self):
        for session in (1, 2):
            self.reports[session, "original_step64_merged"]["provenance"]["merge_manifest"]["source_adapter"]["base_provenance"]["manifest_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "source base"):
            self.result()

    def test_unrelated_pruning_source_rejected(self):
        for session in (1, 2):
            p = self.reports[session, "calibrated_5888_step0"]["provenance"]
            p["source"]["files"][0]["sha256"] = "d" * 64
            merged = self.reports[session, "calibrated_5888_step64_merged"]["provenance"]
            merged["merge_manifest"]["source_adapter"]["base_provenance"] = copy.deepcopy(p)
        with self.assertRaisesRegex(ValueError, "Calibrated checkpoint source"):
            self.result()

    def test_unavailable_telemetry_recorded_without_invention(self):
        with patch.object(bench, "_command", return_value={"available": False, "error": "not installed"}):
            result = bench.nvidia_snapshot()
        self.assertFalse(result["available"])
        self.assertNotIn("gpus", result)
        with patch.object(bench.platform, "system", return_value="Linux"):
            self.assertEqual(bench.windows_power_snapshot(), {"available": False, "reason": "not Windows"})


if __name__ == "__main__":
    unittest.main()
