"""Select bounded recovery checkpoints from eight compatible validation reports.

This is a read-only report audit and numeric proposal, not qualitative acceptance.
Model file identities are checked through the evaluation reports' hash chains;
this script does not load weights, read final data, or initiate more training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from compare_quality import compare
from run_recovery_quality import PRECISION_POLICY


STEPS = [0, 16, 32, 64]
ORDERING = [
    "highest validation F1 among numerically passing candidates",
    "lowest PPL when F1 exactly tied",
    "earliest training step when both exactly tied",
]


def _read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _base_identity(base):
    files = base.get("files_sha256")
    if (base.get("checkpoint_kind") not in ("original", "pruned")
            or base.get("recovery_applied", False)
            or not _digest(base.get("manifest_sha256"))
            or not isinstance(files, dict) or "config.json" not in files
            or not any(name.endswith(".safetensors") for name in files)
            or not all(_digest(value) for value in files.values())):
        raise ValueError("Invalid base provenance or actual base file SHA256 inventory")
    return {"checkpoint_kind": base["checkpoint_kind"],
            "manifest_sha256": base["manifest_sha256"], "files_sha256": files}


def _policy_arms(policy):
    selection = policy["selection"]
    if (policy.get("schema_version") != 1 or selection.get("candidate_steps") != STEPS
            or selection.get("ordering") != ORDERING
            or selection.get("apply_separately_to_each_arm") is not True
            or selection.get("step0_is_no_training_fallback") is not True):
        raise ValueError("Unsupported checkpoint selection policy")
    gates = selection["numeric_gates_relative_to_new_original_step0"]
    if gates != {"max_f1_drop_percentage_points": 5, "max_ppl_relative_increase": 0.1}:
        raise ValueError("Expected frozen 5-point F1 / 10-percent PPL policy gates")
    settings = policy["settings"]
    if settings.get("checkpoint_steps") != STEPS[1:] or settings.get("optimizer_steps") != 64:
        raise ValueError("Expected the completed 64-step training policy")
    arms = policy["arms"]
    if len(arms) != 2:
        raise ValueError("Expected exactly two training arms")
    by_kind = {}
    for name, arm in arms.items():
        kind = _base_identity(arm["base_provenance"])["checkpoint_kind"]
        if kind in by_kind:
            raise ValueError("Duplicate base kind in policy arms")
        by_kind[kind] = name
    if set(by_kind) != {"original", "pruned"}:
        raise ValueError("Expected original and pruned policy arms")
    return by_kind, gates


def _validate_report(report, arm, step, policy, policy_sha, baseline=None):
    provenance = report["provenance"]
    kind = arm["base_provenance"]["checkpoint_kind"]
    if (report.get("status") != "completed" or report.get("split") != "validation"
            or report.get("source_model_revision") != policy["revision"]):
        raise ValueError("Require completed validation reports at the policy revision")
    helpers = report.get("helper_sha256", {})
    if (not _digest(report.get("runner_sha256"))
            or helpers.get("run_recovery_quality.py") != report["runner_sha256"]
            or not {"run_quality.py", "cmrc_official.py", "eval_ppl.py"}.issubset(helpers)
            or report["protocol"].get("inference_precision") != PRECISION_POLICY):
        raise ValueError("Require the new common recovery evaluation version and precision policy")
    if (type(provenance.get("training_step")) is not int
            or provenance["training_step"] != step
            or provenance.get("base_checkpoint_kind") != kind):
        raise ValueError("Wrong base kind or training step")
    expected_kind = "recovered_lora" if step else kind
    if (provenance.get("checkpoint_kind") != expected_kind
            or provenance.get("recovery_applied") is not bool(step)
            or provenance.get("adapter_present") is not bool(step)):
        raise ValueError("Checkpoint kind / recovery / adapter flags disagree")
    actual_base = provenance["base_provenance"] if step else provenance
    identity = _base_identity(actual_base)
    if identity != _base_identity(arm["base_provenance"]):
        raise ValueError("Actual base SHA256 differs from the policy arm")
    if baseline is not None and identity != _base_identity(baseline["provenance"]):
        raise ValueError("Recovered checkpoint base SHA256 differs from step0")
    model = report["model"]
    counts = [model.get(key) for key in
              ("unique_parameter_count", "base_parameter_count", "adapter_parameter_count")]
    if (any(type(value) is not int for value in counts)
            or counts[0] <= 0 or counts[1] <= 0 or counts[2] < 0
            or counts[0] != counts[1] + counts[2]
            or bool(counts[2]) != bool(step) or model.get("adapter_merged") is not False
            or model.get("all_parameters_frozen") is not True
            or model.get("intermediate_size") != arm["expected_base_width"]):
        raise ValueError("Invalid real base/adapter/total parameter accounting")
    if baseline is not None and counts[1] != baseline["model"]["base_parameter_count"]:
        raise ValueError("Base parameter count differs from step0")
    if not isinstance(report["lm"].get("nll"), (int, float)) or not math.isfinite(report["lm"]["nll"]):
        raise ValueError("Invalid NLL")
    if step:
        manifest = provenance["recovery_manifest"]
        if (manifest.get("status") != "completed" or manifest.get("checkpoint_kind") != "recovered_lora"
                or manifest.get("training_step") != step
                or manifest.get("source_model_revision") != policy["revision"]
                or manifest.get("training_policy_sha256") != policy_sha):
            raise ValueError("Incomplete checkpoint or different training policy SHA256")
        if (_base_identity(manifest["base_provenance"]) != identity
                or manifest.get("settings") != policy["settings"]
                or manifest.get("data") != policy["training_data"]):
            raise ValueError("Checkpoint base, training settings, or data differ from policy")
        run = provenance["formal_training_run"]
        if (run.get("status") != "completed" or run.get("optimizer_steps") != 64
                or run.get("sample_count") != policy["training_data"]["row_count"]
                or run.get("supervised_token_count") != policy["training_data"]["supervised_token_count"]
                or run.get("base_parameters_unchanged") is not True
                or not _digest(run.get("sha256")) or not run.get("path")
                or not _digest(provenance.get("manifest_sha256"))):
            raise ValueError("Require a completed, hash-bound formal training run")
        training_source = manifest.get("training_source_sha256")
        if not isinstance(training_source, dict) or not training_source or not all(
                _digest(value) for value in training_source.values()):
            raise ValueError("Missing training source SHA256 provenance")
        reload_result = report.get("reload_validation", {})
        if reload_result.get("status") != "passed" or reload_result.get("probe_exact") is not True:
            raise ValueError("Recovered checkpoint saved-probe reload must pass")


def summarize(policy_path, original_base_path, pruned_base_path, candidate_paths):
    """Audit eight reports and return both numeric proposals without writing files."""
    policy, policy_sha = _read(policy_path)
    by_kind, gates = _policy_arms(policy)
    paths = [Path(original_base_path), Path(pruned_base_path), *map(Path, candidate_paths)]
    if len(paths) != 8 or len({path.resolve() for path in paths}) != 8:
        raise ValueError("Require eight distinct reports: two step0 and six trained candidates")
    loaded = [(path, *_read(path)) for path in paths]
    original = loaded[0][1]
    acceptance = original["protocol"]["acceptance"]
    if any(acceptance.get(key) != value for key, value in gates.items()):
        raise ValueError("Evaluation acceptance gates differ from selection policy")
    if acceptance.get("human_review") != "No widespread repeated outputs.":
        raise ValueError("Missing or changed qualitative human_review requirement")
    grouped = {name: {} for name in by_kind.values()}
    baselines = {"original": loaded[0][1], "pruned": loaded[1][1]}
    original_count = original["model"]["unique_parameter_count"]
    training_source = None
    runs = {}
    adapter_counts = {}
    for index, (path, report, digest) in enumerate(loaded):
        provenance = report["provenance"]
        step = 0 if index < 2 else provenance.get("training_step")
        kind = ("original" if index == 0 else "pruned") if index < 2 else provenance.get("base_checkpoint_kind")
        if kind not in by_kind or type(step) is not int or step not in (STEPS[1:] if index >= 2 else [0]):
            raise ValueError("Invalid candidate arm or scheduled training step")
        name = by_kind[kind]
        if step in grouped[name]:
            raise ValueError("Duplicate arm/training-step report")
        _validate_report(report, policy["arms"][name], step, policy, policy_sha,
                         baseline=baselines[kind] if step else None)
        comparison = compare(original, report)
        if step:
            run = provenance["formal_training_run"]
            run_identity = (run["path"], run["sha256"])
            if name in runs and runs[name] != run_identity:
                raise ValueError("Checkpoints of one arm come from different formal runs")
            runs[name] = run_identity
            source = provenance["recovery_manifest"]["training_source_sha256"]
            if training_source is not None and source != training_source:
                raise ValueError("Training source versions differ across checkpoints or arms")
            training_source = source
            count = report["model"]["adapter_parameter_count"]
            if name in adapter_counts and adapter_counts[name] != count:
                raise ValueError("Adapter parameter counts differ within an arm")
            adapter_counts[name] = count
        row = {"arm": name, "base_checkpoint_kind": kind, "training_step": step,
               "report_path": str(path.resolve()), "report_sha256": digest,
               "em": report["qa_metrics"]["em"], "f1": report["qa_metrics"]["f1"],
               "nll": report["lm"]["nll"], "ppl": report["lm"]["ppl"],
               "total_parameter_count": report["model"]["unique_parameter_count"],
               "base_parameter_count": report["model"]["base_parameter_count"],
               "adapter_parameter_count": report["model"]["adapter_parameter_count"],
               "parameter_reduction_count_vs_original_step0": original_count - report["model"]["unique_parameter_count"],
               **{key: comparison[key] for key in ("parameter_reduction", "f1_drop_percentage_points",
                                                   "ppl_relative_increase", "numeric_quality_gate_passed")}}
        grouped[name][step] = row
    if any(sorted(rows) != STEPS for rows in grouped.values()):
        raise ValueError("Each arm requires exactly steps 0, 16, 32, 64")
    rows, selected = [], {}
    for kind in ("original", "pruned"):
        name = by_kind[kind]
        arm_rows = [grouped[name][step] for step in STEPS]
        rows.extend(arm_rows)
        passing = [row for row in arm_rows if row["numeric_quality_gate_passed"]]
        best = min(passing, key=lambda row: (-row["f1"], row["ppl"], row["training_step"])) if passing else None
        selected[name] = {"checkpoint": best, "qualitative_review_status": "pending",
                          "trained_candidate_passed": any(row["training_step"] > 0 for row in passing),
                          "step0_fallback_selected": best is not None and best["training_step"] == 0,
                          "reason": "fixed numeric ordering; qualitative review required" if best else "no checkpoint passed the fixed gates"}
    differences = []
    for step in STEPS:
        orig = grouped[by_kind["original"]][step]
        pruned = grouped[by_kind["pruned"]][step]
        differences.append({"training_step": step,
                            "definition": "pruned minus original at the same training step",
                            "em_difference_percentage_points": pruned["em"] - orig["em"],
                            "f1_difference_percentage_points": pruned["f1"] - orig["f1"],
                            "nll_difference": pruned["nll"] - orig["nll"],
                            "ppl_relative_difference": pruned["ppl"] / orig["ppl"] - 1,
                            "parameter_difference_count": pruned["total_parameter_count"] - orig["total_parameter_count"]})
    return {"schema_version": 1, "status": "completed", "selection_status": "numeric_proposals_only",
            "qualitative_review_required": True, "split": "validation",
            "policy": {"path": str(Path(policy_path).resolve()), "sha256": policy_sha,
                       "selection": policy["selection"]},
            "evaluation": {key: original[key] for key in ("runner_sha256", "helper_sha256", "protocol")},
            "evaluation_data": original["data"], "training_source_sha256": training_source,
            "base_identities": {name: _base_identity(arm["base_provenance"]) for name, arm in policy["arms"].items()},
            "rows": rows, "matched_step_differences": differences, "selected": selected,
            "limitations": ["Checkpoint choice uses the same fixed validation subset, not final or an independent test.",
                            "No statistical significance or general model improvement is established.",
                            "Unmerged adapters are included in total parameter counts; training budgets match but adapter sizes differ.",
                            "Base identities are audited through evaluation report SHA256 provenance, not by rereading weight tensors here.",
                            "Keep every checkpoint and report; no longer training or relaxed thresholds is authorized by this summary."]}


def markdown(result):
    lines = ["# 有限恢复训练：完整验证对照", "",
             "以下仅按预先记录的数值规则提出检查点选择，定性回答复核仍待完成。全部检查点使用同一版本评测和同一 validation；没有使用 final。", "",
             "门槛统一相对新原模型 step0：F1 下降不超过 5 个百分点，PPL 相对增加不超过 10%。每个分支分别按 F1 降序、PPL 升序、训练步数升序选择；step0 也参与。", "",
             "总参数包含未合并 LoRA adapter。参数减少量以未训练原模型为参照，负数表示参数增加。", "",
             "| 分支 | 步数 | EM | F1 | NLL | PPL | 总参数 | base 参数 | adapter 参数 | 参数减少量 | 数值门槛 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in result["rows"]:
        lines.append(f"| {row['arm']} | {row['training_step']} | {row['em']:.2f} | {row['f1']:.4f} | {row['nll']:.6f} | {row['ppl']:.6f} | {row['total_parameter_count']:,} | {row['base_parameter_count']:,} | {row['adapter_parameter_count']:,} | {row['parameter_reduction_count_vs_original_step0']:,} | {'通过' if row['numeric_quality_gate_passed'] else '未通过'} |")
    lines += ["", "## 数值提议", ""]
    for name, selection in result["selected"].items():
        row = selection["checkpoint"]
        if row is None:
            lines.append(f"- {name}：没有检查点通过门槛，不能选入。")
        else:
            fallback = "（不训练退路）" if row["training_step"] == 0 else ""
            lines.append(f"- {name}：step {row['training_step']}{fallback}，F1 {row['f1']:.4f}、PPL {row['ppl']:.6f}；等待回答定性复核。")
        if not selection["trained_candidate_passed"]:
            lines.append(f"  该分支的训练后检查点均未通过门槛；不延长预算，不放宽阈值。")
    lines += ["", "## 同训练步数对照", "", "差值为剪后分支减原模型分支；PPL 为相对差值。", "",
              "| 步数 | EM 差（百分点） | F1 差（百分点） | NLL 差 | PPL 相对差 | 总参数差 |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in result["matched_step_differences"]:
        lines.append(f"| {row['training_step']} | {row['em_difference_percentage_points']:+.2f} | {row['f1_difference_percentage_points']:+.4f} | {row['nll_difference']:+.6f} | {row['ppl_relative_difference']:+.2%} | {row['parameter_difference_count']:+,} |")
    lines += ["", "两分支训练预算相同，但宽度不同，adapter 参数量并不相同。分别最佳检查点与同 step 对照回答不同的问题，因此两种结果都保留。", "",
              "同一验证集用于检查点选优，不能据此声称 final 表现、统计显著改善或通用能力提升。仍需检查重复输出、抄题和事实退化；助手复核也不等于独立人工验收。全部检查点和报告保留，本轮不追更长训练。", "",
              "## 来源与审计范围", "", f"训练策略 SHA256：`{result['policy']['sha256']}`。这里只核查评测报告记录的实际 base 哈希链，没有重新逐张量审计模型。", "",
              "| 分支 / 步数 | 原始报告 | SHA256 |", "|---|---|---|"]
    for row in result["rows"]:
        lines.append(f"| {row['arm']} / {row['training_step']} | {row['report_path']} | `{row['report_sha256']}` |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--original-base", type=Path, required=True)
    parser.add_argument("--pruned-base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(args.policy, args.original_base, args.pruned_base, args.candidate)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "recovery_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "recovery_summary.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({name: selection["checkpoint"]["training_step"] if selection["checkpoint"] else None
                      for name, selection in result["selected"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
