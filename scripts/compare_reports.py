"""Compare original/pruned evaluation reports only when their settings match."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def require_equal(label, before, after):
    if before != after:
        raise ValueError(f"Cannot compare: {label} differs: {before!r} vs {after!r}")


def escape(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--pruned", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.original.read_text(encoding="utf-8"))
    pruned = json.loads(args.pruned.read_text(encoding="utf-8"))
    for record in (original, pruned):
        require_equal("run status", record["status"], "completed")
    require_equal("original checkpoint type", original["checkpoint_kind"], "original")
    require_equal("pruned checkpoint type", pruned["checkpoint_kind"], "pruned")
    require_equal("source revision", original["source_model_revision"], pruned["source_model_revision"])
    require_equal("evaluation script hash", original["runner_sha256"], pruned["runner_sha256"])
    for key in ("packages", "gpu_name", "selected_dtype", "torch_cuda_build", "tf32_matmul_allowed"):
        require_equal(key, original["environment"][key], pruned["environment"][key])
    for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "attention_implementation"):
        require_equal(key, original["model"][key], pruned["model"][key])
    b, p = original["benchmark"], pruned["benchmark"]
    for key in ("batch_size", "input_tokens", "output_tokens", "warmups", "repeats", "fixture_token_ids", "eos_policy", "first_token_definition", "decode_definition", "memory_definition"):
        require_equal(key, b[key], p[key])
    require_equal("demo count", len(original["demos"]), len(pruned["demos"]))
    demos = []
    for before, after in zip(original["demos"], pruned["demos"]):
        for key in ("id", "document", "question", "input_tokens", "max_new_tokens", "enable_thinking", "do_sample", "use_model_defaults"):
            require_equal(f"{before['id']}: {key}", before[key], after[key])
        demos.append({"question": before["question"], "original": before["answer"], "pruned": after["answer"], "text_identical": before["answer"] == after["answer"]})
    n0 = original["model"]["unique_parameter_count"]
    n1 = pruned["model"]["unique_parameter_count"]
    require_equal("source parameter count", n0, pruned["pruning"]["parameters_before"])
    require_equal("pruned parameter count", n1, pruned["pruning"]["parameters_after"])
    ranges = {
        "original": [min(run["decode_tokens_per_second"] for run in b["runs"]), max(run["decode_tokens_per_second"] for run in b["runs"])],
        "pruned": [min(run["decode_tokens_per_second"] for run in p["runs"]), max(run["decode_tokens_per_second"] for run in p["runs"])],
    }
    overlap = max(ranges["original"][0], ranges["pruned"][0]) <= min(ranges["original"][1], ranges["pruned"][1])
    comparison = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings_match": True,
        "original_report": str(args.original.resolve()),
        "pruned_report": str(args.pruned.resolve()),
        "method": pruned["pruning"]["method"],
        "recovery_applied": pruned["pruning"]["recovery_applied"],
        "mlp_width_before": original["model"]["intermediate_size"],
        "mlp_width_after": pruned["model"]["intermediate_size"],
        "parameters_before": n0,
        "parameters_after": n1,
        "parameters_removed": n0 - n1,
        "total_parameter_reduction_percent": (1 - n1 / n0) * 100,
        "median_decode_tokens_per_second_before": b["median_decode_tokens_per_second"],
        "median_decode_tokens_per_second_after": p["median_decode_tokens_per_second"],
        "observed_median_decode_rate_change_percent": (p["median_decode_tokens_per_second"] / b["median_decode_tokens_per_second"] - 1) * 100,
        "observed_decode_ranges": ranges,
        "ranges_overlap": overlap,
        "demos": demos,
        "limitations": "Sequential short-sequence microbenchmark, not controlled proof of a stable speedup; temperature, frequency and background work can vary. Three manually authored demos are not an accuracy evaluation. No recovery training.",
    }
    gib = 1024 ** 3
    lines = [
        "# 第一次真实 Qwen MLP 剪枝", "",
        "使用同一套脚本和相同设置，比较原模型与剪枝模型。本轮没有恢复训练。", "",
        "## 剪了什么", "",
        f"在每层 MLP 中，从 {comparison['mlp_width_before']} 个中间通道保留 {comparison['mlp_width_after']} 个。将同一通道连接的 gate、up 和 down 权重平方相加，删除分数最低的通道。三块权重同步裁剪，注意力与残差宽度保持不变。这是简单的权重幅值基线，没有使用校准数据或梯度。", "",
        "| 项目 | 原模型 | 剪枝模型 |", "|---|---:|---:|",
        f"| 每层 MLP 中间宽度 | {comparison['mlp_width_before']} | {comparison['mlp_width_after']} |",
        f"| 独立参数个数 | {n0:,} | {n1:,} |",
        f"| 独立参数字节数（GiB） | {original['model']['unique_parameter_storage_bytes'] / gib:.3f} | {pruned['model']['unique_parameter_storage_bytes'] / gib:.3f} |",
        f"| 测速阶段峰值分配显存（GiB） | {b['max_peak_allocated_bytes'] / gib:.3f} | {p['max_peak_allocated_bytes'] / gib:.3f} |",
        f"| 测速阶段峰值保留显存（GiB） | {b['max_peak_reserved_bytes'] / gib:.3f} | {p['max_peak_reserved_bytes'] / gib:.3f} |",
        f"| 首 token 耗时中位数（ms） | {b['median_first_token_seconds'] * 1000:.2f} | {p['median_first_token_seconds'] * 1000:.2f} |",
        f"| 后续生成速度中位数（token/s） | {b['median_decode_tokens_per_second']:.2f} | {p['median_decode_tokens_per_second']:.2f} |",
        f"| 各轮生成速度范围（token/s） | {ranges['original'][0]:.2f}～{ranges['original'][1]:.2f} | {ranges['pruned'][0]:.2f}～{ranges['pruned'][1]:.2f} |", "",
        f"共减少 **{n0 - n1:,} 个参数（{comparison['total_parameter_reduction_percent']:.2f}%）**。通道剪枝比例与整个模型的参数减少比例不同。", "",
        "原下载权重与新保存权重可能采用不同的共享张量去重和分片方式，因此这里比较独立参数字节数，不把两个目录的文件大小差异直接当成剪枝收益。显存仅为 PyTorch 分配器口径，不含全部驱动或其他程序占用。", "",
        "## 回答有没有变化", "",
        "| 问题 | 原模型回答 | 剪后回答 |", "|---|---|---|",
    ]
    lines += [f"| {escape(d['question'])} | {escape(d['original'])} | {escape(d['pruned'])} |" for d in demos]
    lines += [
        "", "这些是三个自编的简单资料问答例子，仅用于查看运行行为。回答相同不能证明整体能力没有下降；回答不同也需要结合资料核查正确性。正式质量评价需要独立测试数据。", "",
        "## 测量条件与结论边界", "",
        f"{original['environment']['gpu_name']}；{original['environment']['selected_dtype']}；同一注意力后端；批量 {b['batch_size']}；固定 {b['input_tokens']} 个输入 token 和 {b['output_tokens']} 个输出 token；预热 {b['warmups']} 次，重复 {b['repeats']} 次。显式逐 token 贪心解码、使用 KV 缓存并进行 CUDA 同步。问答关闭思考与随机采样。", "",
        ("两组速度范围存在重叠，本轮不足以认定稳定加速。" if overlap else "本轮两组速度范围没有重叠，但单次顺序测量仍不足以证明普遍、稳定的加速。") + "后续需在更多长度与重复会话中验证。", "",
        "接下来应先使用独立文本或任务数据比较质量，再决定是否采用校准数据评分和恢复训练，不根据这三个演示问题挑选策略。", "",
        f"原模型记录：[baseline.json]({args.original.resolve().as_posix()})", "",
        f"剪枝模型记录：[baseline.json]({args.pruned.resolve().as_posix()})", "",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
