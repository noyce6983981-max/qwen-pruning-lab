"""Turn a completed baseline JSON into a readable Chinese report (stdlib only)."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    record = json.loads(args.report.read_text(encoding="utf-8-sig"))
    if record["status"] != "completed":
        raise RuntimeError("Only completed baseline reports can be summarized")
    benchmark = record["benchmark"]
    is_pruned = record.get("checkpoint_kind") == "pruned"
    gib = 1024 ** 3
    peak_ram = record["host_memory_at_exit"]["process_peak_working_set_bytes"]
    lines = [
        "# 本地模型评测记录", "",
        ("本轮评测 Qwen3-1.7B 的 MLP 剪枝版本，尚未恢复训练。" if is_pruned else
         "本轮评测原始 Qwen3-1.7B，未对该检查点进行剪枝或训练。") + "已完成中文问答演示及固定长度测速。", "",
        f"运行时间（UTC）：{record['created_at_utc']}。", "",
        "## 设备与结果", "",
        "| 项目 | 本次实测 |", "|---|---|",
        f"| GPU | {record['environment']['gpu_name']} |",
        f"| 模型独立参数量 | {record['model']['unique_parameter_count']:,}，约 {record['model']['unique_parameter_count'] / 1e8:.2f} 亿 |",
        f"| 权重精度 / 注意力实现 | {record['environment']['selected_dtype']} / {record['model']['attention_implementation']} |",
        f"| 运行进程内存峰值，含加载 | {peak_ram / gib:.2f} GiB |",
        f"| 加载阶段 PyTorch 峰值分配显存 | {record['load_peak_gpu_allocated_bytes'] / gib:.2f} GiB |",
        f"| 测速阶段 PyTorch 峰值分配显存 | {benchmark['max_peak_allocated_bytes'] / gib:.2f} GiB |",
        f"| 测速阶段 PyTorch 峰值保留显存 | {benchmark['max_peak_reserved_bytes'] / gib:.2f} GiB |",
        f"| 首 token 时间，中位数 | {benchmark['median_first_token_seconds'] * 1000:.1f} ms |",
        f"| 后续生成速度，中位数 | {benchmark['median_decode_tokens_per_second']:.2f} token/s |", "",
        "内存是 Windows 报告的进程工作集峰值，不等于对整机的最低内存要求。显存是 PyTorch 分配器统计，未包含全部驱动和其他程序占用；系统看到的显卡总用量会更高。训练和长文本的资源需求需另外实测。", "",
        f"测速条件：批量 {benchmark['batch_size']}，{benchmark['input_tokens']} 个输入 token，{benchmark['output_tokens']} 个输出 token；预热 {benchmark['warmups']} 次，重复 {benchmark['repeats']} 次。首 token 时间包含输入处理与首次选词，不含模型加载和分词；后续速度统计余下 {benchmark['output_tokens'] - 1} 个 token。固定长度测速忽略结束符，独立于聊天演示。", "",
        "## 三个自编问答演示", "",
    ]
    for demo in record["demos"]:
        lines += [
            f"### {demo['id']}", "",
            f"资料：{demo['document']}", "",
            f"问题：{demo['question']}", "",
            f"回答：**{demo['answer']}**", "",
        ]
    lines += [
        "这三个简单例子仅证明加载与生成链路可用，不是代表性评测集，也不是模型准确率。问答关闭思考模式与随机采样，后续比较需保持设置一致。", "",
        "## 下一步", "",
        "把原模型与剪枝模型在相同条件下比较；引入独立数据开展正式质量评测后，再研究校准评分和恢复训练。", "",
        "## 可复现记录", "",
        f"- 原始结果：[baseline.json]({args.report.resolve().as_posix()})",
        f"- 模型提交：`{record['source_model_revision']}`",
        f"- 实际运行脚本 SHA256：`{record.get('runner_sha256', '未记录')}`",
        "- 版本：" + ", ".join(f"{key} {value}" for key, value in record["environment"]["packages"].items()),
        "",
    ]
    destination = args.report.with_name("baseline_summary.md")
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "summary": str(destination),
        "peak_process_ram_gib": round(peak_ram / gib, 2),
        "load_peak_gpu_gib": round(record["load_peak_gpu_allocated_bytes"] / gib, 2),
        "inference_peak_gpu_gib": round(benchmark['max_peak_allocated_bytes'] / gib, 2),
        "median_first_token_ms": round(benchmark['median_first_token_seconds'] * 1000, 1),
        "median_decode_tokens_per_second": round(benchmark['median_decode_tokens_per_second'], 2),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
