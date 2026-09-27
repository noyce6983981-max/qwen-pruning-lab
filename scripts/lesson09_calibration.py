"""第九课：用多个手工输入汇总激活大小；标准 Python，CPU，无真实模型。"""
import argparse
import math

# 可选练习：把第二个输入的 B 激活从 20.0 改成 1.0，再观察排序。
B_SECOND_ACTIVATION = 20.0


def rms(values):
    """均方根：先平方，再平均，最后开方；用于概括一批值的大小。"""
    # hypot also avoids overflowing while squaring a large but finite input.
    return math.hypot(*values) / math.sqrt(len(values))


def describe_choice(score_a, score_b):
    if math.isclose(score_a, score_b, rel_tol=1e-12, abs_tol=1e-12):
        return "两者同分"
    return "倾向删除 A" if score_a < score_b else "倾向删除 B"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-second", type=float, default=B_SECOND_ACTIVATION)
    args = parser.parse_args()
    if not math.isfinite(args.b_second):
        parser.error("激活值必须是有限数")
    weights = (0.1, 1.0)
    samples = [(100.0, 1.0), (0.1, args.b_second)]
    contributions = [(weights[0] * a, weights[1] * b) for a, b in samples]

    print("第九课：不要只看一个输入；用一批资料观察通道。")
    print("权重固定为 A=0.1、B=1。激活值在不同输入之间可以变化。")
    print("这里把通道的激活值直接写成手工数字，不加载模型或读取实验题库。")
    for index, ((a, b), (ca, cb)) in enumerate(zip(samples, contributions), 1):
        print(f"\n输入 {index}：激活 A={a:g}、B={b:g}")
        print(f"输出贡献 A={ca:g}、B={cb:g}；原输出={ca + cb:g}")
        print(f"单看这次输出变化：{describe_choice(abs(ca), abs(cb))}")

    activation_rms = [rms([row[channel] for row in samples]) for channel in (0, 1)]
    scores = [abs(weights[channel]) * activation_rms[channel] for channel in (0, 1)]
    deletion_rms = [rms([row[channel] for row in contributions]) for channel in (0, 1)]
    print("\n现在把两次观察汇总，每个输入的权重相同。")
    print("RMS（均方根）= 先平方 → 取平均 → 开方。")
    print(f"A 激活 RMS = sqrt((100² + 0.1²) / 2) = {activation_rms[0]:.6f}")
    print(f"B 激活 RMS = sqrt((1² + ({args.b_second:g})²) / 2) = {activation_rms[1]:.6f}")
    print("\n本课通道分数 = 输出权重的绝对值 × 激活 RMS。")
    for name, weight, activation, score, difference in zip("AB", weights, activation_rms, scores, deletion_rms):
        print(f"{name}：{abs(weight):g} × {activation:.6f} = {score:.6f}；删掉它的输出变化 RMS={difference:.6f}")
    print(f"按这批输入的分数：{describe_choice(*scores)}。")
    print("这个选择允许某个输入变差，追求的是这批输入上的整体输出变化较小。")
    print("\n校准数据：用于观察这些激活、帮助选通道的一批有代表性的资料。")
    print("本步骤只收集统计，不更新权重；校准资料不与验证题或最终题混用。")
    print("本例只有一个输出、每次删一个通道，分数才恰好等于输出变化 RMS。")
    print("真实 Qwen 要汇总一列 down 权重，后续还有其他层，分数不是 EM/F1，也不保证剪后质量。")
    print("可选练习：把 B_SECOND_ACTIVATION 改为 1.0，观察换输入后排序是否改变。")


if __name__ == "__main__":
    main()
