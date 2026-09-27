"""第八课：只看 MLP 最后一次加权求和；手工数字，CPU，无模型加载。"""

import argparse
import math

# 先只改这一行：把 100.0 改为 0.1，运行后观察删哪个通道影响更小。
ACTIVATION_A = 100.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-a", type=float, default=ACTIVATION_A,
                        help="可选：命令行覆盖 A 的激活值；不改变脚本中的默认值")
    args = parser.parse_args()
    if not math.isfinite(args.activation_a):
        parser.error("激活值必须是有限数")

    weight_a, weight_b = 0.1, 1.0
    activation_a, activation_b = args.activation_a, 1.0
    contribution_a = weight_a * activation_a
    contribution_b = weight_b * activation_b
    original = contribution_a + contribution_b
    # 删除一项后的输出：留下另一项。本例没有真正裁剪 Qwen 权重。
    without_a, without_b = contribution_b, contribution_a
    difference_a, difference_b = abs(contribution_a), abs(contribution_b)
    preferred = "A" if difference_a < difference_b else "B" if difference_b < difference_a else "两者一样"

    print("第八课：权重小，不一定说明删除这个通道影响小。")
    print("这里只看两个已经算好的通道值如何经过最后的加权求和。")
    print("权重 = 这次计算中固定的乘法系数；激活值 = 通道遇到输入时算出的数。")
    print(f"\nA：权重 {weight_a:g} × 激活值 {activation_a:g} = 对输出的贡献 {contribution_a:g}")
    print(f"B：权重 {weight_b:g} × 激活值 {activation_b:g} = 对输出的贡献 {contribution_b:g}")
    print(f"原输出 = {contribution_a:g} + {contribution_b:g} = {original:g}")
    print(f"\n删 A：输出 {without_a:g}，相对原输出的绝对变化 {difference_a:g}")
    print(f"删 B：输出 {without_b:g}，相对原输出的绝对变化 {difference_b:g}")
    print("只按此处输出权重的绝对值判断，会倾向删 A，因为 0.1 < 1。")
    print(f"但在这一个输入上，希望输出少变时：{preferred} 更适合删除。" if preferred != "两者一样"
          else "在这一个输入上，删 A 或 B 对输出影响一样大。")
    print("\n练习：只把 ACTIVATION_A 从 100.0 改成 0.1，再运行并观察。")
    print("这里的输出变化不是 EM/F1，也不是‘输出越大越好’。")
    print("只看一个输入不能决定真实剪枝；下一步要观察一批有代表性的文本。")
    print("原实验的幅值评分综合了 gate/up/down 三组权重，本例不是该评分的完整复现。")


if __name__ == "__main__":
    main()
