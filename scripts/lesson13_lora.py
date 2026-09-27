"""第十三课：冻结原权重，只训练 LoRA 修正；四维 CPU 玩具例子。"""

import argparse
import math

import torch


# 可选练习：改为 0.25，让目标的第一个数从 1.5 变成 1.25。
TARGET_SHIFT = 0.5
TOY_STEPS = 10
TOY_LEARNING_RATE = 0.5  # 只适用于此手工回归例子；真实试跑计划用 1e-4。


def format_vector(tensor):
    return "[" + ", ".join(f"{value:.6f}".rstrip("0").rstrip(".")
                           for value in tensor.detach().flatten().tolist()) + "]"


def run_lesson(target_shift=TARGET_SHIFT):
    if not math.isfinite(target_shift) or abs(target_shift) > 0.5:
        raise ValueError("为保持这个固定 10 步例子稳定，目标修正限于 -0.5 到 0.5")
    device = torch.device("cpu")
    dtype = torch.float64
    weight = torch.nn.Parameter(torch.eye(4, device=device, dtype=dtype), requires_grad=False)
    original_weight = weight.detach().clone()
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, dtype=dtype)
    a = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype))
    b = torch.nn.Parameter(torch.zeros((4, 1), device=device, dtype=dtype))
    scale = 1.0

    print("第十三课：LoRA 给原来的计算加上一条可训练的修正支路。")
    print("本课只用 CPU 和手工数字，不加载 Qwen，不读取题库，不运行真实恢复训练。")
    print("W 是 4×4 单位矩阵：输入的四个数原样输出。冻结 W，就是训练时不更新它。")
    print("A 是 1×4，B 是 4×1；r=1 表示修正支路中间只有一个数。")
    print(f"原 W 有 {weight.numel()} 个数；只有 A/B 的 {a.numel() + b.numel()} 个数可以训练。")
    print("训练时仍保留 W；本例总共存 24 个数，不是把总参数量从 16 减到 8。")

    with torch.no_grad():
        old_output = weight @ x
        hand_b = torch.tensor([[0.5], [0.0], [0.0], [0.0]], device=device, dtype=dtype)
        middle = a @ x
        correction = scale * (hand_b @ middle)
        hand_output = old_output + correction
    print("\n先手算一次：A=[1,0,0,0]，B=[0.5,0,0,0]^T，scale=1。")
    print(f"旧输出 W x = {format_vector(old_output)}")
    print(f"A x = {format_vector(middle)}；修正 B(A x) = {format_vector(correction)}")
    print(f"新输出 W x + B(A x) = {format_vector(hand_output)}")

    with torch.no_grad():
        initial_output = weight @ x + scale * (b @ (a @ x))
    initially_equal = torch.equal(initial_output, old_output)
    if not initially_equal:
        raise AssertionError("B 为零时，修正支路必须为零")
    print("\n现在开始训练演示：把 B 的四个数全部设为 0，A 保持上面的初值。")
    print(f"初始输出 = {format_vector(initial_output)}；与旧输出逐值相同：{initially_equal}。")
    print("这表示修正从零开始；不是说训练之后也永远不改变输出。")

    target = old_output.detach().clone()
    target[0] += target_shift
    optimizer = torch.optim.SGD([a, b], lr=TOY_LEARNING_RATE)
    initial_loss = float(torch.mean((initial_output - target).square()))
    losses = []
    print(f"\n训练目标 = {format_vector(target)}。")
    print("本例 loss 是四个数的误差平方后取平均（MSE）；真实语言模型训练用答案 token 的预测损失。")
    print(f"SGD = 每一步根据误差调整参数；本例学习率={TOY_LEARNING_RATE:g}，只演示 {TOY_STEPS} 步。")
    print(f"训练前 MSE = {initial_loss:.8g}")
    for step in range(1, TOY_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        output = weight @ x + scale * (b @ (a @ x))
        loss = torch.mean((output - target).square())
        if not torch.isfinite(loss):
            raise AssertionError("玩具训练出现非有限 loss")
        loss.backward()
        if step == 1:
            print(f"第 1 步更新前：A 梯度全为 0：{bool(torch.all(a.grad == 0))}。")
            if bool(torch.any(b.grad != 0)):
                print("原因是 B 初始为 0；B 先得到更新，后续 A 才能一起学习。")
            else:
                print("当前目标已等于旧输出，A、B 都无需更新。")
        optimizer.step()
        with torch.no_grad():
            output_after = weight @ x + scale * (b @ (a @ x))
            loss_after = float(torch.mean((output_after - target).square()))
        if not math.isfinite(loss_after):
            raise AssertionError("玩具训练出现非有限 loss")
        losses.append(loss_after)
        print(f"第 {step:2d} 步后：MSE={loss_after:.8g}，输出={format_vector(output_after)}")

    weight_unchanged = torch.equal(weight.detach(), original_weight)
    if not weight_unchanged or weight.grad is not None:
        raise AssertionError("冻结的 W 不应变化，也不应累计梯度")
    if target_shift != 0 and not losses[-1] < initial_loss:
        raise AssertionError("这个手工例子的训练应使误差降低")
    print(f"\n检查：W 逐值未变={weight_unchanged}；W 没有梯度={weight.grad is None}。")
    print("W 每一步仍参与前向计算；更新的只有 A 和 B。")
    print("这里只让一个手工输入接近一个手工目标，不说明 Qwen 的回答质量已经恢复。")
    print("可选：把 TARGET_SHIFT 改为 0.25，或加 --target-shift 0.25；无需完成练习才能继续。")
    return {"initial_loss": initial_loss, "final_loss": losses[-1], "losses": losses,
            "initially_equal": initially_equal, "weight_unchanged": weight_unchanged,
            "final_output": output_after.detach().tolist(), "target": target.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-shift", type=float, default=TARGET_SHIFT,
                        help="目标第一个数的改变量；此固定小例子限 -0.5 到 0.5")
    args = parser.parse_args()
    if not math.isfinite(args.target_shift) or abs(args.target_shift) > 0.5:
        parser.error("目标修正必须是 -0.5 到 0.5 之间的有限数")
    run_lesson(args.target_shift)


if __name__ == "__main__":
    main()
