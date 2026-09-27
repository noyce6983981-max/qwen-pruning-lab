"""第十四课：按监督 token 数累积梯度；只用 CPU 的可微手工例子。"""

import torch


def token_losses(parameter, coefficients):
    """手工平方损失，不是语言模型的交叉熵；在 parameter=0 时值等于系数。"""
    return coefficients * (parameter - 1.0).square()


def run_lesson():
    device = torch.device("cpu")
    dtype = torch.float64
    samples = [torch.full((2,), 2.0, device=device, dtype=dtype),
               torch.full((6,), 1.0, device=device, dtype=dtype)]
    total_tokens = sum(sample.numel() for sample in samples)
    accumulated = torch.nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))
    combined = torch.nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))
    per_example = torch.nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))
    optimizer = torch.optim.SGD([accumulated], lr=0.1)

    print("第十四课：先把一组样本的梯度加起来，组末只更新一次参数。")
    print("本例只用 CPU、一个可训练数和 8 个手工 token 损失，不加载模型或题库。")
    print("样本甲：2 个监督 token，每个 loss=2；样本乙：6 个，每个 loss=1。")
    print("按 token 平均：(2×2 + 6×1) / 8 = 1.25。")
    print("若先按题平均再平均：(2+1) / 2 = 1.5，会给短答案的每个 token 更大权重。")
    print("这里选择每个监督 token 权重相同；按题等权是另一种目标，不能混称同一个平均。")

    optimizer.zero_grad(set_to_none=True)
    initial_parameter = accumulated.detach().clone()
    weighted_loss_value = 0.0
    for index, sample in enumerate(samples, 1):
        mean_loss = token_losses(accumulated, sample).mean()
        weight = sample.numel() / total_tokens
        weighted_loss = mean_loss * weight
        weighted_loss_value += float(weighted_loss.detach())
        weighted_loss.backward()
        if not torch.equal(accumulated.detach(), initial_parameter):
            raise AssertionError("组内只累计梯度，不应更新参数")
        print(f"样本 {index}：题内平均 loss={float(mean_loss.detach()):.2f}，"
              f"乘权重 {sample.numel()}/{total_tokens}；"
              f"累计梯度={float(accumulated.grad):.2f}，参数仍为 {float(accumulated.detach()):.2f}。")

    # 独立计算：把 8 个 token 合在一起，在同一初始参数处求平均和梯度。
    combined_loss = token_losses(combined, torch.cat(samples)).mean()
    combined_loss.backward()
    accumulated_gradient = accumulated.grad.detach().clone()
    torch.testing.assert_close(accumulated_gradient, combined.grad, rtol=0, atol=0)
    if weighted_loss_value != float(combined_loss.detach()):
        raise AssertionError("逐题加权损失应等于合并 token 平均损失")

    # 对照：按题平均得到的是另一个目标，其梯度也不同。
    wrong_loss = torch.stack([token_losses(per_example, sample).mean()
                             for sample in samples]).mean()
    wrong_loss.backward()
    if torch.equal(per_example.grad, accumulated_gradient):
        raise AssertionError("这个特意设计的例子应显示两种权重的梯度不同")

    optimizer.step()  # 两条样本都 backward 完成，才在这里更新这一次。
    expected_parameter = initial_parameter - 0.1 * combined.grad
    torch.testing.assert_close(accumulated.detach(), expected_parameter, rtol=0, atol=0)
    print(f"\n合并 8 个 token：loss={float(combined_loss.detach()):.2f}，"
          f"梯度={float(combined.grad):.2f}。")
    print(f"逐题按 token 权重累计：loss={weighted_loss_value:.2f}，"
          f"梯度={float(accumulated_gradient):.2f}；检查完全相同。")
    print(f"按题等权：loss={float(wrong_loss.detach()):.2f}，"
          f"梯度={float(per_example.grad):.2f}；不是本次选择的 token 平均目标。")
    print(f"组末执行一次 SGD：参数 0.00 → {float(accumulated.detach()):.2f}。")
    print("若中途 optimizer.step()，下一条会在新参数上算梯度，就不再是同一次组内累计。")
    print("\n正式预算：512 条样本，每组 8 条 → 64 次优化更新；完整走过数据一次叫 1 epoch。")
    print("本项目监督 token 总数是 6762（答案+EOS）；资料、问题和 padding 不计入。")
    print("第 16/32/64 次更新后的 checkpoint 是保存参数；gradient checkpointing 是重算中间值以省显存。")
    print("本例没有 dropout，且选用可精确表示的数；真实大 batch 与累积可能因随机性和浮点顺序不同而不逐位相等。")
    print("本次仅证明手工目标下的加权与梯度关系，不是正式恢复训练结果。")
    return {"token_mean_loss": weighted_loss_value,
            "per_example_mean_loss": float(wrong_loss.detach()),
            "accumulated_gradient": float(accumulated_gradient),
            "combined_gradient": float(combined.grad),
            "per_example_gradient": float(per_example.grad),
            "updated_parameter": float(accumulated.detach()),
            "optimizer_steps": 1}


if __name__ == "__main__":
    run_lesson()
