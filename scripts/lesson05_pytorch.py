"""把前面的 2 → 3 → 1 小网络写成 PyTorch；仅在 CPU 上运行。"""

import torch
from torch import nn


def main():
    # 外面的一层 [] 表示一批样本：现在只有 1 个样本，每个样本有 2 个数。
    x = torch.tensor([[1.0, 1.0]])

    hidden_layer = nn.Linear(2, 3, bias=False)
    output_layer = nn.Linear(3, 1, bias=False)

    # 设置已知权重，不进行训练。与 lesson02 的权重完全相同。
    with torch.no_grad():
        hidden_layer.weight.copy_(torch.tensor([
            [0.8, -0.6],
            [0.03, 0.02],
            [0.4, 0.5],
        ]))
        output_layer.weight.copy_(torch.tensor([[0.7, 0.1, -0.5]]))

        weighted_sums = hidden_layer(x)
        hidden = torch.relu(weighted_sums)
        output = output_layer(hidden)

    print("这次用 PyTorch 算前面的同一个小网络。")
    print("输入：", x.tolist())
    print("输入形状：", list(x.shape), "= [样本个数, 每个样本的输入数]")
    print("第一层权重形状：", list(hidden_layer.weight.shape), "= [输出数, 输入数]")
    print("第一层就是：输入 × 权重的转置（本例没有偏置）")
    print("加权求和：", [round(v, 4) for v in weighted_sums[0].tolist()])
    print("经过 ReLU：", [round(v, 4) for v in hidden[0].tolist()])
    print("隐藏层形状：", list(hidden.shape))
    print("第二层权重形状：", list(output_layer.weight.shape))
    print(f"最后输出：{output.item():.4f}")
    print("\n对应关系：")
    print("nn.Linear(2, 3)：每个样本的 2 个数变为 3 个数。")
    print("nn.Linear(3, 1)：每个样本的 3 个数变为 1 个数。")
    print("张量可以先理解成带形状的数字表，PyTorch 帮我们完成整批运算。")
    print("\n动手：把输入改成 [[1.0, 2.0]]，看第一个隐藏神经元是否变为 0。")


if __name__ == "__main__":
    main()
