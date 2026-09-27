"""用 CPU 小模型观察 Qwen 式 gate/up/down 连接；权重是手工设置的。"""

import copy

import torch
import torch.nn.functional as F
import torch_pruning as tp
from torch import nn


class TinyGatedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(2, 3, bias=False)
        self.gate_proj = nn.Linear(2, 3, bias=False)
        self.down_proj = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            # up 沿用前面小实验的三组权重。
            self.up_proj.weight.copy_(torch.tensor([
                [0.8, -0.6], [0.03, 0.02], [0.4, 0.5],
            ]))
            self.gate_proj.weight.copy_(torch.tensor([
                [1.0, 0.0], [0.0, 0.1], [0.5, 0.5],
            ]))
            self.down_proj.weight.copy_(torch.tensor([
                [0.7, 0.1, -0.5], [-0.2, 0.3, 0.6],
            ]))

    def stages(self, x):
        up = self.up_proj(x)
        gate_raw = self.gate_proj(x)
        gate = F.silu(gate_raw)
        middle = gate * up  # 相同位置相乘，不是矩阵乘法。
        output = self.down_proj(middle)
        return up, gate_raw, gate, middle, output

    def forward(self, x):
        return self.stages(x)[-1]


def values(tensor):
    return [round(value, 6) for value in tensor[0].tolist()]


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def main():
    torch.set_num_threads(1)
    x = torch.tensor([[1.0, 1.0]])
    # 这里按人的习惯从 1 开始编号，可以改成 1 或 3，再观察输出变化。
    remove_channel = 2
    if remove_channel not in (1, 2, 3):
        raise ValueError("remove_channel 必须是 1、2 或 3")

    original = TinyGatedMLP().eval()
    pruned = copy.deepcopy(original)
    removed = [remove_channel - 1]
    # 中间同一个位置连着三块矩阵：两个投影的输出行、一个投影的输入列。
    tp.prune_linear_out_channels(pruned.gate_proj, idxs=removed)
    tp.prune_linear_out_channels(pruned.up_proj, idxs=removed)
    tp.prune_linear_in_channels(pruned.down_proj, idxs=removed)

    with torch.inference_mode():
        up, gate_raw, gate, middle, before = original.stages(x)
        pruned_stages = pruned.stages(x)
        after = pruned_stages[-1]

    print("第六课：同一个输入，经过 gate 和 up 两条路，再汇合。")
    print("本例只展示一个 MLP 小模块，使用手工权重，不是完整语言模型。")
    print("\n输入：", values(x))
    print("up： 2 → 3，加权求和：", values(up))
    print("gate：2 → 3，加权求和：", values(gate_raw))
    print("gate 再经过 SiLU 激活：", values(gate))
    print("按位置相乘：", values(middle))
    print("down：3 → 2，得到输出：", values(before))
    print("\nSiLU 和上一课的 ReLU 是不同的激活函数。")
    print("gate 是连续数值的调节，不是只有 0/1 的开关，也不是剪枝评分。")

    print(f"\n现在删除中间第 {remove_channel} 个通道（Python 索引 {removed[0]}）：")
    print("gate 权重删对应行；up 权重删对应行；down 权重删对应列。")
    for name in ("gate_proj", "up_proj", "down_proj"):
        print(f"{name} 权重形状：{list(getattr(original, name).weight.shape)} → {list(getattr(pruned, name).weight.shape)}")
    print("中间的数：", values(middle), "→", values(pruned_stages[3]))
    print("最终输出：", values(before), "→", values(after))
    print("每个输出的变化绝对值：", values((after - before).abs()))
    print(f"参数个数：{count_parameters(original)} → {count_parameters(pruned)}")
    print("输入仍是 2 个数，输出仍是 2 个数；减少的是中间通道。")
    print("\n真实 Qwen3-1.7B 的对应尺寸：")
    print("gate、up：各自 2048 → 6144；down：6144 → 2048。")
    print("这些是每个 token 的特征维度，不是输入文字个数。")
    print("本例仅比较单个输入的输出变化和参数量，没有测准确率或加速。")


if __name__ == "__main__":
    main()
