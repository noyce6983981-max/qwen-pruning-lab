# 从零理解这个项目

建议边读边跑下面的小例子。它们只用 CPU，不下载或修改真实模型；做完练习也不是运行项目的前提。先按 README 准备环境，命令都在仓库根目录执行。

| 想理解的问题 | 命令 | 应观察什么 |
| --- | --- | --- |
| 一个神经网络层怎样计算、怎样剪掉中间通道 | `./run.ps1 lesson -WorkRoot ./.work` | 矩阵形状与输出怎样改变 |
| Qwen 的 gate/up/down 怎样联动 | `./run.ps1 lesson6 -WorkRoot ./.work` | 一个通道对应两个行和一个列 |
| EM、F1、PPL 有什么区别 | `./run.ps1 lesson7 -WorkRoot ./.work` | 完全匹配、内容重合和文本预测是不同测量 |
| 权重小的通道一定不重要吗 | `./run.ps1 lesson8 -WorkRoot ./.work` | 激活大小也会影响输出贡献 |
| 激活平方和与有效观察数是什么 | `./run.ps1 lesson9 -WorkRoot ./.work` | 2、−3、1 的平方和为14，观察3次，RMS为√(14/3) |
| LoRA 怎样用小矩阵修正大矩阵 | `./run.ps1 lesson13 -WorkRoot ./.work` | base 不更新，A/B 学到修正；rank不是剪枝比例 |
| 显存不足时怎样累积几题再更新 | `./run.ps1 lesson14 -WorkRoot ./.work` | 按答案 token 数加权，与一次合并计算一致 |

`score_qa.py` 是教学用字符评分；正式实验用 CMRC 官方适配器，两者不能混称。完整定义和实验流程见 [METHODS](METHODS.md) 与 [REPRODUCING](REPRODUCING.md)。

做实验时始终分清三个问题：模型是否真的变小；回答质量能否保住；在实际硬件上是否更省资源或更快。本项目分别用参数审计、质量评测和受控微基准回答，不能用一个数字替代另外两个。
