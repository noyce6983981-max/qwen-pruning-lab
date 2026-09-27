# Qwen Pruning Lab

**在一张 8GB 显卡上，完成 Qwen3-1.7B 的结构化剪枝、有限恢复训练和质量／资源对照。**

这是一个有明确支持范围的中文学习与实验项目：提供校准、通道评分、MLP 裁剪、LoRA 恢复、合并导出、质量评测、自定义输入与资源测试工具。保留失败结果，不把参数下降直接说成速度提升。

## 本轮得到什么

28 层 MLP 宽度从 6144 减至 5888，实际参数 **1,720,574,976 → 1,676,534,784，减少 2.56%**。两边各用 512 题、64 次更新训练；根据独立 validation 选择 checkpoint 后合并，最后一次性检验固定的四个产物。

**最终留出集：200 题；PPL 使用 16,384 个有效预测目标。** 这里是长度筛选的 CMRC2018 本地子集，不是官方隐藏测试。

| 固定产物 | 参数减少 | EM ↑ | F1 ↑ | PPL ↓ |
| --- | ---: | ---: | ---: | ---: |
| 原模型，未训练 | 0 | 67.50 | 84.7006 | 22.7745 |
| 校准剪枝 5888，未恢复 | 2.56% | 63.50 | 82.7126 | 23.7624 |
| 原模型，64 步 LoRA 后合并 | 0 | 67.00 | 86.0512 | 21.0195 |
| 校准剪枝 5888，64 步恢复后合并 | 2.56% | 61.00 | 83.3852 | 21.6728 |

剪后恢复版相对原模型 F1 降低 **1.3154 个百分点**、PPL 降低 **4.84%**，通过事先规定的 F1/PPL 数值门槛；但相对同样训练 64 步的未剪模型，F1 低 **2.6660 个百分点**，EM 低 **6 个百分点**。成果是完成可审计的小幅压缩与对照流程，不能解释为剪枝提升了通用能力或已通过工业部署验收。

完整的速度、显存、失败候选、回答错误及测试证据见 [实验报告](docs/RESULTS.md)。[PDF 成果海报](reports/qwen-pruning-progress-poster.pdf) 可用于一页汇报；[公开指标 JSON](reports/metrics.json) 保留完整精度和原始来源报告 SHA256。

## 先运行 CPU 小例子

实际模型流程已测 Windows x64、Python 3.12、RTX 4060 Laptop 8GB、16GB 系统内存。固定 Torch 2.9.1+cu126、Transformers 4.57.6、PEFT 0.17.1；代码仓库不含模型和环境。

在仓库根目录的 PowerShell 执行：

```powershell
$taskPython312 = (& py -3.12 -c 'import sys; print(sys.executable)').Trim()
.\setup.ps1 -PythonExecutable $taskPython312 -WorkRoot .\.work
$taskPython = '.\.work\qwen-env\Scripts\python.exe'
& $taskPython -X utf8 third_party/cmrc2018/fetch_punkt.py
.\run.ps1 test -WorkRoot .\.work
.\run.ps1 lesson8 -WorkRoot .\.work
```

初次安装包含 PyTorch，不是轻量下载。若没有 `py`，为 `$taskPython312` 填入实际的 Python 3.12 路径。CPU 单元测试用临时小模型；干净仓库会明确跳过六项依赖完整真实数据的检查。CI 运行 CPU 测试，不运行完整 GPU 实验。

只想在 Linux 做 CPU 检查：先从 `https://download.pytorch.org/whl/cpu` 安装 `torch==2.9.1`，再安装 `requirements-cpu.txt`，准备 Punkt 后运行 `python -m unittest discover -s tests -v`。这不表示真实模型工具支持 CPU 推理。

## 使用自己的题目或校准资料

题库每行一个 JSON，可带资料与多个参考答案。没有参考答案也能生成，明确不参与 EM/F1 评分：

```json
{"id":"demo_1","context":"虚构通知：活动周三下午两点开始。","question":"活动什么时候开始？","answers":["周三下午两点"]}
```

下载固定原模型后运行三道自编示例：

```powershell
.\run.ps1 download -WorkRoot .\.work
.\run.ps1 questions -WorkRoot .\.work -CheckpointKind original -QuestionsPath .\examples\custom_questions.jsonl -OutputDir .\results\my_questions
```

用实际剪后／合并目录时，显式传 `-ModelDir` 及正确的 `-CheckpointKind`。不要把只有 adapter 的目录当完整模型。详见 [自定义题库](CUSTOM_QUESTIONS.md)；自己的长文本可通过 [自定义校准](CUSTOM_CALIBRATION.md) 进入既有采集、评分和剪枝流程。改变语料后应重新评测，不能套用本轮分数。

## 从学习到完整复现

- [指标和原理入门](docs/LEARNING.md)：权重、激活、RMS、LoRA 与梯度累积的小例子。
- [方法与评测边界](docs/METHODS.md)：数据隔离、官方评分、有限恢复、公平对照、合并舍入和测速口径。
- [完整复现流程](docs/REPRODUCING.md)：从空环境到校准剪枝、两边恢复、验证选择、合并、最终评测和资源测试。
- [如何记录实际答案复核](REVIEW_GUIDE.md)：记录真实观察，不伪造人工验收。
- [完整结果](docs/RESULTS.md)：包含失败的幅值剪枝、更重剪枝与全部恢复 checkpoint。

脚本位于 `scripts/`，CPU 测试位于 `tests/`，人工编写的小输入位于 `examples/`。各工具有 `--help`。真实运行输出保存在本地新目录，包含 manifest、来源 SHA、逐题原文和机器可读报告；拒绝覆盖旧结果。

## 范围和许可

当前固定恢复配方为校准 5888、每边 64 步；导出与发布流程只支持两边 validation 都选择 step64 的情形。它不是任意架构、任意预算的通用剪枝框架。量化、更小原生模型与具体硬件部署是后续独立对照，本版未实现这些实验。

自有代码和文档使用 [MIT](LICENSE)；CMRC、Qwen、NLTK 等来源及例外见 [第三方说明](THIRD_PARTY_NOTICES.md)。Punkt 数据缓存许可未明确，不随仓库再分发；固定下载器负责在本地获取和校验。权重、数据缓存、个人路径及完整实验目录均不入库。公开报告是明确标注的派生摘要，不冒充原始哈希绑定记录。
