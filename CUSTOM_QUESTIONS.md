# 用自己的题目检查模型

`scripts/evaluate_questions.py`读取本地JSONL题库，逐题生成答案，保存JSON和Markdown报告。它支持本项目已经验证身份的原模型、未恢复剪枝模型，以及最终合并导出的LoRA模型；不支持只含adapter的小目录，也不承诺任意Hugging Face架构通用。

这个入口不读取项目的固定CMRC验证或最终题库。自定义结果单独标记为`custom`，不能混作固定数据集成绩；它也不计算PPL，不把单题生成耗时当作稳定测速。

## 先看三道示例

[examples/custom_questions.jsonl](examples/custom_questions.jsonl) 是项目自编的三条虚构资料，不引用第三方题目。前两条附有参考答案，第三条故意不提供参考，演示“可以生成，但不打分”。三道演示不能证明模型的通用能力。

JSONL的意思是“一行一个JSON对象”。下面这条必须放在文件的一整行：

```json
{"id":"my_1","context":"虚构通知：活动周三下午两点开始。","question":"活动什么时候开始？","answers":["周三下午两点","星期三下午两点"]}
```

| 字段 | 要求 | 作用 |
|---|---|---|
| `id` | 必填，不重复的非空字符串，首尾不带空格 | 把同一道题的答案和分数对应起来 |
| `question` | 必填，非空字符串 | 你希望模型回答的问题 |
| `context` | 可省略；提供时必须为字符串 | 给模型看的资料。没有资料或资料为空时使用普通简短问答提示 |
| `answers` | 可省略或为`[]`；提供时为非空字符串的列表 | 人工准备的参考答案。多个答案按官方评分规则取最佳匹配 |

只接受这四个字段，拼错字段会报错；不把`answers`里的参考内容放入模型提示。`answers: null`、空白参考、重复ID、重复JSON键和空题库都会报错。空行允许跳过，文件使用UTF-8，Windows的UTF-8 BOM也可读取。

资料中的换行写成JSON字符串内的`\n`，不要把一个JSON对象拆成多行。通常先准备几题，检查输出，再扩大题库。

## 运行原模型

在项目目录打开PowerShell。以下命令只读取本地权重，实际生成需要支持BF16的CUDA显卡；输出使用新的时间戳目录。

```powershell
$taskPython = '.\.work\qwen-env\Scripts\python.exe'
$taskOutput = Join-Path '.\results' ('custom_original_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& $taskPython -X utf8 .\scripts\evaluate_questions.py --model-dir '.\.work\models\Qwen3-1.7B' --checkpoint-kind original --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --questions '.\examples\custom_questions.jsonl' --output-dir $taskOutput
if ($LASTEXITCODE -ne 0) { throw 'Custom question evaluation failed; inspect the new report directory.' }
```

将`--questions`改成你自己的JSONL路径即可。示例使用 setup.ps1 -WorkRoot .\.work 的布局；剪后目录替换成你本轮实际生成的路径。

## 换成剪后或合并模型

脚本要求显式写明模型身份，不会在验证失败时自动猜成另一种模型。

| `--checkpoint-kind` | `--model-dir`应指向 | 目录里的身份文件 |
|---|---|---|
| `original` | 原Qwen3完整权重目录 | `download_manifest.json` |
| `pruned` | 本项目生成的未恢复剪枝权重目录 | `pruning_manifest.json` |
| `merged_lora` | 本项目最终合并导出的完整模型目录 | `merge_manifest.json` |

校准5888的未恢复模型示例：

```powershell
$taskOutput = Join-Path '.\results' ('custom_pruned_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& $taskPython -X utf8 .\scripts\evaluate_questions.py --model-dir '.\.work\models\Qwen3-1.7B-calibrated-mlp5888' --checkpoint-kind pruned --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --questions '.\examples\custom_questions.jsonl' --output-dir $taskOutput
if ($LASTEXITCODE -ne 0) { throw 'Custom question evaluation failed.' }
```

合并模型使用同一命令形式，把`--checkpoint-kind`改为`merged_lora`，并将`--model-dir`设为实际导出目录。

`--revision`始终填写原始模型的固定源提交；它不是合并后的文件SHA。脚本分别调用已有base身份校验或`merged_checkpoint.merged_model_provenance`核查实际文件与来源链，报告记录真实文件SHA。不要把未合并的`step-0064` adapter目录当成完整模型，也不要给合并模型伪造旧download／pruning manifest。

## 生成设置与长度

默认非思考、贪心、batch=1、seed=42，完整BF16权重、SDPA、普通BF16推理，不启用autocast，不合并或修改任何权重。生成设置独立构造，`use_model_defaults=False`；不接纳模型默认的采样配置。

带资料的系统提示要求只根据资料给出简短答案，资料未给出时回答“资料未提供”；没有资料时要求简短回答，不确定则明确说明。报告保留完整messages和prompt token IDs。模型回答只去首尾空白，不删除解释、重复句或错误前缀来“修好分数”。

默认完整提示上限448 token、输出上限64 token，总预算512。完整提示包含系统提示和聊天模板，不能仅按资料文字数估算。如果任何一题超长，脚本在加载模型前报出题目ID和token数，整次不生成，不偷偷截断。请主动缩短资料或拆题。

可显式设置`--max-input-tokens`和`--max-new-tokens`，但两者之和必须≤512、输出范围为1–64。例如输出32时输入最多480。这仍是本项目的短文本支持范围，不代表任意长上下文都经过测试。可选`--device`默认为0。

## 怎样读输出

每次新目录包含：

- `predictions.jsonl`：每生成一题立即落盘，保存原始回答、含特殊token版本、生成token IDs、停止原因、提示、参考与耗时。
- `questions_report.json`：完整设置、题库与代码SHA、tokenizer签名、模型来源和参数量、逐题答案及评分覆盖数。
- `questions_report.md`：方便阅读的答案表与有参考子集的EM/F1。

EM是参考文本匹配程度，F1按项目固定的CMRC官方规则计算；两者范围0–100。只有带参考答案的题参与均值，无参考题的EM/F1为`null`，Markdown显示“—”。整个题库都无参考时，均值也为空，不伪造为0分。原始输出为空则是一个真实生成结果，有参考时照常得到相应分数，不会从分母里删除。

例如10题中只有6题有参考，报告会写“评分6题、未评分4题”。这6题的均分不能代表全部10题。开放式回答可能存在同义表述；高文字匹配也不保证事实正确，需一起检查原文。

对照两个模型时，使用同一题库文件和相同长度设置，分别写入两个新目录；先核对题库SHA、编码后题目SHA、tokenizer签名及协议，再比较共同有参考的题和原始回答。此入口提供单模型记录，不自动宣称候选通过固定CMRC质量门槛。

## 出错不会覆盖旧结果

已有输出目录即使为空也拒绝复用。生成中途出错时，`predictions.jsonl`保留已完成题目，报告标为`failed`并记录阶段与错误；不对缺题结果报一个貌似完整的均分。修复后换新目录重新运行，旧失败记录继续保留。题库格式错误时也会保存失败报告，此时可能还没有`predictions.jsonl`。

当前CPU测试覆盖schema、真实官方评分、长度边界、显式身份接口、完整报告和失败保留。真实GPU运行记录由项目主流程单独执行并保留；不要把CPU模拟生成测试当成真实模型效果验证。
