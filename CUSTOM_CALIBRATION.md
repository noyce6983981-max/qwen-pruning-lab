# 用自己的资料准备剪枝校准输入

`scripts/prepare_custom_calibration.py`把用户自己的长文本转成现有激活采集器能读取的校准文件。它只使用CPU tokenizer，不加载模型权重、不运行GPU、不剪枝。输入来自你的本地文件，输出明确标作`custom_user_calibration`，不冒充CMRC训练资料。

## 输入格式

UTF-8 JSONL，一行一个对象，包含必填`id`和`text`，可选`group_id`：

```json
{"id":"manual_1","group_id":"same_manual","text":"这里放你自己有权使用的一段完整长资料……"}
```

`id`必须唯一，`text`不能为空。同一份文档的不同段落可以使用相同`group_id`，程序最多选这个组的一段，避免把一份文档重复计为多个独立观察。没有填写组时，程序根据输入关系生成组标识。示例 [examples/custom_calibration.jsonl](examples/custom_calibration.jsonl) 是项目自编的长篇虚构课堂说明，只有一条，仅用于检查流程，不足以代表真实使用场景。

程序会把完整文本按NFKC、去空白、casefold规范化后识别完全重复；重复文本与用户提供的同组关系会合并，合并具有传递性。它不检测语义近重复，不保证自动排除所有资料相似或数据泄漏。

## 构建校准目录

从项目目录运行：

```powershell
$taskPython = '.\.work\qwen-env\Scripts\python.exe'
$taskData = Join-Path '.\results' ('custom_calibration_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& $taskPython -X utf8 .\scripts\prepare_custom_calibration.py --input '.\examples\custom_calibration.jsonl' --tokenizer-dir '.\.work\models\Qwen3-1.7B' --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --output-dir $taskData --exclude-questions '.\examples\custom_questions.jsonl'
if ($LASTEXITCODE -ne 0) { throw 'Custom calibration preparation failed; inspect its audit files.' }
```

`--exclude-questions`可省略。如果提供，题库格式与 [自定义问答入口](CUSTOM_QUESTIONS.md) 相同，程序检查每道题非空`context`是否与校准完整文本规范化重复；命中时排除整个关联组。没有`context`的普通问答题无法做这种资料匹配。程序不进行子串或语义相似检查，也不搜索外部测试集。

默认最多选64段，可用`--max-documents N`指定任意正整数。只有1段合格时就记录1段，不伪装成64段；实际每层观察次数为`N×256`。按原文件顺序选择，达到上限的剩余文本会记录为跳过。

## 长度与截取规则

每段先用原模型tokenizer编码完整文本，`add_special_tokens=False`。不足256 token明确跳过；合格段只取最前256个token，保存对应token IDs及其原样解码文字。这是**专用于激活校准的明确前缀截取**，与问答入口“完整提示超长就报错、不截断”的规则不同。

原文和完整SHA、规范化SHA、原始token数、输入ID／组、选中前缀及跳过原因都可以审计。长度足够不意味着资料具有代表性，过多近似内容或只看每段开头会影响校准统计，应在报告中说明资料选择方式。

输出包含：

- `calibration.jsonl`：供采集器读取的256-token前缀。
- `groups.json`：分组、合并关系和校准用途。
- `source_documents.jsonl`：完整原始资料及元数据，可能包含你的私有内容，不会自动上传。
- `excluded.jsonl`：短文本、重复、同组、题库资料重合和数量上限等跳过原因。
- `manifest.json`：成功产物的文件SHA、tokenizer文件SHA、真实数量和规则。
- `build_report.json`：构建成功或失败状态。

如果没有合格段，命令失败并保留跳过记录，不写完成的`manifest.json`。已有目录一律拒绝覆盖。兼容旧采集器所需的`source="train"`只表示校准用途，真正来源在`source_kind="custom_user_jsonl"`中，不表示来自CMRC或其他公共训练集。

## 接入原有采集、评分和剪枝

成功生成目录后，`collect_mlp_stats.py --data-dir`可以直接使用它；采集器会核对数据、tokenizer SHA和真实数量。下面只有第一条会运行真实GPU采集，第二条在CPU计算通道分数；请把路径设为自己本轮的新目录：

```powershell
$taskCollection = Join-Path '.\results' ('custom_collection_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
$taskScores = Join-Path '.\results' ('custom_scores_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& $taskPython -X utf8 .\scripts\collect_mlp_stats.py --model-dir '.\.work\models\Qwen3-1.7B' --data-dir $taskData --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --output-dir $taskCollection
if ($LASTEXITCODE -ne 0) { throw 'Calibration collection failed.' }
& $taskPython -X utf8 .\scripts\score_mlp_channels.py --model-dir '.\.work\models\Qwen3-1.7B' --calibration-dir $taskCollection --output-dir $taskScores
if ($LASTEXITCODE -ne 0) { throw 'Channel scoring failed.' }
```

需要制作新候选时，将所得`$taskScores`传给现有`prune_qwen_mlp.py --scores-dir`，并自行指定新权重目录、新报告目录和保留宽度。随后使用同一自定义题库分别检查原模型与候选模型，见 [CUSTOM_QUESTIONS.md](CUSTOM_QUESTIONS.md)。改变校准资料会产生一个新实验，不能套用项目既有CMRC分数或宣称必然改善。

此功能开放的是自定义校准与题库检查。恢复训练仍是本轮固定CMRC的512条／6762监督token配方；这个校准manifest不能直接当成恢复训练数据使用。本阶段没有因此运行新的GPU剪枝实验。
