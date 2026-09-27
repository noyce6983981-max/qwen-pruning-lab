# 从干净仓库复现一次完整实验

这份流程运行的是固定版本 Qwen3-1.7B、校准后 MLP 宽度 5888，以及原模型／剪后模型各 64 步 LoRA 的对照。它把每一步的输入和输出目录显式连接起来，不依赖作者电脑上的历史目录。幅值基线和更重剪枝的历史失败仍是结果的一部分；下面的主流程复现进入恢复阶段的校准 5888 分支，不自动重跑全部探索实验。

已测试平台是 Windows x64、Python 3.12、NVIDIA CUDA；真实实验设备为 RTX 4060 Laptop 8GB、16GB 系统内存。模型操作需要支持 BF16 的 CUDA GPU。CPU 测试通过不代表你的机器已完成真实模型验证。请为原模型、剪后模型、两个合并模型和下载缓存预留空间；它们远大于代码仓库。

按顺序在**同一个 PowerShell 窗口、仓库根目录**运行各段。不要一次盲目执行整页：第 8 节需要实际阅读回答、写下自己的复核记录。后续严格比较会核查代码、环境、数据与文件 SHA；同一实验中途不要更新依赖、编辑 helper、移动模型目录或手改 manifest。相同 seed 也不保证不同 GPU、驱动或软件环境下逐位相同；失败记录同样是实验结果。

## 1. 安装环境并检查 CPU 功能

先安装 Python 3.12 和对应 NVIDIA 驱动，然后克隆仓库并进入根目录。下面通过 Windows Python Launcher 找到 Python 3.12；如果没有 `py` 命令，将 `$taskPython312` 改成你已安装的 Python 3.12 可执行文件绝对路径。

```powershell
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN = '1'

$taskRepo = (Get-Location).Path
if (-not (Test-Path -LiteralPath (Join-Path $taskRepo 'setup.ps1'))) {
    throw '请先进入 qwen-pruning-lab 仓库根目录。'
}
$taskPython312 = (& py -3.12 -c 'import sys; print(sys.executable)').Trim()
if ($LASTEXITCODE -ne 0) { throw '没有找到 Python 3.12。' }
$taskWork = Join-Path $taskRepo '.work'
& .\setup.ps1 -PythonExecutable $taskPython312 -WorkRoot $taskWork
$taskPython = Join-Path $taskWork 'qwen-env\Scripts\python.exe'

# 后面的 Python 命令都经此函数执行；任何非零退出立即停止。
function Invoke-LabPython {
    & $taskPython -X utf8 @args
    if ($LASTEXITCODE -ne 0) { throw "Python 命令失败，退出码 $LASTEXITCODE。请保留日志并检查错误。" }
}

Invoke-LabPython .\third_party\cmrc2018\fetch_punkt.py
& .\run.ps1 test -WorkRoot $taskWork
& .\run.ps1 lesson13 -WorkRoot $taskWork
```

`fetch_punkt.py` 下载固定来源并核对 SHA。该数据缓存因许可未明确不随仓库分发；官方评分脚本与许可在 `third_party/cmrc2018/`。CPU 测试会用临时小模型，不下载完整 Qwen。还未构建数据时，真实冻结数据检查会明确 skip；请同时报告测试数与 skip，不能把 skip 写成已执行。

所有入口都显式传 `-WorkRoot`。不传时保留的 `../../work` 是旧工作目录兼容行为，本教程不使用它。

## 2. 固定本轮路径，下载原模型并构建数据

`reproduction_01` 是你为这次实验起的名称。若已经存在，换新名称，保留旧结果。只创建总结果目录；各阶段输出目录由脚本创建，不要预先创建它们。模型与数据路径一旦写入来源记录，就不要在本轮中途移动。

```powershell
$taskRunName = 'reproduction_01'
$taskRun = Join-Path $taskRepo ('results\' + $taskRunName)
if (Test-Path -LiteralPath $taskRun) { throw '本轮结果目录已存在，请使用新 RunName。' }
New-Item -ItemType Directory -Path $taskRun -Force | Out-Null

$taskRevision = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
$taskOriginal = Join-Path $taskWork 'models\Qwen3-1.7B'
$taskPruned = Join-Path $taskWork ('models\Qwen3-1.7B-calibrated-5888-' + $taskRunName)
$taskData = Join-Path $taskRepo 'data\quality_v1'
$taskCalibration = Join-Path $taskRun 'calibration'
$taskScores = Join-Path $taskRun 'channel_scores'
$taskPruningReport = Join-Path $taskRun 'pruning'

& .\run.ps1 download -WorkRoot $taskWork -ModelDir $taskOriginal
& .\run.ps1 prepare-data -WorkRoot $taskWork -ModelDir $taskOriginal -DataDir $taskData
```

数据构建器下载固定提交的 CMRC2018 train/dev、核对原始文件 SHA，按文档组隔离用途，再用固定原 tokenizer 保存 token。它拒绝覆盖已有 `$taskData`。若你已经按同一版本构建了数据，本轮可复用经校验的完整目录，跳过第二条命令；不要只复制作者的 manifest 或替换其中的本机路径。

固定规模为：calibration 64 条，每条 256 token；recovery 512 条；validation 100 题与 8192 个 LM 目标；final 200 题与 16384 个 LM 目标。final 文件此时被构建和封存，后续不能拿来选通道、训练、选 checkpoint 或调提示。新的本机 manifest 有自己的 SHA，不必与作者包含不同路径的 manifest 字节相同。

## 3. 采集激活、评分并结构化剪枝

从这里开始会使用真实 GPU。采集器观察原模型 MLP 中间激活；评分阶段在 CPU 上结合 down-projection 的平均绝对权重与激活 RMS，逐层稳定排序。评分值不是任务准确率，较高分也不保证恢复训练成功。

```powershell
& .\run.ps1 collect -WorkRoot $taskWork -ModelDir $taskOriginal -DataDir $taskData -OutputDir $taskCalibration
& .\run.ps1 score-channels -WorkRoot $taskWork -ModelDir $taskOriginal -CalibrationDir $taskCalibration -OutputDir $taskScores

# 显式指定保存目录，后面的恢复流程使用同一个 $taskPruned。
Invoke-LabPython .\scripts\prune_qwen_mlp.py --model-dir $taskOriginal --save-dir $taskPruned --output-dir $taskPruningReport --keep-width 5888 --scores-dir $taskScores --revision $taskRevision
```

这里会同步裁剪 gate/up 的输出行与 down 的输入列，并保存、重载检查。省略 `--scores-dir` 会改用本项目自定义联合平方幅值基线，所以此处不能漏写。这个基线也不能代表所有幅值剪枝方法。所有模型只在新目录生成，原始权重保留。

如果只想观察自定义语料的激活分布，可使用独立的 `prepare_custom_calibration.py` 流程；它不替换本页固定 CMRC 数据，也不能把自定义统计直接冒充本次恢复／final 对照。其格式和运行方式以自定义数据说明及 `--help` 为准。

## 4. 为本次运行生成恢复策略

策略记录本机两个 base 的真实文件哈希、数据、训练预算和验证选择规则。它是现在新建的实验记录，不是作者历史策略的复制品，也不声称你已经不知道公开结果。

```powershell
$taskRecoveryPolicy = Join-Path $taskRun 'recovery_policy.json'
Invoke-LabPython .\scripts\prepare_recovery_policy.py --original-model $taskOriginal --pruned-model $taskPruned --data-dir $taskData --revision $taskRevision --output $taskRecoveryPolicy

$taskArms = @('original', 'calibrated_5888')
$taskBaseModels = @{
    original = $taskOriginal
    calibrated_5888 = $taskPruned
}
```

两个分支都从自己的 base 重新初始化 LoRA，不复用教学试跑 adapter。预算固定为 r=8、alpha=16、dropout=0.05、学习率 1e-4，batch=1、累积 8 条后更新一次，64 次更新覆盖 512 条、6762 个答案／结束符监督 token。两边预算一致，adapter 参数量因 MLP 宽度不同而不同。

## 5. 先预检，再完成两边训练

`memory` 用八条最长真实训练样本完成一次累积更新，检查后丢弃；它不是续训起点，也不是任意 512 token 输入的显存保证。正式训练必须提供同代码、同数据、同 base 的成功预检报告。

```powershell
foreach ($taskArm in $taskArms) {
    $taskMemoryOutput = Join-Path $taskRun ('memory_' + $taskArm)
    Invoke-LabPython .\scripts\train_recovery.py --policy $taskRecoveryPolicy --arm $taskArm --mode memory --data-dir $taskData --output-dir $taskMemoryOutput
}

foreach ($taskArm in $taskArms) {
    $taskMemoryReport = Join-Path $taskRun ('memory_' + $taskArm + '\run.json')
    $taskTrainingOutput = Join-Path $taskRun ('train_' + $taskArm)
    Invoke-LabPython .\scripts\train_recovery.py --policy $taskRecoveryPolicy --arm $taskArm --mode train --data-dir $taskData --memory-report $taskMemoryReport --output-dir $taskTrainingOutput
}
```

每条命令是独立进程，同一时刻只加载一个模型。基础权重冻结，但仍参与前向与反向所需计算；只更新 LoRA。gradient checkpointing 是用重算节约显存，与保存 `step-0016`、`step-0032`、`step-0064` 检查点不同。预检失败时先检查错误并保留记录，不为了继续而手改成功标志、缩预算后仍称同一实验。

## 6. 统一评测两边 0／16／32／64 步

训练完成后，使用同一个新入口重测 step0 与所有 adapter。step0 不传 adapter；其余 checkpoint 必须来自已完成的 64 步训练 run，不能用半途产物抢先选优。这里统一使用 BF16 autocast，不混用旧历史评测器的报告。

```powershell
$taskValidation = Join-Path $taskRun 'validation'
foreach ($taskArm in $taskArms) {
    $taskBaseOutput = Join-Path $taskValidation ($taskArm + '_step0')
    Invoke-LabPython .\scripts\run_recovery_quality.py --model-dir $taskBaseModels[$taskArm] --data-dir $taskData --split validation --revision $taskRevision --output-dir $taskBaseOutput

    foreach ($taskStep in @(16, 32, 64)) {
        $taskAdapter = Join-Path $taskRun ('train_' + $taskArm + '\step-' + ('{0:D4}' -f $taskStep))
        $taskCheckpointOutput = Join-Path $taskValidation ($taskArm + '_step' + $taskStep)
        Invoke-LabPython .\scripts\run_recovery_quality.py --model-dir $taskBaseModels[$taskArm] --adapter-dir $taskAdapter --data-dir $taskData --split validation --revision $taskRevision --output-dir $taskCheckpointOutput
    }
}
```

每份目录保留 `quality.json` 与逐题 `qa_predictions.jsonl`。保存探针重载必须完全一致；报告另记录真实 base＋adapter 参数数和字节。不要把未合并 adapter 的参数漏掉。生成答案仅去掉首尾空白，不能手动删解释、重复或错误内容后还称它为原始指标。

## 7. 汇总八份验证，检查数值提议

汇总器严格检查协议、代码、数据、环境、训练来源和全部 checkpoint。门槛相对本轮原模型 step0：F1 下降不超过 5 个百分点、PPL 相对增加不超过 10%；每个分支再按 F1、PPL、较早步数顺序提出选择，step0 也参与。

```powershell
$taskRecoverySummaryDir = Join-Path $taskRun 'recovery_summary'
$taskSummaryArgs = @(
    '.\scripts\summarize_recovery.py',
    '--policy', $taskRecoveryPolicy,
    '--original-base', (Join-Path $taskValidation 'original_step0\quality.json'),
    '--pruned-base', (Join-Path $taskValidation 'calibrated_5888_step0\quality.json'),
    '--output-dir', $taskRecoverySummaryDir
)
foreach ($taskArm in $taskArms) {
    foreach ($taskStep in @(16, 32, 64)) {
        $taskSummaryArgs += @('--candidate', (Join-Path $taskValidation ($taskArm + '_step' + $taskStep + '\quality.json')))
    }
}
Invoke-LabPython @taskSummaryArgs
$taskSelection = Join-Path $taskRecoverySummaryDir 'recovery_summary.json'
Get-Content -LiteralPath (Join-Path $taskRecoverySummaryDir 'recovery_summary.md') -Encoding UTF8
```

**当前导出和发布配方只支持两个分支都选中 step64。** 如果新实验选中 0、16、32，或没有 checkpoint 通过，停在这里、报告真实结果。不能改 summary 的步数、替换报告路径、延长训练或放宽门槛来凑成 step64。支持其他选择需要单独设计、测试新的导出协议，本页不提供绕过方式。

```powershell
$taskSelected = Get-Content -Raw -LiteralPath $taskSelection -Encoding UTF8 | ConvertFrom-Json
foreach ($taskArm in $taskArms) {
    $taskChosen = $taskSelected.selected.$taskArm.checkpoint
    if ($null -eq $taskChosen -or $taskChosen.training_step -ne 64 -or $taskChosen.numeric_quality_gate_passed -ne $true) {
        throw "$taskArm 未满足本版两个 step64 导出的条件；保留结果，不修改选择。"
    }
}
```

## 8. 读回答，再记录自己的复核笔记

先看两个 step64 的 `quality.json`／`qa_predictions.jsonl`，与各自 step0 比较。记录连续重复、抄题、事实错误、达到输出上限的具体 ID，以及哪些结果变好或变差。数值过线不等于人工验收；如果实际回答存在不可接受的退化，先停下并保留失败案例。

在 `$taskRun\review_notes.md` 中写下**你实际观察到的内容**，不能复制一段“已通过”来代替阅读。可用任意文本编辑器；以下第一段只打开文件，不自动填写结论。保存为 UTF-8 后，再单独执行第二段。

```powershell
$taskReviewNotes = Join-Path $taskRun 'review_notes.md'
notepad $taskReviewNotes
```

```powershell
$taskReview = Join-Path $taskRun 'validation_review.json'
Invoke-LabPython .\scripts\record_quality_review.py --summary $taskSelection --notes $taskReviewNotes --output $taskReview
```

这个命令把你的笔记绑定到数值汇总和被选报告，写成 `completed_operator_review`。它只记录自述并核查文件绑定，不验证你真的读过、观察是否正确，也不颁发独立人工验收结论。不要把状态改成作者的 `completed_assistant_review` 来冒充其他人的复核。

## 9. 合并并重载验证两个 step64

完整模型写入新的两个目录；原 base 与 adapter 都保留。合并把 FP32 LoRA 修正写回 BF16 权重，舍入可能改变 logits 和回答；“保存后重载一致”不等于“合并前后所有输出一致”。因此后面评测的是实际导出模型。

```powershell
$taskMergedOriginal = Join-Path $taskWork ('models\Qwen3-1.7B-original-step64-merged-' + $taskRunName)
$taskMergedPruned = Join-Path $taskWork ('models\Qwen3-1.7B-calibrated-5888-step64-merged-' + $taskRunName)
$taskMergedModels = @{
    original = $taskMergedOriginal
    calibrated_5888 = $taskMergedPruned
}

foreach ($taskArm in $taskArms) {
    $taskAdapter = Join-Path $taskRun ('train_' + $taskArm + '\step-0064')
    $taskExportReport = Join-Path $taskRun ('export_' + $taskArm)
    Invoke-LabPython .\scripts\export_lora.py --model-dir $taskBaseModels[$taskArm] --adapter-dir $taskAdapter --data-dir $taskData --selection $taskSelection --arm $taskArm --revision $taskRevision --output-dir $taskMergedModels[$taskArm] --report-dir $taskExportReport
}
```

`--output-dir` 保存完整模型，`--report-dir` 保存 `export.json`。二者都必须是新的目录，且在源 base／adapter 之外。导出会检查 84 个被合并 MLP 权重、非 MLP 权重不变，以及保存重载后的参数／探针一致性；八个恢复样本探针不能代替全量质量评测。

## 10. 在读取 final 之前冻结四个产物

冻结原 step0、校准 step0、原 step64 合并、校准 step64 合并，以及评测代码、数据和门槛。这里不按 final 重新选择。发布策略生成以后，保持代码与产物原样，否则后面的身份核查应当拒绝运行。

```powershell
$taskReleasePolicy = Join-Path $taskRun 'release_policy.json'
Invoke-LabPython .\scripts\freeze_release_policy.py --original-model $taskOriginal --pruned-model $taskPruned --original-merged $taskMergedOriginal --pruned-merged $taskMergedPruned --data-dir $taskData --selection $taskSelection --review $taskReview --revision $taskRevision --output $taskReleasePolicy

$taskModelOrder = @(
    'original_step0',
    'calibrated_5888_step0',
    'original_step64_merged',
    'calibrated_5888_step64_merged'
)
$taskDeploymentModels = @{
    original_step0 = $taskOriginal
    calibrated_5888_step0 = $taskPruned
    original_step64_merged = $taskMergedOriginal
    calibrated_5888_step64_merged = $taskMergedPruned
}
```

如果复核绑定或来源不匹配，修复真实输入／流程并新建记录；不要手改 hash 或伪造状态。final 是对这四个固定管线的本地一次性评价，不是官方隐藏测试，也不支持看到结果后追加训练、改参考答案或改筛选条件。

## 11. 对实际发布产物先测 validation，再测 final

四个模型都经同一个发布入口重新评测。目录必须采用下面的 `validation_<model-id>`／`final_<model-id>` 扁平形式，汇总器据此查找八份报告。

```powershell
$taskReleaseQuality = Join-Path $taskRun 'release_quality'
foreach ($taskSplit in @('validation', 'final')) {
    foreach ($taskModelId in $taskModelOrder) {
        $taskQualityOutput = Join-Path $taskReleaseQuality ($taskSplit + '_' + $taskModelId)
        Invoke-LabPython .\scripts\run_release_quality.py --policy $taskReleasePolicy --model-dir $taskDeploymentModels[$taskModelId] --model-id $taskModelId --data-dir $taskData --split $taskSplit --revision $taskRevision --output-dir $taskQualityOutput
    }
}

$taskReleaseSummaryDir = Join-Path $taskRun 'release_summary'
Invoke-LabPython .\scripts\summarize_release.py --policy $taskReleasePolicy --quality-dir $taskReleaseQuality --output-dir $taskReleaseSummaryDir
Get-Content -LiteralPath (Join-Path $taskReleaseSummaryDir 'release_quality_summary.md') -Encoding UTF8
```

保留全部八份报告，不能只展示最好的模型。汇总同时给出同 step 的原／剪后对照；EM、F1、PPL 可能指向不同结论。final 数值失败要原样报告，不用它重选 checkpoint。完成后还需阅读实际导出模型的回答，记录重复或事实错误；汇总 JSON 明确没有替你完成这项判断。

## 12. 质量条件满足后做两轮资源测试

本版微基准固定需要四个产物各两次，共八份报告。只有这四个固定产物都满足 validation 与 final 的数值条件，且你已处理回答复核中的实际问题，才继续这个完整对照。下面检查仅覆盖数值，不替代人工判断；若有未过线产物，停下保留质量结论，不删掉它或改名凑满八份。

```powershell
$taskReleaseSummaryPath = Join-Path $taskReleaseSummaryDir 'release_quality_summary.json'
$taskReleaseSummary = Get-Content -Raw -LiteralPath $taskReleaseSummaryPath -Encoding UTF8 | ConvertFrom-Json
$taskEligible = @($taskReleaseSummary.validation_and_final_numeric_gate_passed_model_ids)
$taskIneligible = @($taskModelOrder | Where-Object { $taskEligible -notcontains $_ })
if ($taskIneligible.Count -gt 0) {
    throw ('以下固定产物未满足完整资源对照的数值前提：' + ($taskIneligible -join ', '))
}
```

使用同一台机器、同一电源／散热条件，关闭会争用 GPU 的游戏或其他模型进程，记录实际情况。不要为了比较中的某个模型单独更改电源设置。脚本记录温度、功耗、频率、整卡占用与 Windows 电源状态；不可得字段仍写为不可得，不会替你修改设置。

第一轮按四个固定身份正序，第二轮反序。下面每次调用都会启动、结束一个独立 Python 进程；不要并行执行，也不要在同一个模型进程里省略加载来替代它。

```powershell
$taskBenchmarkRoot = Join-Path $taskRun 'benchmark'
$taskBenchmarkReports = @()
foreach ($taskSession in @(1, 2)) {
    $taskOrder = if ($taskSession -eq 1) {
        $taskModelOrder
    } else {
        @($taskModelOrder[3], $taskModelOrder[2], $taskModelOrder[1], $taskModelOrder[0])
    }
    for ($taskIndex = 0; $taskIndex -lt $taskOrder.Count; $taskIndex++) {
        $taskModelId = $taskOrder[$taskIndex]
        $taskPosition = $taskIndex + 1
        $taskBenchmarkOutput = Join-Path $taskBenchmarkRoot ('session' + $taskSession + '_' + $taskPosition + '_' + $taskModelId)
        Invoke-LabPython .\scripts\benchmark_deployment.py run --model-dir $taskDeploymentModels[$taskModelId] --canonical-tokenizer-dir $taskOriginal --revision $taskRevision --model-id $taskModelId --session $taskSession --position $taskPosition --output-dir $taskBenchmarkOutput
        $taskBenchmarkReports += Join-Path $taskBenchmarkOutput 'benchmark.json'
    }
}

$taskBenchmarkSummary = Join-Path $taskRun 'benchmark_summary'
Invoke-LabPython .\scripts\benchmark_deployment.py summarize --reports @taskBenchmarkReports --output-dir $taskBenchmarkSummary
Get-Content -LiteralPath (Join-Path $taskBenchmarkSummary 'deployment_summary.md') -Encoding UTF8
```

微基准使用固定 128／512 输入 token、强制 32 输出、两次预热和七次计时。忽略 EOS 只是控制测速工作量，不是用户回答策略；512＋32 也不是前面 QA 的总长 512 规则。首 token 延迟不含分词和模型磁盘加载，后续速率含 Python 控制开销。allocated／reserved 不等于整卡占用；磁盘文件大小还受 tied weights 的序列化方式影响。

读中位数时也看每轮范围与方向。两轮方向不同或范围重叠，不据此声称稳定加速；即使两轮符合保守描述条件，也不是统计显著性检验或其他机器上的速度保证。资源测试不能补救质量失败。

## 出错、重跑与保留证据

大多数命令拒绝覆盖已有输出；中断时保留已经写出的逐题答案和失败报告。先看失败阶段与异常，不把 `failed` 改成 `completed`。需要重跑某一步时，选择新的输出路径，并将后续变量显式指向该产物；如果上游策略／汇总已绑定旧 SHA，需要重新生成相应下游链，而不是编辑旧记录。

请一起保留原始与剪后 base、各 checkpoint、策略、完整报告、代码版本和环境快照；但不要把 `.work`、模型权重、Punkt 缓存、个人路径和完整运行记录直接提交到公共 GitHub。公开精选结果应注明来源和局限，原始文件的 SHA 与公开整理摘要的 SHA 分开记录。

本流程是一个受限实验配方：固定模型、固定小规模中文阅读理解任务、一个训练 seed、短上下文、两个 LoRA 分支。它能帮助核查本次压缩与恢复是否成立，不保证更换语料、模型、硬件或随机环境后得到相同分数，也不把失败复现变成“复现失败”。
