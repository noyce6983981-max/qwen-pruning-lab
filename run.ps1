param(
    [ValidateSet('lesson', 'lesson6', 'lesson7', 'lesson8', 'lesson9', 'lesson13', 'lesson14', 'score', 'score-channels', 'check', 'collect', 'baseline', 'download', 'prepare-data', 'questions', 'test', 'prune', 'prune-calibrated')]
    [string]$Action = 'lesson',
    [string]$WorkRoot,
    [string]$ModelDir,
    [string]$DataDir,
    [string]$CalibrationDir,
    [string]$ScoresDir,
    [string]$QuestionsPath,
    [ValidateSet('original', 'pruned', 'merged_lora')]
    [string]$CheckpointKind,
    [string]$OutputDir
)
$ErrorActionPreference = 'Stop'
$taskChatRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$taskWorkRoot = if ($WorkRoot) { $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($WorkRoot) } else { Join-Path $taskChatRoot 'work' }
$taskPython = Join-Path $taskWorkRoot 'qwen-env\Scripts\python.exe'
$taskModel = if ($ModelDir) { $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ModelDir) } else { Join-Path $taskWorkRoot 'models\Qwen3-1.7B' }
$taskData = if ($DataDir) { $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($DataDir) } else { Join-Path $PSScriptRoot 'data\quality_v1' }
$taskRevision = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
if ($Action -eq 'score-channels' -and [string]::IsNullOrWhiteSpace($CalibrationDir)) {
    throw 'score-channels requires the completed collection directory. Usage: .\run.ps1 score-channels -WorkRoot .\.work -CalibrationDir <collection-directory> [-OutputDir <new-score-directory>]'
}
if ($Action -eq 'prune-calibrated' -and [string]::IsNullOrWhiteSpace($ScoresDir)) {
    throw 'prune-calibrated requires the completed channel score directory. Usage: .\run.ps1 prune-calibrated -WorkRoot .\.work -ScoresDir <score-directory> [-OutputDir <new-report-directory>]'
}
if ($Action -eq 'questions' -and ([string]::IsNullOrWhiteSpace($QuestionsPath) -or [string]::IsNullOrWhiteSpace($CheckpointKind))) {
    throw 'questions requires explicit input and model identity. Usage: .\run.ps1 questions -WorkRoot .\.work -ModelDir <model-directory> -CheckpointKind original|pruned|merged_lora -QuestionsPath <questions.jsonl> [-OutputDir <new-report-directory>]'
}
if ($Action -eq 'questions' -and $CheckpointKind -ne 'original' -and [string]::IsNullOrWhiteSpace($ModelDir)) {
    throw 'Pruned or merged questions require -ModelDir <the-actual-complete-model-directory>; an adapter directory alone is not a complete model.'
}
function Get-TaskOutput([string]$Prefix) {
    if ($OutputDir) { return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir) }
    return Join-Path $PSScriptRoot ('results\' + $Prefix + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
}
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw "Project Python is missing. Run setup.ps1 with the same -WorkRoot first. Expected: $taskPython"
}
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN = '1'
$env:PYTHONUTF8 = '1'
switch ($Action) {
    'lesson' {
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\lesson05_pytorch.py')
    }
    'lesson6' {
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\lesson06_gated_mlp.py')
    }
    'lesson7' {
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\lesson07_metrics.py')
    }
    'lesson8' {
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\lesson08_channel_importance.py')
    }
    'lesson9' {
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\lesson09_calibration.py')
    }
    'lesson13' {
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\lesson13_lora.py')
    }
    'lesson14' {
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\lesson14_accumulation.py')
    }
    'score' {
        $taskOutput = Get-TaskOutput 'scoring'
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\score_qa.py') --questions (Join-Path $PSScriptRoot 'examples\scoring\questions.jsonl') --predictions (Join-Path $PSScriptRoot 'examples\scoring\predictions_handwritten.jsonl') --output-dir $taskOutput
    }
    'check' {
        $taskOutput = Get-TaskOutput 'compat'
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\check_pruning_compat.py') --output-dir $taskOutput
    }
    'collect' {
        $taskOutput = Get-TaskOutput 'calibration'
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\collect_mlp_stats.py') --model-dir $taskModel --data-dir $taskData --output-dir $taskOutput --revision $taskRevision
    }
    'score-channels' {
        $taskOutput = Get-TaskOutput 'channel_scores'
        $taskCalibration = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($CalibrationDir)
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\score_mlp_channels.py') --model-dir $taskModel --calibration-dir $taskCalibration --output-dir $taskOutput
    }
    'prune-calibrated' {
        $taskRunTag = Get-Date -Format 'yyyyMMdd_HHmmss'
        $taskOutput = Get-TaskOutput 'calibrated_prune'
        $taskPrunedModel = Join-Path $taskWorkRoot ('models\Qwen3-1.7B-calibrated-mlp5888-' + $taskRunTag)
        $taskScores = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ScoresDir)
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\prune_qwen_mlp.py') --model-dir $taskModel --save-dir $taskPrunedModel --output-dir $taskOutput --keep-width 5888 --scores-dir $taskScores
    }
    'download' {
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\download_model.py') --model-dir $taskModel
    }
    'prepare-data' {
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\prepare_quality_data.py') --tokenizer-dir $taskModel --raw-dir (Join-Path $PSScriptRoot 'data\raw\cmrc2018') --output-dir $taskData
    }
    'questions' {
        $taskOutput = Get-TaskOutput 'questions'
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\evaluate_questions.py') --model-dir $taskModel --checkpoint-kind $CheckpointKind.ToLowerInvariant() --revision $taskRevision --questions ($ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($QuestionsPath)) --output-dir $taskOutput
    }
    'test' {
        & $taskPython -X utf8 -m unittest discover -s (Join-Path $PSScriptRoot 'tests') -v
    }
    'baseline' {
        $taskOutput = Get-TaskOutput 'baseline'
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\run_baseline.py') --model-dir $taskModel --output-dir $taskOutput --revision $taskRevision
        if ($LASTEXITCODE -ne 0) { throw "Baseline failed with exit code $LASTEXITCODE" }
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\summarize_baseline.py') (Join-Path $taskOutput 'baseline.json')
    }
    'prune' {
        $taskRunTag = Get-Date -Format 'yyyyMMdd_HHmmss'
        $taskOutput = Get-TaskOutput 'prune'
        $taskPrunedModel = Join-Path $taskWorkRoot ('models\Qwen3-1.7B-mlp5632-' + $taskRunTag)
        $taskOriginalOutput = Join-Path $taskOutput 'original'
        $taskPrunedOutput = Join-Path $taskOutput 'pruned'
        $taskPruningOutput = Join-Path $taskOutput 'pruning'
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\run_baseline.py') --model-dir $taskModel --output-dir $taskOriginalOutput --revision $taskRevision --dtype bfloat16 --bench-warmups 2 --bench-repeats 7
        if ($LASTEXITCODE -ne 0) { throw 'Original-model evaluation failed.' }
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\prune_qwen_mlp.py') --model-dir $taskModel --save-dir $taskPrunedModel --output-dir $taskPruningOutput --keep-width 5632
        if ($LASTEXITCODE -ne 0) { throw 'Pruning or save/reload validation failed.' }
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\run_baseline.py') --model-dir $taskPrunedModel --output-dir $taskPrunedOutput --revision $taskRevision --dtype bfloat16 --bench-warmups 2 --bench-repeats 7
        if ($LASTEXITCODE -ne 0) { throw 'Pruned-model evaluation failed.' }
        & $taskPython -X utf8 (Join-Path $PSScriptRoot 'scripts\compare_reports.py') --original (Join-Path $taskOriginalOutput 'baseline.json') --pruned (Join-Path $taskPrunedOutput 'baseline.json') --output-dir $taskOutput
    }
}
if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
