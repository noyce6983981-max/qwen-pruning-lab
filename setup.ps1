param(
    [Parameter(Mandatory=$true)]
    [string]$PythonExecutable,
    [string]$WorkRoot
)
$ErrorActionPreference = 'Stop'
$taskChatRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$taskWorkRoot = if ($WorkRoot) { $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($WorkRoot) } else { Join-Path $taskChatRoot 'work' }
$taskEnv = Join-Path $taskWorkRoot 'qwen-env'
$taskPython = Join-Path $taskEnv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw 'Supply an existing Python 3.12 executable with -PythonExecutable.'
}
& $PythonExecutable -c "import sys; assert sys.version_info[:2] == (3, 12), 'This environment requires Python 3.12'"
if ($LASTEXITCODE -ne 0) { throw 'Python version check failed.' }
if (-not (Test-Path -LiteralPath $taskPython)) {
    & $PythonExecutable -m venv $taskEnv
    if ($LASTEXITCODE -ne 0) { throw 'Creating the environment failed.' }
}
& $taskPython -m pip install --disable-pip-version-check uv==0.12.19
if ($LASTEXITCODE -ne 0) { throw 'Installing uv failed.' }
$env:UV_CACHE_DIR = Join-Path $taskWorkRoot 'uv-cache'
& $taskPython -m uv pip install --python $taskPython --requirements (Join-Path $PSScriptRoot 'requirements-windows.txt')
if ($LASTEXITCODE -ne 0) { throw 'Installing project dependencies failed.' }
& $taskPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency validation failed.' }
Write-Output "Environment ready: $taskPython"
Write-Output "Use the same work directory for subsequent commands: .\run.ps1 download -WorkRoot '$taskWorkRoot'"
Write-Output "Prepare the local official scorer before tests: & '$taskPython' third_party/cmrc2018/fetch_punkt.py"
