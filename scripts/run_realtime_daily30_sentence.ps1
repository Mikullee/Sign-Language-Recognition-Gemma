$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$defaultPython = "C:\Users\User\miniconda3\envs\slr_mp310\python.exe"
$pythonExe = if ($env:SLR_PYTHON) { $env:SLR_PYTHON } elseif (Test-Path $defaultPython) { $defaultPython } else { "python" }

Push-Location $repoRoot
try {
    & $pythonExe -m recognition.realtime.realtime_infer_daily30_sentence @args
}
finally {
    Pop-Location
}
