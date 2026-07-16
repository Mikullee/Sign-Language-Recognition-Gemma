$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = if ($env:SLR_PYTHON) { $env:SLR_PYTHON } else { "python" }

Push-Location $repoRoot
try {
    & $pythonExe -m recognition.evaluation.eval_auto_trigger_boundaries @args
}
finally {
    Pop-Location
}
