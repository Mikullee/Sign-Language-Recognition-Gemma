$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = if ($env:SLR_PYTHON) { $env:SLR_PYTHON } else { "python" }

Push-Location $repoRoot
try {
    & $pythonExe -m recognition.realtime.realtime_infer_daily30_sentence @args
}
finally {
    Pop-Location
}
