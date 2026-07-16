param(
  [string]$PythonCommand = "",
  [switch]$SkipZip
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$specPath = Join-Path $repoRoot "packaging\windows\SignLanguageRecognition.spec"
$distRoot = Join-Path $repoRoot "dist"
$buildRoot = Join-Path $repoRoot "build"
$portableDir = Join-Path $distRoot "SignLanguageRecognition"
$releaseDir = Join-Path $repoRoot "release"

if ([string]::IsNullOrWhiteSpace($PythonCommand)) {
  $PythonCommand = if ($env:SLR_PYTHON) { $env:SLR_PYTHON } else { "python" }
}

foreach ($target in @($buildRoot, $portableDir)) {
  $fullTarget = [System.IO.Path]::GetFullPath($target)
  if (-not $fullTarget.StartsWith($repoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to clean a path outside the repository: $fullTarget"
  }
  if (Test-Path -LiteralPath $fullTarget) {
    Remove-Item -LiteralPath $fullTarget -Recurse -Force
  }
}

& $PythonCommand -m PyInstaller --noconfirm --clean --distpath $distRoot --workpath $buildRoot $specPath
if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller build failed."
}

Copy-Item -LiteralPath (Join-Path $repoRoot "app_config.json") -Destination $portableDir -Force
Copy-Item -LiteralPath (Join-Path $repoRoot "packaging\windows\start_ivcam.cmd") -Destination $portableDir -Force
New-Item -ItemType Directory -Path (Join-Path $portableDir "logs") -Force | Out-Null

& (Join-Path $portableDir "SignLanguageRecognition.exe") --help
if ($LASTEXITCODE -ne 0) {
  throw "Packaged executable smoke test failed."
}

if (-not $SkipZip) {
  New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null
  $zipPath = Join-Path $releaseDir "SignLanguageRecognition-v0.1.0-windows-x64.zip"
  if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
  }
  Compress-Archive -LiteralPath $portableDir -DestinationPath $zipPath -CompressionLevel Optimal
  & $PythonCommand (Join-Path $repoRoot "scripts\verify_release_safety.py") --portable-dir $portableDir --zip $zipPath
  if ($LASTEXITCODE -ne 0) {
    throw "Release safety scan failed."
  }
  Write-Host "Portable ZIP: $zipPath"
}

Write-Host "Portable directory: $portableDir"
