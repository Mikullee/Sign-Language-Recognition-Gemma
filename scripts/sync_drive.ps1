param(
  [ValidateSet("upload","download")]
  [string]$Mode = "upload",
  [string]$DriveRoot = "",
  [string]$RemoteName = "gdrive",
  [string]$RcloneExe = ".\\.local_tools\\rclone\\rclone.exe",
  [string]$RcloneConfig = ".\\.rclone.conf",
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$localDatasetRoot = Join-Path $repoRoot "dataset_staging"

function Ensure-Rclone([string]$ExePath, [string]$ConfigPath) {
  if (-not (Test-Path $ExePath)) {
    throw "rclone executable not found: $ExePath"
  }
  if (-not (Test-Path $ConfigPath)) {
    throw "rclone config not found: $ConfigPath"
  }
}

function Build-Args([string]$Src, [string]$Dst) {
  $args = @("copy", $Src, $Dst, "--create-empty-src-dirs", "--progress")
  if ($DryRun) { $args += "--dry-run" }
  return $args
}

Push-Location $repoRoot
try {
  $resolvedExe = (Resolve-Path $RcloneExe).Path
  $resolvedConfig = (Resolve-Path $RcloneConfig).Path
  Ensure-Rclone -ExePath $resolvedExe -ConfigPath $resolvedConfig

  $remotePath = if ([string]::IsNullOrWhiteSpace($DriveRoot)) { "$RemoteName`:" } else { "$RemoteName`:$DriveRoot" }

  if ($Mode -eq "upload") {
    if (-not (Test-Path $localDatasetRoot)) {
      throw "Local dataset root not found: $localDatasetRoot"
    }
    $args = Build-Args -Src $localDatasetRoot -Dst $remotePath
    & $resolvedExe @args --config $resolvedConfig
  }
  else {
    New-Item -ItemType Directory -Force -Path $localDatasetRoot | Out-Null
    $args = Build-Args -Src $remotePath -Dst $localDatasetRoot
    & $resolvedExe @args --config $resolvedConfig
  }
}
finally {
  Pop-Location
}
