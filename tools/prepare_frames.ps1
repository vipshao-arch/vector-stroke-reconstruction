param(
  [Parameter(Mandatory = $true)][string]$Video,
  [string]$Work = "analysis"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Work | Out-Null
$target = Join-Path $Work "frames_bgr24.raw"
ffmpeg -y -v error -i $Video -vf "scale=752:352" -pix_fmt bgr24 -f rawvideo $target
if ($LASTEXITCODE -ne 0) { throw "ffmpeg failed with exit code $LASTEXITCODE" }
Write-Host "Decoded frames: $target"
