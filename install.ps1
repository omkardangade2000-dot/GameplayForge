# One-time setup on Windows (PowerShell). Run from the GameplayForge folder:  .\install.ps1
$ErrorActionPreference = "Stop"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements; Write-Host "FFmpeg installed - close and reopen this window, then run install.ps1 again."; exit }
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
# Official PyTorch wheels built with CUDA 12.8 support RTX 50-series (Blackwell, sm_120) and older GPUs. Do not use third-party "patched" builds.
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m gameplayforge doctor
