@echo off
setlocal enabledelayedexpansion
title SLOP - GPU CUDA Acceleration Setup (RTX 4050)

cd /d "%~dp0"

echo =======================================================
echo     SLOP - NVIDIA GPU CUDA Hardware Acceleration Setup
echo =======================================================
echo.
echo Detected GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)
echo Target: PyTorch with CUDA 12.4 Acceleration
echo.

set "VENV_PIP=%~dp0slop\backend\.venv\Scripts\pip.exe"
set "VENV_PYTHON=%~dp0slop\backend\.venv\Scripts\python.exe"

if not exist "%VENV_PIP%" (
  echo [ERROR] Virtual environment pip not found at %VENV_PIP%
  echo Please run run.bat first to initialize the environment.
  pause
  exit /b 1
)

echo [*] Ensuring port 8000 and any active Python processes are closed to avoid Windows file locks...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
  taskkill /F /PID %%a >nul 2>&1
)

echo [*] Installing PyTorch with CUDA 12.4 support (this downloads ~2.5GB)...
echo [*] Progress will be shown below:
echo.
"%VENV_PIP%" install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo.
echo [*] Verifying CUDA GPU installation...
"%VENV_PYTHON%" -c "import torch; print('\n>>> CUDA Available:', torch.cuda.is_available()); print('>>> Active Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Fallback'); print('>>> Device Count:', torch.cuda.device_count(), '\n')"

echo =======================================================
echo GPU SETUP COMPLETE!
echo You can now launch SLOP by running run.bat!
echo =======================================================
pause
