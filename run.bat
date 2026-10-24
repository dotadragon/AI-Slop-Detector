@echo off
setlocal enabledelayedexpansion
title SLOP - AI Content Detector (GPU Accelerated)

cd /d "%~dp0"

echo =======================================================
echo       SLOP - AI Content Detector Launcher
echo       (Unified Full-Stack + NVIDIA GPU CUDA)
echo =======================================================
echo.

set "SLOP_DIR=%~dp0slop"
set "BACKEND_DIR=%SLOP_DIR%\backend"
set "FRONTEND_DIR=%SLOP_DIR%\frontend"
set "VENV_DIR=%BACKEND_DIR%\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"

if not exist "%SLOP_DIR%" (
  echo [ERROR] Could not find the 'slop' folder in "%~dp0"
  pause
  exit /b 1
)
REM -------------------------------------------------------------
REM Step 1: Free network ports (8000, 3000)
REM -------------------------------------------------------------
echo [*] Freeing network ports 8000 and 3000...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000" ^| findstr "LISTENING"') do (
  taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":3000" ^| findstr "LISTENING"') do (
  taskkill /F /PID %%a >nul 2>&1
)

REM -------------------------------------------------------------
REM Step 2: Virtual Environment & Core Setup
REM -------------------------------------------------------------
if not exist "%VENV_PYTHON%" (
  echo [!] Python virtual environment not found. Creating .venv...
  pushd "%BACKEND_DIR%"
  py -3.12 -m venv .venv 2>nul || python -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create Python virtual environment.
    popd
    pause
    exit /b 1
  )
  echo [*] Installing backend dependencies...
  "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
  if exist "requirements.txt" (
    "%VENV_PIP%" install -r requirements.txt
  ) else if exist "requirements-core.txt" (
    "%VENV_PIP%" install -r requirements-core.txt
  )
  popd
)

REM -------------------------------------------------------------
REM Step 3: NVIDIA GPU CUDA Hardware Verification & Auto-Connect
REM -------------------------------------------------------------
echo [*] Checking NVIDIA CUDA GPU acceleration...
"%VENV_PYTHON%" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
  echo [!] CUDA GPU acceleration not detected in PyTorch.
  echo [*] Connecting NVIDIA GPU via PyTorch with CUDA 12.4 support...
  "%VENV_PIP%" install torch torchvision --index-url https://download.pytorch.org/whl/cu124
)

echo [*] GPU Hardware Status:
"%VENV_PYTHON%" -c "import torch; print('    >>> Active GPU:', torch.cuda.get_device_name(0), '| CUDA:', torch.version.cuda) if torch.cuda.is_available() else print('    [!] CUDA unavailable - running on CPU fallback')"
echo.

REM -------------------------------------------------------------
REM Step 4: Frontend Node Modules Verification
REM -------------------------------------------------------------
if not exist "%FRONTEND_DIR%\node_modules" (
  echo [*] Frontend dependencies not found. Running npm install...
  pushd "%FRONTEND_DIR%"
  call npm install
  popd
)

REM -------------------------------------------------------------
REM Step 5: Start Backend Server
REM -------------------------------------------------------------
set "DEVICE=auto"
echo [*] Starting Backend Server at http://127.0.0.1:8000...
start "SLOP Backend [8000]" /d "%BACKEND_DIR%" cmd /k "set "DEVICE=auto" && "%VENV_PYTHON%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir app"

echo [*] Waiting for backend health check (http://127.0.0.1:8000/health)...
set /a WAIT_COUNT=0
:wait_backend
"%VENV_PYTHON%" -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1).getcode()==200 else 1)" >nul 2>&1
if errorlevel 1 (
  set /a WAIT_COUNT+=1
  if !WAIT_COUNT! GTR 30 (
    echo [WARNING] Backend health check timed out. Launching frontend anyway...
    goto start_frontend
  )
  ping -n 2 127.0.0.1 >nul
  goto wait_backend
)
echo [OK] Backend is live and healthy.

REM -------------------------------------------------------------
REM Step 6: Start Frontend Server & Open Browser
REM -------------------------------------------------------------
:start_frontend
echo [*] Starting Frontend Server at http://127.0.0.1:3000...
start "SLOP Frontend [3000]" /d "%FRONTEND_DIR%" cmd /k "npm run dev"

ping -n 3 127.0.0.1 >nul
echo [*] Opening browser at http://127.0.0.1:3000...
start http://127.0.0.1:3000

echo.
echo =======================================================
echo   SLOP IS RUNNING WITH GPU ACCELERATION!
echo =======================================================
echo   - Frontend: http://127.0.0.1:3000
echo   - Backend:  http://127.0.0.1:8000
echo   - API Docs: http://127.0.0.1:8000/docs
echo   - Health:   http://127.0.0.1:8000/health
echo =======================================================
echo.
echo Keep this window open. Close it or press Ctrl+C to exit.
pause
