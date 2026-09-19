@echo off
setlocal enabledelayedexpansion
title SLOP Launcher

cd /d "%~dp0"

echo ===================================================
echo       SLOP - AI Content Detector Launcher
echo ===================================================
echo.

set "SLOP_DIR=%~dp0slop"
set "BACKEND_DIR=%SLOP_DIR%\backend"
set "FRONTEND_DIR=%SLOP_DIR%\frontend"
set "VENV_PYTHON=%BACKEND_DIR%\.venv\Scripts\python.exe"

if not exist "%SLOP_DIR%" (
  echo [ERROR] Could not find the 'slop' folder in "%~dp0"
  pause
  exit /b 1
)

REM Check Python venv
if not exist "%VENV_PYTHON%" (
  echo [!] Python 3.12 virtual environment not found in %BACKEND_DIR%\.venv
  echo [*] Creating virtual environment with Python 3.12...
  pushd "%BACKEND_DIR%"
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create Python 3.12 venv. Ensure Python 3.12 is installed.
    popd
    pause
    exit /b 1
  )
  echo [*] Installing core requirements...
  "%BACKEND_DIR%\.venv\Scripts\pip.exe" install -r requirements-core.txt
  popd
)

REM Check Frontend node_modules
if not exist "%FRONTEND_DIR%\node_modules" (
  echo [*] Frontend dependencies not found. Running npm install...
  pushd "%FRONTEND_DIR%"
  call npm install
  popd
)

echo [*] Starting Backend Server at http://127.0.0.1:8000...
start "SLOP Backend [8000]" cmd /k "cd /d "%BACKEND_DIR%" && "%VENV_PYTHON%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir app"

echo [*] Starting Frontend Server at http://127.0.0.1:3000...
start "SLOP Frontend [3000]" cmd /k "cd /d "%FRONTEND_DIR%" && npm run dev"

echo.
echo [*] Waiting 3 seconds for services to initialize...
timeout /t 3 /nobreak >nul

echo [*] Opening browser at http://127.0.0.1:3000...
start http://127.0.0.1:3000

echo.
echo ===================================================
echo   SLOP IS RUNNING!
echo ===================================================
echo   - Frontend: http://127.0.0.1:3000
echo   - Backend:  http://127.0.0.1:8000
echo   - API Docs: http://127.0.0.1:8000/docs
echo ===================================================
echo.
echo Keep this window open. Close it or press Ctrl+C to exit.
pause >nul
