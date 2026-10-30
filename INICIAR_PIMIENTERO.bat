@echo off
chcp 65001 >nul
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The .venv environment is missing. See docs\INTERFAZ_LOCAL.md for dependency setup.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m scripts.local_app
if errorlevel 1 pause
