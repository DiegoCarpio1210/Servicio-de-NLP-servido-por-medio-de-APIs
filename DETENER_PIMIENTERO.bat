@echo off
chcp 65001 >nul
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Falta el entorno .venv. Cierra la ventana de inicio con Ctrl+C.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m scripts.local_app --stop
