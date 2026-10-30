@echo off
chcp 65001 >nul
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Falta el entorno .venv. Consulta docs\INTERFAZ_LOCAL.md para instalar las dependencias.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m scripts.local_app
if errorlevel 1 pause
