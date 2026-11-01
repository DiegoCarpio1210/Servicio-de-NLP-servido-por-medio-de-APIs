@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m experiments.baseqa_process start %*
if errorlevel 1 pause
