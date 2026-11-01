@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -m experiments.baseqa_process status
pause
