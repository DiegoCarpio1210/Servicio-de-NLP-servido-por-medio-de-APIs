@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -u -m experiments.train_lora_pilot %*
exit /b %errorlevel%
