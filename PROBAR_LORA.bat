@echo off
setlocal
cd /d "%~dp0"
if not exist "tmp\lora-deps\peft\__init__.py" (
  echo Install experiment-only PEFT first; see docs\DEBERTA_LORA_SMOKE_TEST.md.
  exit /b 1
)
".venv\Scripts\python.exe" -u -m experiments.deberta_lora_smoke %*
exit /b %errorlevel%
