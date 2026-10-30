@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set CUDA_VISIBLE_DEVICES=
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Install the documented dependencies first.
  exit /b 1
)
if not exist "logs\pretrained_benchmark" mkdir "logs\pretrained_benchmark"
powershell -NoProfile -Command "$stamp=Get-Date -Format 'yyyyMMdd-HHmmss-fff'; Start-Process -FilePath '.\.venv\Scripts\python.exe' -ArgumentList '-u','-m','experiments.run_pretrained_benchmark' -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -RedirectStandardOutput ('logs\pretrained_benchmark\launcher-'+$stamp+'.log') -RedirectStandardError ('logs\pretrained_benchmark\launcher-'+$stamp+'-error.log')"
echo Background benchmark requested. Check logs\pretrained_benchmark\progress.json and pid.json.
