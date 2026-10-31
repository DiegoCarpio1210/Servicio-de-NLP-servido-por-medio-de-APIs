@echo off
setlocal
cd /d "%~dp0"
if not exist "logs\deberta_lora\pilot200" exit /b 1
type nul > "logs\deberta_lora\pilot200\STOP"
echo Stop requested. The current optimizer update will finish and be checkpointed.
