@echo off
cd /d "%~dp0"
python gui2.py
if %errorlevel% neq 0 pause
