@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  python start.py
) else (
  py start.py
)
pause
