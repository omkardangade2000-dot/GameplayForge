@echo off
REM GameplayForge dashboard: http://127.0.0.1:8770/
REM Runs started from the dashboard use the same GPU/FFmpeg settings as run.bat and keep going if this window closes.
cd /d "%~dp0"
start "" "http://127.0.0.1:8770/"
"%~dp0.venv\Scripts\python.exe" "%~dp0dashboard\server.py"
