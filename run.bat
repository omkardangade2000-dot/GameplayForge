@echo off
REM Usage:  run.bat "MyRecording.mp4"      (file must be inside C:/Users/YOU/Videos)
if "%~1"=="" (echo Usage: run.bat "FileName.mp4" & exit /b 1)
"%~dp0.venv\Scripts\python.exe" -m gameplayforge run %*
