@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=C:\Users\Shitij Chatterjee\AppData\Local\Programs\Python\Python312\python.exe"

"%PYTHON_EXE%" scripts\demo_voice_agent.py %*
if "%~1"=="" pause
