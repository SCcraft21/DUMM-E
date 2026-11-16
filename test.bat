@echo off
setlocal
cd /d "%~dp0\dumm-e-220m"
set "PYTHON_EXE=C:\Users\Shitij Chatterjee\AppData\Local\Programs\Python\Python312\python.exe"

"%PYTHON_EXE%" -m unittest discover tests -v
if "%~1"=="" pause
