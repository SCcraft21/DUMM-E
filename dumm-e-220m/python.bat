@echo off
setlocal
set "PYTHON_EXE=C:\Users\Shitij Chatterjee\AppData\Local\Programs\Python\Python312\python.exe"

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" %*
) else (
    py -3.12 %*
)

