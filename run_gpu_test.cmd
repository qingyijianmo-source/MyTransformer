@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title MyTransformer GPU Translation Test
cd /d "%~dp0"
if "%~1"=="" (
    "C:\Users\1\AppData\Local\Programs\Python\Python314\python.exe" translator_gui.py
) else (
    "C:\Users\1\AppData\Local\Programs\Python\Python314\python.exe" accurate_translator.py %*
)
if errorlevel 1 pause
