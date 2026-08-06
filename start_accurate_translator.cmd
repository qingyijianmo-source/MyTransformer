@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title MyTransformer Accurate Translator
cd /d "%~dp0"
"C:\Users\1\AppData\Local\Programs\Python\Python314\python.exe" translator_gui.py
if errorlevel 1 pause
