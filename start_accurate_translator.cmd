@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title MyTransformer Accurate Translator
cd /d "%~dp0"
set "HF_HOME=%~d0\MyTransformer_HF_Cache"
set "HF_HUB_CACHE=%~d0\MyTransformer_HF_Cache\hub"
set "HF_DATASETS_CACHE=%~d0\MyTransformer_HF_Cache\datasets"
set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python314\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" translator_gui.py
if errorlevel 1 pause
