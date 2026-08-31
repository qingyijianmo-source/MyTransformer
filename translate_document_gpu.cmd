@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title MyTransformer Document Translator
cd /d "%~dp0"
set "HF_HOME=%~d0\MyTransformer_HF_Cache"
set "HF_HUB_CACHE=%~d0\MyTransformer_HF_Cache\hub"
set "HF_DATASETS_CACHE=%~d0\MyTransformer_HF_Cache\datasets"
set "TRANSFORMERS_CACHE=%~d0\MyTransformer_HF_Cache\hub"
set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python314\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
set /p INPUT=请把文档拖到此窗口，或输入完整路径：
set INPUT=%INPUT:"=%
"%PYTHON_EXE%" accurate_translator.py --input "%INPUT%"
pause
