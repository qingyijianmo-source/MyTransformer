@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title MyTransformer Accurate Fine Tuning
cd /d "%~dp0"
set "HF_HOME=%~d0\MyTransformer_HF_Cache"
set "HF_HUB_CACHE=%~d0\MyTransformer_HF_Cache\hub"
set "HF_DATASETS_CACHE=%~d0\MyTransformer_HF_Cache\datasets"
set "TRANSFORMERS_CACHE=%~d0\MyTransformer_HF_Cache\hub"
set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python314\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" -u finetune_accurate.py --config config.finetune.json %*
set TRAIN_EXIT=%ERRORLEVEL%
echo.
echo Training finished. See output\accurate_finetuned\training.log for details.
pause
exit /b %TRAIN_EXIT%
