@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title MyTransformer Accurate Fine Tuning
cd /d "%~dp0"
"C:\Users\1\AppData\Local\Programs\Python\Python314\python.exe" -u finetune_accurate.py --config config.finetune.json %*
set TRAIN_EXIT=%ERRORLEVEL%
echo.
echo Training finished. See output\accurate_finetuned\training.log for details.
pause
exit /b %TRAIN_EXIT%
