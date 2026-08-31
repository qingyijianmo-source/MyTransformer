@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set ZH_EN_REJECTED=0
title MyTransformer - Bidirectional 20-Epoch Training
cd /d "%~dp0"
set "HF_HOME=%~d0\MyTransformer_HF_Cache"
set "HF_HUB_CACHE=%~d0\MyTransformer_HF_Cache\hub"
set "HF_DATASETS_CACHE=%~d0\MyTransformer_HF_Cache\datasets"

set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python314\python.exe"
if exist "%PYTHON_EXE%" goto python_ready
set "PYTHON_EXE=python"

:python_ready
echo ============================================================
echo MyTransformer GPU fine-tuning: 20 epochs in each direction
echo [1/2] Chinese to English will run first.
echo [2/2] English to Chinese will start automatically afterward.
echo Keep this window open to monitor loss, GPU memory, and ETA.
echo ============================================================
echo.

echo [1/2] Starting Chinese to English...
"%PYTHON_EXE%" -u finetune_accurate.py --config config.finetune.json
if errorlevel 5 goto failed_zh_en
if errorlevel 4 goto rejected_zh_en
if errorlevel 1 goto failed_zh_en

echo.
echo [1/2] Chinese to English completed and passed the quality gate.
goto start_en_zh

:rejected_zh_en
set ZH_EN_REJECTED=1
echo.
echo [1/2] Chinese to English completed, but the candidate was rejected.
echo The previous best model remains active. Continuing with direction 2.

:start_en_zh
echo [2/2] Starting English to Chinese...
"%PYTHON_EXE%" -u finetune_accurate.py --config config.finetune.en_zh.json
if errorlevel 5 goto failed_en_zh
if errorlevel 4 goto rejected_en_zh
if errorlevel 1 goto failed_en_zh

if "%ZH_EN_REJECTED%"=="1" goto completed_with_rejection
echo.
echo ============================================================
echo Both 20-epoch training runs completed successfully.
echo Restart the translation UI or click Reload model.
echo ============================================================
pause
exit /b 0

:rejected_en_zh
:completed_with_rejection
echo.
echo ============================================================
echo Both 20-epoch training runs completed.
echo At least one candidate did not pass the quality gate.
echo The previous best model remains active for any rejected direction.
echo ============================================================
pause
exit /b 0

:failed_zh_en
echo.
echo Chinese-to-English training stopped or failed.
echo Check output\accurate_finetuned\training.log
pause
exit /b 1

:failed_en_zh
echo.
echo English-to-Chinese training stopped or failed.
echo Check output\accurate_finetuned_en_zh\training.log
pause
exit /b 1
