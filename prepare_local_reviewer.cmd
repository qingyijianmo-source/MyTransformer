@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title MyTransformer Local Reviewer Setup
cd /d "%~dp0"
set "HF_HOME=%~d0\MyTransformer_HF_Cache"
set "HF_HUB_CACHE=%~d0\MyTransformer_HF_Cache\hub"
set "HF_DATASETS_CACHE=%~d0\MyTransformer_HF_Cache\datasets"
set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python314\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" -m pip install -r requirements-accurate.txt -r requirements-reviewer.txt
if errorlevel 1 goto failed
"%PYTHON_EXE%" prepare_local_reviewer.py
if errorlevel 1 goto failed
echo.
echo 本地深度审校器准备完成。
pause
exit /b 0
:failed
echo.
echo 准备失败，请查看上面的错误信息。
pause
exit /b 1
