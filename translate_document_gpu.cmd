@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title MyTransformer Document Translator
cd /d "%~dp0"
set /p INPUT=请把文档拖到此窗口，或输入完整路径：
set INPUT=%INPUT:"=%
"C:\Users\1\AppData\Local\Programs\Python\Python314\python.exe" accurate_translator.py --input "%INPUT%"
pause
