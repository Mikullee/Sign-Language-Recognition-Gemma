@echo off
setlocal
cd /d "%~dp0"
SignLanguageRecognition.exe --source 0 --backend dshow --trigger-mode auto --save-log
if errorlevel 1 pause
