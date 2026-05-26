@echo off
call conda activate kfashion
powershell -ExecutionPolicy Bypass -File "%~dp0recommend.ps1"
