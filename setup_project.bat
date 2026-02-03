@echo off
REM AdaPrune 项目初始化脚本 (Windows Batch)
REM 使用方法: setup_project.bat

echo ============================================================
echo AdaPrune Project Setup
echo ============================================================

REM 创建目录
mkdir adaprune 2>nul
mkdir adaprune\core 2>nul
mkdir adaprune\strategies 2>nul
mkdir adaprune\utils 2>nul
mkdir configs 2>nul
mkdir data 2>nul
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir experiments 2>nul
mkdir notebooks 2>nul
mkdir results 2>nul
mkdir results\tables 2>nul
mkdir results\figures 2>nul
mkdir scripts 2>nul
mkdir tests 2>nul

REM 创建 __init__.py
echo # AdaPrune > adaprune\__init__.py
echo # AdaPrune > adaprune\core\__init__.py
echo # AdaPrune > adaprune\strategies\__init__.py
echo # AdaPrune > adaprune\utils\__init__. py

echo. 
echo Project structure created!
echo.
echo Next steps:
echo   1. Copy the Python files to their directories
echo   2. Run: pip install -r requirements.txt
echo   3. Run: python scripts\download_datasets.py
echo   4. Run: python scripts\quick_test.py

pause