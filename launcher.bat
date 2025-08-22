@echo off
set "SCRIPT_DIR=%~dp0"
set "PYTHON=python"
"%PYTHON%" "%SCRIPT_DIR%scripts\launcher.py" %*
