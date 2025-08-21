@echo off
chcp 65001 > nul
for /F "delims=" %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"

echo %ESC%[95m
echo ███████╗███████╗██████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██╗     
echo ╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██║     
echo    ██║   █████╗  ██████╔╝██╔████╔██║██║██╔██╗ ██║███████║██║     
echo    ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║╚██╗██║██╔══██║██║     
echo    ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║███████╗
echo    ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝
echo %ESC%[0m
echo.
echo 1^) Scan Shodan
echo 2^) Use TerminalAI
set /p choice=Select option: 

if "%choice%"=="1" (
    python shodan_scan.py %*
) else if "%choice%"=="2" (
    python TerminalAI.py %*
) else (
    echo Invalid selection
    exit /b 1
)
