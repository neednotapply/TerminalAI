@echo off
chcp 65001 > nul
for /F "delims=" %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"

echo %ESC%[32m
echo ████████╗███████╗██████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██╗      █████╗ ██╗
echo ╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██║     ██╔══██╗██║
echo    ██║   █████╗  ██████╔╝██╔████╔██║██║██╔██╗ ██║███████║██║     ███████║██║
echo    ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║╚██╗██║██╔══██║██║     ██╔══██║██║
echo    ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║███████╗██║  ██║██║
echo    ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝
echo %ESC%[0m
echo.
echo %ESC%[32m┌────────────────────────────────┐
echo │ 1) Start TerminalAI           │
echo │ 2) Scan Shodan                │
echo └────────────────────────────────┘%ESC%[0m
start "Rain" /B python3 rain.py --persistent --exclude 0,5,1,76 --exclude 8,11,23,56
set /p choice=Select option:

taskkill /FI "WINDOWTITLE eq Rain" > nul

if "%choice%"=="1" (
    python3 TerminalAI.py %*
) else if "%choice%"=="2" (
    python3 shodanscan.py %*
) else (
    echo Invalid selection
    exit /b 1
)
