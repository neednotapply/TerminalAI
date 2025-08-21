@echo off
chcp 65001 > nul
for /F "delims=" %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"

echo %ESC%[32m
echo ███████╗███████╗██████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██╗     
echo ╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██║     
echo    ██║   █████╗  ██████╔╝██╔████╔██║██║██╔██╗ ██║███████║██║     
echo    ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║╚██╗██║██╔══██║██║     
echo    ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║███████╗
echo    ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝
echo %ESC%[0m
echo.
echo %ESC%[32m┌────────────────────────────────┐
echo │ 1) Start TerminalAI           │
echo │ 2) Scan Shodan                │
echo └────────────────────────────────┘%ESC%[0m
start "Rain" /B python rain.py --header-top 0 --header-bottom 5 --box-top 8 --box-bottom 11 --box-left 23 --box-right 56 --prompt-row 12
set /p choice=Select option:

taskkill /FI "WINDOWTITLE eq Rain" > nul

if "%choice%"=="1" (
    python TerminalAI.py %*
) else if "%choice%"=="2" (
    python shodanscan.py %*
) else (
    echo Invalid selection
    exit /b 1
)
