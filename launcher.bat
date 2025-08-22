@echo off
chcp 65001 > nul
for /F "delims=" %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "PYTHON=python"
where python3 > nul 2>&1 && set "PYTHON=python3"

cls
echo %ESC%[?25l

for /F "tokens=2 delims=:" %%a in ('mode con ^| findstr /R "Columns"') do set "COLS=%%a"
for /F "tokens=2 delims=:" %%a in ('mode con ^| findstr /R "Lines"') do set "ROWS=%%a"
set /a COLS=%COLS: =%
set /a ROWS=%ROWS: =%

set /a HEADER_HEIGHT=6
set /a HEADER_WIDTH=76
set /a HEADER_TOP=1
set /a HEADER_LEFT=(COLS-HEADER_WIDTH)/2+1
set /a HEADER_BOTTOM=HEADER_TOP+HEADER_HEIGHT-1
set /a HEADER_RIGHT=HEADER_LEFT+HEADER_WIDTH-1

set /a BOX_WIDTH=34
set /a BOX_HEIGHT=4
set /a BOX_TOP=HEADER_BOTTOM+2
set /a BOX_LEFT=(COLS-BOX_WIDTH)/2+1
set /a BOX_BOTTOM=BOX_TOP+BOX_HEIGHT-1
set /a BOX_RIGHT=BOX_LEFT+BOX_WIDTH-1

start "Rain" /B %PYTHON% "%SCRIPT_DIR%scripts\rain.py" --persistent --no-clear ^
  --exclude !HEADER_TOP!,!HEADER_BOTTOM!,!HEADER_LEFT!,!HEADER_RIGHT! ^
  --exclude !BOX_TOP!,!BOX_BOTTOM!,!BOX_LEFT!,!BOX_RIGHT!

set /a ROW=HEADER_TOP
echo %ESC%[!ROW!;!HEADER_LEFT!H%ESC%[32;1m████████╗███████╗██████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██╗      █████╗ ██╗
set /a ROW+=1
echo %ESC%[!ROW!;!HEADER_LEFT!H╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██║     ██╔══██╗██║
set /a ROW+=1
echo %ESC%[!ROW!;!HEADER_LEFT!H   ██║   █████╗  ██████╔╝██╔████╔██║██║██╔██╗ ██║███████║██║     ███████║██║
set /a ROW+=1
echo %ESC%[!ROW!;!HEADER_LEFT!H   ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║╚██╗██║██╔══██║██║     ██╔══██║██║
set /a ROW+=1
echo %ESC%[!ROW!;!HEADER_LEFT!H   ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║███████╗██║  ██║██║
set /a ROW+=1
echo %ESC%[!ROW!;!HEADER_LEFT!H   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝%ESC%[0m

set /a ROW=BOX_TOP
set "HORZ="
for /L %%i in (1,1,!BOX_WIDTH!-2) do set "HORZ=!HORZ!─"
echo %ESC%[!ROW!;!BOX_LEFT!H%ESC%[32m┌!HORZ!┐
for /L %%i in (1,1,!BOX_HEIGHT!-2) do (
  set /a ROW=BOX_TOP+%%i
  echo %ESC%[!ROW!;!BOX_LEFT!H│%ESC%[!ROW!;!BOX_RIGHT!H│
)
set /a ROW=BOX_BOTTOM
echo %ESC%[!ROW!;!BOX_LEFT!H└!HORZ!┘%ESC%[0m

set /a OPTION_COL=BOX_LEFT+2
set /a ROW=BOX_TOP+1
echo %ESC%[!ROW!;!OPTION_COL!H%ESC%[32m1) Start TerminalAI
set /a ROW+=1
echo %ESC%[!ROW!;!OPTION_COL!H2) Scan Shodan%ESC%[0m

choice /c 12 /n >nul
set "CHOICE=%errorlevel%"

call :cleanup

if "%CHOICE%"=="1" (
  %PYTHON% "%SCRIPT_DIR%scripts\TerminalAI.py" %*
) else (
  %PYTHON% "%SCRIPT_DIR%scripts\shodanscan.py" %*
)
exit /b

:cleanup
taskkill /FI "WINDOWTITLE eq Rain" > nul 2>&1
echo %ESC%[?25h
cls
exit /b
