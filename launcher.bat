@echo off
chcp 65001 > nul
for /F "delims=" %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "PYTHON=python3"

set "GREEN=%ESC%[32m"
set "RESET=%ESC%[0m"
set "BOLD=%ESC%[1m"

cls
echo %ESC%[?25l

for /F "tokens=2 delims=:" %%a in ('mode con ^| findstr /R "Columns"') do (
  for %%b in (%%a) do set "COLS=%%b"
)
for /F "tokens=2 delims=:" %%a in ('mode con ^| findstr /R "Lines"') do (
  for %%b in (%%a) do set "ROWS=%%b"
)

set /a HEADER_HEIGHT=6
set /a HEADER_WIDTH=76
set /a HEADER_TOP=0
set /a HEADER_LEFT=(COLS-HEADER_WIDTH)/2
set /a HEADER_BOTTOM=HEADER_TOP+HEADER_HEIGHT-1
set /a HEADER_RIGHT=HEADER_LEFT+HEADER_WIDTH-1

set /a BOX_WIDTH=34
set /a BOX_HEIGHT=4
set /a BOX_TOP=HEADER_BOTTOM+2
set /a BOX_LEFT=(COLS-BOX_WIDTH)/2
set /a BOX_BOTTOM=BOX_TOP+BOX_HEIGHT-1
set /a BOX_RIGHT=BOX_LEFT+BOX_WIDTH-1

set /a RAIN_HEADER_TOP=HEADER_TOP+1
set /a RAIN_HEADER_BOTTOM=HEADER_BOTTOM+1
set /a RAIN_HEADER_LEFT=HEADER_LEFT+1
set /a RAIN_HEADER_RIGHT=HEADER_RIGHT+1
set /a RAIN_BOX_TOP=BOX_TOP+1
set /a RAIN_BOX_BOTTOM=BOX_BOTTOM+1
set /a RAIN_BOX_LEFT=BOX_LEFT+1
set /a RAIN_BOX_RIGHT=BOX_RIGHT+1

set /a ROW=HEADER_TOP+1
set /a COL=HEADER_LEFT+1
echo %ESC%[!ROW!;!COL!H!BOLD!!GREEN!████████╗███████╗██████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██╗      █████╗ ██╗
set /a ROW+=1
echo %ESC%[!ROW!;!COL!H╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██║     ██╔══██╗██║
set /a ROW+=1
echo %ESC%[!ROW!;!COL!H   ██║   █████╗  ██████╔╝██╔████╔██║██║██╔██╗ ██║███████║██║     ███████║██║
set /a ROW+=1
echo %ESC%[!ROW!;!COL!H   ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║╚██╗██║██╔══██║██║     ██╔══██║██║
set /a ROW+=1
echo %ESC%[!ROW!;!COL!H   ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║███████╗██║  ██║██║
set /a ROW+=1
echo %ESC%[!ROW!;!COL!H   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝!RESET!

set /a ROW=BOX_TOP+1
set /a COL=BOX_LEFT+1
set "HORZ="
for /L %%i in (1,1,!BOX_WIDTH!-2) do set "HORZ=!HORZ!─"
echo %ESC%[!ROW!;!COL!H!GREEN!┌!HORZ!┐
for /L %%i in (1,1,!BOX_HEIGHT!-2) do (
  set /a ROW=BOX_TOP+1+%%i
  echo %ESC%[!ROW!;!COL!H│%ESC%[!ROW!;!BOX_RIGHT+1!H│
)
set /a ROW=BOX_BOTTOM+1
echo %ESC%[!ROW!;!COL!H└!HORZ!┘!RESET!

set /a OPTION_COL=BOX_LEFT+3
set /a ROW=BOX_TOP+2
echo %ESC%[!ROW!;!OPTION_COL!H!GREEN!1) Start TerminalAI
set /a ROW+=1
echo %ESC%[!ROW!;!OPTION_COL!H2) Scan Shodan!RESET!

start "Rain" /B %PYTHON% "%SCRIPT_DIR%scripts\rain.py" --persistent --no-clear ^
  --exclude !RAIN_HEADER_TOP!,!RAIN_HEADER_BOTTOM!,!RAIN_HEADER_LEFT!,!RAIN_HEADER_RIGHT! ^
  --exclude !RAIN_BOX_TOP!,!RAIN_BOX_BOTTOM!,!RAIN_BOX_LEFT!,!RAIN_BOX_RIGHT!

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
