@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Install 64-bit Python 3.11 or 3.12 from python.org first.
  pause
  exit /b 1
)
py -3 -m venv .venv
if errorlevel 1 goto failed
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Setup complete. Run preview_windows.bat first. No clicks are enabled.
pause
exit /b 0
:failed
echo Setup failed. Read the error above before continuing.
pause
exit /b 1
