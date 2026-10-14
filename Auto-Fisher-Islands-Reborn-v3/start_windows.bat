@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
echo Only use automation where permitted by the game's rules.
echo Starts PAUSED. Focus Roblox, then F8 arms or pauses. F9 stops.
.venv\Scripts\python.exe main.py --enable-clicks
pause
