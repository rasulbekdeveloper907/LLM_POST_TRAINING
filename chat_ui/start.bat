@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo Model yuklanmoqda... Brauzer 20 soniyadan keyin ochiladi.
start "" /min cmd /c "timeout /t 20 >/dev/null & start http://127.0.0.1:8472"
python server.py
pause
