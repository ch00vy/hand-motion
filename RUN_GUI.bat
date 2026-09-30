@echo off
cd /d "%~dp0"

where conda >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    call conda activate hand-demo
)

set "HAND_QPOS_SOURCE=udp"
set "HAND_QPOS_UDP_HOST=127.0.0.1"
set "HAND_QPOS_UDP_PORT=15000"

python scripts\gui\run_gui.py
