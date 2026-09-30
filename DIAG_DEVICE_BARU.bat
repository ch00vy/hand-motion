@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo ============================================================
echo Hand Demo - Diagnostik Device Baru
echo Repo: %CD%
echo ============================================================
echo.

where conda
where python
python --version

echo.
echo [INFO] Git:
where git
git --version

echo.
echo [INFO] Ultraleap folder standar:
if exist "C:\Program Files\Ultraleap" (
    dir "C:\Program Files\Ultraleap"
) else (
    echo C:\Program Files\Ultraleap tidak ditemukan.
)

echo.
echo [INFO] Code resolver:
python -c "from app.realtime.leapc_qpos_publisher import resolve_ultraleap_sdk_dir; print('LeapSDK resolved:', resolve_ultraleap_sdk_dir())"
if errorlevel 1 (
    echo [ERROR] resolve_ultraleap_sdk_dir belum ada. Repo di device ini masih versi lama.
    pause
    exit /b 1
)

echo.
echo [INFO] Smoke test:
python scripts\gui\run_gui.py --smoke-test

echo.
echo [INFO] Probe LeapC 8 detik. Letakkan tangan di atas Leap Motion.
python -c "import time; from app.realtime.leapc_qpos_publisher import resolve_ultraleap_sdk_dir; from app.gui.recorder_controller import LeapCQposProvider; sdk=resolve_ultraleap_sdk_dir(); p=LeapCQposProvider(sdk_dir=str(sdk), qpos_dim=23, tracking_mode='auto'); print('sdk=', sdk); print('start=', p.start(), p.message); [(time.sleep(0.8), print(p.status().message)) for _ in range(10)]; p.close()"

echo.
echo [INFO] Selesai. Kirim semua output ini kalau masih gagal.
pause
endlocal
