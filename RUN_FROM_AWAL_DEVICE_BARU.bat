@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo ============================================================
echo Hand Demo - Setup dan Run dari Awal
echo Repo: %CD%
echo ============================================================
echo.

where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Conda tidak ditemukan di PATH.
    echo Buka dari "Anaconda Prompt", atau install Anaconda/Miniconda dulu.
    pause
    exit /b 1
)

call conda env list | findstr /R /C:"^hand-demo " >nul 2>nul
if errorlevel 1 (
    echo [INFO] Env hand-demo belum ada. Membuat env Python 3.8...
    call conda create -n hand-demo python=3.8 -y
    if errorlevel 1 (
        echo [ERROR] Gagal membuat env hand-demo.
        pause
        exit /b 1
    )
)

echo [INFO] Mengaktifkan env hand-demo...
call conda activate hand-demo
if errorlevel 1 (
    echo [ERROR] Gagal activate env hand-demo.
    pause
    exit /b 1
)

echo.
echo [INFO] Python yang dipakai:
where python
python --version

echo.
echo [INFO] Cek versi code LeapC resolver...
python -c "from app.realtime.leapc_qpos_publisher import resolve_ultraleap_sdk_dir; print('LeapSDK resolved:', resolve_ultraleap_sdk_dir())"
if errorlevel 1 (
    echo.
    echo [ERROR] Code repo ini masih versi lama atau file belum lengkap.
    echo Pastikan file terbaru sudah dicopy/pull ke device baru.
    echo Minimal harus ada fungsi resolve_ultraleap_sdk_dir di:
    echo app\realtime\leapc_qpos_publisher.py
    pause
    exit /b 1
)

echo.
echo [INFO] Install/update dependencies dari requirements.txt...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Install requirements gagal.
    echo Cek koneksi internet, versi Python, dan error package di atas.
    pause
    exit /b 1
)

echo.
echo [INFO] Cek artifact model final...
if not exist "artifacts\dynamic_temporal_gcn_node_classifier_labeled_records\checkpoints\best_temporal_gcn.pt" (
    echo [WARN] Model final belum ditemukan:
    echo artifacts\dynamic_temporal_gcn_node_classifier_labeled_records\checkpoints\best_temporal_gcn.pt
    echo GUI bisa terbuka, tetapi analisis GCN final tidak lengkap.
)

echo.
echo [INFO] Smoke test GUI...
python scripts\gui\run_gui.py --smoke-test
if errorlevel 1 (
    echo.
    echo [ERROR] Smoke test gagal. Perbaiki error di atas dulu.
    pause
    exit /b 1
)

echo.
echo [INFO] Probe LeapC 8 detik. Letakkan tangan di atas Leap Motion.
python -c "import time; from app.realtime.leapc_qpos_publisher import resolve_ultraleap_sdk_dir; from app.gui.recorder_controller import LeapCQposProvider; sdk=resolve_ultraleap_sdk_dir(); p=LeapCQposProvider(sdk_dir=str(sdk), qpos_dim=23, tracking_mode='auto'); print('sdk=', sdk); print('start=', p.start(), p.message); [(time.sleep(0.8), print(p.status().message)) for _ in range(10)]; p.close()"

echo.
echo [INFO] Membuka GUI utama...
python scripts\gui\run_gui.py

endlocal
