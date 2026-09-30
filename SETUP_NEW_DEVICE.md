# Setup Device Baru

Panduan ini untuk Windows + Anaconda + Ultraleap/Leap Motion.

## 1. Install Prasyarat

Install di device baru:

- Git
- Anaconda atau Miniconda
- Ultraleap Gemini / Tracking Software
- Ultraleap LeapSDK

Lokasi SDK yang dicoba otomatis oleh aplikasi:

```text
C:\Program Files\Ultraleap\LeapSDK
C:\Program Files\Ultraleap\*\LeapSDK
C:\Program Files (x86)\Ultraleap\*\LeapSDK
```

Jika SDK berada di lokasi lain, set sekali:

```bat
setx ULTRALEAP_SDK_DIR "D:\path\ke\LeapSDK"
```

Pastikan Ultraleap Tracking Service berjalan dan Leap Motion terbaca di aplikasi Ultraleap Control Panel.

## 2. Copy Repo

Letakkan repo di lokasi seperti:

```bat
C:\Users\<USER>\Documents\hand
```

Masuk ke folder repo:

```bat
cd /d C:\Users\<USER>\Documents\hand
```

## 3. Pastikan Artifact Wajib Ada

Folder `artifacts/` harus ikut disalin dari device lama atau backup. Minimal wajib ada:

```text
artifacts/dynamic_temporal_gcn_node_classifier_labeled_records/checkpoints/best_temporal_gcn.pt
artifacts/retrain_labeled_records/reference_sequence/reference/reference_sequence.npz
artifacts/retrain_labeled_records/reference_sequence/reference/reference_summary.json
artifacts/retrain_labeled_records/reference_sequence/reference/reference_sequence_edges.csv
artifacts/retrain_labeled_records/reference_sequence/reference/reference_sequence_nodes.csv
artifacts/retrain_labeled_records/graph_sequence_dataset/dataset_manifest.csv
```

Folder canonical dataset juga perlu punya minimal satu file graph JSON di:

```text
artifacts/retrain_labeled_records/graph_sequence_dataset/
```

Tanpa artifact ini, GUI masih bisa merekam qpos, tetapi analisis GCN final tidak akan lengkap.

## 4. Buat Conda Environment

Gunakan Python 3.8 karena kombinasi Leap/MyoSuite di project ini sudah pernah tervalidasi dengan env `leapc-py38`.

```bat
conda create -n hand-demo python=3.8 -y
conda activate hand-demo
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Jika install `torch` dari pip gagal atau terlalu besar, install CPU wheel sesuai rekomendasi PyTorch untuk Windows, lalu ulangi:

```bat
python -m pip install -r requirements.txt
```

## 5. Smoke Test

Jalankan:

```bat
python scripts\gui\run_gui.py --smoke-test
```

Output yang diharapkan memuat:

```text
qpos_source_default: leapc
leap_tracking_mode_default: auto
ultraleap_sdk_resolved: ...LeapSDK
ctrl_preview_mode_default: pd_dynamic
prediction_deviating_threshold: 0.990
model_path: ...best_temporal_gcn.pt
reference_dir: ...reference_sequence\reference
```

## 6. Jalankan GUI

Command utama:

```bat
python scripts\gui\run_gui.py
```

Default sudah di-code, jadi tidak perlu `set HAND_...` manual.

## 7. Troubleshooting Cepat

Jika muncul `No module named numpy._core`:

- checkpoint yang disalin kemungkinan masih format NumPy 2-only.
- gunakan checkpoint yang sudah dikonversi kompatibel NumPy 1/2, atau jalankan analisis dengan env yang sama seperti device lama.

Jika muncul error LeapC / SDK:

- cek status GUI. Error sekarang menampilkan path SDK yang sedang dicoba.
- pastikan path SDK berisi subfolder `leapc_cffi`
- jika SDK tidak ada di lokasi standar, set `ULTRALEAP_SDK_DIR`
- pastikan Ultraleap Tracking Service berjalan
- pastikan device terbaca di Ultraleap Control Panel

Jika model tidak bergerak walaupun Ultraleap mendeteksi tangan:

- cek status QPOS. Harus muncul `Direct LeapC qpos live`.
- kalau muncul `waiting for hand tracking qpos`, Python belum menerima skeleton frame dari LeapC.
- kalau muncul `MuJoCo preview not rendering`, masalahnya ada di MyoSuite/MuJoCo env, bukan Leap Motion.
- cek `artifacts/dynamic_temporal_gcn_node_classifier_labeled_records/checkpoints/best_temporal_gcn.pt`
- cek folder reference di `artifacts/retrain_labeled_records/reference_sequence/reference`
- jalankan ulang `python scripts\gui\run_gui.py --smoke-test`

Jika muncul `bad git executable` atau `Failed to initialize: Bad git executable` saat MyoSuite dibuat:

- install Git for Windows dari https://git-scm.com/download/win
- tutup Anaconda Prompt, buka lagi
- cek:

```bat
git --version
```

Kode aplikasi juga otomatis mencoba lokasi standar:

```text
C:\Program Files\Git\cmd\git.exe
```

Jika tidak ada Leap Motion di device baru:

- GUI masih bisa membuka trial lama dari `records/`
- mode realtime recording tidak akan aktif sampai device Leap terbaca
