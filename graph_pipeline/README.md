# Muscle Graph Pipeline: Local JSONL -> PyTorch `.pt`

Pipeline ini sekarang bersifat muscle-centric:

- node = actuator / muscle MyoSuite
- edge = relasi antar muscle
- prior biological graph tetap statis
- dynamic graph per frame / exercise adalah subset aktif dari prior graph
- layout visual default sekarang berbentuk diagram zona anatomis, tetapi semantik graph tetap muscle-to-muscle
- `qpos_*` hanya auxiliary traceability, bukan graph node
- `ctrl_*` tidak lagi dianggap label saja, tetapi menjadi bagian dari feature node muscle

## Desain graph

Spesifikasi graph utama ada di:

- `graph_pipeline/muscle_graph_spec.py`

Untuk env default `myoHandPoseRandom-v0`, pipeline membangun:

- 39 node muscle / actuator
- edge tipe:
  - `synergist`
  - `antagonist`
  - `functional_group`
  - `data_driven` opsional

Relasi dibangun dari dua sumber:

- grouping fungsional muscle yang ditentukan eksplisit
- signature moment-arm actuator terhadap joint MyoSuite

## Semantik graph baru

Sistem sekarang memakai dua layer graph:

- `prior biological graph`
  - statis
  - menyimpan semua relasi biologis yang plausible
  - edge type tetap: `synergist`, `antagonist`, `functional_group`, `data_driven`
- `dynamic active graph`
  - berubah per frame atau window pendek
  - hanya edge prior yang sedang aktif pada exercise saat ini
  - visibility edge ditentukan dari activation node + dynamic edge score

Dengan kata lain:

- prior graph = apa yang secara biologis mungkin
- dynamic graph = apa yang benar-benar aktif saat exercise berlangsung

## Sumber data

Untuk file rekaman seperti `records/ctrl_est_*.csv`:

- `ctrl_*` dipetakan langsung ke muscle sesuai urutan actuator MyoSuite
- `qpos_*` dipakai untuk me-recover state muscle pada pose tersebut
- feature node diisi dari signal muscle:
  - `ctrl`
  - `activation`
  - `force`
  - `length`
  - `velocity`

State muscle tersebut direkonstruksi secara statik dari model MyoSuite pada pose `qpos` yang terekam.

## Format graph store lokal

Graph disimpan ke file `JSONL`, default:

- `artifacts/graphs/graph_samples.jsonl`

Setiap baris adalah satu graph sample. Field utama:

- `graph_id`
- `graph_key`
- `split`
- `graph_kind`
- `nodes`
- `edges`
- `joint_qpos`
- `joint_qpos_names`
- `meta`

Contoh ringkas:

```json
{
  "graph_id": 1,
  "graph_key": "record:ctrl_est_20260216_180727:0",
  "split": "train",
  "graph_kind": "muscle_graph",
  "nodes": [
    {
      "node_name": "ECRL",
      "node_type": "muscle",
      "features": [0.02, 0.02, -19.3, 0.34, 0.0],
      "metadata": {
        "muscle_index": 0,
        "functional_groups": ["wrist_extensors"],
        "feature_names": ["ctrl", "activation", "force", "length", "velocity"]
      }
    }
  ],
  "edges": [
    {
      "edge_id": 1,
      "source_name": "ECRL",
      "target_name": "ECRB",
      "edge_type": "functional_group",
      "edge_weight": 1.0,
      "features": [1.0],
      "metadata": {
        "shared_groups": ["wrist_extensors"]
      }
    }
  ],
  "joint_qpos": [1.87, -0.17],
  "meta": {
    "source_csv": "records/ctrl_est_20260216_180727.csv",
    "env_id": "myoHandPoseRandom-v0"
  }
}
```

## Ingest dari CSV

Script utama:

- `graph_pipeline/ingest_qpos_csv_to_local.py`

Contoh:

```powershell
.\venv\Scripts\python.exe .\graph_pipeline\ingest_qpos_csv_to_local.py `
  --csv .\records\ctrl_est_20260216_180727.csv `
  --split train `
  --output .\artifacts\graphs\graph_samples.jsonl
```

Untuk menimpa store lama:

```powershell
.\venv\Scripts\python.exe .\graph_pipeline\ingest_qpos_csv_to_local.py `
  --csv .\records\ctrl_est_20260216_180727.csv `
  --overwrite
```

Untuk menambah edge `data_driven`:

```powershell
.\venv\Scripts\python.exe .\graph_pipeline\ingest_qpos_csv_to_local.py `
  --csv .\records\ctrl_est_20260216_180727.csv `
  --include-data-driven-edges
```

## Runtime live + buffer exercise

Sekarang runtime utama mendukung dua tahap tambahan:

- update feature node muscle secara live per frame
- buffer sequence per exercise menjadi `x_seq`
- render hanya dynamic active edges, bukan semua prior edges
- fixed zone-based node layout supaya posisi muscle stabil frame-ke-frame

Konfigurasi visual muscle graph sekarang punya section terpisah:

- `use_hand_layout`
- `node_activation_threshold`
- `edge_score_threshold`
- `scoring_mode`
- `show_background_hand`
- `show_labels`

Jalankan:

```powershell
.\venv\Scripts\python.exe .\leap_myosuite.py --muscle-graph-window
```

Threshold dan scoring mode bisa diatur:

```powershell
.\venv\Scripts\python.exe .\leap_myosuite.py `
  --muscle-graph-window `
  --dynamic-edge-mode coactivation `
  --node-activation-threshold 0.02 `
  --edge-score-threshold 0.0005 `
  --dynamic-edge-window 12 `
  --debug-dynamic-edges
```

Opsional rendering:

```powershell
.\venv\Scripts\python.exe .\leap_myosuite.py `
  --muscle-graph-window `
  --hide-hand-background `
  --hide-labels
```

Untuk sweep threshold pada rekaman tertentu:

```powershell
.\venv\Scripts\python.exe .\graph_pipeline\calibrate_dynamic_edges.py `
  --csv .\records\ctrl_est_20260216_180727.csv `
  --max-frames 12
```

Untuk rekaman yang ada saat ini, konfigurasi default yang sekarang dipakai cukup seimbang:

- `node_activation_threshold = 0.02`
- `edge_score_threshold = 0.00065`
- `scoring_mode = coactivation`

Pada sample replay yang diuji, konfigurasi ini menghasilkan sekitar `190-204` active edges per frame, turun dari sekitar `440+` edge pada threshold yang terlalu longgar.

Kontrol runtime:

- `r` untuk mulai / stop recording exercise
- `x` untuk keluar
- `0` untuk manual reset tracking

Saat recording dihentikan, sistem sekarang menyimpan:

- CSV raw ke folder `records/`
- artifact sequence `.pt` ke `artifacts/graphs/sequences/`
- summary `.json` ke `artifacts/graphs/sequences/`

Format sequence artifact:

- prior graph statis tetap
- `x_seq`: `Tensor[T, num_muscles, num_features]`
- `joint_qpos_seq`: `Tensor[T, num_joints]`
- `dynamic_edge_mask_seq`: edge prior mana yang aktif per frame
- `dynamic_edge_weight_seq`: skor edge dinamis per frame
- `visible_edge_records_seq`: record edge aktif per frame

Untuk inspeksi artifact sequence:

```powershell
.\venv\Scripts\python.exe .\graph_pipeline\show_exercise_sequence.py `
  .\artifacts\graphs\sequences\exercise_demo_runtime.pt
```

Untuk menguji jalur live muscle graph tanpa Leap device, replay file CSV rekaman:

```powershell
.\venv\Scripts\python.exe .\graph_pipeline\replay_muscle_graph.py `
  --csv .\records\ctrl_est_20260216_180727.csv `
  --save-sequence `
  --node-activation-threshold 0.02 `
  --edge-score-threshold 0.0005 `
  --debug-dynamic-edges
```

Untuk export contoh screenshot `fist`, `finger_stretch`, dan `making_o` dari replay:

```powershell
.\venv\Scripts\python.exe .\graph_pipeline\replay_muscle_graph.py `
  --csv .\records\ctrl_est_20260402_231408.csv `
  --no-window `
  --export-pose-examples .\artifacts\graphs\hand_pose_examples
```

Dokumentasi singkat khusus visual zone-based muscle graph ada di:

- `graph_pipeline/HAND_MUSCLE_VISUALIZATION.md`

Jika hanya ingin validasi pipeline tanpa membuka window:

```powershell
.\venv\Scripts\python.exe .\graph_pipeline\replay_muscle_graph.py `
  --csv .\records\ctrl_est_20260216_180727.csv `
  --no-window `
  --max-frames 30 `
  --save-sequence
```

## Export ke `.pt`

Script export:

- `graph_pipeline/export_to_pt.py`

Contoh:

```powershell
.\venv\Scripts\python.exe .\graph_pipeline\export_to_pt.py `
  --input .\artifacts\graphs\graph_samples.jsonl `
  --split train `
  --output .\artifacts\train_muscle_graphs.pt
```

## Format artifact `.pt`

Artifact hasil export sekarang muscle-centric. Top-level:

```python
{
    "schema_version": 2,
    "graph_kind": "muscle_graph",
    "edge_type_names": ["synergist", "antagonist", "functional_group", "data_driven"],
    "graphs": [...]
}
```

Setiap graph sample berisi:

- `node_names`
- `x`
- `edge_index`
- `edge_type`
- `edge_weight`
- `meta`
- `joint_qpos_seq` opsional

Secara bentuk:

```python
{
    "graph_id": 1,
    "graph_key": "...",
    "node_names": ["ECRL", "ECRB", ...],
    "x": Tensor[num_muscles, 5],
    "edge_index": Tensor[2, num_edges],
    "edge_type": Tensor[num_edges],
    "edge_weight": Tensor[num_edges],
    "meta": {...},
    "joint_qpos_seq": Tensor[1, num_qpos]
}
```

## Dependency

Script ingest dan spec graph muscle butuh environment yang bisa membuka MyoSuite hand model:

- `mujoco`
- `myosuite`

Script export butuh:

- `torch`

## Langkah praktis

1. ingest sebagian data CSV ke `JSONL`
2. cek satu sample graph untuk melihat 39 muscle nodes dan relation edges
3. export ke `.pt`
4. pakai artifact itu untuk PyTorch / Graph Transformer
