# Konversi Rekaman Rokoko Smartgloves (FBX)

Converter ini digunakan untuk alur tanpa fitur livestream berlangganan:

```text
Smartgloves -> Rokoko Studio recording -> FBX -> qpos.csv -> replay/analisis aplikasi
```

## Menjalankan

Di PowerShell, dari akar proyek:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\convert_rokoko_fbx_to_qpos.ps1 `
  -InputFbx 'C:\Progres PA\Clip.fbx' `
  -OutputDir '.\records\normal\subject_01\rokoko_open_to_fist_01' `
  -Hand right `
  -MotionType open_to_fist `
  -SubjectId subject_01
```

`-ExecutionPolicy Bypass` hanya berlaku untuk proses perintah tersebut dan diperlukan pada beberapa instalasi Windows yang memblokir berkas `.ps1` lokal.

Keluaran di `OutputDir`:

- `qpos.csv`: 23 kolom `qpos_0` sampai `qpos_22`, kompatibel dengan loader GUI;
- `meta.json`: metadata trial;
- `rokoko_fbx_mapping_report.json`: bone, frame rate, frame kalibrasi, dan axis yang terdeteksi.

## Kalibrasi dan validasi

Untuk rekaman `open_to_fist`, tahan tangan terbuka pada 15 frame pertama. Untuk `fist_to_open`, tahan tangan terbuka pada 15 frame terakhir. Converter memakai bagian itu sebagai pose nol.

Axis dan tanda fleksi dideteksi otomatis dari animasi FBX karena skeleton/Rokoko export preset dapat memakai orientasi lokal yang berbeda. Hasil harus dibuka melalui replay MuJoCo sebelum dipakai untuk dataset: pose awal harus terbuka dan gerak menutup harus membengkokkan semua jari ke arah yang benar.

Untuk tangan kiri, ganti `-Hand left`. Rekaman satu file sebaiknya hanya memuat satu arah gerak agar inferensi axis dan tanda stabil.
