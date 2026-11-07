# ADR 0007 — Repo publik tidak menyimpan dataset maupun keluaran per akun

- Status: diterima
- Konteks: repo ini publik. Dataset Kaggle punya lisensi dan pemilik masing-masing, dan keluaran per akun
  (skor, level risiko, prediksi) melekat pada identitas akun nyata (id X, handle TikTok).
- Keputusan: tidak ada dataset dan tidak ada keluaran per akun di repo. Kedua jalur membaca file Kaggle
  asli yang diunduh `scripts/download_datasets.py` ke `data/landing/` (di-gitignore). Yang di-commit
  hanya kode, konfigurasi, dokumentasi, metrik/gambar agregat yang sudah diperiksa bebas identitas,
  dan demo kecil dari dataset **sintetis** (`twitter_bot_detection`, username buatan Faker).
  `.gitignore` menutup `data/**`, `outputs/predictions/`, `outputs/powerbi/`, `outputs/models/`,
  `outputs/logs/`, `.env`, dan `kaggle.json`. Contoh id akun di dokumen disamarkan (`x_<id>`).
- Konsekuensi: hasil lengkap hanya ada setelah orang menjalankan pipeline sendiri. File metrik yang memuat
  id baris/akun (mis. hasil audit duplikasi atau verifikasi MongoDB) tidak di-commit; dashboard demo
  memakai agregat yang dipindai pola identitas (`scripts/export_demo_snapshot.py`).
