# ADR 0006 — Satu `common.py`, dua jalur pemodelan dengan peran berbeda

- Status: diterima
- Konteks: dua hasil kerja digabung: pipeline operasional Rafi dan pipeline analisis lintas platform,
  masing-masing dengan `common.py` sendiri dan prosedur pemodelan sendiri.
- Keputusan:
  - **Satu `src/common.py`** yang memuat API Rafi apa adanya (`load_config`, `path_of`, `snake`,
    `standardize`, …) ditambah helper analisis (`stratified_split`, `evaluate_binary`, …) dan
    `get_mongo()`. Satu `get_spark(name, driver_memory="2g", cores=None)` melayani kedua gaya panggilan.
  - **Dua jalur pemodelan dengan peran yang jelas**: jalur operasional (split 70/15/15, LR/RF/GBT dipilih
    di data validasi) menjalankan data lake, streaming dan dashboard; **jalur analisis** (split 80:20
    berstrata, RF + LR, CV di dalam data latih, audit leakage) **menghasilkan setiap angka yang
    dilaporkan** di `docs/FINAL_REPORT.md`.
  - Kedua jalur membaca `data/landing/` yang sama, tetapi menulis ke zona berbeda
    (`data/{bronze,silver,gold}` vs `data/curated` + `outputs/`).
- Konsekuensi: angka dashboard halaman "Model & Evaluasi" (jalur operasional) bisa berbeda sedikit dari
  laporan (jalur analisis); README dan laporan menyebutkan ini. Jalur analisis memanggil `get_spark`
  dengan `cores=3` eksplisit agar partisi (dan split ber-seed) sama dengan run acuan.
