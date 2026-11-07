# Metodologi

Kelompok "Penyayang Claude" · Teknologi Big Data, SV IPB University.

Aturan di bawah dipakai oleh **semua** pipeline analisis per platform (`src/twitter_pipeline.py`,
`src/instagram_pipeline.py`, `src/facebook_pipeline.py`, `src/tiktok_pipeline.py`) dan oleh sintesis
lintas platform (`src/synthesis.py`), supaya hasilnya dapat dibandingkan. Aturan ini berasal dari
catatan dosen dan dokumen konsep kelompok. Helper bersama ada di `src/common.py`
(`stratified_split`, `evaluate_binary`, `importance_table`, `save_metrics`).

## Tujuh aturan metode

Dokumen temuan dan JSON metrik menyebut aturan ini sebagai *contract rule 1–7* (nomornya sama).

1. **Identitas tidak pernah menjadi fitur.** `profile id`, `User ID`, `Username`, `id`, `screen_name`,
   `author`, `video_id` (dan indeks baris seperti `c0`/`split`) dibuang. ID yang berkorelasi dengan
   label berarti file tersusun menurut label: laporkan sebagai temuan leakage, jangan dimodelkan.
2. **Jangan menyeimbangkan kelas sebelum split.** Split berstrata 80:20 lewat
   `common.stratified_split`, seed 42. Pembobotan kelas *di dalam* model boleh.
3. **Akurasi saja tidak pernah menjadi kesimpulan.** Selalu laporkan precision, recall, F1,
   balanced accuracy, MCC, ROC-AUC, PR-AUC dan confusion matrix, dan bandingkan dengan baseline kelas
   mayoritas dari `common.evaluate_binary`. Model yang tidak jelas melampaui baseline itu tidak
   menemukan apa-apa.
4. **±80% adalah target, bukan kuota.** Angka rendah boleh asal jujur; angka mendekati 99% wajib
   diaudit dulu (leakage, duplikasi, file yang tersusun urut) sebelum dilaporkan, dan sebutkan apa
   saja yang sudah diperiksa.
5. **Kosong bukan nol.** Imputasi secara eksplisit dan jelaskan caranya, atau biarkan NA.
6. **Jangan pernah menyetel (tuning) di data uji.** Bila ada tuning, lakukan cross-validation di dalam
   data latih.
7. **Target operasional `0 = Authentic`, `1 = Inauthentic`.** Label sumber (spam / fake / bot) **bukan**
   konsep yang sama; pemetaannya adalah konvensi operasional dan keterbatasan itu harus disebutkan.

## Model

Random Forest (Spark MLlib) adalah metode utama; Logistic Regression adalah baseline. Keduanya dilatih
pada split yang sama dan keduanya dilaporkan. Bila kelas timpang, dipakai kolom bobot setara
`class_weight`. Setiap hyperparameter yang diset dicatat di JSON metrik.

## Keluaran per track

| Path | Isi |
|---|---|
| `outputs/metrics/<track>.json` | metrik, split, fitur yang dipakai/dibuang, audit leakage, keterbatasan (via `common.save_metrics`) |
| `outputs/predictions/<track>_scored.csv` | skor per akun — **tidak di-commit** (`.gitignore`) |
| `outputs/figures/<track>_*.png` | confusion matrix, importance, dan gambar audit |
| `docs/<track>_findings.md` | temuan tertulis |

Bentuk JSON metrik: `platform`, `dataset` (nama, sumber, `n_rows`, kolom label, arti kelas positif),
`class_balance`, `split`, `features` (`used`, `dropped` + alasan, `engineered` + rumus), `models`
(`random_forest`, `logistic_regression`: `params`, `metrics`, `feature_importance`), `leakage_audit`
(`checks_run`, `findings`), `limitations`.

## Isi laporan temuan

Dalam bahasa sederhana: dataset dan asal-usulnya yang sebenarnya; keseimbangan kelas; apa yang dibuang
dan kenapa; split; metrik utama kedua model terhadap baseline mayoritas; fitur teratas dan artinya
secara perilaku; audit leakage dan hasilnya; serta keterbatasan yang wajib diketahui pembaca sebelum
memercayai angkanya. *False positive* berarti orang sungguhan ikut ditandai: keluaran sistem ini adalah
**antrean tinjauan manusia**, bukan sanksi otomatis.

## Konvensi

Seed 42 di mana pun. Angka yang dilaporkan dibulatkan 4 desimal. Prosa di `docs/` boleh berbahasa
Indonesia; kode, nama identifier dan kunci JSON tetap berbahasa Inggris.

## Hubungan dengan pipeline operasional

Pipeline `ingest.py → process.py → train.py → stream.py` memakai split 70/15/15 dan memilih model di
data validasi; perannya operasional (alur data lake, streaming, dashboard). **Semua angka di
[FINAL_REPORT.md](FINAL_REPORT.md) berasal dari jalur analisis di atas.** Lihat
[ADR 0006](adr/0006-common-module-and-two-modelling-paths.md).
