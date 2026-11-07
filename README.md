# Sistem Big Data — Deteksi Akun Bot/Spam Media Sosial

Prototipe pipeline end-to-end (Spark + Streamlit) untuk proyek Teknologi Big Data.

```
Landing (CSV mentah) → Ingestion (Spark batch, manifest sha256) → Bronze (Parquet)
  → Processing (Spark: dedup, label 0/1, fitur teks/waktu) → Silver → Gold (agregat, EDA)
  → Modeling (Spark MLlib: LogReg / RandomForest / GBT, dipilih di data validasi)
  → Streaming scoring (Structured Streaming, micro-batch) → Dashboard Streamlit
```

## Tech stack
| Lapisan | Teknologi |
|---|---|
| Bahasa | Python 3.12 (versi yang dipakai CI) |
| Pemrosesan batch & streaming | Apache Spark 3.5.3 (PySpark, Spark SQL, Structured Streaming) di atas **Java 17** |
| Machine learning | Spark MLlib (Logistic Regression, Random Forest, GBT) |
| Penyimpanan data lake | Parquet (zona bronze/silver/gold), pyarrow, pandas |
| Dashboard | Streamlit + Plotly |
| Sumber data | Kaggle lewat `kagglehub` (`scripts/download_datasets.py`) |
| Kualitas kode | ruff (lint), pytest (test), GitHub Actions (CI) — lihat [CONTRIBUTING.md](CONTRIBUTING.md) |

## ⚠️ Perbaikan link dataset (kenapa link lama tidak bisa dibuka)
Ketiga slug di dokumen "LINK DATASET" dan notebook EDA kehilangan tanda hubung. Yang benar:

| Platform | Slug yang SALAH | Slug yang BENAR |
|---|---|---|
| Twitter | `goyaladi/twitter-botdetection-dataset` | `goyaladi/twitter-bot-detection-dataset` |
| Instagram | `free4ever1/instagram-fakespammer-genuine-accounts` | `free4ever1/instagram-fake-spammer-genuine-accounts` |
| Facebook | `khajahussainsk/facebookspam-dataset` | `khajahussainsk/facebook-spam-dataset` |

## Cara menjalankan
Butuh Python 3.12 (3.10+ seharusnya jalan), **Java 17** (Spark), dan `pip install -r requirements.txt`
(di Colab: `!apt-get -qq install openjdk-17-jdk-headless` lalu `pip install pyspark==3.5.3`).

```bash
python scripts/download_datasets.py     # unduh ke data/landing/<kunci>/ (butuh kaggle.json), ATAU taruh CSV manual
bash run_all.sh                         # ingest → process → train
cd src && python simulate_feed.py twitter_bot_detection --reset --batches 5 --size 200 --interval 3 &
python stream.py twitter_bot_detection  # (tambah --once untuk memproses lalu berhenti)
cd .. && streamlit run app/streamlit_app.py
```
Menambah dataset baru = tambah satu blok di `config.yaml` (label, kolom ID/teks/waktu). Kolom label
dideteksi otomatis jika `label_column: null`; nilai asli label dicatat di dashboard → **verifikasi pemetaan 0/1**.

## Catatan migrasi
- **Nama kolom berawalan `#` berubah** (perbaikan `snake()`): `#friends` → `n_friends`,
  `#tags/post` → `n_tags_post` (sebelumnya `#tags/post` dan `tags/post` sama-sama menjadi `tags_post`
  sehingga Spark gagal dengan `AMBIGUOUS_REFERENCE`). Dataset yang terdampak: `facebook_spam` dan
  `instagram_fake_spammer`. Bronze/silver yang dibangun dengan nama lama harus dibangun ulang.
  Manifest ingest ikut dihapus, karena `ingest.py` melewati file yang sudah tercatat di manifest
  walaupun bronze-nya sudah tidak ada:
  ```bash
  rm -rf data/bronze data/silver reports/*/ingest_manifest.json && bash run_all.sh
  ```
- **Kolom identitas tidak lagi menjadi fitur**: `facebook_spam` (`profile id`) dan
  `twitter_human_bots` (`c0` = indeks baris tanpa nama, `split` = pembagian buatan penyusun dataset).
  Bronze tidak berubah; cukup ulangi processing dan training:
  ```bash
  cd src && python process.py && python train.py
  ```

## Jalur analisis lintas platform
Selain pipeline operasional di atas, `src/` memuat pipeline analisis per platform yang menghasilkan
angka-angka di laporan. Semuanya membaca file mentah yang sama di `data/landing/<kunci>/` dan menulis
ke `outputs/` (metrik JSON, gambar) serta `outputs/predictions/` (skor per akun, tidak di-commit).

| Skrip | Dataset (`data/landing/…`) | Keluaran utama |
|---|---|---|
| `src/twitter_faker_check.py` | `twitter_bot_detection` | `outputs/metrics/twitter_synthetic_faker_check.json` (bukti dataset sintetis) |
| `src/twitter_pipeline.py` | `twitter_bot_detection` (kontrol negatif), `twitter_human_bots` (hasil utama X) | `outputs/metrics/twitter_{synthetic,real}.json`, gambar `outputs/figures/twitter_*.png` |
| `src/instagram_pipeline.py` | `instagram_fake_spammer` (`train.csv` + `test.csv`, digabung lalu di-split ulang) | `outputs/metrics/instagram.json`, `outputs/figures/instagram_*.png` |
| `src/facebook_pipeline.py` | `facebook_spam` (kasus peringatan: file urut per label) | `outputs/metrics/facebook.json`, `outputs/figures/facebook_*.png` |

```bash
python src/twitter_faker_check.py          # beberapa detik, tanpa Spark
python src/twitter_pipeline.py             # ±20 menit (CV 3-fold Random Forest pada 37 ribu akun)
python src/twitter_pipeline.py --dataset real
python src/instagram_pipeline.py           # ±2 menit
python src/facebook_pipeline.py            # ±2 menit
```
Prosedurnya sama untuk setiap dataset: split berstrata 80:20 (seed 42), bobot kelas dari data latih,
CV 3-fold di dalam data latih, Random Forest (model utama) dan Logistic Regression (baseline), lalu
satu kali evaluasi di data uji terhadap baseline kelas mayoritas.
Instagram dan Facebook berbagi definisi model di `src/profile_tracks_utils.py`; karena datanya kecil
(696 dan 600 akun), keduanya juga melaporkan CV berulang di dalam data latih dan interval bootstrap.

## Modul bersama `src/common.py`
Satu modul dipakai semua skrip di `src/`:
- **konfigurasi & skema**: `load_config`, `path_of`, `landing_path`, `report_dir`, `snake`/`snake_all`,
  `detect_label`, `standardize` (dipakai batch *dan* streaming, jadi fitur training = fitur scoring);
- **Spark**: `get_spark(name, driver_memory="2g", cores=None)`; `cores=None` = semua core (`local[*]`),
  driver 2 GB, 8 shuffle partition, UI mati, zona waktu UTC;
- **helper analisis**: `stratified_split` (80:20 berstrata, seed 42, duplikat tidak hilang),
  `evaluate_binary` (confusion matrix, precision/recall/F1, balanced accuracy, MCC, ROC-AUC, PR-AUC,
  baseline kelas mayoritas), `importance_table`, `save_metrics`, `save_predictions`;
- **MongoDB**: `get_mongo()` (lihat *Environment variables*).

Zona data mengikuti `paths` di `config.yaml`. Pipeline ingest/process/train memakai
`data/bronze|silver|gold/<kunci>`; jalur analisis memakai zona sendiri, **`data/curated/`** dan
**`outputs/`**, sehingga keduanya tidak saling menimpa. Bagian `outputs/` yang berisi data per-akun
(`predictions/`, `powerbi/`, `models/`, `logs/`) ada di `.gitignore`.

## Environment variables
| Variabel | Wajib? | Dipakai oleh | Keterangan |
|---|---|---|---|
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | salah satu cara login Kaggle | `scripts/download_datasets.py` | Alternatif: file `~/.kaggle/kaggle.json` (`chmod 600`), atau token baru di `KAGGLE_API_TOKEN` / `~/.kaggle/access_token`. Tidak perlu bila CSV ditaruh manual di `data/landing/<kunci>/`. |
| `JAVA_HOME` | bila Java 17 bukan `java` default | Spark (semua skrip di `src/`) | Tidak ada path Java yang di-hard-code di kode; mis. `export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64` di Ubuntu. |
| `MONGO_URI`, `MONGO_DB` | hanya untuk lapisan MongoDB | `common.get_mongo()` | Dibaca dari lingkungan atau file `.env` di root repo (tidak di-commit). Tanpa `MONGO_URI`, URI dirakit dari `MONGO_INITDB_ROOT_USERNAME`/`MONGO_INITDB_ROOT_PASSWORD` + `MONGO_HOST`/`MONGO_PORT`. |

Jangan pernah commit kredensial: `.env`, `.env.*` (kecuali `.env.example`) dan `kaggle.json` sudah ada di `.gitignore`.

## Lint dan test
```bash
pip install -r requirements-dev.txt
ruff check .     # aturan dikunci di pyproject.toml
pytest           # test unit di tests/ (sebagian memakai Spark lokal kecil)
```
CI (GitHub Actions) menjalankan keduanya plus cek format judul PR di setiap pull request.
Alur kerja tim (branch, PR, review, merge) ada di [CONTRIBUTING.md](CONTRIBUTING.md).

## Hasil uji pada data yang tersedia (Twitter-Bot Detection, 50.000 baris)
Pipeline berjalan penuh (ingest → process → train → streaming 3 micro-batch → 6 halaman dashboard tanpa error).
Namun **model tidak menemukan sinyal**: ROC-AUC uji ≈ 0,50 pada ketiga algoritma, akurasi 0,498 vs baseline 0,500.
Penyebab ada di data, bukan di kode: tweet berupa kalimat acak, distribusi fitur seragam, rata-rata fitur
identik antara bot dan bukan-bot → dataset ini **sintetis**: kosakata kolom `Tweet` persis 971 kata daftar
`lorem` milik library Faker. Dataset ini dipertahankan sebagai **kontrol negatif**. Jangan melaporkannya sebagai "akurasi deteksi".
Dashboard menampilkan peringatan ini secara otomatis. Pipeline yang sama diuji pada fixture berlabel
yang SENGAJA diberi sinyal (hanya untuk uji kode, sudah dihapus) → AUC ≈ 0,97, jadi kode memang bisa belajar bila datanya punya sinyal.

## Syarat 100.000 baris
Twitter (50.000) + Instagram (±696) + Facebook (600) ≈ 51 ribu → **belum** memenuhi syarat. Dashboard (halaman
Arsitektur & Data Lake) menghitung total otomatis. Tambahkan sumber berlabel yang nyata, mis. *Twitter Human-Bots*
(37.438 akun, fitur level-akun) — verifikasi slug di Kaggle — lalu jalankan ulang `run_all.sh`.

## Keterbatasan
- Dataset diproses terpisah per platform; label bot/spam/fake belum tentu sepadan antar-platform.
- `facebook_spam` berisi 497 profil label 0 dan 103 label 1 (bukan 500/100 seperti kartu Kaggle) yang
  tersusun **dua blok berurutan menurut label**. Akurasi tinggi pada dataset ini terutama mencerminkan
  blok file, bukan perilaku spam; laporkan sebagai kasus peringatan, bukan kemampuan deteksi.
- Streaming memutar ulang akun dari data uji (held-out) lewat file; produksi memakai Kafka.
- Skor model = sinyal bantu untuk review manusia, bukan dasar tunggal untuk menghukum akun.
