# Sistem Big Data — Deteksi Akun Bot/Spam Media Sosial

Prototipe pipeline end-to-end (Spark + MongoDB + Streamlit) untuk proyek Teknologi Big Data.

```
Landing (CSV mentah) → Ingestion (Spark batch, manifest sha256) → Bronze (Parquet)
  → Processing (Spark: dedup, label 0/1, fitur teks/waktu) → Silver → Gold (agregat, EDA)
  → Modeling (Spark MLlib: LogReg / RandomForest / GBT, dipilih di data validasi)
  → Streaming scoring (Structured Streaming, micro-batch)
  → MongoDB (lapisan penyajian) → Dashboard Streamlit  (+ ekspor CSV untuk Power BI)
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
cp .env.example .env && chmod 600 .env  # isi user/password MongoDB
docker compose -f deploy/mongodb/docker-compose.yml --env-file .env up -d    # MongoDB (lihat di bawah)
python scripts/download_datasets.py     # unduh ke data/landing/<kunci>/ (butuh kaggle.json), ATAU taruh CSV manual
bash run_all.sh                         # ingest → process → train → muat hasil ke MongoDB
cd src && python simulate_feed.py twitter_bot_detection --reset --batches 5 --size 200 --interval 3 &
python stream.py twitter_bot_detection  # skor micro-batch → MongoDB stream_scores (--once: proses lalu berhenti)
cd .. && streamlit run app/streamlit_app.py
```
Tanpa Kaggle dan tanpa Spark, dashboard tetap bisa dibuka dengan data demo (lihat *Dashboard*):
`python scripts/seed_demo.py`.
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
| `src/tiktok_pipeline.py` | `tiktok_videos` (blok `unlabelled_datasets`, **tanpa label**) | `outputs/metrics/tiktok.json`, `outputs/figures/tiktok_*.png`; profil per author di `outputs/predictions/` |
| `src/synthesis.py` | keluaran keempat track di atas | perbandingan lintas platform, **level risiko 0–4**, tabel per akun ke `data/curated/` + **MongoDB**, ekspor CSV Power BI ke `outputs/powerbi/` |

```bash
python src/twitter_faker_check.py          # beberapa detik, tanpa Spark
python src/twitter_pipeline.py             # ±20 menit (CV 3-fold Random Forest pada 37 ribu akun)
python src/twitter_pipeline.py --dataset real
python src/instagram_pipeline.py           # ±2 menit
python src/facebook_pipeline.py            # ±2 menit
python src/tiktok_pipeline.py              # ±2 menit; jalankan SETELAH track lain (membaca metriknya)
python src/synthesis.py                    # ±3 menit; butuh MongoDB (lihat di bawah) atau --skip-mongo
```
Prosedurnya sama untuk setiap dataset: split berstrata 80:20 (seed 42), bobot kelas dari data latih,
CV 3-fold di dalam data latih, Random Forest (model utama) dan Logistic Regression (baseline), lalu
satu kali evaluasi di data uji terhadap baseline kelas mayoritas.
Instagram dan Facebook berbagi definisi model di `src/profile_tracks_utils.py`; karena datanya kecil
(696 dan 600 akun), keduanya juga melaporkan CV berulang di dalam data latih dan interval bootstrap.

TikTok berbeda: datanya **tidak berlabel**, jadi tidak ada classifier dan tidak ada akurasi. Pipeline
membangun profil perilaku per author (handle diganti kunci pseudonim) dan indikator anomali tanpa
supervisi untuk tinjauan manusia. Dataset ini ada di blok `unlabelled_datasets` di `config.yaml`:
ikut diunduh `scripts/download_datasets.py`, tetapi tidak ikut ingest/process/train (tahap itu butuh label).

## Lapisan penyajian MongoDB
Arsitektur penyajian: **Spark → MongoDB → {dashboard Streamlit, ekspor Power BI}**. MongoDB berjalan di
Docker, kecil (512 MB), dan hanya terbuka di loopback.

```bash
cp .env.example .env          # lalu isi user/password (jangan di-commit)
chmod 600 .env
docker compose -f deploy/mongodb/docker-compose.yml --env-file .env up -d     # nyalakan
docker compose -f deploy/mongodb/docker-compose.yml --env-file .env ps        # status (healthy)
python src/synthesis.py                                                       # isi koleksi + verifikasi
docker compose -f deploy/mongodb/docker-compose.yml --env-file .env down      # matikan (data tetap)
docker compose -f deploy/mongodb/docker-compose.yml --env-file .env down -v   # matikan + hapus data
```
`synthesis.py` menulis `account_scores` lewat Spark (`mongo-spark-connector` 10.4.0, diunduh dari Maven
saat Spark start; bila gagal otomatis memakai `pymongo`, atau paksa dengan `--no-connector`), lalu
koleksi kecil lain (`model_metrics`, `datasets`, `feature_importance`, `risk_level_definitions`,
`risk_level_summary`, `concept_comparison`, `coordination_evidence`, `high_score_profiles`,
`cross_platform_summary`) dengan `pymongo`, lalu membaca ulang dan mencocokkan hitungannya.
`--skip-mongo` menjalankan analisis + CSV saja. Tabel yang sama diekspor sebagai CSV datar ke
`outputs/powerbi/` untuk Power BI (**Get Data → Text/CSV**); folder itu berisi data per akun sehingga
ada di `.gitignore`.

**Level risiko** 0 (*tidak ada indikasi*), 1 (*indikasi lemah, pantau*), 2 (*indikasi kuat, prioritas
tinjauan manusia*), serta `NA` (*tidak dapat dinilai*, bukan 0). Ambang ditentukan dengan cross-fitting
2 bagian dan hanya diberikan bila model lolos *evidence gate*; level 3–4 (koordinasi/amplifikasi) butuh
data jaringan yang tidak ada di dataset mana pun, sehingga tidak pernah diberikan. Tidak ada aksi otomatis:
level tertinggi berarti antrean tinjauan manusia.

## Dashboard (Streamlit, hanya membaca MongoDB)
`streamlit run app/streamlit_app.py` membuka 8 halaman. **Semua halaman membaca MongoDB saja**; bila
MongoDB tidak terjangkau, dashboard menampilkan perintah untuk menyalakannya lalu berhenti (tidak ada
cadangan berupa file).

| Halaman | Koleksi MongoDB | Diisi oleh |
|---|---|---|
| Arsitektur & Data Lake | `pipeline_ingest_manifests`, `pipeline_datasets`, jumlah dokumen tiap koleksi | `run_all.sh` → `src/load_mongo.py` |
| Kualitas Data, EDA, Model & Evaluasi | `pipeline_dq`, `pipeline_eda_sample`, `pipeline_feature_stats`, `pipeline_monthly_counts`, `pipeline_metrics`, `pipeline_roc`, `pipeline_feature_importance`, `pipeline_test_predictions` | `src/load_mongo.py` |
| Streaming (simulasi) | `stream_scores` | `src/stream.py` (setiap micro-batch) |
| Antrean Review Moderator | `account_scores` (filter dataset + level risiko; precision di atas ambang vs base rate bila ada label) | `src/synthesis.py` |
| Level Risiko | `risk_level_definitions`, `risk_level_summary`, agregasi `account_scores` | `src/synthesis.py` |
| Perbandingan Lintas Platform | `datasets`, `model_metrics`, `concept_comparison`, `coordination_evidence` | `src/synthesis.py` |

**Data demo** (`python scripts/seed_demo.py`, butuh database kosong; `--reset` menghapus isi database
`MONGO_DB` dulu) memuat hanya data tanpa identitas akun nyata:
- hasil batch + skor streaming dataset **sintetis** `twitter_bot_detection` yang di-commit di `reports/` dan
  `data/gold/` (username buatan Faker) → halaman 1–5 berisi dataset itu saja;
- agregat hasil analisis dari run lengkap (`demo/analysis_aggregates.json`, dibuat
  `scripts/export_demo_snapshot.py` yang menolak menulis bila ada pola identitas) → halaman Level Risiko
  dan Perbandingan Lintas Platform menampilkan angka hasil run lengkap;
- `account_scores` demo = akun uji sintetis, diberi level dengan aturan `synthesis.py`; evidence gate
  gagal (model tanpa sinyal), jadi antrean moderator demo menunjukkan semua akun *Tidak dapat dinilai*.

Yang hanya muncul setelah **run lengkap**: hasil batch dataset nyata (Treiman, Instagram, Facebook),
antrean moderator untuk akun nyata (X, Instagram, Facebook, TikTok) dan skor streaming dataset lain.

Database demo diberi penanda (koleksi `meta`, dokumen `demo_seed`): selama penanda itu ada, **setiap
halaman diawali banner "Mode demo"** dan hitungan 100.000 baris di halaman Arsitektur disebut sebagai
hitungan data demo, bukan sistem lengkap. `src/load_mongo.py` (langkah terakhir `run_all.sh`) dan
`src/synthesis.py` menghapus penanda itu (beserta akun demo di `account_scores`) saat memuat hasil run lengkap.

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
| `MONGO_INITDB_ROOT_USERNAME`, `MONGO_INITDB_ROOT_PASSWORD` | ya, untuk MongoDB | `docker-compose.yml`, `common.get_mongo()` | User root MongoDB; compose menolak start bila kosong. |
| `MONGO_PORT` | tidak (default 27017) | compose (port host), `common.get_mongo()` | Ganti bila 27017 sudah dipakai. |
| `MONGO_DB` | tidak (default `socmed_bot_detection`) | `synthesis.py`, `common.get_mongo()` | Nama database. |
| `MONGO_URI` | tidak | `common.get_mongo()` | Hanya bila MongoDB ada di tempat lain (mis. Atlas); tanpa ini URI dirakit dari variabel di atas + `MONGO_HOST`. |

Semua variabel ada di [`.env.example`](.env.example) dengan nilai contoh saja. Salin menjadi `.env`; kode
(`common.load_env()`) dan `docker compose --env-file .env` membacanya, tanpa menimpa variabel yang sudah di-set.
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
