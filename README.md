# Sistem Big Data — Deteksi Akun Bot/Spam Media Sosial

Prototipe pipeline end-to-end (Spark + Streamlit) untuk proyek Teknologi Big Data.

```
Landing (CSV mentah) → Ingestion (Spark batch, manifest sha256) → Bronze (Parquet)
  → Processing (Spark: dedup, label 0/1, fitur teks/waktu) → Silver → Gold (agregat, EDA)
  → Modeling (Spark MLlib: LogReg / RandomForest / GBT, dipilih di data validasi)
  → Streaming scoring (Structured Streaming, micro-batch) → Dashboard Streamlit
```

## ⚠️ Perbaikan link dataset (kenapa link lama tidak bisa dibuka)
Ketiga slug di dokumen "LINK DATASET" dan notebook EDA kehilangan tanda hubung. Yang benar:

| Platform | Slug yang SALAH | Slug yang BENAR |
|---|---|---|
| Twitter | `goyaladi/twitter-botdetection-dataset` | `goyaladi/twitter-bot-detection-dataset` |
| Instagram | `free4ever1/instagram-fakespammer-genuine-accounts` | `free4ever1/instagram-fake-spammer-genuine-accounts` |
| Facebook | `khajahussainsk/facebookspam-dataset` | `khajahussainsk/facebook-spam-dataset` |

## Cara menjalankan
Butuh Python 3.10+, **Java 17** (Spark), dan `pip install -r requirements.txt`
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

## Hasil uji pada data yang tersedia (Twitter-Bot Detection, 50.000 baris)
Pipeline berjalan penuh (ingest → process → train → streaming 3 micro-batch → 6 halaman dashboard tanpa error).
Namun **model tidak menemukan sinyal**: ROC-AUC uji ≈ 0,50 pada ketiga algoritma, akurasi 0,498 vs baseline 0,500.
Penyebab ada di data, bukan di kode: tweet berupa kalimat acak, distribusi fitur seragam, rata-rata fitur
identik antara bot dan bukan-bot → dataset ini hampir pasti **sintetis**. Jangan melaporkannya sebagai "akurasi deteksi".
Dashboard menampilkan peringatan ini secara otomatis. Pipeline yang sama diuji pada fixture berlabel
yang SENGAJA diberi sinyal (hanya untuk uji kode, sudah dihapus) → AUC ≈ 0,97, jadi kode memang bisa belajar bila datanya punya sinyal.

## Syarat 100.000 baris
Twitter (50.000) + Instagram (±696) + Facebook (600) ≈ 51 ribu → **belum** memenuhi syarat. Dashboard (halaman
Arsitektur & Data Lake) menghitung total otomatis. Tambahkan sumber berlabel yang nyata, mis. *Twitter Human-Bots*
(37.438 akun, fitur level-akun) — verifikasi slug di Kaggle — lalu jalankan ulang `run_all.sh`.

## Keterbatasan
- Dataset diproses terpisah per platform; label bot/spam/fake belum tentu sepadan antar-platform.
- Streaming memutar ulang akun dari data uji (held-out) lewat file; produksi memakai Kafka.
- Skor model = sinyal bantu untuk review manusia, bukan dasar tunggal untuk menghukum akun.
