# Lapisan penyajian: MongoDB + tabel Power BI

Group "Penyayang Claude", Teknologi Big Data, SV IPB University.
Koleksi analisis dibuat oleh `src/synthesis.py`, koleksi pipeline operasional oleh `src/load_mongo.py` dan
`src/stream.py`. Ringkasan hasil verifikasi ada di [FINAL_REPORT.md](FINAL_REPORT.md) §9.

Arsitektur Worksheet 2: Kaggle → batch ingest → data lake (raw CSV, curated Parquet) → Spark →
Spark MLlib → **MongoDB** → **Power BI**. Dua komponen terakhir sebelumnya belum ada. Dokumen ini
menjelaskan cara menyalakannya, cara terhubung, dan isi setiap koleksi dan tabel. Di repo ini dashboard
Streamlit juga membaca MongoDB, dan hanya MongoDB ([ADR 0004](adr/0004-streamlit-reads-mongodb.md)):
Spark → MongoDB → {Streamlit, ekspor Power BI}.

## 1. Menyalakan dan mematikan MongoDB

MongoDB berjalan di Docker, sengaja dibuat kecil supaya muat di laptop/VM kecil.

| Item | Nilai |
|---|---|
| Image | `mongo:7.0` |
| Container | `<project>-mongo-1`; project compose default `klp4-bot-detection` (bisa diganti dengan `-p`) |
| Batas memori | `mem_limit: 512m`, `cpus: 1.0`, cache WiredTiger 0,25 GB |
| Port | `127.0.0.1:${MONGO_PORT}` (default 27017, **hanya loopback**, tidak terbuka ke jaringan) |
| Volume data | Docker volume `<project>_mongo_data` (data bertahan saat container di-restart) |
| Compose file | `deploy/mongodb/docker-compose.yml` |
| Kredensial | `.env` di root repo (salin dari `.env.example`, chmod 600). **Jangan dibagikan, jangan di-commit.** |
| Database | `MONGO_DB` (default `socmed_bot_detection`) |
| User | `MONGO_INITDB_ROOT_USERNAME` di `.env` (root, `authSource=admin`) |

```bash
cp .env.example .env && chmod 600 .env       # isi user/password
docker compose -f deploy/mongodb/docker-compose.yml --env-file .env up -d
```

```bash
docker compose -f deploy/mongodb/docker-compose.yml --env-file .env down
```

Mengisi ulang koleksi analisis (idempoten: setiap koleksi di-drop lalu ditulis ulang) dan hasil pipeline
operasional:

```bash
python src/synthesis.py          # koleksi analisis (§2)
cd src && python load_mongo.py   # koleksi pipeline_* (§2.3); juga dijalankan otomatis oleh run_all.sh
```

### Connection string

```
mongodb://<user>:<password>@127.0.0.1:<MONGO_PORT>/?authSource=admin
```

`<user>` dan `<password>` adalah `MONGO_INITDB_ROOT_USERNAME` dan `MONGO_INITDB_ROOT_PASSWORD` di `.env`.
Kode di repo tidak perlu connection string: `common.get_mongo()` merakitnya dari `.env` (atau memakai
`MONGO_URI` bila di-set). Karena port hanya terbuka di loopback, anggota kelompok yang bekerja dari
laptop memakai SSH tunnel:

```bash
ssh -L 27017:127.0.0.1:27017 <user>@<host>
```

lalu terhubung ke `127.0.0.1:27017` dari MongoDB Compass atau `mongosh`. Untuk membuka shell langsung
di host:

```bash
docker compose -f deploy/mongodb/docker-compose.yml --env-file .env exec mongo \
  mongosh -u <user> -p --authenticationDatabase admin socmed_bot_detection
```

## 2. Koleksi MongoDB

| Koleksi | Satu dokumen = | Dokumen | Indeks |
|---|---|---|---|
| `account_scores` | satu akun (atau author TikTok) | 25.335 | `{platform, risk_level}`, `{dataset, rf_p_inauthentic: -1}`, `{account_uid}` unik |
| `model_metrics` | satu dataset (semua model, params, metrik, CV) | 5 | `{dataset}` |
| `datasets` | satu dataset: provenance, peran, verdict, angka yang boleh dikutip | 5 | `{dataset}` |
| `feature_importance` | satu (dataset, model, fitur, jenis importance) | 338 | `{dataset, importance_type, rank}` |
| `risk_level_definitions` | satu level 0–4 + NA | 6 | `{code}` |
| `risk_level_summary` | satu (dataset, level) | 34 | `{dataset}` |
| `concept_comparison` | satu (indikator perilaku, dataset) | 32 | `{indicator}` |
| `coordination_evidence` | satu (dataset, jenis bukti koordinasi) | 25 | `{dataset}` |
| `high_score_profiles` | satu (dataset, indikator): profil 10% skor teratas vs semua akun (Worksheet Q2) | 37 | `{dataset}` |
| `cross_platform_summary` | ringkasan sintesis (1 dokumen) | 1 | – |

### 2.1 `account_scores`

| Field | Tipe | Arti |
|---|---|---|
| `account_uid` | string | kunci unik lintas platform: `x_<id>`, `xs_<user_id>`, `ig_<file_row>`, `fb_<profile id>`, `tt_<author_key>` |
| `platform` | string | `twitter` / `instagram` / `facebook` / `tiktok` |
| `dataset` | string | `twitter_real`, `twitter_synthetic`, `instagram`, `facebook`, `tiktok` |
| `dataset_role` | string | `primary_result`, `negative_control`, `supporting_small_sample`, `cautionary_artefact`, `unlabelled_indicators` |
| `source_ref` | string | rujukan ke baris sumber: id X, `train.csv:<baris>` (IG), profile id (FB), `author_key` pseudonim (TikTok). **Handle/screen_name tidak dimuat.** |
| `evaluation_scope` | string | `test_holdout`, `cv_out_of_fold`, `unsupervised_all_authors` |
| `label` | int / null | 0 = Authentic, 1 = Inauthentic (konvensi operasional). **null untuk TikTok** (tidak ada label) |
| `label_concept` | string | arti label sumber: bot / fake / spam / none |
| `rf_p_inauthentic`, `lr_p_inauthentic` | double / null | probabilitas out-of-sample dari track platform; null untuk TikTok |
| `rf_prediction_at_0_5` | int / null | prediksi RF pada ambang 0,5 (hanya pembanding) |
| `rank_in_dataset` | int / null | 1 = skor RF tertinggi dalam dataset |
| `risk_level` | int / null | 0, 1, 2. **null = tidak dapat dinilai** (bukan 0). Level 3–4 tidak pernah muncul |
| `risk_level_code` | string | `"0"`, `"1"`, `"2"`, `"NA"` — kunci relasi ke `dim_risk_level` |
| `risk_level_name` | string | nama level dalam Bahasa Indonesia |
| `risk_level_basis` | string | `model_probability_crossfit`, `tiktok_indicator_rule`, `not_assessable` |
| `suggested_action` | string | `none`, `monitor`, `human_review`, `none_not_assessable`. Tidak ada aksi otomatis |
| `crossfit_fold` | string / null | paruh (A/B) tempat akun berada; ambang yang dipakai berasal dari paruh lainnya |
| `t1_applied`, `t2_applied` | double / null | ambang yang benar-benar dipakai untuk akun ini |
| `tiktok_review_tier` | string / null | tier dari track TikTok |
| `tiktok_video_count`, `tiktok_behavioural_indicators_assessable` | int / null | konteks TikTok |
| `ind_burst_posting`, `ind_caption_duplication`, `ind_regular_posting`, `ind_engagement_outlier`, `ind_iforest_outlier` | int / null | indikator tri-state TikTok: 1 menyala, 0 dinilai dan tidak menyala, null tidak dapat dinilai |

### 2.2 Koleksi lain

- `model_metrics`: `dataset`, `platform`, `role`, `verdict`, `number_to_quote`, `dataset_info`,
  `class_balance`, `split`, `limitations`, dan `models.<random_forest|logistic_regression>` berisi
  `params`, `metrics` (persis seperti JSON track) dan `cross_validation`. Dokumen TikTok berisi
  `review_tiers` dan `coordination_assessment`, tanpa `models`.
- `feature_importance`: `dataset`, `model`, `feature`, `concept` (konsep perilaku; null untuk X
  sintetis), `importance_type` (`rf_impurity_train`, `lr_coefficient_per_sd`,
  `permutation_test_roc_auc_drop`), `value`, `rank`.
- `risk_level_definitions`: `code`, `level`, `name_id` (nama di laporan ini),
  `name_concept_doc` dan `description_concept_doc` (kategori dan deskripsi **verbatim** dari dokumen
  konsep §13; null untuk NA, yang tidak ada di §13), `name_en`, `definition`, `required_evidence`,
  `suggested_action`, `reachable` (false untuk 3 dan 4). Pemetaan dan alasan penamaan ulang ada di
  `FINAL_REPORT.md` §8.1.
- `high_score_profiles`: `dataset`, `indicator`, `statistic` (`median` atau `share`), `top_decile`,
  `all_scored`, `authentic_all`, `inauthentic_all`, `top_over_all`, `top_decile_p25/p75`, `n_top`,
  `n_scored`, `share_inauthentic_top_decile`, `share_inauthentic_all`.
- `risk_level_summary`: `dataset`, `level_code` (`0`/`1`/`2`, `>=1`/`>=2`, `3`/`4`, `NA`), `kind`
  (`per_level`, `cumulative`, `unreachable`, `indicator_rule`, `not_assessable`), `n_accounts`,
  `volume_share`, `n_inauthentic`, `n_authentic`, `precision_in_level`,
  `precision_ci95_low/high`, `share_of_all_inauthentic`, `share_of_all_authentic`,
  `recall_at_or_above`, `t1_deployment`, `t2_deployment`.
- `concept_comparison`: `indicator`, `concept`, `dataset`, `source_column`, `equivalence`
  (`equivalent`/`partial`), `note`, `n`, `median_authentic`, `median_inauthentic` (atau
  `share_*` untuk biner), `auc`, `auc_ci95_low/high`, `cliffs_delta`, `direction`,
  `lr_coefficient_per_sd_same_coding`, `lr_direction`, `facebook_caveat`, `tiktok_status`.
- `coordination_evidence`: `dataset`, `evidence_type`, `status` (`available`/`partial`/`absent`), `note`.

### 2.3 Koleksi pipeline operasional (`src/load_mongo.py`, `src/stream.py`)

Ditulis oleh `run_all.sh` (langkah terakhir) dan `stream.py`. Satu dokumen per baris; setiap dokumen
membawa `dataset` (kunci di `config.yaml`), dan pemuatan ulang hanya mengganti dokumen dataset itu.

| Koleksi | Satu dokumen = | Sumber |
|---|---|---|
| `pipeline_datasets` | satu dataset: platform, slug Kaggle, catatan, ambang alert | `config.yaml` |
| `pipeline_ingest_manifests` | manifest ingest satu dataset (file, sha256, jumlah baris, skema bronze) | `reports/<ds>/ingest_manifest.json` |
| `pipeline_dq` | laporan kualitas data satu dataset | `reports/<ds>/dq.json` |
| `pipeline_metrics` | metrik model terbaik + verdict | `reports/<ds>/metrics.json` |
| `pipeline_roc` | satu titik kurva ROC | `data/gold/<ds>/roc.csv` |
| `pipeline_feature_importance` | satu fitur | `data/gold/<ds>/feature_importance.csv` |
| `pipeline_feature_stats` | satu (fitur, label) | `data/gold/<ds>/feature_stats_by_label.csv` |
| `pipeline_monthly_counts` | satu (bulan, label) | `data/gold/<ds>/monthly_counts.csv` |
| `pipeline_eda_sample` | satu baris sampel EDA, **tanpa** `record_id`/`username_ref` | `data/gold/<ds>/eda_sample.parquet` |
| `pipeline_test_predictions` | satu akun data uji: skor dan prediksi | `data/gold/<ds>/test_predictions.parquet` |
| `stream_scores` | satu akun yang diberi skor di satu micro-batch | `src/stream.py` (`foreachBatch`) |

Dashboard Streamlit membaca koleksi-koleksi ini dan koleksi analisis di §2; tidak ada file yang dibaca
langsung. `scripts/seed_demo.py` mengisi database kosong dengan data demo tanpa identitas akun nyata.

## 3. Tabel Power BI (`outputs/powerbi/`)

Power BI tidak punya konektor MongoDB bawaan yang sederhana (perlu MongoDB BI/Atlas SQL Connector
lewat ODBC), jadi tabel yang sama diekspor sebagai CSV datar. Kelompok bisa membangun dashboard
tanpa menyentuh Spark: **Get Data → Text/CSV**, untuk setiap file.

Format: UTF-8, pemisah koma, titik desimal, header di baris pertama, nilai kosong = null
(**jangan** ganti null dengan 0 di Power Query: missing is not zero). Set kolom `code`,
`risk_level_code`, dan `level_code` ke tipe **Text**: `"NA"` adalah kode level (tidak dapat dinilai),
bukan null, dan deteksi tipe otomatis bisa salah membacanya.

| File | Grain | Kunci relasi |
|---|---|---|
| `fact_account_scores.csv` | satu akun / author | `dataset` → `dim_dataset`, `risk_level_code` → `dim_risk_level[code]` |
| `dim_dataset.csv` | satu dataset | `dataset` |
| `dim_risk_level.csv` | satu level (0–4, NA); membawa `name_id` dan `name_concept_doc` (§13) | `code` |
| `fact_model_metrics.csv` | (dataset, model, evaluasi, metrik) — format panjang | `dataset` |
| `fact_feature_importance.csv` | (dataset, model, fitur, jenis importance) | `dataset` |
| `fact_importance_by_concept.csv` | (dataset, konsep, jenis share) | `dataset` |
| `fact_concept_comparison.csv` | (indikator, dataset) | `dataset` |
| `fact_risk_level_summary.csv` | (dataset, level, jenis) | `dataset`, `level_code` |
| `fact_coordination_evidence.csv` | (dataset, jenis bukti) | `dataset` |
| `fact_high_score_profile.csv` | (dataset, indikator): profil 10% skor teratas | `dataset` |

Saran halaman dashboard:

1. **Antrean tinjauan moderator**: filter `suggested_action = human_review` dan
   `dataset_role = primary_result`, urutkan `rf_p_inauthentic` turun. Tampilkan `source_ref`,
   skor, level. Jangan tampilkan kolom `label` di halaman operasional (label hanya untuk evaluasi).
2. **Level risiko**: dari `fact_risk_level_summary` (`kind = per_level`), tampilkan
   `n_accounts`, `volume_share`, `precision_in_level` dengan interval kepercayaannya per dataset.
   Tampilkan level 3 dan 4 sebagai "tidak dapat dicapai dengan data ini" dari `dim_risk_level`.
3. **Perbandingan konsep lintas platform**: `fact_concept_comparison`, sumbu `auc` dengan garis 0,5.
4. **Kepercayaan per dataset**: `dim_dataset[verdict]` dan `number_to_quote`. **Jangan** membuat
   visual yang mengurutkan platform berdasarkan akurasi (lihat FINAL_REPORT §4).
