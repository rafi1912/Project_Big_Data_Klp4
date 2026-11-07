# Arsitektur

```mermaid
flowchart LR
    K["Kaggle<br/>(5 dataset)"] -->|scripts/download_datasets.py| L["data/landing/&lt;kunci&gt;/<br/>CSV mentah, di-gitignore"]

    subgraph SPARK["Apache Spark 3.5 (lokal)"]
        direction TB
        subgraph OPS["Pipeline operasional (Rafi)"]
            I["ingest.py<br/>manifest sha256"] --> B["Bronze<br/>Parquet"]
            B --> P["process.py<br/>dedup, label 0/1, fitur"] --> S["Silver / Gold"]
            S --> T["train.py<br/>LR / RF / GBT, split 70/15/15"]
            T --> ST["stream.py<br/>Structured Streaming"]
        end
        subgraph ANA["Pipeline analisis"]
            X["twitter_pipeline.py"]
            IG["instagram_pipeline.py"]
            FB["facebook_pipeline.py"]
            TT["tiktok_pipeline.py<br/>tanpa label"]
            SY["synthesis.py<br/>level risiko 0–4"]
            X & IG & FB & TT --> SY
        end
    end

    L --> I
    L --> X & IG & FB & TT
    SY --> C["data/curated/<br/>Parquet per akun"]

    S -->|load_mongo.py| M[("MongoDB<br/>lapisan penyajian")]
    T -->|load_mongo.py| M
    ST -->|foreachBatch| M
    SY -->|mongo-spark-connector / pymongo| M

    M --> UI["Dashboard Streamlit<br/>(hanya membaca MongoDB)"]
    M --> PBI["Ekspor Power BI<br/>outputs/powerbi/*.csv"]
```

| Komponen | Kode | Keluaran | Di-commit? |
|---|---|---|---|
| Unduh data | `scripts/download_datasets.py` | `data/landing/<kunci>/` | tidak |
| Pipeline operasional | `src/ingest.py`, `process.py`, `train.py`, `stream.py`, `simulate_feed.py` | `data/{bronze,silver,gold}/<kunci>/`, `models/`, `reports/<kunci>/` | tidak (kecuali demo sintetis `twitter_bot_detection`) |
| Pipeline analisis | `src/*_pipeline.py`, `src/profile_tracks_utils.py`, `src/twitter_faker_check.py` | `outputs/metrics/`, `outputs/figures/`, `outputs/predictions/` | metrik & gambar tanpa identitas: ya; prediksi per akun: tidak |
| Sintesis | `src/synthesis.py` | `data/curated/`, MongoDB, `outputs/powerbi/` | tidak (berisi baris per akun) |
| Penyajian | `src/load_mongo.py`, `deploy/mongodb/docker-compose.yml` | koleksi MongoDB | — |
| Dashboard | `app/streamlit_app.py`, `scripts/seed_demo.py` | — | — |

Kedua jalur membaca file mentah yang **sama** di `data/landing/` dan menulis ke zona yang **berbeda**
(`data/{bronze,silver,gold}/<kunci>` vs `data/curated/` + `outputs/`), sehingga tidak saling menimpa.
Spark dipakai karena arsitektur menuntutnya (lihat [ADR 0003](adr/0003-spark-is-an-architecture-requirement.md)),
MongoDB adalah satu-satunya sumber data dashboard ([ADR 0004](adr/0004-streamlit-reads-mongodb.md)).
Koleksi dan tabel Power BI dijelaskan di [mongodb_powerbi.md](mongodb_powerbi.md).
