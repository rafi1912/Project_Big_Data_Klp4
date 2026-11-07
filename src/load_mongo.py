"""TAHAP 5 — PENYAJIAN: muat keluaran ingest/process/train ke MongoDB (dibaca dashboard Streamlit).

Satu dokumen per baris; setiap dokumen membawa field `dataset` (= kunci di config.yaml).

  pipeline_datasets            platform, slug Kaggle, catatan, ambang alert (dari config.yaml)
  pipeline_ingest_manifests    reports/<ds>/ingest_manifest.json (file, sha256, jumlah baris)
  pipeline_dq                  reports/<ds>/dq.json
  pipeline_metrics             reports/<ds>/metrics.json
  pipeline_roc                 data/gold/<ds>/roc.csv
  pipeline_feature_importance  data/gold/<ds>/feature_importance.csv
  pipeline_feature_stats       data/gold/<ds>/feature_stats_by_label.csv
  pipeline_monthly_counts      data/gold/<ds>/monthly_counts.csv (bila ada)
  pipeline_eda_sample          data/gold/<ds>/eda_sample.parquet, TANPA record_id/username_ref
  pipeline_test_predictions    data/gold/<ds>/test_predictions.parquet

Idempoten: dokumen lama dataset yang sama dihapus lalu ditulis ulang. Skor streaming
(`stream_scores`) ditulis langsung oleh stream.py. Pemuatan ini juga menghapus penanda database demo
(`meta`/`demo_seed`, ditulis scripts/seed_demo.py), karena isinya sekarang hasil run sungguhan.

  python load_mongo.py                    # semua dataset yang sudah punya hasil
  python load_mongo.py facebook_spam      # satu dataset
"""
import json
import math
import sys

import numpy as np
import pandas as pd

import common

COLLECTIONS = ["pipeline_datasets", "pipeline_ingest_manifests", "pipeline_dq", "pipeline_metrics",
               "pipeline_roc", "pipeline_feature_importance", "pipeline_feature_stats",
               "pipeline_monthly_counts", "pipeline_eda_sample", "pipeline_test_predictions"]
JSON_FILES = {"pipeline_ingest_manifests": "ingest_manifest.json", "pipeline_dq": "dq.json",
              "pipeline_metrics": "metrics.json"}
GOLD_FILES = {"pipeline_roc": "roc.csv", "pipeline_feature_importance": "feature_importance.csv",
              "pipeline_feature_stats": "feature_stats_by_label.csv",
              "pipeline_monthly_counts": "monthly_counts.csv", "pipeline_eda_sample": "eda_sample.parquet",
              "pipeline_test_predictions": "test_predictions.parquet"}
EDA_DROP = ["record_id", "username_ref"]   # grafik EDA tidak butuh identitas akun
META_COLLECTION = "meta"
DEMO_MARKER_ID = "demo_seed"               # ditulis scripts/seed_demo.py; dashboard menampilkan banner demo


def clean(v):
    """Nilai Python/numpy/pandas -> nilai BSON. NaN/NA/NaT -> None (missing is not zero)."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    if isinstance(v, np.ndarray):
        return [clean(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    return v


def records(df: pd.DataFrame, dataset: str) -> list:
    return [{"dataset": dataset, **{k: clean(v) for k, v in row.items()}}
            for row in df.to_dict(orient="records")]


def read_table(path):
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def dataset_documents(cfg: dict, key: str) -> dict:
    """Semua dokumen satu dataset, per koleksi. Tidak menyentuh MongoDB (mudah diuji)."""
    ds = cfg["datasets"][key]
    reports = common.ROOT / cfg["paths"]["reports"] / key
    gold = common.path_of(cfg, "gold", key)
    docs = {"pipeline_datasets": [{"dataset": key, "platform": ds["platform"], "kaggle": ds["kaggle"],
                                   "note": ds.get("note"),
                                   "alert_threshold": cfg["modeling"]["alert_threshold"]}]}
    for coll, name in JSON_FILES.items():
        p = reports / name
        docs[coll] = [{**clean(json.loads(p.read_text(encoding="utf-8"))), "dataset": key}] if p.exists() else []
    for coll, name in GOLD_FILES.items():
        p = gold / name
        df = read_table(p) if p.exists() else pd.DataFrame()
        if coll == "pipeline_eda_sample":
            df = df.drop(columns=[c for c in EDA_DROP if c in df.columns])
        docs[coll] = records(df, key)
    return docs


def write_dataset(db, key: str, docs: dict) -> dict:
    counts = {}
    for coll, rows in docs.items():
        db[coll].delete_many({"dataset": key})
        if rows:
            db[coll].insert_many(rows)
        counts[coll] = len(rows)
    return counts


def clear_demo_marker(db) -> bool:
    """Hapus penanda database demo beserta baris account_scores demo. Dipanggil oleh pemuat run lengkap
    (load_mongo.py dan synthesis.py). True bila penanda tadinya ada."""
    removed = db[META_COLLECTION].delete_one({"_id": DEMO_MARKER_ID}).deleted_count
    if removed:
        db["account_scores"].delete_many({"dataset_role": "demo_synthetic"})
    return bool(removed)


def ensure_indexes(db):
    for coll in COLLECTIONS:
        db[coll].create_index("dataset")
    db["stream_scores"].create_index([("dataset", 1), ("batch_id", 1)])


def connect():
    client, db = common.get_mongo(timeout_ms=3000)
    try:
        client.admin.command("ping")
    except Exception as e:
        raise SystemExit(f"[load_mongo] MongoDB tidak terjangkau ({type(e).__name__}: {str(e)[:120]}).\n"
                         "  Nyalakan dari root repo: docker compose -f deploy/mongodb/docker-compose.yml "
                         "--env-file .env up -d\n  lalu ulangi: cd src && python load_mongo.py")
    return client, db


def main(only=None):
    cfg = common.load_config()
    client, db = connect()
    ensure_indexes(db)
    if clear_demo_marker(db):
        print("[load_mongo] penanda demo dihapus: database ini sekarang memuat hasil run lengkap")
    for key in cfg["datasets"]:
        if only and key != only:
            continue
        if not (common.ROOT / cfg["paths"]["reports"] / key / "dq.json").exists():
            print(f"[load_mongo] {key}: belum ada hasil process.py -> dilewati")
            continue
        counts = write_dataset(db, key, dataset_documents(cfg, key))
        print(f"[load_mongo] {key}: " + ", ".join(f"{c.removeprefix('pipeline_')}={n}" for c, n in counts.items()))
    print(f"[load_mongo] database '{db.name}' di {common.redact_uri(common.mongo_uri())}")
    client.close()

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
