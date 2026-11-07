"""Isi MongoDB dengan data DEMO supaya dashboard bisa dibuka tanpa Kaggle dan tanpa Spark.

Yang dimuat — semuanya sudah ada di repo dan TIDAK berisi identitas akun nyata:
  1. Hasil ingest/process/train untuk dataset SINTETIS `twitter_bot_detection` (reports/ dan
     data/gold/ yang di-commit; username buatan Faker), termasuk skor streaming yang di-commit.
  2. Agregat hasil analisis lintas platform dari run lengkap (demo/analysis_aggregates.json):
     metrik model, ringkasan level risiko, perbandingan konsep, bukti koordinasi. Tanpa baris per akun.
  3. `account_scores` demo = akun uji sintetis di atas, dengan level risiko dihitung memakai aturan
     synthesis.py (cross-fitting 2 bagian + evidence gate). Model tidak punya sinyal pada data
     sintetis, jadi gate gagal dan semua akun 'Tidak dapat dinilai' (NA) — perilaku yang benar.

Yang TIDAK ada di demo: hasil pipeline batch untuk dataset nyata, antrean moderator untuk akun nyata,
dan skor streaming selain dataset sintetis. Itu hanya muncul setelah run lengkap (lihat README).

  python scripts/seed_demo.py            # database harus kosong
  python scripts/seed_demo.py --reset    # HAPUS seluruh database MONGO_DB lalu isi demo
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import common  # noqa: E402
import load_mongo  # noqa: E402

DEMO_KEY = "twitter_bot_detection"
SNAPSHOT = ROOT / "demo" / "analysis_aggregates.json"


def demo_levels(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, dict]:
    """Level risiko dengan aturan synthesis.py: ambang dari bagian lain (cross-fitting), lalu evidence
    gate (batas bawah Wilson precision Level>=2 harus > base rate). Gate gagal -> semua NA."""
    import synthesis as syn
    fold = syn.stratified_halves(y)
    lvl = np.empty(len(y), dtype=int)
    for f in (0, 1):
        cal, eva = fold == f, fold != f
        t1, t2, _ = syn.derive_thresholds(y[cal], s[cal])
        lvl[eva] = syn.assign(s[eva], t1, t2)
    _, cum = syn.level_table(y, lvl)
    low = cum[1]["precision_ci95"][0]
    gate = {"level2_precision_ci95_low": low, "base_rate": round(float(y.mean()), 4),
            "passed": bool(low is not None and low > y.mean())}
    return (lvl.astype(float) if gate["passed"] else np.full(len(y), np.nan)), gate


def demo_accounts(pred: pd.DataFrame) -> tuple[list, dict]:
    import synthesis as syn
    y = pred["label"].to_numpy().astype(int)
    s = pred["prob_bot"].to_numpy().astype(float)
    lvl, gate = demo_levels(y, s)
    code = ["NA" if np.isnan(v) else str(int(v)) for v in lvl]
    order = np.argsort(-s, kind="mergesort")
    rank = np.empty(len(s), dtype=int)
    rank[order] = np.arange(1, len(s) + 1)
    docs = []
    for i, r in enumerate(pred.itertuples(index=False)):
        docs.append({
            "account_uid": f"demo_{r.record_id}", "platform": "twitter", "dataset": f"{DEMO_KEY}_demo",
            "dataset_role": "demo_synthetic", "source_ref": str(r.record_id),
            "evaluation_scope": "test_holdout (pipeline batch, data sintetis)", "label": int(r.label),
            "label_concept": "bot (data sintetis buatan Faker)",
            "rf_p_inauthentic": round(float(r.prob_bot), 6), "lr_p_inauthentic": None,
            "rf_prediction_at_0_5": int(r.pred), "rank_in_dataset": int(rank[i]),
            "risk_level": None if code[i] == "NA" else int(code[i]), "risk_level_code": code[i],
            "risk_level_name": next(lv["name_id"] for lv in syn.LEVELS if lv["code"] == code[i]),
            "risk_level_basis": "model_probability_crossfit" if gate["passed"] else "evidence_gate_failed_no_signal",
            "suggested_action": syn.ACTION[code[i]],
            "score_source": "model terbaik pipeline batch (train.py), bukan Random Forest jalur analisis"})
    return docs, gate


def seed(db) -> dict:
    """Isi `db` (dianggap kosong) dengan seluruh data demo. Mengembalikan jumlah dokumen per koleksi."""
    cfg = common.load_config()
    load_mongo.ensure_indexes(db)
    counts = load_mongo.write_dataset(db, DEMO_KEY, load_mongo.dataset_documents(cfg, DEMO_KEY))
    parts = sorted(common.path_of(cfg, "stream_out", DEMO_KEY).glob("*.parquet"))
    if parts:
        s = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        db["stream_scores"].insert_many(load_mongo.records(s, DEMO_KEY))
        counts["stream_scores"] = len(s)
    snap = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    for coll, docs in snap.items():
        if docs:
            db[coll].insert_many(docs)
        counts[coll] = len(docs)
    pred = pd.read_parquet(common.path_of(cfg, "gold", DEMO_KEY, "test_predictions.parquet"))
    docs, gate = demo_accounts(pred)
    db["account_scores"].insert_many(docs)
    db["account_scores"].create_index([("dataset", 1), ("rf_p_inauthentic", -1)])
    counts["account_scores"] = len(docs)
    counts["_demo_evidence_gate"] = gate
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reset", action="store_true", help="hapus seluruh database MONGO_DB dulu")
    a = ap.parse_args()
    client, db = load_mongo.connect()
    existing = db.list_collection_names()
    if existing and not a.reset:
        raise SystemExit(f"Database '{db.name}' tidak kosong ({len(existing)} koleksi). Demo hanya diisi ke "
                         "database kosong; jalankan dengan --reset untuk MENGHAPUS seluruh isinya dulu.")
    if existing:
        client.drop_database(db.name)
        print(f"[seed_demo] database '{db.name}' dikosongkan")
    counts = seed(db)
    gate = counts.pop("_demo_evidence_gate")
    print("[seed_demo] " + ", ".join(f"{c}={n}" for c, n in counts.items()))
    print(f"[seed_demo] evidence gate akun demo: {gate}")
    print(f"[seed_demo] selesai: database '{db.name}' di {common.redact_uri(common.mongo_uri())}")
    client.close()

if __name__ == "__main__":
    main()
