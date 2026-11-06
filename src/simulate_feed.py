"""Simulator sumber data streaming: memutar ulang akun dari DATA UJI (held-out) sebagai
micro-batch CSV ke data/stream/incoming/<dataset>/ (di produksi: producer Kafka).

  python simulate_feed.py <dataset> --batches 5 --size 200 --interval 3
"""
import argparse, shutil, time

import pandas as pd

from common import load_config, path_of, snake, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--size", type=int, default=200); ap.add_argument("--interval", type=float, default=3)
    ap.add_argument("--reset", action="store_true", help="hapus micro-batch & checkpoint & skor lama")
    a = ap.parse_args()
    cfg = load_config(); key, ds = a.dataset, cfg["datasets"][a.dataset]
    dest = path_of(cfg, "stream_in", key)
    if a.reset:
        for p in (dest, path_of(cfg, "checkpoint", key), path_of(cfg, "stream_out", key)):
            shutil.rmtree(p, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)

    test_ids = set(pd.read_parquet(path_of(cfg, "gold", key, "test_predictions.parquet"))["record_id"].astype(str))
    raw_files = sorted(path_of(cfg, "landing", key).glob("**/*.csv"))
    df = pd.concat([pd.read_csv(f) for f in raw_files], ignore_index=True)
    id_col = next(c for c in df.columns if snake(c) == snake((ds.get("id_columns") or [df.columns[0]])[0])) \
        if ds.get("id_columns") else None
    if id_col is None:
        df = df.reset_index().rename(columns={"index": "__row"})   # tanpa ID: tidak bisa dibatasi ke data uji
        print("[feed] PERINGATAN: dataset tanpa kolom ID -> memutar ulang sampel acak (BUKAN murni data uji).")
    else:
        df = df[df[id_col].astype(str).isin(test_ids)]
    df = df.sample(frac=1.0, random_state=cfg["modeling"]["seed"]).head(a.batches * a.size)
    existing = len(list(dest.glob("batch_*.csv")))
    for i in range(a.batches):
        chunk = df.iloc[i * a.size:(i + 1) * a.size].drop(columns=["__row"], errors="ignore")
        if chunk.empty:
            break
        tmp = dest / f".tmp_{i}.csv"; final = dest / f"batch_{existing + i + 1:04d}.csv"
        chunk.to_csv(tmp, index=False); tmp.rename(final)          # rename atomik
        print(f"[feed] {final.name}: {len(chunk)} akun", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
