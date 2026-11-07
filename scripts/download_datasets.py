"""Unduh dataset Kaggle ke data/landing/<kunci>/ memakai slug yang BENAR (dengan tanda hubung).

Mengunduh semua dataset di config.yaml: `datasets` (berlabel, untuk ingest/process/train) dan
`unlabelled_datasets` (tanpa label, hanya untuk jalur analisis, mis. TikTok).

Butuh kredensial Kaggle: file ~/.kaggle/kaggle.json ATAU env KAGGLE_USERNAME + KAGGLE_KEY
(atau token baru di KAGGLE_API_TOKEN / ~/.kaggle/access_token).
  pip install kagglehub
  python scripts/download_datasets.py            # semua dataset di config.yaml
  python scripts/download_datasets.py facebook_spam
"""
import shutil, sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXT = {".csv", ".tsv", ".json", ".jsonl"}


def targets(cfg: dict, only=None) -> dict:
    """Dataset yang diunduh: berlabel dan tanpa label, atau hanya `only` bila diberikan."""
    every = {**cfg["datasets"], **(cfg.get("unlabelled_datasets") or {})}
    return {k: v for k, v in every.items() if not only or k == only}


def main(only=None):
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    try:
        import kagglehub
    except ImportError:
        sys.exit("Jalankan dulu: pip install kagglehub")

    for key, ds in targets(cfg, only).items():
        dest = ROOT / cfg["paths"]["landing"] / key
        dest.mkdir(parents=True, exist_ok=True)
        slug = ds["kaggle"]
        try:
            src = Path(kagglehub.dataset_download(slug))
        except Exception as e:                      # slug salah / belum login / dataset dihapus
            print(f"[GAGAL] {key} ({slug}): {type(e).__name__}: {str(e)[:200]}")
            continue
        files = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in EXT]
        for p in files:
            shutil.copy2(p, dest / p.name)
        print(f"[OK] {key} <- {slug}: {[p.name for p in files]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
