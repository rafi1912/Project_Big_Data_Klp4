"""Unduh dataset Kaggle ke data/landing/<kunci>/ memakai slug yang BENAR (dengan tanda hubung).

Butuh kredensial Kaggle: file ~/.kaggle/kaggle.json ATAU env KAGGLE_USERNAME + KAGGLE_KEY.
  pip install kagglehub
  python scripts/download_datasets.py            # semua dataset di config.yaml
  python scripts/download_datasets.py facebook_spam
"""
import shutil, sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
only = sys.argv[1] if len(sys.argv) > 1 else None
try:
    import kagglehub
except ImportError:
    sys.exit("Jalankan dulu: pip install kagglehub")

for key, ds in cfg["datasets"].items():
    if only and key != only:
        continue
    dest = ROOT / cfg["paths"]["landing"] / key
    dest.mkdir(parents=True, exist_ok=True)
    slug = ds["kaggle"]
    try:
        src = Path(kagglehub.dataset_download(slug))
    except Exception as e:                      # slug salah / belum login / dataset dihapus
        print(f"[GAGAL] {key} ({slug}): {type(e).__name__}: {str(e)[:200]}")
        continue
    files = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".json", ".jsonl"}]
    for p in files:
        shutil.copy2(p, dest / p.name)
    print(f"[OK] {key} <- {slug}: {[p.name for p in files]}")
