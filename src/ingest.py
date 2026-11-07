"""TAHAP 1 — INGESTION (batch): landing zone (CSV mentah) -> bronze (Parquet).

- File mentah TIDAK diubah; hanya ditambah metadata (_source_file, _ingested_at, _batch_id).
- Idempoten: file yang sudah tercatat (sha256 sama) di manifest dilewati, selama data batch-nya
  (`_batch_id`) memang ada di bronze. Entri manifest tanpa data di bronze (bronze dihapus/kosong,
  atau manifest ikut ter-commit) dibuang dari manifest dan file-nya di-ingest ulang.
- Menyimpan skema bronze untuk dipakai jalur streaming.
"""
import hashlib, json, sys
from datetime import datetime, timezone

from pyspark.errors import AnalysisException
from pyspark.sql import functions as F

from common import get_spark, load_config, path_of, report_dir, snake_all

EXT = {".csv", ".tsv", ".json", ".jsonl"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_any(spark, p):
    s = p.suffix.lower()
    if s in {".csv", ".tsv"}:
        return (spark.read.option("header", True).option("inferSchema", True)
                .option("multiLine", True).option("escape", '"')
                .option("sep", "\t" if s == ".tsv" else ",").csv(str(p)))
    return spark.read.json(str(p))


def bronze_batch_ids(spark, path):
    """Nilai `_batch_id` yang benar-benar ada di bronze; kosong bila bronze belum ada atau kosong."""
    if not path.exists():
        return set()
    try:
        return {r[0] for r in spark.read.parquet(str(path)).select("_batch_id").distinct().collect()}
    except AnalysisException:          # folder ada tetapi tidak berisi file parquet
        return set()


def main(only=None):
    cfg, spark = load_config(), get_spark("ingest")
    for key in cfg["datasets"]:
        if only and key != only:
            continue
        land = path_of(cfg, "landing", key)
        files = sorted(p for p in land.glob("**/*") if p.is_file() and p.suffix.lower() in EXT) if land.exists() else []
        if not files:
            print(f"[ingest] {key}: tidak ada file di {land} -> dilewati")
            continue
        rd = report_dir(cfg, key)
        mpath = rd / "ingest_manifest.json"
        manifest = json.loads(mpath.read_text()) if mpath.exists() else {"files": []}
        present = bronze_batch_ids(spark, path_of(cfg, "bronze", key)) if manifest["files"] else set()
        stale = [f for f in manifest["files"] if f.get("batch_id") not in present]
        if stale:
            print(f"[ingest] {key}: {len(stale)} entri manifest tanpa data di bronze "
                  f"({', '.join(f['name'] for f in stale)}) -> dibuang dari manifest, file di-ingest ulang")
            manifest["files"] = [f for f in manifest["files"] if f not in stale]
        seen = {f["sha256"] for f in manifest["files"]}
        for p in files:
            digest = sha256(p)
            if digest in seen:
                print(f"[ingest] {key}/{p.name}: sudah pernah di-ingest -> dilewati")
                continue
            df = read_any(spark, p)
            orig_cols = df.columns
            df = df.toDF(*snake_all(df.columns))
            batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            schema_json = df.schema.json()
            n = df.count()
            (df.withColumn("_source_file", F.lit(p.name))
               .withColumn("_ingested_at", F.current_timestamp())
               .withColumn("_batch_id", F.lit(batch_id))
               .write.mode("append").parquet(str(path_of(cfg, "bronze", key))))
            manifest["files"].append({"name": p.name, "sha256": digest, "rows": n,
                                      "batch_id": batch_id, "size_bytes": p.stat().st_size})
            manifest["original_columns"] = orig_cols
            manifest["bronze_schema"] = json.loads(schema_json)
            print(f"[ingest] {key}/{p.name}: {n:,} baris -> bronze (batch {batch_id})")
        mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    spark.stop()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
