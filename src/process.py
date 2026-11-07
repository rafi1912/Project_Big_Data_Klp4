"""TAHAP 2 — PROCESSING (batch, Spark): bronze -> silver (bersih) -> gold (agregat).

Aturan pembersihan (semua tercatat di reports/<dataset>/dq.json, tidak ada baris
yang dibuang diam-diam):
  1. buang baris duplikat persis;  2. buang baris tanpa label;
  3. petakan label ke 0/1 (nilai asli dicatat);  4. turunkan fitur teks/waktu;
  5. tandai fitur yang mencurigakan (korelasi |r| tinggi dengan label = indikasi leakage).
"""
import json, sys

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import types as T

from common import detect_label, get_spark, load_config, path_of, report_dir, standardize

META = ["_source_file", "_ingested_at", "_batch_id"]


def numeric_cols(df, exclude):
    return [f.name for f in df.schema.fields
            if isinstance(f.dataType, (T.IntegerType, T.LongType, T.DoubleType, T.FloatType, T.ShortType, T.DecimalType))
            and f.name not in exclude]


def run(spark, cfg, key):
    ds = cfg["datasets"][key]
    bronze_path = path_of(cfg, "bronze", key)
    if not bronze_path.exists():
        print(f"[process] {key}: bronze belum ada -> dilewati")
        return
    rd = report_dir(cfg, key)
    bronze = spark.read.parquet(str(bronze_path)).drop(*META)
    n_raw = bronze.count()
    lab = detect_label(bronze.columns, ds.get("label_column"))
    if lab is None:
        raise SystemExit(f"[{key}] kolom label tidak terdeteksi. Isi `label_column` di config.yaml. "
                         f"Kolom: {bronze.columns}")
    raw_label_values = {str(r[0]): r[1] for r in bronze.groupBy(F.col(lab).cast("string")).count().collect()}

    deduped = bronze.dropDuplicates()
    n_dedup = deduped.count()
    std = standardize(deduped, ds)
    n_nolabel = std.filter(F.col("label").isNull()).count()
    std = std.filter(F.col("label").isNotNull()).withColumn("label", F.col("label").cast("int"))
    std = std.cache()
    n_clean = std.count()
    dup_ids = n_clean - std.select("record_id").distinct().count()

    exclude = {"label", "record_id", "username_ref"}
    feats = numeric_cols(std, exclude)
    null_pct = {c: round(float(v) * 100, 3) for c, v in
                std.agg(*[F.avg(F.col(c).isNull().cast("int")).alias(c) for c in std.columns]).first().asDict().items()}
    lab_counts = {int(r["label"]): r["count"] for r in std.groupBy("label").count().collect()}
    corr = {c: (None if v is None else round(float(v), 4)) for c, v in
            std.agg(*[F.corr(F.col(c).cast("double"), F.col("label").cast("double")).alias(c) for c in feats]).first().asDict().items()}
    thr = cfg["modeling"]["leakage_corr_threshold"]
    leak = [c for c, v in corr.items() if v is not None and abs(v) >= thr]
    cats = [f.name for f in std.schema.fields if isinstance(f.dataType, T.StringType) and f.name not in exclude]
    card = {c: std.select(c).distinct().count() for c in cats}

    dq = {"dataset": key, "platform": ds["platform"], "note": ds.get("note"),
          "rows_raw": n_raw, "duplicate_rows_removed": n_raw - n_dedup,
          "rows_without_label_removed": n_nolabel, "rows_clean": n_clean,
          "duplicate_record_ids": dup_ids, "label_column": lab, "label_raw_values": raw_label_values,
          "label_counts": lab_counts, "null_pct_by_column": null_pct,
          "numeric_features": feats, "categorical_cardinality": card,
          "corr_with_label": corr, "leakage_suspects": leak}
    (rd / "dq.json").write_text(json.dumps(dq, indent=2, ensure_ascii=False))

    # ---- SILVER
    std.write.mode("overwrite").parquet(str(path_of(cfg, "silver", key)))

    # ---- GOLD: statistik fitur per label, tren bulanan, sampel EDA
    rows = []
    for c in feats:
        g = std.groupBy("label").agg(F.mean(c).alias("mean"), F.stddev(c).alias("std"),
                                     F.expr(f"percentile_approx({c}, 0.5)").alias("median"),
                                     F.min(c).alias("min"), F.max(c).alias("max")).collect()
        for r in g:
            rows.append({"feature": c, "label": int(r["label"]), **{k: r[k] for k in ["mean", "std", "median", "min", "max"]}})
    gold = path_of(cfg, "gold", key)
    gold.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(gold / "feature_stats_by_label.csv", index=False)
    if "event_ts" in std.columns:
        (std.filter(F.col("event_ts").isNotNull())
            .groupBy(F.date_format(F.date_trunc("month", "event_ts"), "yyyy-MM").alias("month"), "label")
            .count().orderBy("month").toPandas().to_csv(gold / "monthly_counts.csv", index=False))
    sample = std.drop("event_ts").sample(False, min(1.0, 20000 / max(n_clean, 1)), seed=cfg["modeling"]["seed"]).toPandas()
    sample.to_parquet(gold / "eda_sample.parquet", index=False)
    print(f"[process] {key}: raw={n_raw:,} dup_removed={n_raw - n_dedup:,} nolabel={n_nolabel:,} "
          f"clean={n_clean:,} fitur_numerik={len(feats)} leakage={leak or '-'}")
    std.unpersist()


def main(only=None):
    cfg, spark = load_config(), get_spark("process")
    for key in cfg["datasets"]:
        if only and key != only:
            continue
        run(spark, cfg, key)
    spark.stop()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
