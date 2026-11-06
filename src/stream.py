"""TAHAP 4 — STREAMING (Spark Structured Streaming): scoring micro-batch hampir real-time.

Membaca file CSV micro-batch yang 'jatuh' ke data/stream/incoming/<dataset>/ (disimulasikan oleh
simulate_feed.py; di produksi sumbernya Kafka/Kinesis), menerapkan standardize() yang SAMA dengan
jalur batch, lalu memberi skor memakai model terbaik hasil training.
Keluaran: data/gold/stream_scores/<dataset>/ (parquet) -> dibaca dashboard Streamlit.

  python stream.py <dataset> --once        # proses semua file yang ada lalu berhenti
  python stream.py <dataset>               # jalan terus (Ctrl+C untuk berhenti)
"""
import argparse, json

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from pyspark.sql import types as T

from common import get_spark, load_config, path_of, report_dir, standardize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", default="5 seconds")
    a = ap.parse_args()
    cfg, spark = load_config(), get_spark("stream")
    key, ds = a.dataset, cfg["datasets"][a.dataset]
    manifest = json.loads((report_dir(cfg, key) / "ingest_manifest.json").read_text())
    schema = T.StructType.fromJson(manifest["bronze_schema"])
    model = PipelineModel.load(str(path_of(cfg, "models", key, "best")))
    thr = cfg["modeling"]["alert_threshold"]

    src = path_of(cfg, "stream_in", key); src.mkdir(parents=True, exist_ok=True)
    out = path_of(cfg, "stream_out", key); out.mkdir(parents=True, exist_ok=True)
    ckpt = path_of(cfg, "checkpoint", key)

    raw = (spark.readStream.schema(schema).option("header", True).option("multiLine", True)
           .option("escape", '"').option("maxFilesPerTrigger", 1).csv(str(src)))
    feats = standardize(raw, ds)            # label (bila ada) = kunci jawaban simulasi, bukan input model
    scored = model.transform(feats)
    sel = ["record_id", "prob_bot", "pred_bot", "alert", "scored_at"]
    keep = [c for c in ["username_ref", "label"] if c in scored.columns]
    result = (scored.withColumn("prob_bot", vector_to_array("probability")[1])
              .withColumn("pred_bot", F.col("prediction").cast("int"))
              .withColumn("alert", (F.col("prob_bot") >= F.lit(thr)).cast("int"))
              .withColumn("scored_at", F.current_timestamp())
              .select(*sel[:1], *keep, *sel[1:]))

    def sink(batch_df, batch_id):
        n = batch_df.count()
        if n:
            batch_df.withColumn("batch_id", F.lit(int(batch_id))).write.mode("append").parquet(str(out))
        print(f"[stream] micro-batch {batch_id}: {n} akun diberi skor", flush=True)

    w = result.writeStream.foreachBatch(sink).option("checkpointLocation", str(ckpt))
    q = (w.trigger(availableNow=True) if a.once else w.trigger(processingTime=a.interval)).start()
    q.awaitTermination()


if __name__ == "__main__":
    main()
