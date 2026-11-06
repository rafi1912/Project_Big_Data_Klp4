"""TAHAP 3 — MODELING (Spark MLlib): silver -> model terbaik + metrik jujur.

Praktik anti-leakage / anti-klaim-berlebihan:
  * ID, username, label, teks mentah TIDAK dipakai sebagai fitur.
  * split train/validasi/uji berstrata per label; imputasi & encoding hanya di-fit pada train
    (semua ada di dalam Pipeline).
  * model dipilih pada data VALIDASI; angka yang dilaporkan dari data UJI.
  * dilaporkan baseline mayoritas, precision/recall/F1 kelas bot, confusion matrix, ROC-AUC.
  * `verdict` otomatis memperingatkan bila AUC ~ 0.5 (tidak ada sinyal) atau terlalu sempurna.
"""
import json, sys

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Imputer, OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from common import get_spark, load_config, path_of, report_dir


def feature_names(df, col="features_raw"):
    attrs = df.schema[col].metadata.get("ml_attr", {}).get("attrs", {})
    pairs = [(a["idx"], a["name"]) for grp in attrs.values() for a in grp]
    return [n for _, n in sorted(pairs)]


def confusion(pdf):
    tp = int(((pdf.label == 1) & (pdf.pred == 1)).sum()); fp = int(((pdf.label == 0) & (pdf.pred == 1)).sum())
    tn = int(((pdf.label == 0) & (pdf.pred == 0)).sum()); fn = int(((pdf.label == 1) & (pdf.pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy": (tp + tn) / max(len(pdf), 1),
            "precision_bot": prec, "recall_bot": rec, "f1_bot": f1}


def roc_points(y, s, n=200):
    order = np.argsort(-s); y = np.asarray(y)[order]
    tps, fps = np.cumsum(y), np.cumsum(1 - y)
    tpr, fpr = tps / max(tps[-1], 1), fps / max(fps[-1], 1)
    idx = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
    return pd.DataFrame({"fpr": fpr[idx], "tpr": tpr[idx]})


def run(spark, cfg, key):
    spath = path_of(cfg, "silver", key)
    if not spath.exists():
        print(f"[train] {key}: silver belum ada -> dilewati"); return
    m = cfg["modeling"]; seed = m["seed"]
    rd = report_dir(cfg, key); dq = json.loads((rd / "dq.json").read_text())
    df = spark.read.parquet(str(spath)).withColumn("label", F.col("label").cast("double"))

    num = [c for c in dq["numeric_features"] if c not in dq["leakage_suspects"]]
    cats = [c for c, k in dq["categorical_cardinality"].items() if 1 < k <= m["max_categories"]]
    skipped = [c for c, k in dq["categorical_cardinality"].items() if k > m["max_categories"] or k <= 1]
    num = [c for c in num if df.agg(F.countDistinct(c)).first()[0] > 1]
    if not num and not cats:
        print(f"[train] {key}: tidak ada fitur yang bisa dipakai"); return

    # ---- split berstrata (train/val/test)
    w = Window.partitionBy("label").orderBy(F.rand(seed))
    df = df.withColumn("_p", F.percent_rank().over(w))
    a, b = m["split"][0], m["split"][0] + m["split"][1]
    train, val, test = (df.filter(F.col("_p") < a), df.filter((F.col("_p") >= a) & (F.col("_p") < b)),
                        df.filter(F.col("_p") >= b))
    train, val, test = train.cache(), val.cache(), test.cache()
    n_tr, n_va, n_te = train.count(), val.count(), test.count()

    stages = []
    inputs = []
    if num:
        stages.append(Imputer(strategy="median", inputCols=num, outputCols=[f"{c}__i" for c in num]))
        inputs += [f"{c}__i" for c in num]
    if cats:
        stages += [StringIndexer(inputCols=cats, outputCols=[f"{c}__x" for c in cats], handleInvalid="keep"),
                   OneHotEncoder(inputCols=[f"{c}__x" for c in cats], outputCols=[f"{c}__o" for c in cats], handleInvalid="keep")]
        inputs += [f"{c}__o" for c in cats]
    stages += [VectorAssembler(inputCols=inputs, outputCol="features_raw", handleInvalid="keep"),
               StandardScaler(inputCol="features_raw", outputCol="features", withMean=False, withStd=True)]

    candidates = {
        "LogisticRegression": LogisticRegression(labelCol="label", maxIter=100, regParam=0.01),
        "RandomForest": RandomForestClassifier(labelCol="label", numTrees=100, maxDepth=8, seed=seed),
        "GBT": GBTClassifier(labelCol="label", maxIter=40, maxDepth=5, seed=seed),
    }
    ev = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    fitted, results = {}, {}
    for name, clf in candidates.items():
        pm = Pipeline(stages=stages + [clf]).fit(train)
        fitted[name] = pm
        results[name] = {"val_auc": float(ev.evaluate(pm.transform(val)))}
        print(f"[train] {key} {name}: val AUC={results[name]['val_auc']:.4f}")
    best = max(results, key=lambda k: results[k]["val_auc"])
    pm = fitted[best]

    # ---- evaluasi di data UJI
    pred = pm.transform(test)
    cols = ["record_id", "label"] + (["username_ref"] if "username_ref" in pred.columns else [])
    pdf = (pred.select(*cols, vector_to_array("probability")[1].alias("prob_bot"),
                       F.col("prediction").cast("int").alias("pred")).toPandas())
    pdf["label"] = pdf["label"].astype(int)
    test_auc = float(ev.evaluate(pred))
    base_acc = float(max(pdf.label.mean(), 1 - pdf.label.mean()))
    conf = confusion(pdf)
    for name in results:
        p2 = fitted[name].transform(test)
        results[name]["test_auc"] = float(ev.evaluate(p2))

    if test_auc < 0.60:
        verdict = ("LEMAH — ROC-AUC uji < 0.60: model tidak lebih baik dari tebakan. Fitur pada dataset ini "
                   "tampaknya tidak memuat sinyal pembeda bot/bukan-bot (indikasi data sintetis acak). "
                   "Jangan diklaim sebagai kemampuan deteksi.")
    elif test_auc > 0.99:
        verdict = "WASPADA — skor hampir sempurna; audit leakage/duplikasi sebelum dipercaya."
    else:
        verdict = "WAJAR — ada sinyal yang dapat dipelajari; tetap evaluasi false positive sebelum dipakai moderator."

    # ---- importances
    names = feature_names(pred)
    last = pm.stages[-1]
    imp = (last.featureImportances.toArray() if hasattr(last, "featureImportances") else np.abs(last.coefficients.toArray()))
    imp_df = (pd.DataFrame({"feature": names[:len(imp)], "importance": imp[:len(names)]})
              .sort_values("importance", ascending=False).head(15))

    out = path_of(cfg, "models", key)
    pm.write().overwrite().save(str(out / "best"))
    gold = path_of(cfg, "gold", key); gold.mkdir(parents=True, exist_ok=True)
    pdf.to_parquet(gold / "test_predictions.parquet", index=False)
    roc_points(pdf.label.values, pdf.prob_bot.values).to_csv(gold / "roc.csv", index=False)
    imp_df.to_csv(gold / "feature_importance.csv", index=False)
    metrics = {"dataset": key, "best_model": best, "n_train": n_tr, "n_val": n_va, "n_test": n_te,
               "features_numeric": num, "features_categorical": cats, "features_skipped": skipped,
               "leakage_excluded": dq["leakage_suspects"], "baseline_majority_accuracy": base_acc,
               "test_auc": test_auc, "test": conf, "models": results, "verdict": verdict}
    (rd / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[train] {key}: terbaik={best} test AUC={test_auc:.4f} acc={conf['accuracy']:.4f} "
          f"(baseline {base_acc:.4f}) F1_bot={conf['f1_bot']:.4f}\n[train] verdict: {verdict}")
    for d in (train, val, test): d.unpersist()


def main(only=None):
    cfg, spark = load_config(), get_spark("train")
    for key in cfg["datasets"]:
        if only and key != only:
            continue
        run(spark, cfg, key)
    spark.stop()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
