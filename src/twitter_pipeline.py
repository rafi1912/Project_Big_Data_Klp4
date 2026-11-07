"""X / Twitter track - bot / inauthentic-account detection (Spark MLlib).

Group "Penyayang Claude" - Teknologi Big Data, SV IPB University.
Follows the method rules in docs/methodology.md. Uses src/common.py for the Spark session, the
stratified split, the metric set and the output writers.

This track runs TWO datasets through the identical procedure:

* ``synthetic`` - goyaladi/twitter-bot-detection-dataset, the dataset named in
  Worksheets 2 and 3. It is kept as a documented NEGATIVE CONTROL: the script
  first validates it statistically (uniform noise columns, closed Faker word
  list, one tweet per user), then models it exactly like a real dataset,
  including TF-IDF on Tweet / Hashtags, and reports that it lands on the
  majority-class baseline.
* ``real`` - danieltreiman/twitter-human-bots-dataset, 37,438 real accounts.
  This is the headline X / Twitter result.

Per dataset: stratified 80:20 split (seed 42) -> class weights from train ->
3-fold cross-validation INSIDE train for a small grid -> refit on all of train
-> one evaluation on the held-out test set for Random Forest (main model) and
Logistic Regression (baseline) -> impurity importance, LR coefficients and
test-set permutation importance -> leakage audit -> JSON, scored CSV, figures.

Input: data/landing/twitter_bot_detection/bot_detection_data.csv and
data/landing/twitter_human_bots/twitter_human_bots_dataset.csv (scripts/download_datasets.py).

Run from the repo root (Java 17 on PATH or JAVA_HOME set):

    python src/twitter_pipeline.py                  # both datasets
    python src/twitter_pipeline.py --dataset real   # one of them

Outputs (track names ``twitter_synthetic`` and ``twitter_real``):
    outputs/metrics/<track>.json, outputs/predictions/<track>_scored.csv,
    outputs/figures/<track>_*.png, outputs/figures/twitter_side_by_side.png,
    outputs/figures/twitter_distributions.png
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration - every hyperparameter the track sets is recorded in the JSON
# ---------------------------------------------------------------------------
SEED = common.RANDOM_SEED
SYN_PATH = common.landing_path("twitter_bot_detection") / "bot_detection_data.csv"
REAL_PATH = common.landing_path("twitter_human_bots") / "twitter_human_bots_dataset.csv"

CV_FOLDS = 3
CV_METRIC = "areaUnderPR"          # threshold-free, sensitive to the positive class
RF_FIXED = {"numTrees": 150, "featureSubsetStrategy": "sqrt", "impurity": "gini",
            "maxBins": 32, "minInstancesPerNode": 5, "subsamplingRate": 1.0,
            "bootstrap": True, "seed": SEED}
RF_GRID = {"maxDepth": [8, 12, 16, 20]}
LR_FIXED = {"maxIter": 200, "elasticNetParam": 0.0, "standardization": True,
            "threshold": 0.5, "family": "binomial"}
LR_GRID = {"regParam": [0.001, 0.01, 0.1]}
SCALER = {"withMean": False, "withStd": True}   # withMean=False keeps TF-IDF sparse

TFIDF = {"vocabSize": 1000, "minDF": 5, "minDocFreq": 5, "minTokenLength": 2,
         "pattern": "[^a-z]+"}
LANG_MIN_TRAIN_COUNT = 30          # rarer `lang` levels collapse to __other__
PERM_REPEATS = 5
REVIEW_FRACTIONS = (0.05, 0.10, 0.20)
SINGLE_FEATURE_AUC_ALARM = 0.95

# Palette (validated with the dataviz skill's validator: blue/orange pass all checks)
C_AUTH = "#2a78d6"      # authentic / human  (categorical slot 1)
C_INAUTH = "#eb6834"    # inauthentic / bot  (categorical slot 2)
C_POS = "#e34948"       # LR coefficient pushing toward inauthentic (diverging pole)
C_NEG = "#2a78d6"       # LR coefficient pushing toward authentic (diverging pole)
C_SURFACE = "#fcfcfb"
C_TEXT = "#0b0b0b"
C_TEXT2 = "#52514e"
C_GRID = "#e4e3df"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def r4(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else round(x, 4)


# ---------------------------------------------------------------------------
# Generic helpers (track-local)
# ---------------------------------------------------------------------------
def bool_col(name: str):
    """'True'/'False' string -> 1.0/0.0. Anything else stays NULL (never 0)."""
    from pyspark.sql import functions as F
    c = F.trim(F.col(name))
    return (F.when(c == "True", 1.0).when(c == "False", 0.0)
            .otherwise(F.lit(None).cast("double")))


def assert_no_nulls(df, cols: list[str], what: str) -> dict:
    """Missing is not zero: fail loudly instead of silently filling."""
    from pyspark.sql import functions as F
    row = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in cols]).first()
    nulls = {c: int(row[c] or 0) for c in cols}
    bad = {c: n for c, n in nulls.items() if n}
    if bad:
        raise ValueError(f"{what}: unexpected NULLs in model inputs {bad}")
    return nulls


def add_class_weights(train):
    """Inverse-frequency class weights computed on TRAIN only (n / (2 * n_class))."""
    from pyspark.sql import functions as F
    counts = {int(r["label"]): int(r["count"]) for r in train.groupBy("label").count().collect()}
    n = sum(counts.values())
    w = {k: n / (2.0 * v) for k, v in counts.items()}
    out = train.withColumn("weight", F.when(F.col("label") == 1.0, F.lit(w[1]))
                           .otherwise(F.lit(w[0])))
    return out, {"0": round(w[0], 4), "1": round(w[1], 4), "train_counts": counts,
                 "formula": "n_train / (2 * n_class)"}


def class_balance(df) -> dict:
    counts = {int(r["label"]): int(r["count"]) for r in df.groupBy("label").count().collect()}
    n0, n1 = counts.get(0, 0), counts.get(1, 0)
    return {"n_authentic": n0, "n_inauthentic": n1, "positive_rate": r4(n1 / (n0 + n1))}


def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [r4(centre - half), r4(centre + half)]


def auc_null_band(n_pos: int, n_neg: int, z: float = 1.96) -> list[float]:
    """95% band of ROC-AUC for a model with NO signal (Mann-Whitney null)."""
    sd = math.sqrt((n_pos + n_neg + 1) / (12.0 * n_pos * n_neg))
    return [r4(0.5 - z * sd), r4(0.5 + z * sd)]


def single_feature_auc(y: np.ndarray, x: np.ndarray) -> float:
    """Direction-free ROC-AUC of one raw feature (max(AUC, 1-AUC))."""
    from sklearn.metrics import roc_auc_score
    ok = ~pd.isna(x)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return float("nan")
    a = roc_auc_score(y[ok], x[ok].astype(float))
    return max(a, 1 - a)


def vif_table(pdf: pd.DataFrame, cols: list[str]) -> dict:
    """Variance inflation factor of each column against the others (OLS, intercept)."""
    X = pdf[cols].to_numpy(dtype=float)
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    out = {}
    for j, c in enumerate(cols):
        others = np.delete(X, j, axis=1)
        A = np.column_stack([np.ones(len(X)), others])
        beta, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
        resid = X[:, j] - A @ beta
        r2 = 1 - resid.var() / X[:, j].var()
        out[c] = "inf (exact linear combination)" if r2 > 1 - 1e-9 else round(1.0 / (1 - r2), 2)
    return out


def leakage_order_checks(pdf: pd.DataFrame, label: str, id_col: str | None) -> list[dict]:
    """Is the file sorted by label, or does an identifier track the label?"""
    from scipy import stats
    y = pdf[label].to_numpy().astype(int)
    out = []
    r, p = stats.pointbiserialr(y, np.arange(len(y)))
    out.append({"check": "row_order_vs_label", "pointbiserial_r": r4(r), "p_value": r4(p),
                "verdict": "not sorted by label" if abs(r) < 0.05 else "FILE ORDER TRACKS LABEL"})
    same = float((y[1:] == y[:-1]).mean())
    pr = y.mean()
    out.append({"check": "adjacent_label_agreement", "observed": r4(same),
                "expected_if_shuffled": r4(pr * pr + (1 - pr) * (1 - pr)),
                "verdict": "consistent with unsorted file"
                if abs(same - (pr * pr + (1 - pr) ** 2)) < 0.02 else "LABEL RUNS - file may be sorted"})
    if id_col:
        rho = stats.spearmanr(pdf[id_col].to_numpy(), y).statistic
        out.append({"check": f"{id_col}_rank_correlation_with_label", "spearman_rho": r4(rho),
                    "note": "identifier is excluded from features regardless"})
    return out


# ---------------------------------------------------------------------------
# Modelling core - identical for both datasets
# ---------------------------------------------------------------------------
def build_pipelines(prep_stages_fn, input_cols: list[str]):
    """Return (rf_pipeline, rf_grid, lr_pipeline, lr_grid).

    `prep_stages_fn()` must return FRESH feature stages (tokenizers, encoders)
    so the two pipelines never share a stage object.
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
    from pyspark.ml.feature import StandardScaler, VectorAssembler
    from pyspark.ml.tuning import ParamGridBuilder

    rf = RandomForestClassifier(labelCol="label", featuresCol="features",
                                weightCol="weight", **RF_FIXED)
    rf_pipe = Pipeline(stages=prep_stages_fn() + [
        VectorAssembler(inputCols=input_cols, outputCol="features"), rf])
    rf_grid = ParamGridBuilder().addGrid(rf.maxDepth, RF_GRID["maxDepth"]).build()

    lr = LogisticRegression(labelCol="label", featuresCol="features_scaled",
                            weightCol="weight", **LR_FIXED)
    lr_pipe = Pipeline(stages=prep_stages_fn() + [
        VectorAssembler(inputCols=input_cols, outputCol="features"),
        StandardScaler(inputCol="features", outputCol="features_scaled", **SCALER), lr])
    lr_grid = ParamGridBuilder().addGrid(lr.regParam, LR_GRID["regParam"]).build()
    return rf_pipe, rf_grid, lr_pipe, lr_grid


def cross_validate(pipe, grid, train, name: str):
    """k-fold CV strictly inside TRAIN; returns (best PipelineModel, cv report)."""
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.tuning import CrossValidator

    ev = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="probability",
                                       metricName=CV_METRIC)
    cv = CrossValidator(estimator=pipe, estimatorParamMaps=grid, evaluator=ev,
                        numFolds=CV_FOLDS, seed=SEED, parallelism=1)
    t0 = time.time()
    cvm = cv.fit(train)
    avg = list(cvm.avgMetrics)
    std = list(getattr(cvm, "stdMetrics", []) or [float("nan")] * len(avg))
    table = []
    for pm, a, s in zip(grid, avg, std):
        table.append({**{p.name: v for p, v in pm.items()},
                      f"cv_mean_{CV_METRIC}": r4(a), f"cv_std_{CV_METRIC}": r4(s)})
    best = int(np.argmax(avg))
    log(f"  {name}: CV {CV_FOLDS}-fold done in {time.time() - t0:.0f}s, "
        f"best {table[best]}")
    return cvm.bestModel, {"folds": CV_FOLDS, "metric": CV_METRIC, "seed": SEED,
                           "scope": "inside training split only", "grid": table,
                           "selected": {p.name: v for p, v in grid[best].items()}}


def p1_col():
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F
    return vector_to_array(F.col("probability")).getItem(1)


def score_pandas(spark, model, pdf: pd.DataFrame, schema) -> np.ndarray:
    """Score a pandas frame with a fitted PipelineModel; order-safe via _pos."""
    pdf = pdf.copy()
    pdf["_pos"] = np.arange(len(pdf), dtype=np.int64)
    from pyspark.sql import types as T
    sch = T.StructType(list(schema.fields) + [T.StructField("_pos", T.LongType())])
    sdf = spark.createDataFrame(pdf, schema=sch)
    out = (model.transform(sdf).select("_pos", p1_col().alias("p1"))
           .toPandas().sort_values("_pos"))
    return out["p1"].to_numpy()


def permutation_importance(spark, model, test_pdf: pd.DataFrame, schema,
                           groups: dict[str, list[str]], repeats: int = PERM_REPEATS) -> list[dict]:
    """Test-set permutation importance: drop in ROC-AUC / PR-AUC when one feature
    (or one group, e.g. all TF-IDF terms of a text field) is shuffled across rows.
    Columns inside a group are shuffled with the SAME permutation."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(SEED)
    y = test_pdf["label"].to_numpy().astype(int)
    base = score_pandas(spark, model, test_pdf, schema)
    base_auc, base_ap = roc_auc_score(y, base), average_precision_score(y, base)
    n = len(test_pdf)
    rows = []
    for gname, cols in groups.items():
        frames = []
        for rep in range(repeats):
            perm = rng.permutation(n)
            f = test_pdf.copy()
            for c in cols:
                f[c] = test_pdf[c].to_numpy()[perm]
            frames.append(f)
        stacked = pd.concat(frames, ignore_index=True)
        p = score_pandas(spark, model, stacked, schema).reshape(repeats, n)
        d_auc = [base_auc - roc_auc_score(y, p[i]) for i in range(repeats)]
        d_ap = [base_ap - average_precision_score(y, p[i]) for i in range(repeats)]
        rows.append({"feature": gname, "columns": cols,
                     "roc_auc_drop_mean": round(float(np.mean(d_auc)), 6),
                     "roc_auc_drop_std": round(float(np.std(d_auc)), 6),
                     "pr_auc_drop_mean": round(float(np.mean(d_ap)), 6),
                     "pr_auc_drop_std": round(float(np.std(d_ap)), 6)})
    rows.sort(key=lambda r: r["roc_auc_drop_mean"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows, {"baseline_roc_auc_sklearn": r4(base_auc), "baseline_avg_precision_sklearn": r4(base_ap),
                  "repeats": repeats, "seed": SEED, "evaluated_on": "held-out test split",
                  "metric_impl": "sklearn roc_auc_score / average_precision_score on Spark-scored probabilities"}


def review_queue(y: np.ndarray, p: np.ndarray) -> list[dict]:
    """Operational view for moderators: if they can only review the top k% of
    accounts by score, how many of those are labelled inauthentic?"""
    order = np.argsort(-p, kind="stable")
    n, n_pos = len(y), int(y.sum())
    out = []
    for f in REVIEW_FRACTIONS:
        k = max(1, int(round(f * n)))
        top = y[order[:k]]
        out.append({"top_fraction": f, "k_accounts": k,
                    "precision_at_k": r4(top.mean()),
                    "recall_at_k": r4(top.sum() / n_pos if n_pos else 0),
                    "base_rate": r4(n_pos / n)})
    return out


def extra_metrics(y: np.ndarray, p: np.ndarray, m: dict) -> dict:
    cm = m["confusion_matrix"]
    n = m["n_test"]
    n_pos = cm["true_positive"] + cm["false_negative"]
    n_neg = n - n_pos
    return {
        "accuracy_wilson_95ci": wilson_ci(cm["true_positive"] + cm["true_negative"], n),
        "roc_auc_no_signal_95band": auc_null_band(n_pos, n_neg),
        "pr_auc_no_signal_reference": r4(n_pos / n),
        "review_queue": review_queue(y, p),
        "decision_threshold": 0.5,
        "threshold_note": "0.5 on class-weighted models (equivalent to a balanced-prior cut-off)",
    }


def fit_and_evaluate(spark, train, test, prep_stages_fn, input_cols, model_input_cols,
                     key_cols, track):
    """Shared modelling block. Returns dict with fitted models, metrics, scored test."""
    from pyspark.sql import functions as F

    train_w, weights = add_class_weights(train)
    train_w = train_w.cache()
    rf_pipe, rf_grid, lr_pipe, lr_grid = build_pipelines(prep_stages_fn, input_cols)

    log(f"[{track}] cross-validating Random Forest")
    rf_model, rf_cv = cross_validate(rf_pipe, rf_grid, train_w, "random_forest")
    log(f"[{track}] cross-validating Logistic Regression")
    lr_model, lr_cv = cross_validate(lr_pipe, lr_grid, train_w, "logistic_regression")

    res = {"weights": weights, "rf_model": rf_model, "lr_model": lr_model,
           "rf_cv": rf_cv, "lr_cv": lr_cv}
    test = test.cache()
    for tag, model in (("rf", rf_model), ("lr", lr_model)):
        pred = model.transform(test).cache()
        m = common.evaluate_binary(pred)
        pp = pred.select("label", p1_col().alias("p1")).toPandas()
        m.update(extra_metrics(pp["label"].to_numpy().astype(int), pp["p1"].to_numpy(), m))
        res[f"{tag}_metrics"] = m
        res[f"{tag}_pred"] = pred
        log(f"[{track}] {tag.upper()} test: acc {m['accuracy']} (baseline "
            f"{m['baseline_majority_class']['accuracy']}), bal_acc {m['balanced_accuracy']}, "
            f"F1 {m['f1_positive']}, MCC {m['mcc']}, ROC-AUC {m.get('roc_auc')}, PR-AUC {m.get('pr_auc')}")

    # Names of the assembled vector, recovered from the RF pipeline's own output.
    res["feature_names"] = common.assembled_feature_names(res["rf_pred"], "features")

    # Scored per-account CSV (held-out test rows only - out-of-sample scores).
    rf_s = res["rf_pred"].select(*key_cols, F.col("label").cast("int").alias("label"),
                                 F.round(p1_col(), 6).alias("rf_probability_inauthentic"),
                                 F.col("prediction").cast("int").alias("rf_prediction"))
    lr_s = res["lr_pred"].select(*key_cols[:1],
                                 F.round(p1_col(), 6).alias("lr_probability_inauthentic"),
                                 F.col("prediction").cast("int").alias("lr_prediction"))
    scored = (rf_s.join(lr_s, on=key_cols[0], how="inner")
              .withColumn("evaluated_on", F.lit("test_split_80_20_seed42"))
              .orderBy(F.col("rf_probability_inauthentic").desc(), F.col(key_cols[0])))
    res["scored"] = scored

    # Model inputs of the test split, for permutation importance.
    res["test_pdf"] = test.select(*model_input_cols, "label").toPandas()
    res["test_schema"] = test.select(*model_input_cols, "label").schema
    return res


def model_block(params: dict, cv: dict, metrics: dict, fi: list, perm: list, perm_meta: dict,
                extra: dict | None = None) -> dict:
    out = {"params": params, "cross_validation": cv, "metrics": metrics,
           "feature_importance": fi, "permutation_importance": perm,
           "permutation_importance_meta": perm_meta}
    if extra:
        out.update(extra)
    return out


def rf_params(model) -> dict:
    stage = model.stages[-1]
    return {**RF_FIXED, "maxDepth": stage.getOrDefault("maxDepth"),
            "weightCol": "weight (inverse class frequency, train)", "labelCol": "label",
            "n_trees_fitted": stage.getNumTrees, "total_nodes": stage.totalNumNodes}


def lr_params(model) -> dict:
    stage = model.stages[-1]
    return {**LR_FIXED, "regParam": stage.getOrDefault("regParam"),
            "weightCol": "weight (inverse class frequency, train)",
            "scaler": {"type": "StandardScaler", **SCALER},
            "coefficients_are_on": "standardised features (per 1 SD)",
            "intercept": round(float(stage.intercept), 6)}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE,
        "savefig.facecolor": C_SURFACE, "axes.edgecolor": C_GRID,
        "axes.labelcolor": C_TEXT2, "xtick.color": C_TEXT2, "ytick.color": C_TEXT2,
        "text.color": C_TEXT, "axes.titlecolor": C_TEXT, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False, "legend.frameon": False,
    })
    return plt


def fig_confusion(track: str, title: str, metrics: dict[str, dict]) -> Path:
    plt = _mpl()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    for ax, (name, m) in zip(axes, metrics.items()):
        cm = m["confusion_matrix"]
        mat = np.array([[cm["true_negative"], cm["false_positive"]],
                        [cm["false_negative"], cm["true_positive"]]], dtype=float)
        rowpct = mat / mat.sum(1, keepdims=True)
        ax.imshow(rowpct, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ink = "#ffffff" if rowpct[i, j] > 0.55 else C_TEXT
                ax.text(j, i, f"{int(mat[i, j]):,}\n{rowpct[i, j]:.1%} of row",
                        ha="center", va="center", color=ink, fontsize=10)
        ax.set_xticks([0, 1], ["Predicted\nauthentic (0)", "Predicted\ninauthentic (1)"])
        ax.set_yticks([0, 1], ["Actual\nauthentic (0)", "Actual\ninauthentic (1)"])
        for s in ax.spines.values():
            s.set_visible(False)
        b = m["baseline_majority_class"]["accuracy"]
        ax.set_title(f"{name}\naccuracy {m['accuracy']:.3f} (majority baseline {b:.3f})\n"
                     f"balanced accuracy {m['balanced_accuracy']:.3f} · MCC {m['mcc']:.3f}",
                     fontsize=10)
    fig.suptitle(title, fontweight="bold", color=C_TEXT)
    fig.text(0.5, 0.01, "Shading = share of the actual class (row). Held-out 20% test split, seed 42.",
             ha="center", color=C_TEXT2, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path = common.OUT_FIGURES / f"{track}_confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_importance(track: str, title: str, impurity: list[dict], perm_rf: list[dict],
                   lr_coef: list[dict], top: int = 12, perm_xlim: tuple | None = None,
                   impurity_title: str = "Random Forest — impurity importance (train)") -> Path:
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))

    def barh(ax, labels, vals, colors, xlabel, err=None):
        ypos = np.arange(len(labels))[::-1]
        ax.barh(ypos, vals, color=colors, height=0.62, xerr=err,
                error_kw={"ecolor": C_TEXT2, "elinewidth": 1, "capsize": 2},
                edgecolor=C_SURFACE, linewidth=2)
        ax.set_yticks(ypos, labels, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.axvline(0, color=C_TEXT2, linewidth=0.8)
        ax.xaxis.grid(True, color=C_GRID, linewidth=0.6)
        ax.set_axisbelow(True)

    imp = impurity[:top]
    barh(axes[0], [r["feature"] for r in imp], [r["importance"] for r in imp],
         [C_AUTH] * len(imp), "Mean decrease in impurity (sums to 1)")
    axes[0].set_title(impurity_title)

    pr = perm_rf[:top]
    barh(axes[1], [r["feature"] for r in pr], [r["roc_auc_drop_mean"] for r in pr],
         [C_AUTH] * len(pr), "Drop in test ROC-AUC when shuffled (mean ± SD)",
         err=[r["roc_auc_drop_std"] for r in pr])
    axes[1].set_title("Random Forest — permutation importance (test)")
    if perm_xlim:
        axes[1].set_xlim(*perm_xlim)
        axes[1].set_xlabel("Drop in test ROC-AUC when shuffled (mean ± SD)\n"
                           "same x-scale in the real and synthetic figures")

    lc = lr_coef[:top]
    barh(axes[2], [r["feature"] for r in lc], [r["coefficient"] for r in lc],
         [C_POS if r["coefficient"] > 0 else C_NEG for r in lc],
         "Coefficient per 1 SD (log-odds of inauthentic)")
    axes[2].set_title("Logistic Regression — coefficients")
    from matplotlib.patches import Patch
    axes[2].legend(handles=[Patch(color=C_POS, label="toward inauthentic"),
                            Patch(color=C_NEG, label="toward authentic")],
                   loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8.5)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    path = common.OUT_FIGURES / f"{track}_feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_side_by_side(syn: dict, real: dict) -> Path:
    """Same pipeline, two datasets: RF test metrics against what chance gives."""
    plt = _mpl()
    keys = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
            ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1_positive", "F1 (inauthentic)"),
            ("mcc", "MCC")]

    def chance(m, k):
        cm = m["confusion_matrix"]
        pos = (cm["true_positive"] + cm["false_negative"]) / m["n_test"]
        # F1 has no single "chance" value (it depends on how often you guess 1), so no tick.
        return {"accuracy": m["baseline_majority_class"]["accuracy"], "balanced_accuracy": 0.5,
                "roc_auc": 0.5, "pr_auc": pos, "f1_positive": None, "mcc": 0.0}[k]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ypos = np.arange(len(keys))[::-1]
    h = 0.36
    for off, (lab, m, col) in zip((h / 2 + 0.02, -h / 2 - 0.02),
                                  (("Synthetic (goyaladi) — RF", syn, C_INAUTH),
                                   ("Real (Treiman) — RF", real, C_AUTH))):
        vals = [m[k] for k, _ in keys]
        ax.barh(ypos + off, vals, height=h, color=col, label=lab, edgecolor=C_SURFACE, linewidth=2)
        for y, v, (k, _) in zip(ypos + off, vals, keys):
            c = chance(m, k)
            x = max(v, 0, c if (c is not None and abs(c - v) < 0.03) else 0) + 0.012
            ax.text(x, y, f"{v:.3f}", va="center", fontsize=8.5, color=C_TEXT)
        for y, (k, _) in zip(ypos + off, keys):
            c = chance(m, k)
            if c is not None:
                ax.plot([c, c], [y - h / 2, y + h / 2], color=C_TEXT, linewidth=2)
    ax.plot([], [], color=C_TEXT, linewidth=2, label="what no-signal guessing scores")
    ax.set_yticks(ypos, [lab for _, lab in keys])
    ax.set_xlim(-0.05, 1.08)
    ax.axvline(0, color=C_TEXT2, linewidth=0.8)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Same Spark pipeline, two datasets: held-out test metrics (Random Forest)")
    fig.text(0.01, 0.01, "Black ticks: majority-class accuracy, 0.5 for balanced accuracy / ROC-AUC, "
             "positive rate for PR-AUC, 0 for MCC.", fontsize=8.5, color=C_TEXT2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = common.OUT_FIGURES / "twitter_side_by_side.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_distributions(syn_pdf: pd.DataFrame, real_pdf: pd.DataFrame) -> Path:
    """Class-conditional distributions: synthetic (top) vs real (bottom)."""
    plt = _mpl()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2))
    spec = [
        (syn_pdf, "follower_count", "Follower Count", False, 0),
        (syn_pdf, "retweet_count", "Retweet Count", False, 0),
        (syn_pdf, "mention_count", "Mention Count", True, 0),
        (real_pdf, "followers_count", "followers_count (log10 of x+1)", "log", 1),
        (real_pdf, "friends_count", "friends_count (log10 of x+1)", "log", 1),
        (real_pdf, "average_tweets_per_day", "average_tweets_per_day (log10 of x+0.01)", "log", 1),
    ]
    for k, (pdf, col, lab, kind, row) in enumerate(spec):
        ax = axes[row, k % 3]
        x = pdf[col].astype(float)
        if kind == "log":
            x = np.log10(x + (0.01 if col == "average_tweets_per_day" else 1))
        y = pdf["label"].astype(int)
        if kind is True:   # discrete 0..5
            vals = np.arange(int(x.min()), int(x.max()) + 1)
            for off, cls, col_, name in ((-0.18, 0, C_AUTH, "authentic (0)"),
                                         (0.18, 1, C_INAUTH, "inauthentic (1)")):
                share = [(x[y == cls] == v).mean() for v in vals]
                ax.bar(vals + off, share, width=0.34, color=col_, label=name,
                       edgecolor=C_SURFACE, linewidth=2)
            ax.set_ylabel("share of class")
        else:
            if kind is False and x.nunique() <= 200:   # integer-valued: align bins to integers
                w = int(np.ceil((x.max() - x.min() + 1) / 20))
                bins = np.append(np.arange(x.min() - 0.5, x.max() + 0.5, w), x.max() + 0.5)
            else:
                bins = np.linspace(x.min(), x.max(), 41)
            for cls, col_, name in ((0, C_AUTH, "authentic (0)"), (1, C_INAUTH, "inauthentic (1)")):
                ax.hist(x[y == cls], bins=bins, density=True, histtype="step", linewidth=2,
                        color=col_, label=name)
            ax.set_ylabel("density")
        ax.set_xlabel(lab)
        ax.yaxis.grid(True, color=C_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=C_AUTH, linewidth=2, label="authentic / human (0)"),
                        Line2D([], [], color=C_INAUTH, linewidth=2, label="inauthentic / bot (1)")],
               loc="upper center", ncol=2, fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
    axes[0, 0].set_title("Synthetic dataset: the two classes are the same distribution", loc="left")
    axes[1, 0].set_title("Real dataset: bots and humans separate", loc="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = common.OUT_FIGURES / "twitter_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Dataset 1: synthetic negative control
# ---------------------------------------------------------------------------
SYN_NUMERIC = ["retweet_count", "mention_count", "follower_count", "verified",
               "tweet_hour", "tweet_day_of_week", "tweet_days_since_2020_01_01",
               "tweet_len_chars", "tweet_len_words", "has_hashtags", "hashtag_count"]


def load_synthetic(spark):
    from pyspark.sql import functions as F
    from pyspark.sql import types as T
    schema = T.StructType([
        T.StructField("User ID", T.LongType()), T.StructField("Username", T.StringType()),
        T.StructField("Tweet", T.StringType()), T.StructField("Retweet Count", T.IntegerType()),
        T.StructField("Mention Count", T.IntegerType()), T.StructField("Follower Count", T.IntegerType()),
        T.StructField("Verified", T.StringType()), T.StructField("Bot Label", T.IntegerType()),
        T.StructField("Location", T.StringType()), T.StructField("Created At", T.StringType()),
        T.StructField("Hashtags", T.StringType())])
    raw = spark.read.csv(str(SYN_PATH), header=True, schema=schema, escape='"', mode="FAILFAST")
    ts = F.to_timestamp("Created At", "yyyy-MM-dd HH:mm:ss")
    ht = F.trim(F.coalesce(F.col("Hashtags"), F.lit("")))
    df = raw.select(
        F.col("User ID").alias("user_id"), F.col("Username").alias("username"),
        F.col("Tweet").alias("tweet"), F.col("Location").alias("location"),
        F.col("Hashtags").alias("hashtags_raw"),
        # NULL Hashtags = the tweet carried no hashtags (an empty list), so an
        # empty string / count 0 is the true value here, not an imputation.
        ht.alias("hashtags_text"),
        F.col("Retweet Count").cast("double").alias("retweet_count"),
        F.col("Mention Count").cast("double").alias("mention_count"),
        F.col("Follower Count").cast("double").alias("follower_count"),
        bool_col("Verified").alias("verified"),
        F.col("Bot Label").cast("double").alias("label"),
        ts.alias("created_at"))
    df = (df.withColumn("tweet_hour", F.hour("created_at").cast("double"))
            .withColumn("tweet_day_of_week", F.dayofweek("created_at").cast("double"))
            .withColumn("tweet_days_since_2020_01_01",
                        F.datediff(F.to_date("created_at"), F.lit("2020-01-01")).cast("double"))
            .withColumn("tweet_len_chars", F.length("tweet").cast("double"))
            .withColumn("tweet_len_words", F.size(F.split(F.trim("tweet"), r"\s+")).cast("double"))
            .withColumn("has_hashtags", (F.col("hashtags_text") != "").cast("double"))
            .withColumn("hashtag_count", F.when(F.col("hashtags_text") == "", 0.0)
                        .otherwise(F.size(F.split("hashtags_text", r"\s+")).cast("double"))))
    return df


def syn_prep_stages():
    from pyspark.ml.feature import IDF, CountVectorizer, RegexTokenizer
    st = []
    for src in ("tweet", "hashtags_text"):
        base = "tweet" if src == "tweet" else "hashtags"
        st += [RegexTokenizer(inputCol=src, outputCol=f"{base}_tokens", pattern=TFIDF["pattern"],
                              gaps=True, toLowercase=True, minTokenLength=TFIDF["minTokenLength"]),
               CountVectorizer(inputCol=f"{base}_tokens", outputCol=f"{base}_tf",
                               vocabSize=TFIDF["vocabSize"], minDF=TFIDF["minDF"]),
               IDF(inputCol=f"{base}_tf", outputCol=f"{base}_tfidf", minDocFreq=TFIDF["minDocFreq"])]
    return st


def synthetic_validation(pdf: pd.DataFrame) -> dict:
    """Statistical evidence that the file is generated noise, computed from the data."""
    from collections import Counter

    from scipy import stats
    y = pdf["label"].to_numpy().astype(int)
    n = len(pdf)
    v = {"n_rows": n}
    v["label_balance_binomial_test_p_vs_0.5"] = r4(stats.binomtest(int(y.sum()), n, 0.5).pvalue)
    v["distinct_user_id"] = int(pdf["user_id"].nunique())
    v["one_row_per_user_id"] = bool(pdf["user_id"].nunique() == n)
    v["distinct_tweet_text"] = int(pdf["tweet"].nunique())
    rep = pdf[pdf["username"].duplicated(keep=False)].groupby("username")["label"].nunique()
    v["usernames_shared_by_several_user_ids"] = int(len(rep))
    v["shared_usernames_with_conflicting_labels_share"] = r4((rep > 1).mean())

    num = {}
    for col, lo, hi in (("retweet_count", 0, 100), ("mention_count", 0, 5),
                        ("follower_count", 0, 10000)):
        x = pdf[col].to_numpy()
        counts = np.bincount(x.astype(int) - lo, minlength=hi - lo + 1)
        r, rp = stats.pointbiserialr(y, x)
        num[col] = {"min": float(x.min()), "max": float(x.max()),
                    "pearson_r_with_label": r4(r), "pearson_p": r4(rp),
                    "ks_bot_vs_human_p": r4(stats.ks_2samp(x[y == 1], x[y == 0]).pvalue),
                    "chi2_uniform_over_integer_range_p": r4(stats.chisquare(counts).pvalue)}
    v["numeric_columns"] = num
    v["max_abs_numeric_correlation_with_label"] = r4(max(abs(d["pearson_r_with_label"]) for d in num.values()))

    ct = pd.crosstab(pdf["verified"], pdf["label"])
    v["verified_by_label"] = {f"verified={int(i)}": {f"label={int(j)}": int(ct.loc[i, j]) for j in ct.columns}
                              for i in ct.index}
    v["verified_chi2_p"] = r4(stats.chi2_contingency(ct)[1])

    ts = pd.to_datetime(pdf["created_at"])
    sec = ts.astype("int64").to_numpy() // 10**9
    v["created_at_range"] = [str(ts.min()), str(ts.max())]
    v["created_at_ks_bot_vs_human_p"] = r4(stats.ks_2samp(sec[y == 1], sec[y == 0]).pvalue)
    v["tweet_hour_chi2_p"] = r4(stats.chi2_contingency(pd.crosstab(ts.dt.hour, y))[1])

    tok = pdf["tweet"].str.lower().str.findall(r"[a-z]+")
    vocab = Counter(w for ws in tok for w in ws)
    htok = pdf["hashtags_text"].str.lower().str.findall(r"[a-z]+")
    hvocab = Counter(w for ws in htok for w in ws)
    v["tweet_vocabulary_size"] = len(vocab)
    v["tweet_tokens_total"] = int(sum(vocab.values()))
    v["hashtag_vocabulary_size"] = len(hvocab)
    v["hashtag_words_also_in_tweet_vocabulary_share"] = r4(len(set(hvocab) & set(vocab)) / max(1, len(hvocab)))
    v["tweet_words_min_max"] = [int(tok.str.len().min()), int(tok.str.len().max())]
    # Document-frequency of each word per class, then one 2x2 chi-square per word.
    from sklearn.feature_extraction.text import CountVectorizer as SkCV
    X = SkCV(binary=True, token_pattern=r"[a-z]+", lowercase=True).fit(pdf["tweet"])
    M = X.transform(pdf["tweet"])
    df_pos = np.asarray(M[y == 1].sum(0)).ravel()
    df_neg = np.asarray(M[y == 0].sum(0)).ravel()
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    tested, sig, minp = 0, 0, 1.0
    for a, b in zip(df_pos, df_neg):
        if a + b < 200:
            continue
        tested += 1
        p = stats.chi2_contingency([[a, n_pos - a], [b, n_neg - b]])[1]
        sig += p < 0.05
        minp = min(minp, p)
    v["word_label_association"] = {
        "words_tested_(>=200_docs)": tested, "words_with_p_lt_0.05": int(sig),
        "expected_by_chance_at_0.05": round(0.05 * tested, 1), "smallest_p": r4(minp),
        "bonferroni_threshold": float(f"{0.05 / max(1, tested):.2e}"),
        "words_significant_after_bonferroni": int(minp < 0.05 / max(1, tested))}

    loc = pdf["location"]
    v["location_distinct"] = int(loc.nunique())
    prefix = loc.str.extract(r"^(North|South|East|West|Lake|Port|New)\s")[0].fillna("none")
    suffix = loc.str.extract(r"(town|ton|land|ville|berg|burgh|borough|bury|view|port|mouth|"
                             r"stad|furt|chester|fort|haven|side|shire)$")[0].fillna("none")
    v["location_faker_city_pattern"] = {
        "share_matching_faker_city_suffix": r4((suffix != "none").mean()),
        "prefix_vs_label_chi2_p": r4(stats.chi2_contingency(pd.crosstab(prefix, y))[1]),
        "suffix_vs_label_chi2_p": r4(stats.chi2_contingency(pd.crosstab(suffix, y))[1])}
    v["single_feature_auc_max"] = r4(max(single_feature_auc(y, pdf[c].to_numpy()) for c in SYN_NUMERIC))
    return v


def run_synthetic(spark) -> dict:
    track = "twitter_synthetic"
    log(f"[{track}] loading {SYN_PATH.relative_to(common.PROJECT_ROOT)}")
    df = load_synthetic(spark).cache()
    n_rows = df.count()
    assert_no_nulls(df, SYN_NUMERIC + ["label", "tweet", "hashtags_text"], track)

    full_pdf = df.select("user_id", "username", "tweet", "hashtags_text", "location",
                         "created_at", "label", *SYN_NUMERIC).toPandas()
    log(f"[{track}] validating the dataset itself ({n_rows} rows)")
    validation = synthetic_validation(full_pdf)
    faker_json = common.OUT_METRICS / "twitter_synthetic_faker_check.json"
    import json
    validation["faker_provenance_check"] = (
        json.loads(faker_json.read_text()) if faker_json.exists()
        else "not run - see src/twitter_faker_check.py (needs the Faker library)")

    train, test = common.stratified_split(df, label_col="label")
    n_train, n_test = train.count(), test.count()
    log(f"[{track}] split: train {n_train}, test {n_test}")

    input_cols = SYN_NUMERIC + ["tweet_tfidf", "hashtags_tfidf"]
    model_inputs = SYN_NUMERIC + ["tweet", "hashtags_text"]
    res = fit_and_evaluate(spark, train, test, syn_prep_stages, input_cols, model_inputs,
                           ["user_id", "username"], track)

    # Map TF-IDF slot names back to words, using each fitted model's OWN vocabulary.
    def word_names(model):
        vocab = {st.getOutputCol().replace("_tf", "_tfidf"): st.vocabulary
                 for st in model.stages if type(st).__name__ == "CountVectorizerModel"}
        out = []
        for nm in res["feature_names"]:
            for col, words in vocab.items():
                if nm.startswith(col + "_"):
                    nm = f"{col.split('_')[0]}:{words[int(nm.rsplit('_', 1)[1])]}"
                    break
            out.append(nm)
        return out

    names = word_names(res["rf_model"])
    rf_stage, lr_stage = res["rf_model"].stages[-1], res["lr_model"].stages[-1]
    rf_fi = common.importance_table(rf_stage, names)
    lr_fi = common.importance_table(lr_stage, word_names(res["lr_model"]))

    def grouped(rows, key):
        g = {}
        for r in rows:
            k = r["feature"].split(":")[0] + "_tfidf (all terms)" if ":" in r["feature"] else r["feature"]
            g[k] = g.get(k, 0.0) + (r[key] if key == "importance" else abs(r[key]))
        out = [{"feature": k, key if key == "importance" else "abs_coefficient_sum": round(v, 6)}
               for k, v in sorted(g.items(), key=lambda kv: -kv[1])]
        return out

    groups = {c: [c] for c in SYN_NUMERIC}
    groups["tweet_tfidf (Tweet text)"] = ["tweet"]
    groups["hashtags_tfidf (Hashtags text)"] = ["hashtags_text"]
    log(f"[{track}] permutation importance (test, {PERM_REPEATS} repeats per feature)")
    rf_perm, rf_perm_meta = permutation_importance(spark, res["rf_model"], res["test_pdf"],
                                                   res["test_schema"], groups)
    lr_perm, lr_perm_meta = permutation_importance(spark, res["lr_model"], res["test_pdf"],
                                                   res["test_schema"], groups)

    # Leakage audit
    tr_pdf = train.select("label", *SYN_NUMERIC).toPandas()
    ytr = tr_pdf["label"].to_numpy().astype(int)
    sf_auc = {c: r4(single_feature_auc(ytr, tr_pdf[c].to_numpy())) for c in SYN_NUMERIC}
    order = leakage_order_checks(full_pdf, "label", "user_id")
    dups_all = int(full_pdf.drop(columns=["user_id", "username"]).duplicated().sum())
    rf_m, lr_m = res["rf_metrics"], res["lr_metrics"]
    band = rf_m["roc_auc_no_signal_95band"]
    audit = {
        "checks_run": [
            "identifier columns (User ID, Username) excluded from features",
            "row order vs label (point-biserial) and adjacent-label agreement (sorted-file check)",
            "User ID rank correlation with label",
            "exact duplicate rows excluding identifiers",
            "single-feature ROC-AUC scan on TRAIN (alarm if > 0.95)",
            "TF-IDF vocabulary and IDF weights fitted inside each training fold only",
            "near-99% check: headline accuracy far below 0.95, so no ceiling audit was triggered",
            "dataset-validity checks (see dataset_validation): uniformity, label independence, "
            "closed vocabulary, one row per user",
            "test ROC-AUC compared with the analytic no-signal band",
        ],
        "order_checks": order,
        "exact_duplicate_rows_excluding_ids": dups_all,
        "single_feature_auc_train": sf_auc,
        "findings": [
            f"No leakage found - and nothing to leak: every column is independent of the label "
            f"(max |r| = {validation['max_abs_numeric_correlation_with_label']}, "
            f"Verified chi2 p = {validation['verified_chi2_p']}, "
            f"max single-feature AUC on train = {max(sf_auc.values())}).",
            f"The dataset is synthetic. Retweet/Mention/Follower counts are uniform over integer "
            f"ranges 0-100 / 0-5 / 0-10000 (chi2 uniformity p = "
            f"{validation['numeric_columns']['retweet_count']['chi2_uniform_over_integer_range_p']}, "
            f"{validation['numeric_columns']['mention_count']['chi2_uniform_over_integer_range_p']}, "
            f"{validation['numeric_columns']['follower_count']['chi2_uniform_over_integer_range_p']}); "
            f"Tweet and Hashtags draw from the same closed list of "
            f"{validation['tweet_vocabulary_size']} words"
            + (" - exactly the Faker library's en_US lorem word list"
               if isinstance(validation["faker_provenance_check"], dict)
               and validation["faker_provenance_check"]["verdict"].startswith("Tweet and Hashtags")
               else "")
            + f"; every Location has a Faker city prefix or suffix; "
            f"{validation['distinct_user_id']} distinct User IDs for {n_rows} rows (one tweet per user).",
            f"Impurity importance gives the TF-IDF terms "
            f"{sum(r['importance'] for r in grouped(rf_fi, 'importance') if 'tfidf' in r['feature']):.0%} "
            f"of the forest, yet shuffling the text on the test set changes ROC-AUC by "
            f"{next(r['roc_auc_drop_mean'] for r in rf_perm if r['feature'].startswith('tweet_tfidf')):+.4f} "
            f"(Tweet) and "
            f"{next(r['roc_auc_drop_mean'] for r in rf_perm if r['feature'].startswith('hashtags_tfidf')):+.4f} "
            f"(Hashtags): impurity importance rewards many-valued features even when they carry no signal.",
            f"Words associated with the label at p<0.05: "
            f"{validation['word_label_association']['words_with_p_lt_0.05']} of "
            f"{validation['word_label_association']['words_tested_(>=200_docs)']} "
            f"(about {validation['word_label_association']['expected_by_chance_at_0.05']} expected "
            f"by chance); none survive Bonferroni.",
            f"Random Forest test ROC-AUC {rf_m['roc_auc']} and LR {lr_m['roc_auc']} both sit inside "
            f"the no-signal 95% band {band}; accuracy {rf_m['accuracy']} / {lr_m['accuracy']} vs "
            f"majority baseline {rf_m['baseline_majority_class']['accuracy']}.",
            "Result: a suspiciously LOW score that is fully explained by the data, not the model. "
            "The dataset is retained as a negative control and not used for the X/Twitter result.",
        ],
    }

    payload = {
        "platform": "twitter",
        "dataset": {"name": "Twitter Bot Detection Dataset (synthetic negative control)",
                    "source": "kaggle: goyaladi/twitter-bot-detection-dataset",
                    "file": str(SYN_PATH.relative_to(common.PROJECT_ROOT)),
                    "n_rows": n_rows, "label_col": "Bot Label",
                    "positive_means": "Bot Label = 1 (bot) -> 1 = Inauthentic (operational mapping)",
                    "role_in_study": "negative control - original Worksheet 2/3 dataset, shown to be "
                                     "generated noise and set aside"},
        "class_balance": class_balance(df),
        "split": {"method": "stratified_80_20", "seed": SEED, "n_train": n_train, "n_test": n_test,
                  "implementation": "common.stratified_split (sampleBy per class, anti-join)"},
        "features": {
            "used": input_cols,
            "n_assembled_dimensions": len(names),
            "dropped": {
                "User ID": "identifier (contract rule 1)",
                "Username": "identifier (contract rule 1)",
                "Location": f"free-text place name, {validation['location_distinct']} distinct values "
                            f"in {n_rows} rows (median 2 rows per value); Faker city pattern with "
                            f"prefix/suffix unrelated to label "
                            f"(chi2 p = {validation['location_faker_city_pattern']['prefix_vs_label_chi2_p']}"
                            f" / {validation['location_faker_city_pattern']['suffix_vs_label_chi2_p']})",
                "Created At": "decomposed into tweet_hour, tweet_day_of_week, tweet_days_since_2020_01_01",
            },
            "engineered": {
                "verified": "Verified == 'True' -> 1.0, 'False' -> 0.0 (no other values present)",
                "tweet_hour": "hour(Created At)",
                "tweet_day_of_week": "dayofweek(Created At), 1 = Sunday",
                "tweet_days_since_2020_01_01": "datediff(Created At, 2020-01-01)",
                "tweet_len_chars": "length(Tweet)",
                "tweet_len_words": "number of whitespace-separated words in Tweet",
                "has_hashtags": "Hashtags not empty (NULL = tweet had no hashtags)",
                "hashtag_count": "number of hashtags; 0 when the tweet had none (a true zero)",
                "tweet_tfidf": f"RegexTokenizer -> CountVectorizer(vocabSize={TFIDF['vocabSize']}, "
                               f"minDF={TFIDF['minDF']}) -> IDF(minDocFreq={TFIDF['minDocFreq']}) on Tweet",
                "hashtags_tfidf": "same TF-IDF chain on Hashtags",
            },
            "missing_values": "Only Hashtags has NULLs (16.7%); NULL means the tweet carried no "
                              "hashtags, so it becomes an empty list. Every model input is "
                              "asserted NULL-free; nothing is zero-filled.",
            "class_weights": res["weights"],
        },
        "models": {
            "random_forest": model_block(rf_params(res["rf_model"]), res["rf_cv"], rf_m, rf_fi[:60],
                                         rf_perm, rf_perm_meta,
                                         {"feature_importance_grouped": grouped(rf_fi, "importance")}),
            "logistic_regression": model_block(lr_params(res["lr_model"]), res["lr_cv"], lr_m,
                                               lr_fi[:60], lr_perm, lr_perm_meta,
                                               {"coefficients_grouped": grouped(lr_fi, "coefficient")}),
        },
        "dataset_validation": validation,
        "leakage_audit": audit,
        "limitations": [
            "This dataset is synthetic (Faker-style text, uniform counts, labels independent of "
            "every column). No model can or should beat the baseline on it; its numbers say nothing "
            "about real Twitter bots.",
            "Each row is a single tweet with its own User ID, so it cannot represent account-level "
            "behaviour (posting rhythm, follower growth) at all.",
            "The label 'bot' is undefined in the source; even if it were real, 'bot' is not the same "
            "concept as Facebook 'spam' or Instagram 'fake'. 0/1 is an operational mapping.",
            "Kept on purpose as a negative control: it shows the validation step that catches a "
            "dataset which would otherwise be reported as 'model only reaches 50%'.",
        ],
    }
    p = common.save_metrics(payload, track)
    log(f"[{track}] wrote {p.relative_to(common.PROJECT_ROOT)}")
    sp = common.save_predictions(res["scored"], track)
    log(f"[{track}] wrote {sp.relative_to(common.PROJECT_ROOT)}")

    for pred in (res["rf_pred"], res["lr_pred"]):
        pred.unpersist()
    return {"payload": payload, "full_pdf": full_pdf}


# ---------------------------------------------------------------------------
# Dataset 2: real accounts (Treiman)
# ---------------------------------------------------------------------------
REAL_BOOL = ["default_profile", "default_profile_image", "geo_enabled", "verified"]
REAL_NUMERIC = REAL_BOOL + [
    "log_favourites_count", "log_followers_count", "log_friends_count",
    "follower_share", "log_average_tweets_per_day", "account_age_days",
    "has_description", "description_length", "has_location"]


def load_real(spark):
    from pyspark.sql import functions as F
    raw = (spark.read.option("header", True).option("multiLine", True).option("escape", '"')
           .option("mode", "FAILFAST").csv(str(REAL_PATH)))
    desc = F.col("description")
    loc = F.lower(F.trim(F.col("location")))
    lang = F.trim(F.col("lang"))
    df = raw.select(
        F.col("id").cast("long").alias("id"), F.col("screen_name"),
        F.col("split").cast("int").alias("vendor_split"),
        F.to_timestamp("created_at", "yyyy-MM-dd HH:mm:ss").alias("created_at"),
        *[bool_col(c).alias(c) for c in REAL_BOOL],
        *[F.col(c).cast("double").alias(c) for c in
          ("favourites_count", "followers_count", "friends_count", "statuses_count",
           "average_tweets_per_day", "account_age_days")],
        F.when(F.col("account_type") == "bot", 1.0).when(F.col("account_type") == "human", 0.0)
         .alias("label"),
        # Kept only for the leakage audit (download-completeness artifact), never a feature:
        F.col("profile_image_path").isNull().cast("double").alias("_image_file_missing"),
        F.col("profile_background_image_url").isNull().cast("double").alias("_bg_url_missing"),
        # Empty bio: the API returns "", the CSV stores nothing -> NULL. Length 0 is the
        # true length of an empty bio, so this is not a zero-fill of unknown data.
        F.when(desc.isNotNull() & (F.trim(desc) != ""), 1.0).otherwise(0.0).alias("has_description"),
        F.when(desc.isNotNull(), F.length(desc)).otherwise(0).cast("double").alias("description_length"),
        # 'unknown' is the vendor's placeholder for an empty location field.
        F.when(loc.isNotNull() & (loc != "") & (loc != "unknown"), 1.0).otherwise(0.0).alias("has_location"),
        # lang NULL (21%) is kept as its own explicit category, never dropped or zero-filled.
        F.when(lang.isNull() | (lang == ""), F.lit("__missing__")).otherwise(lang).alias("lang_raw"),
    )
    df = (df.withColumn("log_favourites_count", F.log1p("favourites_count"))
            .withColumn("log_followers_count", F.log1p("followers_count"))
            .withColumn("log_friends_count", F.log1p("friends_count"))
            # Followers/friends ratio as a bounded share: r / (1 + r) with
            # r = (followers + 1) / (friends + 1). Denominator >= 2, so no divide-by-zero.
            .withColumn("follower_share", (F.col("followers_count") + 1)
                        / (F.col("followers_count") + F.col("friends_count") + 2))
            # First design, kept ONLY for the collinearity diagnostic (never a feature):
            .withColumn("_ratio_log_v1",
                        F.log((F.col("followers_count") + 1) / (F.col("friends_count") + 1)))
            .withColumn("log_average_tweets_per_day", F.log1p("average_tweets_per_day"))
            .withColumn("log_statuses_count", F.log1p("statuses_count")))
    return df


_LANG_LEVELS: list[str] = []


def real_prep_stages():
    from pyspark.ml.feature import OneHotEncoder, StringIndexer
    return [StringIndexer(inputCol="lang_cat", outputCol="lang_idx", stringOrderType="alphabetAsc",
                          handleInvalid="keep"),
            OneHotEncoder(inputCols=["lang_idx"], outputCols=["lang_onehot"], dropLast=True,
                          handleInvalid="keep")]


def apply_lang_levels(df, levels: list[str]):
    from pyspark.sql import functions as F
    return df.withColumn("lang_cat", F.when(F.col("lang_raw").isin(levels), F.col("lang_raw"))
                         .otherwise(F.lit("__other__")))


def collinearity_diagnostic(spark, train_w, test, best_rf_depth, best_lr_reg) -> dict:
    """statuses_count = average_tweets_per_day * account_age_days by construction.
    Measure it, then show what keeping all three does to LR and RF importances."""
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
    from pyspark.ml.feature import StandardScaler, VectorAssembler

    pdf = train_w.select(*dict.fromkeys(["statuses_count", "average_tweets_per_day",
                                         "account_age_days", "log_statuses_count",
                                         "_ratio_log_v1", *REAL_NUMERIC])).toPandas()
    prod = pdf["average_tweets_per_day"] * pdf["account_age_days"]
    abs_err = (pdf["statuses_count"] - prod).abs()
    # rate is published rounded to 3 dp, so |error| <= 0.0005 * age (+1 for rounding the count)
    within = (abs_err <= 0.0005 * pdf["account_age_days"] + 1).mean()
    ok = ((pdf["statuses_count"] > 0) & (pdf["average_tweets_per_day"] > 0)
          & (pdf["account_age_days"] > 0))
    A = np.column_stack([np.ones(int(ok.sum())), np.log(pdf.loc[ok, "average_tweets_per_day"]),
                         np.log(pdf.loc[ok, "account_age_days"])])
    yv = np.log(pdf.loc[ok, "statuses_count"]).to_numpy()
    beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
    r2 = 1 - (yv - A @ beta).var() / yv.var()
    out = {
        "identity_tested": "statuses_count == average_tweets_per_day * account_age_days",
        "pearson_r_statuses_vs_rate_x_age": r4(np.corrcoef(pdf["statuses_count"], prod)[0, 1]),
        "share_rows_exact_within_rounding": r4(within),
        "median_relative_error": float(f"{(abs_err / pdf['statuses_count'].clip(lower=1)).median():.2e}"),
        "log_space_regression": {"model": "ln(statuses) = b0 + b1 ln(rate) + b2 ln(age)",
                                 "rows_with_all_positive": int(ok.sum()), "r2": round(float(r2), 6),
                                 "b1_ln_rate": round(float(beta[1]), 4),
                                 "b2_ln_age": round(float(beta[2]), 4),
                                 "implied_vif_of_ln_statuses": round(1.0 / (1 - r2), 1)},
    }
    num_cols = [c for c in REAL_NUMERIC if c not in REAL_BOOL]
    out["vif_with_statuses_count"] = vif_table(pdf, num_cols + ["log_statuses_count"])
    out["vif_used_feature_set"] = vif_table(pdf, num_cols)
    v1_cols = [c if c != "follower_share" else "_ratio_log_v1" for c in num_cols]
    out["ratio_redesign"] = {
        "first_version": "ln((followers + 1) / (friends + 1))",
        "problem": "identical to log_followers_count - log_friends_count, an exact linear "
                   "combination of two other features (found by this VIF check)",
        "vif_first_version": {k: v for k, v in vif_table(pdf, v1_cols).items()
                              if k in ("log_followers_count", "log_friends_count", "_ratio_log_v1")},
        "fix": "follower_share = r / (1 + r); same ordering as the ratio (so Random Forest splits "
               "are unchanged) but no longer a linear combination of the logs",
    }

    # Refit both models with the redundant third column added back (same hyperparameters).
    def fit(cols, kind, data):
        stages = real_prep_stages() + [VectorAssembler(inputCols=cols + ["lang_onehot"], outputCol="features")]
        if kind == "rf":
            stages.append(RandomForestClassifier(labelCol="label", weightCol="weight",
                                                 **{**RF_FIXED, "maxDepth": best_rf_depth}))
        else:
            stages += [StandardScaler(inputCol="features", outputCol="features_scaled", **SCALER),
                       LogisticRegression(labelCol="label", featuresCol="features_scaled", weightCol="weight",
                                          **{**LR_FIXED, "regParam": best_lr_reg})]
        return Pipeline(stages=stages).fit(data)

    trio = ["log_statuses_count", "log_average_tweets_per_day", "account_age_days"]
    with_cols = REAL_NUMERIC + ["log_statuses_count"]
    rf_with = fit(with_cols, "rf", train_w)
    names = with_cols + [f"lang_{i}" for i in range(rf_with.stages[-1].numFeatures - len(with_cols))]
    imp = dict(zip(names, rf_with.stages[-1].featureImportances.toArray()))
    out["rf_impurity_importance_with_all_three"] = {c: round(float(imp[c]), 4) for c in trio}
    out["rf_impurity_rank_with_all_three"] = {
        c: int(sorted(imp, key=lambda k: -imp[k]).index(c) + 1) for c in trio}
    # Does the redundant column change predictive power? Answered INSIDE train
    # (stratified 80:20 of the training split), so the test set plays no part in the decision.
    in_tr, in_va = common.stratified_split(train_w, label_col="label", seed=SEED)
    in_tr = in_tr.cache()
    out["rf_inner_validation_roc_auc"] = {
        "with_all_three": common.evaluate_binary(fit(with_cols, "rf", in_tr).transform(in_va))["roc_auc"],
        "without_statuses (used)": common.evaluate_binary(fit(REAL_NUMERIC, "rf", in_tr).transform(in_va))["roc_auc"],
        "scope": "80:20 stratified split inside the training data; test split not used"}
    in_tr.unpersist()

    # LR coefficient stability over 5 bootstrap resamples of train, with vs without.
    boot = {"with_all_three": {c: [] for c in trio},
            "without_statuses (used)": {c: [] for c in trio[1:]}}
    for b in range(5):
        for label, cols in (("with_all_three", with_cols), ("without_statuses (used)", REAL_NUMERIC)):
            m = fit(cols, "lr", train_w.sample(True, 1.0, seed=SEED + b))
            coef = dict(zip(cols, m.stages[-1].coefficients.toArray()[:len(cols)]))
            for c in boot[label]:
                boot[label][c].append(float(coef[c]))
    out["lr_bootstrap_coefficients"] = {
        k: {c: {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v)), 4),
                "min": round(float(np.min(v)), 4), "max": round(float(np.max(v)), 4)}
            for c, v in d.items()} for k, d in boot.items()}
    out["decision"] = ("Dropped statuses_count. It is the product of the other two by construction, "
                       "so it adds no information. Kept average_tweets_per_day (activity intensity) "
                       "and account_age_days (account maturity), which are nearly unrelated "
                       "to each other in log space.")
    return out


def run_real(spark) -> dict:
    from pyspark.sql import functions as F
    track = "twitter_real"
    log(f"[{track}] loading {REAL_PATH.relative_to(common.PROJECT_ROOT)}")
    df = load_real(spark).cache()
    n_rows = df.count()

    # Parse integrity: Spark's multiLine CSV must agree with an independent pandas read.
    chk = pd.read_csv(REAL_PATH, usecols=["id", "account_type"])
    n_bot_spark = df.filter(F.col("label") == 1.0).count()
    if n_rows != len(chk) or n_bot_spark != int((chk["account_type"] == "bot").sum()):
        raise RuntimeError(f"Spark parse mismatch: {n_rows} rows / {n_bot_spark} bots vs pandas "
                           f"{len(chk)} / {(chk['account_type'] == 'bot').sum()}")
    if df.filter(F.col("label").isNull()).count():
        raise ValueError("account_type has values other than bot/human")
    assert_no_nulls(df, [c for c in REAL_NUMERIC] + ["label", "lang_raw", "statuses_count"], track)

    full_cols = list(dict.fromkeys(
        ["id", "label", "created_at", "vendor_split", "followers_count", "friends_count",
         "average_tweets_per_day", "statuses_count", "account_age_days", "_image_file_missing",
         "_bg_url_missing", "lang_raw", *REAL_NUMERIC]))
    full_pdf = df.select(*full_cols).toPandas()

    train, test = common.stratified_split(df, label_col="label")
    # lang levels learned from TRAIN only; rare levels collapse to __other__
    lc = train.groupBy("lang_raw").count().collect()
    levels = sorted(r["lang_raw"] for r in lc if r["count"] >= LANG_MIN_TRAIN_COUNT)
    train, test = apply_lang_levels(train, levels), apply_lang_levels(test, levels)
    n_train, n_test = train.count(), test.count()
    log(f"[{track}] split: train {n_train}, test {n_test}; lang levels kept: {len(levels)}")

    input_cols = REAL_NUMERIC + ["lang_onehot"]
    model_inputs = REAL_NUMERIC + ["lang_cat"]
    res = fit_and_evaluate(spark, train, test, real_prep_stages, input_cols, model_inputs,
                           ["id", "screen_name"], track)
    names = [n.replace("lang_onehot_", "lang=") for n in res["feature_names"]]
    rf_stage, lr_stage = res["rf_model"].stages[-1], res["lr_model"].stages[-1]
    rf_fi = common.importance_table(rf_stage, names)
    lr_fi = common.importance_table(lr_stage, names)

    groups = {c: [c] for c in REAL_NUMERIC}
    groups["lang (all levels)"] = ["lang_cat"]
    log(f"[{track}] permutation importance (test, {PERM_REPEATS} repeats per feature)")
    rf_perm, rf_perm_meta = permutation_importance(spark, res["rf_model"], res["test_pdf"],
                                                   res["test_schema"], groups)
    lr_perm, lr_perm_meta = permutation_importance(spark, res["lr_model"], res["test_pdf"],
                                                   res["test_schema"], groups)
    rf_lang = round(sum(r["importance"] for r in rf_fi if r["feature"].startswith("lang=")), 6)

    log(f"[{track}] collinearity diagnostic (statuses = rate x age)")
    train_w, _ = add_class_weights(train)
    colin = collinearity_diagnostic(spark, train_w.cache(), test,
                                    res["rf_cv"]["selected"]["maxDepth"], res["lr_cv"]["selected"]["regParam"])

    # Secondary sanity check on the vendor's own partition (no re-tuning).
    log(f"[{track}] vendor-split sanity check")
    vtr = apply_lang_levels(df.filter(F.col("vendor_split") == 0), levels)
    vte = apply_lang_levels(df.filter(F.col("vendor_split").isin(1, 2)), levels)
    vtr_w, vw = add_class_weights(vtr)
    rf_p, _, lr_p, _ = build_pipelines(real_prep_stages, input_cols)
    rf_p.getStages()[-1].setMaxDepth(res["rf_cv"]["selected"]["maxDepth"])
    lr_p.getStages()[-1].setRegParam(res["lr_cv"]["selected"]["regParam"])
    vendor = {"train_rows": vtr.count(), "eval_rows": vte.count(),
              "train_definition": "vendor split == 0", "eval_definition": "vendor split in {1, 2}",
              "hyperparameters": "CV-selected on the stratified split; no re-tuning", "class_weights": vw}
    for tag, pipe in (("random_forest", rf_p), ("logistic_regression", lr_p)):
        m = common.evaluate_binary(pipe.fit(vtr_w).transform(vte))
        vendor[tag] = {k: m[k] for k in ("accuracy", "precision_positive", "recall_positive", "f1_positive",
                                         "balanced_accuracy", "mcc", "roc_auc", "pr_auc")}
        vendor[tag]["baseline_accuracy"] = m["baseline_majority_class"]["accuracy"]

    # Leakage audit
    from scipy import stats
    tr_pdf = train.select("label", *REAL_NUMERIC, "_image_file_missing", "_bg_url_missing",
                          "log_statuses_count").toPandas()
    ytr = tr_pdf["label"].to_numpy().astype(int)
    sf_auc = {c: r4(single_feature_auc(ytr, tr_pdf[c].to_numpy())) for c in REAL_NUMERIC + ["log_statuses_count"]}
    sf_auc_sorted = dict(sorted(sf_auc.items(), key=lambda kv: -kv[1]))
    order = leakage_order_checks(full_pdf, "label", "id")
    rho_id_created = stats.spearmanr(full_pdf["id"], full_pdf["created_at"].astype("int64")).statistic
    img_rate = full_pdf.groupby("label")["_image_file_missing"].mean()
    bg = full_pdf.groupby("default_profile")["_bg_url_missing"].mean()
    bg_year = full_pdf.assign(y=full_pdf["created_at"].dt.year).groupby("y")["_bg_url_missing"].mean()
    feat_cols = REAL_NUMERIC + ["lang_cat"]
    tr_keys = set(map(tuple, train.select(*feat_cols).toPandas().round(6).astype(str).to_numpy()))
    te_feat = res["test_pdf"][feat_cols].round(6).astype(str).to_numpy()
    cross = int(sum(tuple(r) in tr_keys for r in te_feat))
    collection = (full_pdf["created_at"] + pd.to_timedelta(full_pdf["account_age_days"], unit="D"))
    rf_m, lr_m = res["rf_metrics"], res["lr_metrics"]
    audit = {
        "checks_run": [
            "identifier columns (id, screen_name, unnamed row index) excluded from features",
            "row order vs label and adjacent-label agreement (sorted-file check)",
            "id rank correlation with label, explained against account creation time",
            "exact duplicate rows; test feature vectors identical to a train feature vector",
            "single-feature ROC-AUC scan on TRAIN (alarm if > 0.95)",
            "label-dependent missingness in dropped artifact columns (image paths, background URL)",
            "construction redundancy: statuses = rate x age, account_age_days = collection date - created_at",
            "lang levels learned from TRAIN only",
            "near-99% check: headline accuracy below 0.95, so no ceiling audit was triggered",
            "secondary evaluation on the vendor's own split",
        ],
        "order_checks": order,
        "id_vs_created_at_spearman": r4(rho_id_created),
        "exact_duplicate_rows": int(full_pdf.drop(columns=["id"]).duplicated().sum()),
        "test_rows_with_feature_vector_identical_to_a_train_row": cross,
        "single_feature_auc_train": sf_auc_sorted,
        "profile_image_file_missing_rate_by_label": {"authentic": r4(img_rate.get(0.0)),
                                                     "inauthentic": r4(img_rate.get(1.0))},
        "profile_image_file_missing_single_flag_auc_train": audit_flag_auc(ytr, tr_pdf),
        "background_url_missing_rate_by_default_profile": {str(bool(k)): r4(v) for k, v in bg.items()},
        "background_url_missing_rate_by_created_year": {int(k): r4(v) for k, v in bg_year.items()},
        "implied_collection_window": [str(collection.min()), str(collection.max())],
        "findings": [
            "File is not sorted by label (row-order r ~ 0, adjacent-label agreement equals the "
            "shuffled expectation).",
            f"id correlates with the label (Spearman {order[-1]['spearman_rho']}) only because Twitter "
            f"ids are issued in time order (id vs created_at Spearman {r4(rho_id_created)}) and bots in "
            "this sample are younger accounts. That information is already in account_age_days; id is "
            "excluded as an identifier.",
            f"profile_image_path / profile_background_image_path are missing for "
            f"{img_rate.get(0.0):.1%} of humans but {img_rate.get(1.0):.1%} of bots - a download-"
            f"completeness artifact correlated with the label (single-flag AUC "
            f"{audit_flag_auc(ytr, tr_pdf)}). Using it would be leakage; dropped.",
            "profile_background_image_url is NULL only for default-profile accounts created from "
            "2015 on (Twitter stopped assigning background themes), so it encodes account era plus "
            "default_profile, both already present. Dropped.",
            f"Highest single-feature train AUC is {next(iter(sf_auc_sorted.values()))} "
            f"({next(iter(sf_auc_sorted))}) - no feature alone gives the answer away.",
            f"statuses_count equals average_tweets_per_day x account_age_days for "
            f"{colin['share_rows_exact_within_rounding']:.1%} of training rows (collinearity by "
            f"construction, not label leakage); handled by dropping statuses_count.",
            f"created_at is redundant with account_age_days (collection on "
            f"{str(collection.min().date())} - {str(collection.max().date())}); dropped.",
            f"Headline RF accuracy {rf_m['accuracy']} is far from 99%; the vendor split gives "
            f"{vendor['random_forest']['accuracy']}, consistent with the stratified split.",
        ],
    }

    payload = {
        "platform": "twitter",
        "dataset": {"name": "Twitter Human/Bots Dataset (real accounts)",
                    "source": "kaggle: danieltreiman/twitter-human-bots-dataset",
                    "file": str(REAL_PATH.relative_to(common.PROJECT_ROOT)),
                    "n_rows": n_rows, "label_col": "account_type",
                    "positive_means": "account_type = 'bot' -> 1 = Inauthentic (operational mapping); "
                                      "'human' -> 0 = Authentic",
                    "role_in_study": "headline X/Twitter result (replaces the synthetic dataset)",
                    "ignored_assets": "profile_images/ and profile_background_images/ JPEG folders"},
        "class_balance": class_balance(df),
        "split": {"method": "stratified_80_20", "seed": SEED, "n_train": n_train, "n_test": n_test,
                  "implementation": "common.stratified_split (sampleBy per class, anti-join)",
                  "vendor_split_column": "ignored for the headline; used only in vendor_split_check"},
        "features": {
            "used": input_cols,
            "n_assembled_dimensions": len(names),
            "lang_levels_kept": levels,
            "dropped": {
                "id": "identifier (contract rule 1); also a proxy for creation time",
                "screen_name": "identifier (contract rule 1)",
                "Unnamed row index (_c0)": "file row number",
                "profile_image_url": "URL artifact (string value carries no behaviour)",
                "profile_background_image_url": "URL artifact; its NULLs encode account era + default_profile",
                "profile_image_path": "local download path; 35% NULL by download completeness, "
                                      "label-dependent -> leakage risk",
                "profile_background_image_path": "same as profile_image_path",
                "split": "vendor's own partition, not an account property",
                "statuses_count": "= average_tweets_per_day x account_age_days by construction (see collinearity)",
                "created_at": "= collection date - account_age_days by construction",
                "description (raw text)": "replaced by has_description and description_length",
                "location (raw text)": "replaced by has_location",
                "lang (raw)": "replaced by lang_cat (one-hot)",
            },
            "engineered": {
                "log_favourites_count": "ln(1 + favourites_count)",
                "log_followers_count": "ln(1 + followers_count)",
                "log_friends_count": "ln(1 + friends_count)",
                "follower_share": "(followers_count + 1) / (followers_count + friends_count + 2) "
                                  "= r / (1 + r), r = (followers + 1) / (friends + 1): the followers/"
                                  "friends ratio as a bounded 0-1 share; +1/+2 smoothing guards "
                                  "divide-by-zero (no followers and no friends -> 0.5)",
                "log_average_tweets_per_day": "ln(1 + average_tweets_per_day)",
                "has_description": "1 if description is non-empty",
                "description_length": "characters in description; 0 for an empty bio (true value)",
                "has_location": "1 if location is set and not the placeholder 'unknown'",
                "lang_cat": f"lang; NULL -> explicit '__missing__' level; levels with < "
                            f"{LANG_MIN_TRAIN_COUNT} training accounts -> '__other__'; one-hot "
                            "(dropLast) for both models",
                "booleans": "default_profile, default_profile_image, geo_enabled, verified: "
                            "'True'/'False' -> 1.0/0.0",
            },
            "missing_values": {
                "lang": "21.3% NULL -> its own category '__missing__' (informative: 13.8% of humans, "
                        "36.2% of bots)",
                "description": "19.4% NULL = empty bio -> has_description 0, length 0",
                "location": "4 NULL + 34.8% 'unknown' placeholder -> has_location 0",
                "all model inputs": "asserted NULL-free after the rules above; nothing zero-filled",
            },
            "collinearity": colin,
            "class_weights": res["weights"],
        },
        "models": {
            "random_forest": model_block(rf_params(res["rf_model"]), res["rf_cv"], rf_m, rf_fi,
                                         rf_perm, rf_perm_meta,
                                         {"lang_onehot_importance_sum": rf_lang}),
            "logistic_regression": model_block(lr_params(res["lr_model"]), res["lr_cv"], lr_m, lr_fi,
                                               lr_perm, lr_perm_meta),
        },
        "vendor_split_check": vendor,
        "leakage_audit": audit,
        "limitations": [
            "Labels come from the dataset author's sources (earlier annotated bot collections), not "
            "from Twitter; 'bot' here is not the same concept as Facebook 'spam' or Instagram 'fake'. "
            "0/1 is an operational mapping.",
            "Snapshot collected around 2020-08-18/19 via the old v1.1 API; account behaviour and the "
            "platform (now X) have changed since. Do not expect the same performance on today's accounts.",
            "Bots in this sample are younger, have fewer followers and emptier profiles; a bot "
            "operator who fills in a profile and buys followers would not look like these bots.",
            "Profile metadata only - no tweet content, timing or network features.",
            "lang missingness is informative but may partly reflect how the author collected the "
            "data rather than account behaviour.",
            "Class weighting at a 0.5 threshold trades precision for recall. False positives are real "
            "people: scores are a review queue for trust-and-safety moderators, never an automated sanction.",
        ],
    }
    p = common.save_metrics(payload, track)
    log(f"[{track}] wrote {p.relative_to(common.PROJECT_ROOT)}")
    sp = common.save_predictions(res["scored"], track)
    log(f"[{track}] wrote {sp.relative_to(common.PROJECT_ROOT)}")
    return {"payload": payload, "full_pdf": full_pdf}


def audit_flag_auc(y, pdf) -> float | None:
    return r4(single_feature_auc(y, pdf["_image_file_missing"].to_numpy()))


# ---------------------------------------------------------------------------
def make_figures() -> list[Path]:
    """Draw every figure from the saved metrics JSON (+ raw CSVs for the
    distribution panel), so figures can be redrawn without re-running Spark."""
    import json
    paths = {t: common.OUT_METRICS / f"{t}.json" for t in ("twitter_synthetic", "twitter_real")}
    js = {t: json.loads(p.read_text()) for t, p in paths.items() if p.exists()}
    out = []
    xlim = None
    if "twitter_real" in js:
        top = max(r["roc_auc_drop_mean"] + r["roc_auc_drop_std"]
                  for r in js["twitter_real"]["models"]["random_forest"]["permutation_importance"])
        xlim = (-0.012, round(top * 1.15, 3))
    for t, d in js.items():
        rf, lr = d["models"]["random_forest"], d["models"]["logistic_regression"]
        label = ("Synthetic dataset (negative control)" if t == "twitter_synthetic"
                 else "Real dataset (Treiman)")
        out.append(fig_confusion(t, f"{label} — confusion matrices",
                                 {"Random Forest": rf["metrics"], "Logistic Regression": lr["metrics"]}))
        if t == "twitter_synthetic":
            out.append(fig_importance(
                t, f"{label} — impurity says 'text matters', permutation says nothing matters",
                rf["feature_importance_grouped"], rf["permutation_importance"],
                lr["feature_importance"], perm_xlim=xlim,
                impurity_title="Random Forest — impurity importance (train)\nTF-IDF terms summed per field"))
        else:
            out.append(fig_importance(t, f"{label} — feature importance, three views",
                                      rf["feature_importance"], rf["permutation_importance"],
                                      lr["feature_importance"], perm_xlim=xlim))
    if len(js) == 2:
        out.append(fig_side_by_side(js["twitter_synthetic"]["models"]["random_forest"]["metrics"],
                                    js["twitter_real"]["models"]["random_forest"]["metrics"]))
        syn = pd.read_csv(SYN_PATH, usecols=["Follower Count", "Retweet Count", "Mention Count", "Bot Label"])
        syn = syn.rename(columns={"Follower Count": "follower_count", "Retweet Count": "retweet_count",
                                  "Mention Count": "mention_count", "Bot Label": "label"})
        real = pd.read_csv(REAL_PATH, usecols=["followers_count", "friends_count",
                                               "average_tweets_per_day", "account_type"])
        real["label"] = (real["account_type"] == "bot").astype(int)
        out.append(fig_distributions(syn, real))
    for p in out:
        log(f"wrote {p.relative_to(common.PROJECT_ROOT)}")
    return out


# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", choices=["synthetic", "real", "both"], default="both")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw figures from outputs/metrics/twitter_*.json without Spark")
    args = ap.parse_args(argv)
    if args.figures_only:
        make_figures()
        return

    t0 = time.time()
    spark = common.get_spark("twitter_pipeline", driver_memory="2g", cores=3)
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    try:
        out = {}
        if args.dataset in ("real", "both"):
            out["real"] = run_real(spark)
        if args.dataset in ("synthetic", "both"):
            out["synthetic"] = run_synthetic(spark)
    finally:
        spark.stop()
    make_figures()
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
