"""Helpers for the two profile-metadata tracks (Instagram, Facebook) only.

Anything the Instagram and Facebook pipelines both need beyond ``common`` lives
here. It is owned by the IG/FB track and is not imported by the X/Twitter or
TikTok tracks.

What it adds on top of ``common``:
  * one place where the Random Forest / Logistic Regression pipelines and their
    hyperparameters are defined, so both platforms use identical settings;
  * repeated stratified k-fold cross-validation *inside the training split*
    (the test split is never touched), with out-of-fold scores;
  * bootstrap confidence intervals and Wilson intervals for test metrics;
  * permutation importance on the test split, computed in a single Spark pass;
  * an exact McNemar test for "is RF really better than LR on this test set?";
  * the shared figures (confusion matrix, importance, CV variability).

Every metric that is reported as *the* metric of a fold or a split comes from
``common.evaluate_binary`` so that numbers stay comparable with other tracks.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

import common

SEED = common.RANDOM_SEED

# --------------------------------------------------------------------------
# Model definitions - fixed a priori, NOT tuned (contract rule 6).
# --------------------------------------------------------------------------
RF_PARAMS = {
    "numTrees": 200,
    "maxDepth": 10,
    "featureSubsetStrategy": "sqrt",
    "impurity": "gini",
    "maxBins": 32,
    "minInstancesPerNode": 1,
    "subsamplingRate": 1.0,
    "bootstrap": True,
    "seed": SEED,
}
LR_PARAMS = {
    "family": "binomial",
    "maxIter": 200,
    # Small ridge penalty: keeps coefficients finite when classes are (nearly)
    # linearly separable, which an unpenalised LR cannot handle.
    "regParam": 0.01,
    "elasticNetParam": 0.0,
    # Features are standardised explicitly by a StandardScaler stage so the
    # coefficients are directly comparable (log-odds per 1 SD of the feature).
    "standardization": False,
    "threshold": 0.5,
}
SCALER_PARAMS = {"withMean": True, "withStd": True}
DECISION_THRESHOLD = 0.5

MODEL_KINDS = ("random_forest", "logistic_regression")
CV_METRICS = ("accuracy", "precision_positive", "recall_positive", "f1_positive",
              "specificity", "balanced_accuracy", "mcc", "roc_auc", "pr_auc")

# Colours (reference palette, light surface). Authentic = blue, Inauthentic = orange.
C_AUTH = "#2a78d6"
C_INAUTH = "#eb6834"
C_NEUTRAL = "#8a8984"
C_TEXT = "#0b0b0b"
C_TEXT2 = "#52514e"
C_SURFACE = "#fcfcfb"
C_GRID = "#e4e3df"


def model_params(kind: str, weighted: bool) -> dict:
    """Every hyperparameter set for a model, for the metrics JSON."""
    if kind == "random_forest":
        p = dict(RF_PARAMS)
    else:
        p = dict(LR_PARAMS)
        p["feature_scaling"] = "StandardScaler(" + ", ".join(
            f"{k}={v}" for k, v in SCALER_PARAMS.items()) + ")"
    p["weightCol"] = "weight (balanced: n / (2 * n_class), from training rows only)" if weighted else None
    p["decision_threshold"] = DECISION_THRESHOLD
    p["tuned"] = False
    return p


def build_pipeline(kind: str, feature_cols: list[str], weight_col: str | None = None):
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
    from pyspark.ml.feature import StandardScaler, VectorAssembler

    if kind == "random_forest":
        asm = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="error")
        clf = RandomForestClassifier(featuresCol="features", labelCol="label", **RF_PARAMS)
        stages = [asm, clf]
    elif kind == "logistic_regression":
        asm = VectorAssembler(inputCols=feature_cols, outputCol="features_raw", handleInvalid="error")
        sc = StandardScaler(inputCol="features_raw", outputCol="features", **SCALER_PARAMS)
        clf = LogisticRegression(featuresCol="features", labelCol="label", **LR_PARAMS)
        stages = [asm, sc, clf]
    else:
        raise ValueError(kind)
    if weight_col:
        clf.setWeightCol(weight_col)
    return Pipeline(stages=stages)


# --------------------------------------------------------------------------
# pandas <-> Spark
# --------------------------------------------------------------------------
def add_balanced_weights(pdf: pd.DataFrame, label: str = "label") -> pd.DataFrame:
    """weight = n / (2 * n_class), computed on the rows passed in (training rows)."""
    out = pdf.copy()
    n = len(out)
    counts = out[label].value_counts().to_dict()
    out["weight"] = out[label].map(lambda c: n / (2.0 * counts[c])).astype(float)
    return out


def to_spark(spark, pdf: pd.DataFrame):
    from pyspark.sql import functions as F
    sdf = spark.createDataFrame(pdf.reset_index(drop=True))
    return sdf.withColumn("label", F.col("label").cast("double"))


def score(model, sdf, keep: Iterable[str]):
    """Apply a fitted pipeline; return Spark DF with p_inauthentic as a plain column."""
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F
    pred = model.transform(sdf)
    return pred.withColumn("p_inauthentic", vector_to_array("probability").getItem(1)) \
               .select(*keep, "label", "prediction", "probability", "p_inauthentic")


def fit_and_evaluate(spark, train_pdf, test_pdf, feature_cols, kind, weighted, keep=("account_key",)):
    """Fit one model on train, evaluate on test with common.evaluate_binary."""
    tr = add_balanced_weights(train_pdf) if weighted else train_pdf
    model = build_pipeline(kind, feature_cols, "weight" if weighted else None).fit(to_spark(spark, tr))
    pred = score(model, to_spark(spark, test_pdf), keep).cache()
    metrics = common.evaluate_binary(pred)
    pred_pdf = pred.select(*keep, "label", "prediction", "p_inauthentic").toPandas()
    pred.unpersist()
    # feature names straight from the assembler metadata, in vector order
    asm_out = model.stages[0].transform(to_spark(spark, test_pdf.head(2)))
    names = common.assembled_feature_names(asm_out, model.stages[0].getOutputCol())
    imp = common.importance_table(model.stages[-1], names)
    return {"model": model, "metrics": metrics, "pred": pred_pdf, "importance": imp,
            "feature_names": names}


# --------------------------------------------------------------------------
# Repeated stratified k-fold CV (inside the training split only)
# --------------------------------------------------------------------------
def repeated_cv(spark, train_pdf, feature_cols, weighted, n_splits=5, n_repeats=10,
                keep=("account_key",), log=print):
    """Repeated stratified k-fold on the TRAINING rows. The test split is never seen.

    Returns per-model: list of per-fold metric dicts (from common.evaluate_binary),
    mean/std/min/max summary, and out-of-fold scores from the first repeat
    (so every training account gets one honest, unseen-by-the-model score).
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    y = train_pdf["label"].values
    res = {k: {"folds": [], "oof": []} for k in MODEL_KINDS}
    for i, (tr_idx, va_idx) in enumerate(rskf.split(np.zeros(len(y)), y)):
        rep, fold = divmod(i, n_splits)
        tr = train_pdf.iloc[tr_idx]
        va = train_pdf.iloc[va_idx]
        if weighted:
            tr = add_balanced_weights(tr)
        tr_s = to_spark(spark, tr).cache()
        va_s = to_spark(spark, va).cache()
        for kind in MODEL_KINDS:
            model = build_pipeline(kind, feature_cols, "weight" if weighted else None).fit(tr_s)
            pred = score(model, va_s, keep).cache()
            m = common.evaluate_binary(pred)
            row = {"repeat": rep, "fold": fold, "n_train": len(tr), "n_val": len(va)}
            row.update({k: m[k] for k in CV_METRICS if k in m})
            res[kind]["folds"].append(row)
            if rep == 0:
                o = pred.select(*keep, "label", "prediction", "p_inauthentic").toPandas()
                res[kind]["oof"].append(o)
            pred.unpersist()
        tr_s.unpersist(); va_s.unpersist()
        if (i + 1) % n_splits == 0:
            log(f"    CV repeat {rep + 1}/{n_repeats} done")
    out = {}
    for kind in MODEL_KINDS:
        folds = pd.DataFrame(res[kind]["folds"])
        summ = {}
        for k in CV_METRICS:
            if k in folds:
                v = folds[k].astype(float)
                summ[k] = {"mean": round(v.mean(), 4), "std": round(v.std(ddof=1), 4),
                           "min": round(v.min(), 4), "max": round(v.max(), 4)}
        out[kind] = {
            "scheme": f"repeated stratified {n_splits}-fold x {n_repeats} repeats on the training split only",
            "n_fits": len(folds),
            "summary": summ,
            "folds": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                      for r in res[kind]["folds"]],
            "_oof": pd.concat(res[kind]["oof"], ignore_index=True),
        }
    # paired RF - LR difference per fold
    rf = pd.DataFrame(res["random_forest"]["folds"])
    lr = pd.DataFrame(res["logistic_regression"]["folds"])
    paired = {}
    for k in ("accuracy", "f1_positive", "balanced_accuracy", "mcc", "roc_auc", "pr_auc"):
        d = rf[k].astype(float) - lr[k].astype(float)
        paired[k] = {"mean_diff_rf_minus_lr": round(d.mean(), 4), "std": round(d.std(ddof=1), 4),
                     "share_of_folds_rf_better": round(float((d > 0).mean()), 4),
                     "share_of_folds_tied": round(float((d == 0).mean()), 4)}
    out["paired_rf_minus_lr"] = paired
    return out


# --------------------------------------------------------------------------
# Uncertainty on a single test split
# --------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(centre - half, 4), round(centre + half, 4)]


def _threshold_metrics(y, pred):
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    n = tp + fp + tn + fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    den = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return {"accuracy": (tp + tn) / n, "precision_positive": prec, "recall_positive": rec,
            "f1_positive": f1, "balanced_accuracy": (rec + spec) / 2, "mcc": mcc}


def bootstrap_ci(pred_pdf: pd.DataFrame, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Percentile 95% CI by resampling test accounts with replacement.

    Answers: "if we had drawn a different test set of the same size from the
    same population, how much would this number move?"  PR-AUC here is
    sklearn average precision (close to, not identical with, Spark's areaUnderPR).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = pred_pdf["label"].astype(int).values
    p = pred_pdf["prediction"].astype(int).values
    s = pred_pdf["p_inauthentic"].astype(float).values
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        m = _threshold_metrics(yy, p[idx])
        m["roc_auc"] = roc_auc_score(yy, s[idx])
        m["pr_auc_average_precision"] = average_precision_score(yy, s[idx])
        rows.append(m)
    b = pd.DataFrame(rows)
    return {k: [round(float(b[k].quantile(0.025)), 4), round(float(b[k].quantile(0.975)), 4)]
            for k in b.columns} | {"n_resamples": len(b), "method": "percentile bootstrap over test accounts"}


def mcnemar_exact(y, pred_a, pred_b) -> dict:
    """Exact (binomial) McNemar test on the same test accounts."""
    from scipy.stats import binomtest
    y = np.asarray(y).astype(int); a = np.asarray(pred_a).astype(int); b = np.asarray(pred_b).astype(int)
    a_only = int(((a == y) & (b != y)).sum())
    b_only = int(((a != y) & (b == y)).sum())
    n = a_only + b_only
    pval = 1.0 if n == 0 else float(binomtest(a_only, n, 0.5).pvalue)
    return {"rf_right_lr_wrong": a_only, "lr_right_rf_wrong": b_only, "p_value_exact": round(pval, 4)}


# --------------------------------------------------------------------------
# Permutation importance on the test split (one Spark pass)
# --------------------------------------------------------------------------
def permutation_importance(spark, model, test_pdf, feature_cols, groups: dict | None = None,
                           n_repeats: int = 30, seed: int = SEED) -> list[dict]:
    """Drop in test ROC-AUC / average precision / balanced accuracy when one
    feature (or a group of features) is shuffled across test accounts.

    All shuffled copies are stacked into one DataFrame and scored in a single
    transform, which is much cheaper than one Spark job per (feature, repeat).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(seed)
    targets = {f: [f] for f in feature_cols}
    if groups:
        targets.update({f"[group] {g}": cols for g, cols in groups.items()})
    base = test_pdf[feature_cols + ["label"]].reset_index(drop=True)
    frames = [base.assign(_key="__baseline__", _rep=0)]
    for key, cols in targets.items():
        for r in range(n_repeats):
            perm = base.copy()
            order = rng.permutation(len(perm))
            for c in cols:   # same permutation for every column of a group keeps the group's joint shape
                perm[c] = perm[c].values[order]
            frames.append(perm.assign(_key=key, _rep=r))
    stacked = pd.concat(frames, ignore_index=True)
    pred = score(model, to_spark(spark, stacked), ["_key", "_rep"]) \
        .select("_key", "_rep", "label", "prediction", "p_inauthentic").toPandas()

    def _m(d):
        y = d["label"].astype(int).values
        return (roc_auc_score(y, d["p_inauthentic"]), average_precision_score(y, d["p_inauthentic"]),
                _threshold_metrics(y, d["prediction"].astype(int).values)["balanced_accuracy"])

    b_auc, b_ap, b_bacc = _m(pred[pred._key == "__baseline__"])
    out = []
    for key, d in pred[pred._key != "__baseline__"].groupby("_key"):
        vals = np.array([_m(g) for _, g in d.groupby("_rep")])
        drops = np.array([b_auc, b_ap, b_bacc]) - vals
        out.append({
            "feature": key,
            "is_group": key.startswith("[group]"),
            "members": targets[key],
            "roc_auc_drop_mean": round(float(drops[:, 0].mean()), 4),
            "roc_auc_drop_std": round(float(drops[:, 0].std(ddof=1)), 4),
            "pr_auc_ap_drop_mean": round(float(drops[:, 1].mean()), 4),
            "balanced_accuracy_drop_mean": round(float(drops[:, 2].mean()), 4),
        })
    out.sort(key=lambda r: (r["is_group"], -r["roc_auc_drop_mean"]))
    rank = 0
    for r in out:
        if not r["is_group"]:
            rank += 1
            r["rank"] = rank
    return out, {"baseline_roc_auc": round(b_auc, 4), "baseline_pr_auc_ap": round(b_ap, 4),
                 "baseline_balanced_accuracy": round(b_bacc, 4), "n_repeats": n_repeats,
                 "evaluated_on": "test split"}


def rank_agreement(impurity: list[dict], perm: list[dict]) -> dict:
    """Spearman rank correlation between impurity and permutation importance."""
    from scipy.stats import spearmanr
    imp = {r["feature"]: r["importance"] for r in impurity}
    pm = {r["feature"]: r["roc_auc_drop_mean"] for r in perm if not r["is_group"]}
    feats = [f for f in imp if f in pm]
    rho, p = spearmanr([imp[f] for f in feats], [pm[f] for f in feats])
    top_imp = [r["feature"] for r in impurity[:5]]
    top_perm = [r["feature"] for r in perm if not r["is_group"]][:5]
    return {"spearman_rho": round(float(rho), 4), "p_value": round(float(p), 4),
            "top5_impurity": top_imp, "top5_permutation": top_perm,
            "top5_overlap": len(set(top_imp) & set(top_perm))}


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE, "savefig.facecolor": C_SURFACE,
        "axes.edgecolor": C_GRID, "axes.labelcolor": C_TEXT2, "xtick.color": C_TEXT2,
        "ytick.color": C_TEXT2, "text.color": C_TEXT, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "legend.frameon": False, "legend.fontsize": 8,
    })
    return plt


def plot_confusion(results: dict, path, title: str):
    """results: {display_name: metrics dict from evaluate_binary}"""
    plt = _style()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f2f7fd", "#9ec5f4", "#2a78d6", "#104281"])
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 3.9))
    axes = np.atleast_1d(axes)
    for ax, (name, m) in zip(axes, results.items()):
        cm = m["confusion_matrix"]
        mat = np.array([[cm["true_negative"], cm["false_positive"]],
                        [cm["false_negative"], cm["true_positive"]]])
        row_share = mat / mat.sum(axis=1, keepdims=True)
        ax.imshow(row_share, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                txt_col = "white" if row_share[i, j] > 0.55 else C_TEXT
                ax.text(j, i, f"{mat[i, j]}\n({row_share[i, j]:.0%} of row)", ha="center",
                        va="center", color=txt_col, fontsize=9)
        ax.set_xticks([0, 1], ["Authentic (0)", "Inauthentic (1)"])
        ax.set_yticks([0, 1], ["Authentic (0)", "Inauthentic (1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True label")
        ax.set_title(f"{name}\nacc {m['accuracy']:.3f} · recall {m['recall_positive']:.3f} · "
                     f"F1 {m['f1_positive']:.3f}", fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(title, fontsize=11, fontweight="bold", color=C_TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(rf_imp, perm, lr_coef, path, title, label_map=None):
    plt = _style()
    label_map = label_map or {}
    lab = lambda f: label_map.get(f, f)
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.36 * len(rf_imp) + 1.8))
    # 1. RF impurity
    ax = axes[0]
    order = list(reversed(rf_imp))
    ax.barh([lab(r["feature"]) for r in order], [r["importance"] for r in order], color=C_AUTH, height=0.6)
    ax.set_title("RF impurity importance (train)")
    ax.set_xlabel("mean decrease in Gini impurity (sums to 1)")
    # 2. Permutation importance (same feature order as panel 1 for comparison)
    ax = axes[1]
    pm = {r["feature"]: r for r in perm if not r["is_group"]}
    means = [pm[r["feature"]]["roc_auc_drop_mean"] for r in order]
    stds = [pm[r["feature"]]["roc_auc_drop_std"] for r in order]
    ax.barh([lab(r["feature"]) for r in order], means, xerr=stds, color=C_AUTH, height=0.6,
            error_kw={"ecolor": C_TEXT2, "elinewidth": 1, "capsize": 2})
    ax.axvline(0, color=C_TEXT2, lw=0.8)
    ax.set_title("RF permutation importance (test)\nsame row order as left panel", fontsize=10)
    ax.set_xlabel("drop in test ROC-AUC when shuffled (mean ± SD)")
    ax.set_yticklabels([])
    # 3. LR standardised coefficients
    ax = axes[2]
    lr_order = sorted(lr_coef, key=lambda r: abs(r["coefficient"]))
    vals = [r["coefficient"] for r in lr_order]
    ax.barh([lab(r["feature"]) for r in lr_order], vals, height=0.6,
            color=[C_INAUTH if v > 0 else C_AUTH for v in vals])
    ax.axvline(0, color=C_TEXT2, lw=0.8)
    ax.set_title("LR coefficients (standardised)\norange → inauthentic · blue → authentic", fontsize=10)
    ax.set_xlabel("log-odds per 1 SD of the feature")
    for a in axes:
        a.xaxis.grid(True, color=C_GRID, lw=0.6)
        a.set_axisbelow(True)
    fig.suptitle(title, fontsize=11, fontweight="bold", color=C_TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cv(cv: dict, single: dict, path, title, extra_points: dict | None = None,
            metrics=("accuracy", "f1_positive", "balanced_accuracy", "mcc", "roc_auc", "pr_auc")):
    """Per-fold metric spread for both models, with the single-split test value marked."""
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    rng = np.random.default_rng(SEED)
    names = {"random_forest": "Random Forest", "logistic_regression": "Logistic Regression"}
    for ax, kind in zip(axes, MODEL_KINDS):
        folds = pd.DataFrame(cv[kind]["folds"])
        for i, m in enumerate(metrics):
            v = folds[m].astype(float).values
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=10, color=C_AUTH, alpha=0.45,
                       linewidths=0, label="CV fold (validation)" if i == 0 else None)
            mu, sd = v.mean(), v.std(ddof=1)
            ax.errorbar(i + 0.32, mu, yerr=sd, fmt="o", color=C_TEXT, ms=5, capsize=3, lw=1.2,
                        label="CV mean ± SD" if i == 0 else None)
            ax.scatter(i, single[kind][m], marker="D", s=46, color=C_INAUTH, zorder=5,
                       edgecolors=C_SURFACE, linewidths=1.5,
                       label="headline 80:20 test split" if i == 0 else None)
            if extra_points:
                for j, (lbl, pts) in enumerate(extra_points.items()):
                    ax.scatter(i - 0.32, pts[kind][m], marker="s", s=34, color=C_NEUTRAL, zorder=5,
                               edgecolors=C_SURFACE, linewidths=1.2, label=lbl if i == 0 else None)
        ax.set_xticks(range(len(metrics)), [m.replace("_positive", "").replace("_", "\n") for m in metrics])
        ax.set_title(names[kind])
        ax.yaxis.grid(True, color=C_GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("metric value")
    axes[0].legend(loc="lower left")
    fig.suptitle(title, fontsize=11, fontweight="bold", color=C_TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def r4(x):
    """Round floats to 4 dp recursively (contract convention)."""
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, dict):
        return {k: r4(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r4(v) for v in x]
    if isinstance(x, (np.floating,)):
        return round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# --------------------------------------------------------------------------
# "Split lottery" and learning curve - quantify how much a single split moves
# --------------------------------------------------------------------------
def split_lottery(spark, full_sdf, prepare, feature_cols, weighted, seeds, log=print):
    """Re-run the contract's stratified 80:20 split with other seeds.

    Nothing is chosen from this (no tuning); it only shows how far the headline
    numbers could have moved had the group picked a different seed.
    ``prepare`` turns a Spark split into the pandas frame the models use.
    """
    keys = ("accuracy", "recall_positive", "precision_positive", "f1_positive",
            "balanced_accuracy", "mcc", "roc_auc", "pr_auc")
    rows = []
    for s in seeds:
        tr_s, te_s = common.stratified_split(full_sdf, "label", seed=s)
        tr, te = prepare(tr_s), prepare(te_s)
        for kind in MODEL_KINDS:
            m = fit_and_evaluate(spark, tr, te, feature_cols, kind, weighted)["metrics"]
            rows.append({"seed": s, "model": kind, "n_test": len(te),
                         "n_test_positive": int(te["label"].sum()), **{k: m[k] for k in keys}})
        log(f"    split seed {s} done")
    df = pd.DataFrame(rows)
    summary = {}
    for kind in MODEL_KINDS:
        d = df[df.model == kind]
        summary[kind] = {k: {"mean": round(d[k].mean(), 4), "std": round(d[k].std(ddof=1), 4),
                             "min": round(d[k].min(), 4), "max": round(d[k].max(), 4)} for k in keys}
    return {"seeds": list(seeds), "summary": summary,
            "n_test_range": [int(df.n_test.min()), int(df.n_test.max())],
            "per_seed": [r4(r) for r in rows]}


def learning_curve(spark, train_pdf, feature_cols, weighted, fractions, n_splits=5, n_repeats=2,
                   kind="random_forest", log=print):
    """Validation score vs. number of training accounts (inside the training split)."""
    from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    y = train_pdf["label"].values
    rows = []
    for i, (tr_idx, va_idx) in enumerate(rskf.split(np.zeros(len(y)), y)):
        tr_full = train_pdf.iloc[tr_idx]
        va_s = to_spark(spark, train_pdf.iloc[va_idx]).cache()
        for f in fractions:
            tr = tr_full if f >= 1.0 else train_test_split(
                tr_full, train_size=f, stratify=tr_full["label"], random_state=SEED + i)[0]
            if weighted:
                tr = add_balanced_weights(tr)
            model = build_pipeline(kind, feature_cols, "weight" if weighted else None).fit(to_spark(spark, tr))
            m = common.evaluate_binary(score(model, va_s, []))
            rows.append({"fraction": f, "n_train": len(tr), "accuracy": m["accuracy"],
                         "f1_positive": m["f1_positive"], "roc_auc": m["roc_auc"],
                         "balanced_accuracy": m["balanced_accuracy"]})
        va_s.unpersist()
        log(f"    learning-curve fold {i + 1}/{n_splits * n_repeats} done")
    df = pd.DataFrame(rows)
    agg = df.groupby("fraction").agg(
        n_train=("n_train", "mean"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
        f1_mean=("f1_positive", "mean"), f1_std=("f1_positive", "std")).reset_index()
    return {"model": kind, "scheme": f"stratified {n_splits}-fold x {n_repeats} inside the training split; "
                                     "training part subsampled (stratified) to each fraction",
            "points": [r4(r) for r in agg.to_dict("records")]}


def plot_learning_curve(lc: dict, path, title):
    plt = _style()
    pts = pd.DataFrame(lc["points"])
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for m, col, lbl in (("accuracy", C_AUTH, "accuracy"), ("roc_auc", C_INAUTH, "ROC-AUC")):
        ax.plot(pts.n_train, pts[f"{m}_mean"], color=col, lw=2, marker="o", ms=5, label=lbl)
        ax.fill_between(pts.n_train, pts[f"{m}_mean"] - pts[f"{m}_std"], pts[f"{m}_mean"] + pts[f"{m}_std"],
                        color=col, alpha=0.15, lw=0)
    ax.set_xlabel("training accounts used")
    ax.set_ylabel("validation score (mean ± SD across folds)")
    ax.yaxis.grid(True, color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def headline_vs_cv(single_metrics: dict, cv_kind: dict,
                   keys=("accuracy", "f1_positive", "recall_positive", "precision_positive",
                         "balanced_accuracy", "mcc", "roc_auc", "pr_auc")) -> dict:
    """z-score of the single headline test value against the CV fold distribution."""
    out = {}
    for k in keys:
        s = cv_kind["summary"].get(k)
        if s and s["std"]:
            out[k] = {"headline": single_metrics[k], "cv_mean": s["mean"], "cv_std": s["std"],
                      "z": round((single_metrics[k] - s["mean"]) / s["std"], 2)}
    return out
