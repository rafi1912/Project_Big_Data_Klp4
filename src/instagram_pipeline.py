"""Instagram track - fake / spammer account detection from profile metadata.

Run from the repo root (Java 17 on PATH or JAVA_HOME set):

    python src/instagram_pipeline.py

Data: Kaggle ``free4ever1/instagram-fake-spammer-genuine-accounts`` in
data/landing/instagram_fake_spammer/ (scripts/download_datasets.py):
train.csv 576 rows + test.csv 120 rows = 696 accounts, 348 genuine / 348 fake.

Method (docs/methodology.md):
  * the two vendor files are combined and re-split with the contract's
    stratified 80:20 split (seed 42) -> HEADLINE result;
  * the vendor's own train/test partition is evaluated as a SECONDARY check;
  * Random Forest (main) and Logistic Regression (baseline) in Spark MLlib,
    fixed hyperparameters, no tuning;
  * because 696 rows make any single split high-variance, the script also runs
    repeated stratified 5-fold x 10 CV *inside the training split*, a bootstrap
    over test accounts, a "split lottery" over 20 other seeds, and a learning
    curve - so the group can see how much the point estimate can move.

Output target: 0 = Authentic (genuine), 1 = Inauthentic (fake/spammer).
Scores are a human-review queue for trust & safety moderators, never an
automated sanction.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import common
import profile_tracks_utils as U

TRACK = "instagram"
RAW = common.landing_path("instagram_fake_spammer")
T0 = time.time()

# original column -> code-safe name
RENAME = {
    "profile pic": "profile_pic",
    "nums/length username": "nums_length_username",
    "fullname words": "fullname_words",
    "nums/length fullname": "nums_length_fullname",
    "name==username": "name_equals_username",
    "description length": "description_length",
    "external URL": "external_url",
    "private": "private",
    "#posts": "n_posts",
    "#followers": "n_followers",
    "#follows": "n_follows",
    "fake": "label",
}
COUNT_COLS = ["n_posts", "n_followers", "n_follows"]
ENGINEERED = {f"log1p_{c}": f"log(1 + {c})  [{[k for k, v in RENAME.items() if v == c][0]}]"
              for c in COUNT_COLS}
FEATURES = ["profile_pic", "nums_length_username", "fullname_words", "nums_length_fullname",
            "name_equals_username", "description_length", "external_url", "private",
            "log1p_n_posts", "log1p_n_followers", "log1p_n_follows"]
LABEL_MAP = {v: k for k, v in RENAME.items()} | {f"log1p_{c}": f"log1p({[k for k, v in RENAME.items() if v == c][0]})"
                                                for c in COUNT_COLS}

CV_SPLITS, CV_REPEATS = 5, 10
LOTTERY_SEEDS = list(range(1, 21))
LC_FRACTIONS = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0]


def log(msg):
    print(f"[{TRACK} {time.time() - T0:6.1f}s] {msg}", flush=True)


def load(spark):
    """Read both vendor files in Spark, keep provenance, rename, engineer."""
    from pyspark.sql import functions as F

    parts = []
    for name in ("train", "test"):
        sdf = spark.read.csv(str(RAW / f"{name}.csv"), header=True, inferSchema=True)
        assert sdf.rdd.getNumPartitions() == 1, "row numbering below assumes one partition"
        sdf = sdf.toDF(*[RENAME[c] for c in sdf.columns])
        sdf = (sdf.withColumn("source_row", F.monotonically_increasing_id())
                  .withColumn("source_file", F.lit(f"{name}.csv")))
        parts.append(sdf)
    full = parts[0].unionByName(parts[1])
    # account_key is a provenance handle (file + row) for the output CSV only - never a feature.
    full = full.withColumn(
        "account_key",
        F.concat(F.regexp_replace("source_file", r"\.csv$", ""), F.lit("_r"),
                 F.lpad(F.col("source_row").cast("string"), 4, "0")))
    for c in COUNT_COLS:
        full = full.withColumn(f"log1p_{c}", F.log1p(F.col(c).cast("double")))
    full = full.withColumn("label", F.col("label").cast("int"))
    return full


def to_model_frame(sdf) -> pd.DataFrame:
    cols = ["account_key", "source_file", "source_row", "label"] + FEATURES
    pdf = sdf.select(*cols).toPandas()
    pdf["label"] = pdf["label"].astype(int)
    pdf["account_key"] = pdf["account_key"].astype(object)
    pdf["source_file"] = pdf["source_file"].astype(object)
    return pdf.sort_values("account_key").reset_index(drop=True)


def data_audit(raw: pd.DataFrame) -> dict:
    """Facts about the file itself, before any modelling."""
    feats_raw = [v for v in RENAME.values() if v != "label"]
    out = {}
    out["missing_values_total"] = int(raw[feats_raw].isna().sum().sum())
    per_file = {}
    for f, d in raw.groupby("source_file"):
        lab = d.sort_values("source_row")["label"].values
        per_file[f] = {"n": len(d), "n_fake": int(lab.sum()),
                       "label_changes_in_row_order": int((np.diff(lab) != 0).sum()),
                       "first_fake_row": int(np.argmax(lab == 1))}
    out["per_vendor_file"] = per_file
    dup_mask = raw.duplicated(feats_raw + ["label"], keep=False)
    groups = raw[dup_mask].groupby(feats_raw + ["label"])["account_key"].apply(list).tolist()
    out["exact_duplicate_groups"] = groups
    out["n_rows_in_duplicate_groups"] = int(dup_mask.sum())
    conflict = raw[raw.duplicated(feats_raw, keep=False)].groupby(feats_raw)["label"].nunique()
    out["identical_features_conflicting_labels"] = int((conflict > 1).sum())
    out["selection_artefacts"] = {
        "no_profile_pic": {"genuine": int(((raw.profile_pic == 0) & (raw.label == 0)).sum()),
                           "fake": int(((raw.profile_pic == 0) & (raw.label == 1)).sum())},
        "has_external_url": {"genuine": int(((raw.external_url == 1) & (raw.label == 0)).sum()),
                             "fake": int(((raw.external_url == 1) & (raw.label == 1)).sum())},
    }
    out["pearson_r_with_label"] = {LABEL_MAP.get(c, c): round(float(raw[c].corr(raw["label"])), 4)
                                   for c in feats_raw}
    out["median_by_class"] = {
        LABEL_MAP.get(c, c): {"genuine": float(raw.loc[raw.label == 0, c].median()),
                              "fake": float(raw.loc[raw.label == 1, c].median())} for c in feats_raw}
    return out


def main():
    # cores=3 as in the reference run: the partition count feeds the seeded splits.
    spark = common.get_spark("instagram_track", driver_memory="2g", cores=3)
    try:
        run(spark)
    finally:
        spark.stop()
    log("done")


def run(spark):
    log("loading")
    full_sdf = load(spark).cache()
    raw = full_sdf.toPandas()
    raw["account_key"] = raw["account_key"].astype(object)
    raw["source_file"] = raw["source_file"].astype(object)
    n = len(raw)
    assert n == 696, n
    audit = data_audit(raw)
    log(f"rows={n}  fake={int(raw.label.sum())}  missing={audit['missing_values_total']}  "
        f"duplicate rows={audit['n_rows_in_duplicate_groups']}")

    # ------------------------------------------------------------------ headline split
    tr_s, te_s = common.stratified_split(full_sdf.select("account_key", "source_file", "source_row",
                                                         "label", *FEATURES), "label")
    train, test = to_model_frame(tr_s), to_model_frame(te_s)
    log(f"headline split: train={len(train)} (fake {train.label.sum()})  test={len(test)} (fake {test.label.sum()})")

    # duplicates straddling the split = the same feature vector seen in training and test
    test_keys = set(test.account_key)
    straddle = [g for g in audit["exact_duplicate_groups"]
                if any(k in test_keys for k in g) and any(k not in test_keys for k in g)]

    weighted = False  # 348/348 overall; class weighting not needed
    res = {}
    for kind in U.MODEL_KINDS:
        res[kind] = U.fit_and_evaluate(spark, train, test, FEATURES, kind, weighted)
        m = res[kind]["metrics"]
        log(f"{kind}: acc={m['accuracy']} f1={m['f1_positive']} mcc={m['mcc']} roc_auc={m['roc_auc']}")

    # uncertainty on the single test split
    for kind in U.MODEL_KINDS:
        m = res[kind]["metrics"]
        k = m["confusion_matrix"]["true_positive"] + m["confusion_matrix"]["true_negative"]
        res[kind]["uncertainty"] = {
            "accuracy_wilson_95ci": U.wilson_ci(k, m["n_test"]),
            "one_account_moves_accuracy_by": round(1 / m["n_test"], 4),
            "bootstrap_95ci": U.bootstrap_ci(res[kind]["pred"]),
        }
    mcn = U.mcnemar_exact(res["random_forest"]["pred"]["label"],
                          res["random_forest"]["pred"]["prediction"],
                          res["logistic_regression"]["pred"].set_index("account_key")
                          .loc[res["random_forest"]["pred"]["account_key"], "prediction"].values)
    log(f"McNemar RF vs LR on test: {mcn}")

    # sensitivity: drop test rows whose exact duplicate sits in train
    dup_sensitivity = None
    if straddle:
        drop = {k for g in straddle for k in g if k in test_keys}
        dup_sensitivity = {"test_rows_removed": sorted(drop)}
        for kind in U.MODEL_KINDS:
            p = res[kind]["pred"][~res[kind]["pred"].account_key.isin(drop)]
            dup_sensitivity[kind] = {k: round(v, 4) for k, v in U._threshold_metrics(
                p.label.astype(int).values, p.prediction.astype(int).values).items()}

    # permutation importance of the RF on the test split
    log("permutation importance")
    perm, perm_base = U.permutation_importance(spark, res["random_forest"]["model"], test, FEATURES,
                                               n_repeats=30)
    agree = U.rank_agreement(res["random_forest"]["importance"], perm)
    log(f"impurity vs permutation: {agree}")

    # ------------------------------------------------------------------ repeated CV (train only)
    log(f"repeated stratified CV {CV_SPLITS}x{CV_REPEATS} on the training split")
    cv = U.repeated_cv(spark, train, FEATURES, weighted, CV_SPLITS, CV_REPEATS, log=log)
    for kind in U.MODEL_KINDS:
        s = cv[kind]["summary"]
        log(f"CV {kind}: acc {s['accuracy']['mean']}±{s['accuracy']['std']}  "
            f"f1 {s['f1_positive']['mean']}±{s['f1_positive']['std']}  roc {s['roc_auc']['mean']}±{s['roc_auc']['std']}")

    # ------------------------------------------------------------------ vendor partition (secondary)
    log("vendor partition (train.csv -> test.csv)")
    v_train = raw[raw.source_file == "train.csv"][["account_key", "label"] + FEATURES].copy()
    v_test = raw[raw.source_file == "test.csv"][["account_key", "label"] + FEATURES].copy()
    vendor = {}
    for kind in U.MODEL_KINDS:
        r = U.fit_and_evaluate(spark, v_train, v_test, FEATURES, kind, weighted)
        m = r["metrics"]
        k = m["confusion_matrix"]["true_positive"] + m["confusion_matrix"]["true_negative"]
        cvs = cv[kind]["summary"]
        vendor[kind] = {
            "metrics": m,
            "accuracy_wilson_95ci": U.wilson_ci(k, m["n_test"]),
            "difference_vs_headline": {x: round(m[x] - res[kind]["metrics"][x], 4)
                                       for x in ("accuracy", "f1_positive", "mcc", "roc_auc", "pr_auc")},
            "z_vs_cv_distribution": {x: (round((m[x] - cvs[x]["mean"]) / cvs[x]["std"], 2)
                                         if cvs[x]["std"] else None)
                                     for x in ("accuracy", "f1_positive", "mcc", "roc_auc")},
        }
        log(f"vendor {kind}: acc={m['accuracy']} f1={m['f1_positive']} roc={m['roc_auc']}")

    # ------------------------------------------------------------------ split lottery
    log(f"split lottery over seeds {LOTTERY_SEEDS[0]}..{LOTTERY_SEEDS[-1]}")
    split_sdf = full_sdf.select("account_key", "source_file", "source_row", "label", *FEATURES)
    lottery = U.split_lottery(spark, split_sdf, to_model_frame, FEATURES, weighted, LOTTERY_SEEDS, log=log)

    # ------------------------------------------------------------------ learning curve
    log("learning curve (RF)")
    lc = U.learning_curve(spark, train, FEATURES, weighted, LC_FRACTIONS, n_splits=5, n_repeats=2, log=log)

    # ------------------------------------------------------------------ outputs
    write_outputs(spark, raw, train, test, res, perm, perm_base, agree, cv, vendor, lottery, lc,
                  audit, straddle, dup_sensitivity, mcn)


def write_outputs(spark, raw, train, test, res, perm, perm_base, agree, cv, vendor, lottery, lc,
                  audit, straddle, dup_sensitivity, mcn):
    # ---- per-account scores: test = held-out scores; train = out-of-fold (CV repeat 1)
    rf_t = res["random_forest"]["pred"].set_index("account_key")
    lr_t = res["logistic_regression"]["pred"].set_index("account_key")
    rf_o = cv["random_forest"]["_oof"].set_index("account_key")
    lr_o = cv["logistic_regression"]["_oof"].set_index("account_key")
    rows = []
    for _, r in pd.concat([train.assign(split="train"), test.assign(split="test")]).iterrows():
        k = r.account_key
        rf, lr, src = ((rf_t.loc[k], lr_t.loc[k], "test_holdout") if r.split == "test"
                       else (rf_o.loc[k], lr_o.loc[k], "cv_out_of_fold"))
        rows.append({"account_key": k, "source_file": r.source_file, "source_row": int(r.source_row),
                     "split": r.split, "score_source": src, "label": int(r.label),
                     "rf_p_inauthentic": round(float(rf.p_inauthentic), 4),
                     "rf_prediction": int(rf.prediction),
                     "lr_p_inauthentic": round(float(lr.p_inauthentic), 4),
                     "lr_prediction": int(lr.prediction)})
    scored = pd.DataFrame(rows).sort_values("rf_p_inauthentic", ascending=False)
    scored["review_priority"] = np.arange(1, len(scored) + 1)
    scored = scored.sort_values("account_key")
    pred_path = common.save_predictions(spark.createDataFrame(scored), TRACK)
    log(f"wrote {pred_path}")

    # who is wrongly flagged? (every account scored once, out-of-sample)
    sc = scored.merge(raw[["account_key", "profile_pic", "description_length", "n_posts", "n_followers",
                           "n_follows"]], on="account_key")
    fp = sc[(sc.rf_prediction == 1) & (sc.label == 0)]
    fn = sc[(sc.rf_prediction == 0) & (sc.label == 1)]
    errors = {
        "scope": "all 696 accounts, each scored once out-of-sample (test holdout or CV out-of-fold)",
        "random_forest": {"errors": int(len(fp) + len(fn)), "false_positives": int(len(fp)),
                          "false_negatives": int(len(fn))},
        "false_positive_profile": {
            "with_profile_pic": int((fp.profile_pic == 1).sum()),
            "empty_bio": int((fp.description_length == 0).sum()),
            "posts_range": [int(fp.n_posts.min()), int(fp.n_posts.max())] if len(fp) else None,
            "followers_under_500": int((fp.n_followers < 500).sum()),
            "median_followers": float(fp.n_followers.median()) if len(fp) else None,
            "account_keys": fp.account_key.tolist()},
        "false_negative_profile": {
            "with_profile_pic": int((fn.profile_pic == 1).sum()),
            "median_follows": float(fn.n_follows.median()) if len(fn) else None,
            "median_followers": float(fn.n_followers.median()) if len(fn) else None},
    }

    # ---- figures
    fig = common.OUT_FIGURES
    U.plot_confusion({"Random Forest": res["random_forest"]["metrics"],
                      "Logistic Regression": res["logistic_regression"]["metrics"]},
                     fig / f"{TRACK}_confusion_matrix.png",
                     f"Instagram - test split (n={len(test)}), stratified 80:20 seed 42")
    U.plot_importance(res["random_forest"]["importance"], perm, res["logistic_regression"]["importance"],
                      fig / f"{TRACK}_feature_importance.png",
                      "Instagram - what separates fake from genuine accounts", LABEL_MAP)
    single = {k: res[k]["metrics"] for k in U.MODEL_KINDS}
    U.plot_cv(cv, single, fig / f"{TRACK}_cv_variability.png",
              f"Instagram - spread across {CV_SPLITS}x{CV_REPEATS} CV folds vs. the single headline split",
              extra_points={"vendor train/test partition": {k: vendor[k]["metrics"] for k in U.MODEL_KINDS}})
    U.plot_learning_curve(lc, fig / f"{TRACK}_learning_curve.png",
                          "Instagram RF - validation score vs. training size")
    log("wrote figures")

    # ---- metrics JSON
    rf_m = res["random_forest"]["metrics"]
    models = {}
    for kind in U.MODEL_KINDS:
        cvk = {k: v for k, v in cv[kind].items() if k != "_oof"}
        entry = {
            "params": U.model_params(kind, weighted=False),
            "metrics": res[kind]["metrics"],
            "feature_importance": res[kind]["importance"],
            "importance_type": ("impurity (mean decrease in Gini), from training"
                                if kind == "random_forest" else
                                "coefficient on standardised feature (log-odds per 1 SD)"),
            "uncertainty_on_test_split": res[kind]["uncertainty"],
            "cross_validation": cvk,
        }
        if kind == "random_forest":
            entry["permutation_importance"] = {"on": perm_base, "features": perm,
                                               "agreement_with_impurity": agree}
        models[kind] = entry

    lim = [
        "Labels are the dataset author's own judgement: one person hand-identified 'fake/spammer' accounts "
        "for a final-year project and states that a couple may be misidentified. 'Fake' and 'spammer' are "
        "merged by the author. Our 1 = Inauthentic is an operational convention, not proof of intent.",
        "Collected by one crawler over 15-19 March 2019. Instagram, its users and fake-account tactics have "
        "changed since; nothing here shows the model would transfer to 2026 accounts.",
        f"Small: {len(raw)} accounts, test split n={len(test)}. One test account moves accuracy by "
        f"{round(100 / len(test), 2)} percentage points; see cross_validation and split_lottery for the spread.",
        "Balanced 50/50 by construction. Real-world prevalence of fake accounts is unknown and certainly not "
        "50%; precision in deployment would be lower than measured here.",
        "Selection: accounts were chosen by the author, plausibly the clearer-cut cases. Borderline accounts "
        "(e.g. new genuine users with no picture and no posts) are likely under-represented - they are exactly "
        "the people a deployed model would wrongly flag.",
        "Profile metadata only - no behaviour over time, no content, no network. Strongly predictive signals "
        "(no profile picture, digits in the username) are cheap for an adversary to change.",
        "Both vendor files are sorted by class (all genuine rows first). Harmless for our shuffled splits, "
        "but any un-shuffled k-fold or head/tail split of these files would be badly biased.",
    ]

    payload = {
        "platform": TRACK,
        "dataset": {
            "name": "Instagram fake spammer genuine accounts",
            "source": "kaggle: free4ever1/instagram-fake-spammer-genuine-accounts (CC BY 3.0)",
            "files": {"train.csv": 576, "test.csv": 120},
            "n_rows": len(raw),
            "label_col": "fake",
            "positive_means": "fake or spammer account as judged by the dataset author "
                              "(mapped to 1 = Inauthentic)",
            "collection": "crawler, 15-19 March 2019; labels hand-assigned by the author",
        },
        "class_balance": {"n_authentic": int((raw.label == 0).sum()), "n_inauthentic": int(raw.label.sum()),
                          "positive_rate": round(float(raw.label.mean()), 4)},
        "split": {"method": "stratified_80_20", "seed": common.RANDOM_SEED, "n_train": len(train),
                  "n_test": len(test), "n_train_inauthentic": int(train.label.sum()),
                  "n_test_inauthentic": int(test.label.sum()),
                  "note": "both vendor files combined, then re-split with common.stratified_split "
                          "(Spark sampleBy, so the split is approximately, not exactly, 80:20)"},
        "features": {
            "used": FEATURES,
            "original_names": {f: LABEL_MAP[f] for f in FEATURES},
            "dropped": {
                "#posts": "replaced by log1p_n_posts",
                "#followers": "replaced by log1p_n_followers (raw values span 0 to 15.3 million)",
                "#follows": "replaced by log1p_n_follows",
            },
            "engineered": ENGINEERED,
            "identifiers": "none in this dataset; account_key (vendor file + row) is a provenance handle in "
                           "the output CSV only and is never a feature",
            "missing_values": "none present (0 of 696 x 11 cells); no imputation performed",
            "note_on_transform": "log1p is monotone, so it does not change the Random Forest at all; it only "
                                 "keeps LR from being dominated by a few accounts with millions of followers",
        },
        "models": models,
        "secondary_evaluation": {
            "vendor_partition": {
                "description": "train on vendor train.csv (576), evaluate on vendor test.csv (120); "
                               "same features and hyperparameters",
                **vendor,
            },
            "split_lottery": {
                "description": "same stratified 80:20 procedure with 20 other seeds; nothing is chosen from "
                               "this, it only shows how far a single split can move the headline",
                **lottery,
            },
            "learning_curve": lc,
            "out_of_sample_errors": errors,
            "mcnemar_rf_vs_lr_on_test": mcn,
            "cv_paired_rf_minus_lr": cv["paired_rf_minus_lr"],
            "headline_vs_cv": {k: U.headline_vs_cv(res[k]["metrics"], cv[k]) for k in U.MODEL_KINDS},
        },
        "leakage_audit": {
            "checks_run": [
                "identifier columns: none exist in this dataset",
                "row order vs label: both vendor files are sorted by class (genuine rows first); "
                "irrelevant to our random stratified split, recorded as a hazard for others",
                "exact duplicate rows across all 11 features + label",
                "identical feature vectors with conflicting labels",
                "exact duplicates straddling the headline train/test boundary (+ sensitivity rerun "
                "excluding them)",
                "impurity vs permutation importance agreement (RF, test split)",
                "single-split result vs repeated CV, other split seeds and the vendor partition",
                "headline accuracy vs the ~99% audit threshold in the contract",
            ],
            "findings": [
                f"{audit['n_rows_in_duplicate_groups']} rows form {len(audit['exact_duplicate_groups'])} exact "
                f"duplicate pairs: {audit['exact_duplicate_groups']}. Each pair sits inside one vendor file; "
                "all four pairs share one label. No ID exists to say whether they are the same account "
                "crawled twice or different accounts with identical metadata, so they were kept.",
                f"{len(straddle)} duplicate pair(s) straddle the headline split: {straddle}."
                + (f" Excluding the affected test rows changes RF accuracy to "
                   f"{dup_sensitivity['random_forest']['accuracy']} (from {rf_m['accuracy']})."
                   if dup_sensitivity else ""),
                f"{audit['identical_features_conflicting_labels']} identical feature vectors carry "
                "conflicting labels.",
                f"Impurity and permutation importance agree: Spearman rho = {agree['spearman_rho']}, "
                f"{agree['top5_overlap']}/5 of the top-5 features shared.",
                f"Headline RF accuracy {rf_m['accuracy']} is well below the 99% audit threshold; the "
                "separating features (profile picture, digits in username, bio length, follower counts) "
                "are plausible behavioural signals, not artefacts of the file.",
                f"Selection artefact (not leakage): only {audit['selection_artefacts']['no_profile_pic']['genuine']} "
                f"genuine accounts lack a profile picture vs {audit['selection_artefacts']['no_profile_pic']['fake']} "
                f"fakes, and {audit['selection_artefacts']['has_external_url']['fake']} fakes have an external URL vs "
                f"{audit['selection_artefacts']['has_external_url']['genuine']} genuine. These rules hold far less "
                "well on real Instagram.",
            ],
            "duplicate_sensitivity": dup_sensitivity,
            "data_facts": {k: v for k, v in audit.items() if k != "exact_duplicate_groups"},
        },
        "limitations": lim,
        "intended_use": "Ranking accounts for HUMAN review by trust & safety moderators. A score is not "
                        "evidence of intent; a false positive is a real person wrongly flagged.",
    }
    path = common.save_metrics(U.r4(payload), TRACK)
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
