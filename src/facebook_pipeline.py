"""Facebook track - spam profile detection from profile + activity metadata.

Run from the repo root (Java 17 on PATH or JAVA_HOME set):

    python src/facebook_pipeline.py

Data: Kaggle ``khajahussainsk/facebook-spam-dataset`` in data/landing/facebook_spam/
(scripts/download_datasets.py) - one file, 600 rows, 497 legit / 103 spam
(~17% positive, genuinely imbalanced).

Method (docs/methodology.md):
  * ``profile id`` is an identifier -> dropped (rule 1). It also turns out to be
    a label leak because the file is ordered by class; that is audited and
    reported, not modelled.
  * stratified 80:20 split, seed 42; Random Forest (main) and Logistic
    Regression (baseline) in Spark MLlib with balanced class weights; fixed
    hyperparameters, no tuning.
  * accuracy is NOT the headline here: PR-AUC, recall and F1 lead.
  * The near-perfect score this dataset produces is audited before it is
    reported: duplicates / near-duplicates, single-threshold separability,
    plausibility of the values (measured vs. generated), impurity vs.
    permutation importance, and feature-family ablation.

Output target: 0 = Authentic (legit), 1 = Inauthentic (spam).
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

TRACK = "facebook"
RAW_FILE = common.landing_path("facebook_spam") / "Facebook Spam Dataset.csv"
T0 = time.time()

RENAME = {
    "profile id": "profile_id",
    "#friends": "n_friends",
    "#following": "n_following",
    "#community": "n_community",
    "age": "account_age_days",
    "#postshared": "n_posts_shared",
    "#urlshared": "n_urls_shared",
    "#photos/videos": "n_photos_videos",
    "fpurls": "frac_posts_with_urls",
    "fpphotos/videos": "frac_posts_with_photos_videos",
    "avgcomment/post": "avg_comments_per_post",
    "likes/post": "avg_likes_per_post",
    "tags/post": "avg_tags_per_post",
    "#tags/post": "avg_hashtags_per_post",
    "Label": "label",
}
ORIG = {v: k for k, v in RENAME.items()}
RAW_FEATURES = [v for v in RENAME.values() if v not in ("profile_id", "label")]
# heavy-tailed counts get log1p (no effect on RF; keeps LR from being driven by outliers,
# e.g. one account with 665 comments per post)
LOG_COLS = ["n_friends", "n_following", "n_community", "n_posts_shared", "n_urls_shared",
            "n_photos_videos", "avg_comments_per_post"]
FEATURES = [f"log1p_{c}" if c in LOG_COLS else c for c in RAW_FEATURES]
LABEL_MAP = {f: (f"log1p({ORIG[f[6:]]})" if f.startswith("log1p_") else ORIG[f]) for f in FEATURES}
LABEL_MAP["profile_id"] = "profile id"
COUNT_COLS = ["n_friends", "n_following", "n_community", "account_age_days", "n_posts_shared",
              "n_urls_shared", "n_photos_videos", "avg_tags_per_post", "avg_hashtags_per_post"]

# the five columns the group flagged for suspiciously high |r| with the label
FLAGGED = ["n_urls_shared", "frac_posts_with_urls", "n_community", "n_photos_videos", "n_posts_shared"]
GROUPS = {
    "network (friends, following)": ["log1p_n_friends", "log1p_n_following"],
    "url sharing (#urlshared, fpurls)": ["log1p_n_urls_shared", "frac_posts_with_urls"],
    "posting volume (#postshared, #photos/videos, fpphotos/videos)":
        ["log1p_n_posts_shared", "log1p_n_photos_videos", "frac_posts_with_photos_videos"],
    "engagement (comments, likes)": ["log1p_avg_comments_per_post", "avg_likes_per_post"],
    "tagging (tags, hashtags)": ["avg_tags_per_post", "avg_hashtags_per_post"],
    "the 5 flagged high-correlation columns": [f"log1p_{c}" if c in LOG_COLS else c for c in FLAGGED],
}

CV_SPLITS, CV_REPEATS = 5, 10
LOTTERY_SEEDS = list(range(1, 21))


def log(msg):
    print(f"[{TRACK} {time.time() - T0:6.1f}s] {msg}", flush=True)


# ==========================================================================
# Load (Spark)
# ==========================================================================
def load(spark):
    from pyspark.sql import functions as F
    sdf = spark.read.csv(str(RAW_FILE), header=True, inferSchema=True)
    assert sdf.rdd.getNumPartitions() == 1, "file_row numbering below assumes one partition"
    sdf = sdf.toDF(*[RENAME[c] for c in sdf.columns])
    sdf = sdf.withColumn("file_row", F.monotonically_increasing_id() + 1)
    # Missing is not zero (contract rule 5): fpurls is by definition #urlshared / #postshared,
    # which holds to 1e-3 on 594/600 rows, so the 2 missing values are reconstructed from the
    # two counts that ARE present for those rows.
    sdf = sdf.withColumn("frac_posts_with_urls_imputed", F.col("frac_posts_with_urls").isNull())
    sdf = sdf.withColumn("frac_posts_with_urls",
                         F.coalesce(F.col("frac_posts_with_urls"),
                                    F.col("n_urls_shared") / F.col("n_posts_shared")))
    for c in LOG_COLS:
        sdf = sdf.withColumn(f"log1p_{c}", F.log1p(F.col(c).cast("double")))
    sdf = sdf.withColumn("account_key", F.concat(F.lit("fb_"), F.col("profile_id").cast("string")))
    return sdf.withColumn("label", F.col("label").cast("int"))


def to_model_frame(sdf) -> pd.DataFrame:
    cols = ["account_key", "profile_id", "label"] + FEATURES
    pdf = sdf.select(*cols).toPandas()
    pdf["label"] = pdf["label"].astype(int)
    pdf["account_key"] = pdf["account_key"].astype(object)
    return pdf.sort_values("profile_id").reset_index(drop=True)


# ==========================================================================
# Audit helpers (pandas / sklearn - diagnostics only, the models are Spark)
# ==========================================================================
def _bacc(y, p):
    tpr = (p[y == 1] == 1).mean() if (y == 1).any() else 0.0
    tnr = (p[y == 0] == 0).mean() if (y == 0).any() else 0.0
    return (tpr + tnr) / 2


def one_rule(train, test, col):
    """Best single threshold on one column, chosen on TRAIN by balanced accuracy, scored on TEST."""
    x = train[col].values.astype(float); y = train.label.values
    v = np.unique(x)
    cands = (v[:-1] + v[1:]) / 2 if len(v) > 1 else v
    best = (-1, None, None)
    for d in (">=", "<="):
        P = (x[None, :] >= cands[:, None]) if d == ">=" else (x[None, :] <= cands[:, None])
        tpr = (P[:, y == 1]).mean(axis=1); tnr = (~P[:, y == 0]).mean(axis=1)
        b = (tpr + tnr) / 2
        i = int(np.argmax(b))
        if b[i] > best[0]:
            best = (float(b[i]), float(cands[i]), d)
    _, thr, d = best
    xt = test[col].values.astype(float); yt = test.label.values
    pt = ((xt >= thr) if d == ">=" else (xt <= thr)).astype(int)
    m = U._threshold_metrics(yt, pt)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(yt, xt)
    # state the rule in the column's original units (log1p columns are back-transformed)
    if col.startswith("log1p_"):
        name, thr_orig = ORIG[col[6:]], float(np.expm1(thr))
    else:
        name, thr_orig = LABEL_MAP.get(col, col), thr
    return {"feature": name, "rule": f"{name} {d} {thr_orig:.4g} -> spam",
            "train_balanced_accuracy": round(best[0], 4),
            "test_accuracy": round(m["accuracy"], 4), "test_balanced_accuracy": round(m["balanced_accuracy"], 4),
            "test_recall": round(m["recall_positive"], 4), "test_precision": round(m["precision_positive"], 4),
            "test_f1": round(m["f1_positive"], 4), "test_univariate_roc_auc": round(max(auc, 1 - auc), 4),
            "_thr": thr_orig, "_dir": d, "_raw_col": col[6:] if col.startswith("log1p_") else col}


def id_ordering_audit(raw, train, test):
    lab = raw.sort_values("file_row")["label"].values
    ids = raw.sort_values("file_row")["profile_id"].values
    blocks = {f"rows {i + 1}-{min(i + 50, len(lab))}": int(lab[i:i + 50].sum()) for i in range(0, len(lab), 50)}
    spam_ids = ids[lab == 1]
    missing = sorted(set(range(int(ids.min()), int(ids.max()) + 1)) - set(ids.tolist()))
    # descriptive boundary on the full file
    accs = [(((ids >= t).astype(int) == lab).mean(), t) for t in np.unique(ids)]
    best_acc, best_t = max(accs)
    tail = lab[ids >= best_t]; head = lab[ids < best_t]
    id_rule = one_rule(train, test, "profile_id")
    return {
        "pearson_r_profile_id_vs_label": round(float(np.corrcoef(ids, lab)[0, 1]), 4),
        "profile_id_monotone_in_file_order": bool(np.all(np.diff(ids) > 0)),
        "profile_id_range": [int(ids.min()), int(ids.max())],
        "missing_profile_ids": missing,
        "label_changes_in_file_order": int((np.diff(lab) != 0).sum()),
        "spam_count_per_50_rows": blocks,
        "spam_in_first_8_rows": int(lab[:8].sum()),
        "spam_ids_below_boundary": [int(i) for i in spam_ids if i < best_t],
        "boundary_profile_id": int(best_t),
        "spam_share_at_or_above_boundary": [int(tail.sum()), int(len(tail))],
        "spam_share_below_boundary": [int(head.sum()), int(len(head))],
        "id_only_threshold_accuracy_full_file": round(float(best_acc), 4),
        "majority_class_accuracy_full_file": round(float(1 - lab.mean()), 4),
        "id_only_rule_learned_on_train_scored_on_test": {k: v for k, v in id_rule.items() if not k.startswith("_")},
    }


def duplicate_audit(raw, train, test):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    full_dup = int(raw.duplicated(RAW_FEATURES + ["label"]).sum())
    feat_dup = int(raw.duplicated(RAW_FEATURES).sum())
    sc = StandardScaler().fit(train[FEATURES].values)
    Xtr, Xte = sc.transform(train[FEATURES].values), sc.transform(test[FEATURES].values)
    nn = NearestNeighbors(n_neighbors=1).fit(Xtr)
    d, idx = nn.kneighbors(Xte)
    d = d[:, 0]; nb_lab = train.label.values[idx[:, 0]]
    yt = test.label.values
    # typical spacing for reference: NN distance within the training set itself
    d_tr = NearestNeighbors(n_neighbors=2).fit(Xtr).kneighbors(Xtr)[0][:, 1]
    m = U._threshold_metrics(yt, nb_lab)
    return {
        "exact_duplicate_rows_excluding_id": full_dup,
        "identical_feature_vectors_excluding_id": feat_dup,
        "space": "13 features (log1p on heavy-tailed counts), standardised with training mean/SD",
        "test_to_nearest_train_distance_quantiles": {q: round(float(np.quantile(d, q)), 4)
                                                     for q in (0.0, 0.05, 0.25, 0.5)},
        "train_to_nearest_other_train_distance_median": round(float(np.median(d_tr)), 4),
        "test_rows_within_0.25_of_a_train_row": int((d < 0.25).sum()),
        "test_rows_within_0.10_of_a_train_row": int((d < 0.10).sum()),
        "one_nearest_neighbour_from_train_on_test": {k: round(v, 4) for k, v in m.items()},
    }


def plausibility_audit(raw):
    from scipy import stats
    y = raw.label.values
    out = {}
    out["share_of_values_ending_in_0"] = {
        ORIG[c]: {"all": round(float((raw[c] % 10 == 0).mean()), 3),
                  "legit": round(float((raw.loc[y == 0, c] % 10 == 0).mean()), 3),
                  "spam": round(float((raw.loc[y == 1, c] % 10 == 0).mean()), 3)} for c in COUNT_COLS}
    out["expected_share_ending_in_0_if_measured"] = 0.10
    ld = {}
    for lab, nm in ((0, "legit"), (1, "spam")):
        digits = np.concatenate([raw.loc[y == lab, c].astype(int).values % 10
                                 for c in COUNT_COLS if raw[c].min() >= 10])
        obs = np.bincount(digits, minlength=10)
        ld[nm] = {"counts_by_last_digit_0_to_9": obs.tolist(),
                  "chi2_uniform_p_value": float(f"{stats.chisquare(obs).pvalue:.3g}")}
    out["last_digit_test_pooled_count_fields"] = ld
    r = raw.sort_values("file_row")
    round_share = (r[COUNT_COLS] % 10 == 0).mean(axis=1).values
    out["round_value_share_rows_1_to_8_vs_rest"] = [round(float(round_share[:8].mean()), 3),
                                                     round(float(round_share[8:].mean()), 3)]
    fp = raw["frac_posts_with_photos_videos"]; fu = raw["frac_posts_with_urls"]
    out["impossible_fractions"] = {
        "fpphotos/videos > 1 (more photos than posts)": {
            "n": int((fp > 1).sum()), "legit": int(((fp > 1) & (y == 0)).sum()),
            "spam": int(((fp > 1) & (y == 1)).sum()), "max": round(float(fp.max()), 4)},
        "fpurls > 1 (more URLs than posts)": {
            "n": int((fu > 1).sum()), "legit": int(((fu > 1) & (y == 0)).sum()),
            "spam": int(((fu > 1) & (y == 1)).sum()), "max": round(float(fu.max()), 4)},
    }
    ratio_u = raw.n_urls_shared / raw.n_posts_shared
    ratio_p = raw.n_photos_videos / raw.n_posts_shared
    orig_fu = raw.frac_posts_with_urls.where(~raw.frac_posts_with_urls_imputed)
    mis_u = raw[(orig_fu.notna()) & ((ratio_u - orig_fu).abs() > 1e-3)]
    mis_p = raw[(ratio_p - fp).abs() > 1e-3]
    out["fraction_formula_consistency"] = {
        "fpurls == #urlshared / #postshared (|diff|<1e-3)": [int(len(raw) - len(mis_u) - raw.frac_posts_with_urls_imputed.sum()),
                                                             int(len(raw) - raw.frac_posts_with_urls_imputed.sum())],
        "fpurls mismatching profile ids": mis_u.profile_id.astype(int).tolist(),
        "fpphotos/videos == #photos/videos / #postshared (|diff|<1e-3)": [int(len(raw) - len(mis_p)), len(raw)],
        "fpphotos/videos mismatching profile ids": mis_p.profile_id.astype(int).tolist(),
    }
    out["averages_stored_as_whole_numbers"] = {
        ORIG[c]: round(float((raw[c] == raw[c].round()).mean()), 3)
        for c in ("avg_tags_per_post", "avg_hashtags_per_post", "avg_comments_per_post", "avg_likes_per_post")}
    ac = raw.avg_comments_per_post
    out["avgcomment/post_outlier"] = {"max": float(ac.max()),
                                      "profile_id": int(raw.loc[ac.idxmax(), "profile_id"]),
                                      "label": int(raw.loc[ac.idxmax(), "label"]),
                                      "second_largest": float(ac.nlargest(2).iloc[-1])}
    out["skewness_by_class"] = {
        ORIG[c]: {"legit": round(float(stats.skew(raw.loc[y == 0, c])), 2),
                  "spam": round(float(stats.skew(raw.loc[y == 1, c])), 2)}
        for c in ("n_friends", "n_community", "n_posts_shared", "n_urls_shared", "n_photos_videos")}
    out["pearson_r_#friends_#following"] = round(float(raw.n_friends.corr(raw.n_following)), 4)
    out["pearson_r_#postshared_#photos/videos"] = round(float(raw.n_posts_shared.corr(raw.n_photos_videos)), 4)
    out["dataset_card_vs_file"] = {"card_claims": {"legit": 500, "spam": 100},
                                   "file_has": {"legit": int((y == 0).sum()), "spam": int(y.sum())}}
    out["pearson_r_with_label"] = {ORIG[c]: round(float(raw[c].corr(raw.label)), 4)
                                   for c in RAW_FEATURES + ["profile_id"]}
    return out


def ablation(spark, train, test):
    """Retrain the RF (same params, same split) on restricted feature sets."""
    fam = lambda cols: [f"log1p_{c}" if c in LOG_COLS else c for c in cols]
    sets = {
        "all 13 features (headline)": FEATURES,
        "without the 5 flagged columns": [f for f in FEATURES if f not in fam(FLAGGED)],
        "without flagged + network (friends, following)":
            [f for f in FEATURES if f not in fam(FLAGGED + ["n_friends", "n_following"])],
        "only #friends": fam(["n_friends"]),
        "only profile id (leak demonstration)": ["profile_id"],
    }
    out = {}
    for name, cols in sets.items():
        m = U.fit_and_evaluate(spark, train, test, cols, "random_forest", True)["metrics"]
        out[name] = {"features": [LABEL_MAP.get(c, c) for c in cols],
                     **{k: m[k] for k in ("pr_auc", "roc_auc", "recall_positive", "precision_positive",
                                          "f1_positive", "balanced_accuracy", "mcc", "accuracy")}}
        log(f"  ablation '{name}': pr_auc={m['pr_auc']} f1={m['f1_positive']} bacc={m['balanced_accuracy']}")
    return out


# ==========================================================================
# Figures specific to this track
# ==========================================================================
def plot_pr(res, test, path):
    from sklearn.metrics import precision_recall_curve
    plt = U._style()
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for kind, col, name in (("random_forest", U.C_AUTH, "Random Forest"),
                            ("logistic_regression", U.C_INAUTH, "Logistic Regression")):
        p = res[kind]["pred"]
        pr, rc, _ = precision_recall_curve(p.label.astype(int), p.p_inauthentic)
        ax.step(rc, pr, where="post", color=col, lw=2,
                label=f"{name}  (PR-AUC {res[kind]['metrics']['pr_auc']:.3f})")
        m = res[kind]["metrics"]
        ax.scatter([m["recall_positive"]], [m["precision_positive"]], color=col, s=50, zorder=5,
                   edgecolors=U.C_SURFACE, linewidths=1.5)
    prev = test.label.mean()
    ax.axhline(prev, color=U.C_NEUTRAL, lw=1.2, ls="--", label=f"no-skill (prevalence {prev:.3f})")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.04)
    ax.set_xlabel("recall (share of spam accounts caught)")
    ax.set_ylabel("precision (share of flags that are spam)")
    ax.yaxis.grid(True, color=U.C_GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="lower left")
    ax.set_title(f"Facebook - precision/recall on test (n={len(test)}, {int(test.label.sum())} spam)\n"
                 "dot = operating point at threshold 0.5", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_id_ordering(raw, audit, path):
    plt = U._style()
    r = raw.sort_values("file_row")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    roll = pd.Series(r.label.values).rolling(25, center=True, min_periods=5).mean()
    ax.fill_between(r.profile_id, 0, roll, color=U.C_INAUTH, alpha=0.18, lw=0)
    ax.plot(r.profile_id, roll, color=U.C_INAUTH, lw=2, label="spam share, rolling 25 rows")
    sp = r[r.label == 1]
    ax.scatter(sp.profile_id, np.full(len(sp), 1.06), marker="|", s=60, color=U.C_INAUTH,
               label="each spam account")
    ax.axvline(audit["boundary_profile_id"], color=U.C_TEXT2, lw=1, ls="--")
    ax.text(audit["boundary_profile_id"] - 8, 0.55,
            f"profile id ≥ {audit['boundary_profile_id']}:\n"
            f"{audit['spam_share_at_or_above_boundary'][0]}/{audit['spam_share_at_or_above_boundary'][1]} spam",
            ha="right", fontsize=8, color=U.C_TEXT)
    ax.set_ylim(0, 1.12); ax.set_xlim(0, r.profile_id.max() + 5)
    ax.set_xlabel("profile id  (= row order in the file)")
    ax.set_ylabel("share of rows that are spam")
    ax.yaxis.grid(True, color=U.C_GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="center left")
    ax.set_title(f"Facebook - the file is ordered by class: profile id alone gives "
                 f"{audit['id_only_threshold_accuracy_full_file']:.1%} accuracy "
                 f"(majority baseline {audit['majority_class_accuracy_full_file']:.1%})")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_separability(raw, rules, path):
    plt = U._style()
    top = rules[:4]
    fig, axes = plt.subplots(len(top), 1, figsize=(9, 1.65 * len(top) + 0.8))
    rng = np.random.default_rng(U.SEED)
    for ax, r in zip(axes, top):
        col = r["_raw_col"]
        for lab, c, nm in ((0, U.C_AUTH, "authentic (legit)"), (1, U.C_INAUTH, "inauthentic (spam)")):
            v = raw.loc[raw.label == lab, col].values
            ax.scatter(v, lab + rng.uniform(-0.25, 0.25, len(v)), s=9, color=c, alpha=0.55, linewidths=0,
                       label=nm)
        ax.axvline(r["_thr"], color=U.C_TEXT, lw=1.2, ls="--")
        ax.set_yticks([0, 1], ["legit", "spam"])
        if raw[col].min() > 0 and raw[col].max() / max(raw[col].min(), 1e-9) > 50:
            ax.set_xscale("log")
        ax.set_title(f"{r['rule']}   →  test balanced acc {r['test_balanced_accuracy']:.3f}, "
                     f"test acc {r['test_accuracy']:.3f}", fontsize=9, loc="left", fontweight="normal")
    axes[0].legend(loc="upper right", ncol=2, bbox_to_anchor=(1, 1.55))
    fig.suptitle("Facebook - one threshold on one column already separates the classes "
                 "(threshold chosen on train, scored on test)", fontsize=10, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_digits(pl, path):
    plt = U._style()
    d = pl["last_digit_test_pooled_count_fields"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(10)
    for i, (nm, col, lbl) in enumerate((("legit", U.C_AUTH, "authentic (legit)"),
                                        ("spam", U.C_INAUTH, "inauthentic (spam)"))):
        c = np.array(d[nm]["counts_by_last_digit_0_to_9"], float)
        ax.bar(x + (i - 0.5) * 0.38, c / c.sum(), width=0.36, color=col, label=lbl)
    ax.axhline(0.1, color=U.C_TEXT2, lw=1, ls="--", label="expected if values were measured (10%)")
    ax.set_xticks(x)
    ax.set_xlabel("last digit of the value (count fields ≥ 10, pooled)")
    ax.set_ylabel("share of values")
    ax.yaxis.grid(True, color=U.C_GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    sh = {nm: d[nm]["counts_by_last_digit_0_to_9"][0] / sum(d[nm]["counts_by_last_digit_0_to_9"])
          for nm in ("legit", "spam")}
    ax.set_title(f"Facebook - last digits are not uniform: '0' makes up {sh['legit']:.0%} (legit) and "
                 f"{sh['spam']:.0%} (spam)\nof values vs 10% expected - a sign of rounded or hand-entered numbers",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def block_audit(spark, raw, train, test, scored, boundary):
    """Does the model learn spam, or which half of the file a row came from?

    The file is two blocks: mostly-legit rows (ids < boundary) then mostly-spam
    rows (ids >= boundary). 'Out-of-block' rows are those whose label disagrees
    with their block. Uses the honest out-of-sample scores (test holdout + CV
    out-of-fold), never in-sample predictions.
    """
    s = scored.copy()
    s["block"] = (s.profile_id >= boundary).astype(int)
    s["out_of_block"] = s.block != s.label
    s["first_8_rows"] = s.profile_id <= 8
    out = {"boundary_profile_id": int(boundary),
           "definition": f"block = 1 if profile id >= {boundary} else 0; out-of-block = label != block",
           "n_out_of_block": int(s.out_of_block.sum()),
           "spam_labelled_rows_in_legit_block": s[(s.label == 1) & (s.block == 0)].profile_id.astype(int).tolist(),
           "legit_labelled_rows_in_spam_block": s[(s.label == 0) & (s.block == 1)].profile_id.astype(int).tolist(),
           "label_agrees_with_block": [int((~s.out_of_block).sum()), len(s)]}
    for kind, col in (("random_forest", "rf_prediction"), ("logistic_regression", "lr_prediction")):
        err = s[col] != s.label
        out[f"{kind}_out_of_sample_errors"] = {
            "total": int(err.sum()),
            "on_out_of_block_rows": [int((err & s.out_of_block).sum()), int(s.out_of_block.sum())],
            "on_in_block_rows": [int((err & ~s.out_of_block).sum()), int((~s.out_of_block).sum())],
            "in_block_error_profile_ids": s[err & ~s.out_of_block].profile_id.astype(int).tolist(),
            "agreement_of_predictions_with_block": round(float((s[col] == s.block).mean()), 4),
            "agreement_of_predictions_with_label": round(float((s[col] == s.label).mean()), 4)}
    # what do the conflicting rows look like? (rows 1-8 kept apart: hand-typed examples)
    r = raw.set_index("profile_id")
    cols = ["n_friends", "n_following", "n_community", "n_urls_shared", "n_posts_shared"]
    grp = {
        "all legit rows": r[r.label == 0].index,
        "all spam rows": r[r.label == 1].index,
        "legit-labelled rows in the spam block":
            s[(s.label == 0) & (s.block == 1)].profile_id.values,
        "spam-labelled rows in the legit block (excl. rows 1-8)":
            s[(s.label == 1) & (s.block == 0) & ~s.first_8_rows].profile_id.values,
    }
    out["median_profile_by_group"] = {g: {ORIG[c]: float(r.loc[idx, c].median()) for c in cols} | {"n": len(idx)}
                                      for g, idx in grp.items()}
    # same RF, same split, but asked to predict the block instead of the label
    blk = {}
    for target, fn in (("label", lambda d: d.label), ("block", lambda d: (d.profile_id >= boundary).astype(int))):
        m = U.fit_and_evaluate(spark, train.assign(label=fn(train)), test.assign(label=fn(test)), FEATURES,
                               "random_forest", True, keep=("account_key", "profile_id"))["metrics"]
        blk[f"rf_predicting_{target}"] = {k: m[k] for k in ("accuracy", "balanced_accuracy", "f1_positive",
                                                            "pr_auc", "roc_auc", "mcc")} | \
                                         {"confusion_matrix": m["confusion_matrix"]}
    out["same_rf_same_split_predicting_label_vs_block"] = blk
    return out, s


def plot_block_conflicts(s, raw, path):
    plt = U._style()
    d = s.merge(raw[["profile_id", "n_friends", "n_urls_shared"]], on="profile_id")
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for lab, c, nm in ((0, U.C_AUTH, "labelled legit"), (1, U.C_INAUTH, "labelled spam")):
        dd = d[(d.label == lab) & ~d.out_of_block]
        ax.scatter(dd.n_friends, dd.n_urls_shared, s=12, color=c, alpha=0.45, linewidths=0,
                   label=f"{nm} (label agrees with file block)")
    for lab, c, nm in ((0, U.C_AUTH, "labelled legit, but sits in the spam block"),
                       (1, U.C_INAUTH, "labelled spam, but sits in the legit block")):
        dd = d[(d.label == lab) & d.out_of_block]
        ax.scatter(dd.n_friends, dd.n_urls_shared, s=70, facecolors=c, edgecolors=U.C_TEXT, linewidths=1.2,
                   marker="D", label=nm, zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("#friends (log scale)")
    ax.set_ylabel("#urlshared (log scale)")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_title("Facebook - rows whose label disagrees with their position in the file\n"
                 "carry the profile of their block, not of their label", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ==========================================================================
# Main
# ==========================================================================
def main():
    # cores=3 as in the reference run: the partition count feeds the seeded splits.
    spark = common.get_spark("facebook_track", driver_memory="2g", cores=3)
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
    assert len(raw) == 600
    n_imputed = int(raw.frac_posts_with_urls_imputed.sum())
    log(f"rows={len(raw)} spam={int(raw.label.sum())} fpurls imputed from formula: {n_imputed}")

    tr_s, te_s = common.stratified_split(full_sdf.select("account_key", "profile_id", "label", *FEATURES), "label")
    train, test = to_model_frame(tr_s), to_model_frame(te_s)
    log(f"split: train={len(train)} (spam {train.label.sum()})  test={len(test)} (spam {test.label.sum()})")

    # ---------------------------------------------------------------- models
    weighted = True
    res = {}
    for kind in U.MODEL_KINDS:
        res[kind] = U.fit_and_evaluate(spark, train, test, FEATURES, kind, weighted,
                                       keep=("account_key", "profile_id"))
        m = res[kind]["metrics"]
        k = m["confusion_matrix"]["true_positive"] + m["confusion_matrix"]["true_negative"]
        res[kind]["uncertainty"] = {
            "accuracy_wilson_95ci": U.wilson_ci(k, m["n_test"]),
            "recall_wilson_95ci": U.wilson_ci(m["confusion_matrix"]["true_positive"],
                                              m["confusion_matrix"]["true_positive"] + m["confusion_matrix"]["false_negative"]),
            "one_spam_account_moves_recall_by": round(1 / max(1, int(test.label.sum())), 4),
            "bootstrap_95ci": U.bootstrap_ci(res[kind]["pred"]),
        }
        log(f"{kind}: pr_auc={m['pr_auc']} recall={m['recall_positive']} f1={m['f1_positive']} "
            f"precision={m['precision_positive']} acc={m['accuracy']} mcc={m['mcc']}")
    mcn = U.mcnemar_exact(res["random_forest"]["pred"]["label"], res["random_forest"]["pred"]["prediction"],
                          res["logistic_regression"]["pred"].set_index("account_key")
                          .loc[res["random_forest"]["pred"]["account_key"], "prediction"].values)

    # ---------------------------------------------------------------- audits
    log("audit: ID ordering")
    id_audit = id_ordering_audit(raw, train, test)
    log(f"  r(id,label)={id_audit['pearson_r_profile_id_vs_label']} boundary id={id_audit['boundary_profile_id']} "
        f"id-only acc={id_audit['id_only_threshold_accuracy_full_file']}")
    log("audit: duplicates / near-duplicates")
    dup = duplicate_audit(raw, train, test)
    log(f"  exact dups={dup['exact_duplicate_rows_excluding_id']}  1-NN test="
        f"{dup['one_nearest_neighbour_from_train_on_test']['balanced_accuracy']}")
    log("audit: single-threshold separability")
    rules = []
    for c in FEATURES:
        r = one_rule(train, test, c); r["_col"] = c; rules.append(r)
    rules.sort(key=lambda r: -r["test_balanced_accuracy"])
    log(f"  best 1R: {rules[0]['rule']} test bacc={rules[0]['test_balanced_accuracy']}")
    from sklearn.tree import DecisionTreeClassifier, export_text
    dt = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=U.SEED)
    dt.fit(train[FEATURES], train.label)
    dt_m = U._threshold_metrics(test.label.values, dt.predict(test[FEATURES]))
    depth2 = {"rule_text": export_text(dt, feature_names=[LABEL_MAP[f] for f in FEATURES]),
              "test": {k: round(v, 4) for k, v in dt_m.items()},
              "note": "sklearn DecisionTree(max_depth=2, class_weight='balanced'), diagnostic only"}
    log(f"  depth-2 tree test bacc={dt_m['balanced_accuracy']:.4f}")
    log("audit: plausibility")
    pl = plausibility_audit(raw)
    log("audit: permutation importance (RF, test)")
    perm, perm_base = U.permutation_importance(spark, res["random_forest"]["model"], test, FEATURES,
                                               groups=GROUPS, n_repeats=30)
    agree = U.rank_agreement(res["random_forest"]["importance"], perm)
    log(f"  impurity vs permutation: {agree}")
    log("audit: ablation")
    abl = ablation(spark, train, test)

    # ---------------------------------------------------------------- uncertainty
    log(f"repeated stratified CV {CV_SPLITS}x{CV_REPEATS} on the training split")
    cv = U.repeated_cv(spark, train, FEATURES, weighted, CV_SPLITS, CV_REPEATS,
                       keep=("account_key", "profile_id"), log=log)
    for kind in U.MODEL_KINDS:
        s = cv[kind]["summary"]
        log(f"CV {kind}: pr_auc {s['pr_auc']['mean']}±{s['pr_auc']['std']}  recall "
            f"{s['recall_positive']['mean']}±{s['recall_positive']['std']}  f1 {s['f1_positive']['mean']}±{s['f1_positive']['std']}")
    log("split lottery")
    lottery = U.split_lottery(spark, full_sdf.select("account_key", "profile_id", "label", *FEATURES),
                              to_model_frame, FEATURES, weighted, LOTTERY_SEEDS, log=log)

    write_outputs(spark, raw, train, test, res, mcn, id_audit, dup, rules, depth2, pl, perm, perm_base,
                  agree, abl, cv, lottery, n_imputed)


def write_outputs(spark, raw, train, test, res, mcn, id_audit, dup, rules, depth2, pl, perm, perm_base,
                  agree, abl, cv, lottery, n_imputed):
    # ---- per-account scores
    rf_t = res["random_forest"]["pred"].set_index("account_key")
    lr_t = res["logistic_regression"]["pred"].set_index("account_key")
    rf_o = cv["random_forest"]["_oof"].set_index("account_key")
    lr_o = cv["logistic_regression"]["_oof"].set_index("account_key")
    rows = []
    for _, r in pd.concat([train.assign(split="train"), test.assign(split="test")]).iterrows():
        k = r.account_key
        rf, lr, src = ((rf_t.loc[k], lr_t.loc[k], "test_holdout") if r.split == "test"
                       else (rf_o.loc[k], lr_o.loc[k], "cv_out_of_fold"))
        rows.append({"account_key": k, "profile_id": int(r.profile_id), "split": r.split, "score_source": src,
                     "label": int(r.label),
                     "rf_p_inauthentic": round(float(rf.p_inauthentic), 4), "rf_prediction": int(rf.prediction),
                     "lr_p_inauthentic": round(float(lr.p_inauthentic), 4), "lr_prediction": int(lr.prediction)})
    scored = pd.DataFrame(rows).sort_values("rf_p_inauthentic", ascending=False)
    scored["review_priority"] = np.arange(1, len(scored) + 1)
    scored = scored.sort_values("profile_id")
    log(f"wrote {common.save_predictions(spark.createDataFrame(scored), TRACK)}")
    log("audit: label vs file block")
    blk, blk_rows = block_audit(spark, raw, train, test, scored, id_audit["boundary_profile_id"])
    log(f"  RF errors on out-of-block rows {blk['random_forest_out_of_sample_errors']['on_out_of_block_rows']}, "
        f"in-block {blk['random_forest_out_of_sample_errors']['on_in_block_rows']}; "
        f"RF->block acc {blk['same_rf_same_split_predicting_label_vs_block']['rf_predicting_block']['accuracy']}")

    # ---- figures
    F = common.OUT_FIGURES
    U.plot_confusion({"Random Forest (class-weighted)": res["random_forest"]["metrics"],
                      "Logistic Regression (class-weighted)": res["logistic_regression"]["metrics"]},
                     F / f"{TRACK}_confusion_matrix.png",
                     f"Facebook - test split (n={len(test)}, {int(test.label.sum())} spam), stratified 80:20 seed 42")
    U.plot_importance(res["random_forest"]["importance"], perm, res["logistic_regression"]["importance"],
                      F / f"{TRACK}_feature_importance.png",
                      "Facebook - what separates spam from legit profiles", LABEL_MAP)
    U.plot_cv(cv, {k: res[k]["metrics"] for k in U.MODEL_KINDS}, F / f"{TRACK}_cv_variability.png",
              f"Facebook - spread across {CV_SPLITS}x{CV_REPEATS} CV folds vs. the single headline split",
              metrics=("pr_auc", "recall_positive", "precision_positive", "f1_positive", "balanced_accuracy", "mcc"))
    plot_pr(res, test, F / f"{TRACK}_pr_curve.png")
    plot_id_ordering(raw, id_audit, F / f"{TRACK}_id_ordering_leak.png")
    plot_separability(raw, rules, F / f"{TRACK}_single_feature_separability.png")
    plot_digits(pl, F / f"{TRACK}_digit_heaping.png")
    plot_block_conflicts(blk_rows, raw, F / f"{TRACK}_block_conflicts.png")
    log("wrote figures")

    rf_m = res["random_forest"]["metrics"]; lr_m = res["logistic_regression"]["metrics"]
    best = rules[0]
    groups = {r["feature"]: r for r in perm if r["is_group"]}
    abl_wo = abl["without the 5 flagged columns"]
    abl_wo2 = abl["without flagged + network (friends, following)"]
    prev_test = round(float(test.label.mean()), 4)

    models = {}
    for kind in U.MODEL_KINDS:
        cvk = {k: v for k, v in cv[kind].items() if k != "_oof"}
        e = {"params": U.model_params(kind, weighted=True), "metrics": res[kind]["metrics"],
             "feature_importance": res[kind]["importance"],
             "importance_type": ("impurity (mean decrease in Gini), from training" if kind == "random_forest"
                                 else "coefficient on standardised feature (log-odds per 1 SD)"),
             "uncertainty_on_test_split": res[kind]["uncertainty"], "cross_validation": cvk}
        if kind == "random_forest":
            e["permutation_importance"] = {"on": perm_base, "features": perm, "agreement_with_impurity": agree}
        models[kind] = e

    cvs = cv["random_forest"]["summary"]
    lot = lottery["summary"]["random_forest"]
    blk_rf = blk["random_forest_out_of_sample_errors"]
    blk_cmp = blk["same_rf_same_split_predicting_label_vs_block"]
    single_max = max(r["roc_auc_drop_mean"] for r in perm if not r["is_group"])
    verdict = (
        f"VERDICT - do not believe the ~97% as a measure of Facebook spam detection. The number itself is "
        f"reproducible (RF accuracy {cvs['accuracy']['mean']} +/- {cvs['accuracy']['std']} over 5x10 CV; "
        f"{lot['accuracy']['min']}-{lot['accuracy']['max']} over 20 other split seeds; {rf_m['accuracy']} on the "
        f"seed-42 split) and it is NOT caused by duplicate rows (0 exact, no near-duplicates) or by the profile id "
        f"(dropped). But: (1) most of it is free - always answering 'legit' scores "
        f"{id_audit['majority_class_accuracy_full_file']} on the full file "
        f"({rf_m['baseline_majority_class']['accuracy']} on the seed-42 test split). (2) One threshold on one "
        f"column ('{best['rule']}', chosen on train) reproduces the RF's test result exactly (balanced accuracy "
        f"{best['test_balanced_accuracy']} vs {rf_m['balanced_accuracy']}), and so does an RF given only the six "
        f"weakest columns. (3) The file is two blocks, legit rows first and spam rows last; the same RF predicts "
        f"which block a row sits in (test accuracy {blk_cmp['rf_predicting_block']['accuracy']}, PR-AUC "
        f"{blk_cmp['rf_predicting_block']['pr_auc']}) better than it predicts the label "
        f"({blk_cmp['rf_predicting_label']['accuracy']}, {blk_cmp['rf_predicting_label']['pr_auc']}). (4) "
        f"{blk_rf['on_out_of_block_rows'][0]} of the RF's {blk_rf['total']} out-of-sample errors fall on the "
        f"{blk['n_out_of_block']} rows whose label disagrees with their block, and those rows carry the profile of "
        f"their block, not of their label; on the other {blk_rf['on_in_block_rows'][1]} rows it makes "
        f"{blk_rf['on_in_block_rows'][0]} errors. The model's accuracy is essentially the share of rows whose label "
        f"agrees with their block ({blk['label_agrees_with_block'][0]}/{blk['label_agrees_with_block'][1]} = "
        f"{blk['label_agrees_with_block'][0] / blk['label_agrees_with_block'][1]:.3f}). (5) The values carry "
        f"fingerprints of construction rather than measurement (round-number heaping, 'fractions' above 1, "
        f"'averages' that are always whole numbers, hand-typed-looking first rows, class counts that do not match "
        f"the dataset card). Impurity and permutation importance agree on the top features (rho "
        f"{agree['spearman_rho']}), so the importance ranking is not an artefact of the impurity measure - but no "
        f"single column matters (max single-column ROC-AUC drop {single_max}) because every column carries the "
        f"same block signal. What the score measures is how different two separately produced groups of rows "
        f"are. Treat it as an upper bound on an easy benchmark; do not compare it head-to-head with X or TikTok, "
        f"and do not cite it as evidence that profile metadata detects Facebook spam."
    )

    payload = {
        "platform": TRACK,
        "dataset": {
            "name": "Facebook Spam Dataset",
            "source": "kaggle: khajahussainsk/facebook-spam-dataset (license: unknown)",
            "n_rows": len(raw), "label_col": "Label",
            "positive_means": "spam profile per the dataset author (mapped to 1 = Inauthentic)",
            "collection_claimed_by_card": "Facebook API / Graph API, public profiles; 500 legit + 100 spam",
            "schema_correction_for_concept_doc": {
                "concept_doc_guessed": ["avg_comments", "avg_likes", "avg_tags", "avg_hashtags"],
                "actual_columns": ["avgcomment/post", "likes/post", "tags/post", "#tags/post"],
                "mapping": {"avg_comments": "avgcomment/post", "avg_likes": "likes/post",
                            "avg_tags": "tags/post (card: 'average number of tags in a post / rate of tagging')",
                            "avg_hashtags": "#tags/post (card: 'average number of hashtags present in a post')"},
                "full_verified_schema": list(RENAME.keys()),
            },
        },
        "class_balance": {"n_authentic": int((raw.label == 0).sum()), "n_inauthentic": int(raw.label.sum()),
                          "positive_rate": round(float(raw.label.mean()), 4)},
        "split": {"method": "stratified_80_20", "seed": common.RANDOM_SEED, "n_train": len(train),
                  "n_test": len(test), "n_train_inauthentic": int(train.label.sum()),
                  "n_test_inauthentic": int(test.label.sum()),
                  "note": "common.stratified_split (Spark sampleBy): approximately, not exactly, 80:20"},
        "features": {
            "used": FEATURES,
            "original_names": {f: LABEL_MAP[f] for f in FEATURES},
            "dropped": {
                "profile id": "identifier (contract rule 1); also a label leak because the file is sorted by "
                              "class - see leakage_audit",
                **{ORIG[c]: f"replaced by log1p_{c}" for c in LOG_COLS},
            },
            "engineered": {f"log1p_{c}": f"log(1 + {ORIG[c]})" for c in LOG_COLS},
            "missing_values": {
                "fpurls": f"{n_imputed} missing (profile ids "
                          f"{raw.loc[raw.frac_posts_with_urls_imputed, 'profile_id'].astype(int).tolist()}); "
                          "reconstructed as #urlshared / #postshared, the column's own definition, which "
                          f"matches the stored value on {pl['fraction_formula_consistency']['fpurls == #urlshared / #postshared (|diff|<1e-3)'][0]}"
                          f"/{pl['fraction_formula_consistency']['fpurls == #urlshared / #postshared (|diff|<1e-3)'][1]} "
                          "non-missing rows. Not zero, not a median.",
                "all other columns": "no missing values",
            },
            "class_weighting": "weight = n / (2 * n_class) on training rows (legit ~0.6, spam ~3.0)",
        },
        "models": models,
        "lead_metrics": {
            "why": "17% positive class: accuracy of always answering 'legit' is already "
                   f"{rf_m['baseline_majority_class']['accuracy']}. PR-AUC (no-skill = prevalence "
                   f"{prev_test}), recall and F1 lead.",
            "random_forest": {k: rf_m[k] for k in ("pr_auc", "recall_positive", "f1_positive",
                                                   "precision_positive", "mcc")},
            "logistic_regression": {k: lr_m[k] for k in ("pr_auc", "recall_positive", "f1_positive",
                                                         "precision_positive", "mcc")},
            "no_skill_pr_auc": prev_test,
        },
        "secondary_evaluation": {
            "split_lottery": {"description": "same stratified 80:20 procedure with 20 other seeds; nothing "
                                             "is chosen from this", **lottery},
            "mcnemar_rf_vs_lr_on_test": mcn,
            "cv_paired_rf_minus_lr": cv["paired_rf_minus_lr"],
            "headline_vs_cv": {k: U.headline_vs_cv(res[k]["metrics"], cv[k]) for k in U.MODEL_KINDS},
        },
        "leakage_audit": {
            "checks_run": [
                "identifier vs label: correlation, file order, label runs, ID-only threshold rule (train->test)",
                "exact duplicate rows and identical feature vectors (excluding the id)",
                "near-duplicates: nearest training neighbour of every test row in standardised feature space; "
                "1-nearest-neighbour classifier from train to test",
                "single-threshold separability per column (threshold chosen on train, scored on test) and a "
                "depth-2 decision tree",
                "plausibility of values: last-digit heaping, impossible fractions, fraction formulas vs stored "
                "fractions, 'averages' that are whole numbers, outliers, skewness by class, card vs file counts",
                "RF impurity importance (train) vs permutation importance (test), single columns and "
                "correlated families",
                "ablation: RF retrained without the flagged columns, without flagged+network, with one column, "
                "with the id alone",
                "label vs file block: where the out-of-sample errors fall, what the conflicting rows look like, "
                "and the same RF trained to predict the block instead of the label",
                "uncertainty: repeated stratified 5x10 CV inside train, 20-seed split lottery, bootstrap CIs",
            ],
            "findings": [
                f"LEAK (removed): profile id correlates r = {id_audit['pearson_r_profile_id_vs_label']} with the "
                f"label because the file is ordered by class. Profile id is monotone in file order; "
                f"{id_audit['spam_share_at_or_above_boundary'][0]} of the {id_audit['spam_share_at_or_above_boundary'][1]} "
                f"rows with id >= {id_audit['boundary_profile_id']} are spam, against "
                f"{id_audit['spam_share_below_boundary'][0]} of {id_audit['spam_share_below_boundary'][1]} below it "
                f"(4 of those in the first 8 rows). An 'id >= {id_audit['boundary_profile_id']}' rule alone scores "
                f"{id_audit['id_only_threshold_accuracy_full_file']} accuracy vs a "
                f"{id_audit['majority_class_accuracy_full_file']} majority baseline; an RF on the id alone reaches "
                f"PR-AUC {abl['only profile id (leak demonstration)']['pr_auc']} on test. Dropped per rule 1.",
                f"No exact duplicates ({dup['exact_duplicate_rows_excluding_id']}) and no identical feature vectors "
                f"({dup['identical_feature_vectors_excluding_id']}). Median distance from a test row to its nearest "
                f"training row is {dup['test_to_nearest_train_distance_quantiles'][0.5]} vs "
                f"{dup['train_to_nearest_other_train_distance_median']} between training rows; "
                f"{dup['test_rows_within_0.10_of_a_train_row']} test rows lie within 0.10. Not a duplication leak.",
                f"Single-threshold separability: '{best['rule']}' (chosen on train) gives test balanced accuracy "
                f"{best['test_balanced_accuracy']}, recall {best['test_recall']}, precision {best['test_precision']}. "
                f"A depth-2 tree gives {depth2['test']['balanced_accuracy']}. The RF reaches "
                f"{rf_m['balanced_accuracy']}.",
                f"Importance: impurity vs permutation Spearman rho = {agree['spearman_rho']} "
                f"(top-5 overlap {agree['top5_overlap']}/5). Single-column permutation drops are tiny because "
                f"the columns are redundant; shuffling the five flagged columns together costs "
                f"{groups['[group] the 5 flagged high-correlation columns']['roc_auc_drop_mean']} ROC-AUC.",
                f"Ablation: without the five flagged columns the RF still scores PR-AUC {abl_wo['pr_auc']}, "
                f"F1 {abl_wo['f1_positive']}; without those AND friends/following, PR-AUC {abl_wo2['pr_auc']}, "
                f"F1 {abl_wo2['f1_positive']}. The classes differ on nearly every column.",
                f"Label vs file block: the same RF predicts the file block (ids >= {blk['boundary_profile_id']}) with "
                f"test accuracy {blk_cmp['rf_predicting_block']['accuracy']} / PR-AUC "
                f"{blk_cmp['rf_predicting_block']['pr_auc']}, better than the label "
                f"({blk_cmp['rf_predicting_label']['accuracy']} / {blk_cmp['rf_predicting_label']['pr_auc']}). "
                f"{blk_rf['on_out_of_block_rows'][0]}/{blk_rf['on_out_of_block_rows'][1]} out-of-block rows are "
                f"misclassified out-of-sample vs {blk_rf['on_in_block_rows'][0]}/{blk_rf['on_in_block_rows'][1]} "
                f"in-block rows.",
                f"Plausibility: {pl['impossible_fractions']['fpphotos/videos > 1 (more photos than posts)']['n']} rows "
                f"have fpphotos/videos > 1 (max {pl['impossible_fractions']['fpphotos/videos > 1 (more photos than posts)']['max']}), "
                f"impossible for a fraction of posts; last digits pile up on 0 and 5 (chi-square p = "
                f"{pl['last_digit_test_pooled_count_fields']['legit']['chi2_uniform_p_value']} legit, "
                f"{pl['last_digit_test_pooled_count_fields']['spam']['chi2_uniform_p_value']} spam); 'tags/post' and "
                f"'#tags/post' are described as averages but are whole numbers in "
                f"{pl['averages_stored_as_whole_numbers']['tags/post']:.0%} / "
                f"{pl['averages_stored_as_whole_numbers']['#tags/post']:.0%} of rows; the first 8 rows are "
                f"{pl['round_value_share_rows_1_to_8_vs_rest'][0]:.0%} round numbers vs "
                f"{pl['round_value_share_rows_1_to_8_vs_rest'][1]:.0%} for the rest; the card says 500/100, "
                f"the file has {pl['dataset_card_vs_file']['file_has']['legit']}/{pl['dataset_card_vs_file']['file_has']['spam']}.",
            ],
            "verdict": verdict,
            "label_vs_file_block": blk,
            "id_ordering": id_audit,
            "duplicates": dup,
            "single_feature_rules": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rules],
            "depth2_tree": depth2,
            "plausibility": pl,
            "ablation_random_forest": abl,
        },
        "limitations": [
            "Provenance is thin: the card claims Facebook (Graph) API collection from public profiles but gives no "
            "date, no sampling method and no labelling procedure, and the license is 'unknown'. The class counts "
            "in the file (497/103) do not match the card (500/100).",
            "The values look partly constructed rather than measured (see leakage_audit.plausibility). Whatever "
            "the model learns here may describe how the file was made more than how spammers behave.",
            "The file is two blocks (legit rows, then spam rows) and the features track the block more closely "
            "than the label (see leakage_audit.label_vs_file_block). The 23 rows whose label disagrees with their "
            "block are either mislabelled or deliberately mixed in; the model cannot tell, and neither can we.",
            "'Spam' here is the author's label for a profile. It is not the same concept as Instagram 'fake' or "
            "X 'bot'; 1 = Inauthentic is an operational convention, not a claim about intent.",
            f"Small and imbalanced: {int(test.label.sum())} spam accounts in the test split, so one missed spam "
            f"account moves recall by {round(100 / max(1, int(test.label.sum())), 1)} percentage points.",
            "Real-world spam prevalence is unknown; precision measured at 17% prevalence would fall at lower "
            "prevalence.",
            "A near-perfect benchmark score must not be compared head-to-head with the harder platforms as if "
            "Facebook spam were easier to detect.",
        ],
        "intended_use": "Ranking accounts for HUMAN review by trust & safety moderators. A score is not "
                        "evidence of intent; a false positive is a real person wrongly flagged.",
    }
    log(f"wrote {common.save_metrics(U.r4(payload), TRACK)}")


if __name__ == "__main__":
    main()
