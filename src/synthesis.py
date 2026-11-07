"""Synthesis track: cross-platform comparison, risk levels 0-4, MongoDB + Power BI serving.

Group "Penyayang Claude" - Teknologi Big Data, SV IPB University.

This track trains no new per-platform model. It reads what the four platform
tracks produced (outputs/metrics/*.json, outputs/predictions/*.csv), plus the
raw files only to measure the direction of comparable behavioural concepts,
and builds four things on top:

  1. A cross-platform comparison of what the models learned, by behavioural
     concept - NOT a ranking by accuracy (that ranking would put the label-sorted
     Facebook file first and the only large real dataset, X, last).
  2. The risk-level 0-4 layer (concept document section 13) on top of the
     out-of-sample model probabilities. Thresholds are derived from the
     precision/recall trade-off with 2-fold cross-fitting, so no account is
     levelled by a threshold that was derived on that account (contract rule 6).
     Levels 3 and 4 need cross-account coordination / amplification evidence
     that none of the datasets contains; they are reported as unreachable.
  3. Architecture-compliant serving: Spark writes the unified per-account table
     to the curated zone (Parquet) and to MongoDB (mongo-spark-connector, with a
     pymongo fallback); metrics, importances, level tables and the concept
     comparison are loaded too, then queried back and checked.
  4. Flat Power BI tables in outputs/powerbi/.

Run:
    source .venv/bin/activate
    export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
    docker compose -f deploy/mongodb/docker-compose.yml --env-file .env.mongo up -d
    python src/synthesis.py              # everything
    python src/synthesis.py --skip-mongo # analysis + CSVs only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ROOT = common.PROJECT_ROOT
SEED = common.RANDOM_SEED
OUT_POWERBI = ROOT / "outputs" / "powerbi"
OUT_POWERBI.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "logs" / "synthesis.log"
CURATED_ACCOUNTS = common.DATA_CURATED / "cross_platform_account_scores.parquet"

RAW_X_REAL = common.DATA_RAW / "twitter_real_candidates" / "treiman" / "twitter_human_bots_dataset.csv"
RAW_IG = [common.DATA_RAW / "instagram" / "train.csv", common.DATA_RAW / "instagram" / "test.csv"]
RAW_FB = common.DATA_RAW / "facebook" / "Facebook Spam Dataset.csv"
RAW_X_SYN = common.DATA_RAW / "twitter" / "bot_detection_data.csv"

MONGO_CONNECTOR = "org.mongodb.spark:mongo-spark-connector_2.12:10.4.0"
N_BOOT = 1000          # bootstrap resamples for univariate AUC intervals
N_BOOT_THRESH = 200    # bootstrap resamples for threshold stability
BETA_L1 = 2.0          # recall-weighted cut: wide "watch" net
BETA_L2 = 0.5          # precision-weighted cut: a person's time is spent on each flag
PREVALENCES = [0.10, 0.05, 0.01]   # illustrative only - real prevalence is unknown

_T0 = time.time()


def log(msg: str) -> None:
    line = f"[{time.time() - _T0:7.1f}s] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as fh:
        fh.write(line + "\n")


def r4(x):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        return None if (math.isnan(x) or math.isinf(x)) else round(float(x), 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def wilson_ci(k: int, n: int, z: float = 1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(centre - half, 4), round(centre + half, 4)]


# ==========================================================================
# 0. Inputs from the platform tracks
# ==========================================================================
TRACKS = ["twitter_real", "twitter_synthetic", "instagram", "facebook", "tiktok"]

DATASET_INFO = {
    "twitter_real": {
        "platform": "twitter", "role": "primary_result",
        "label_concept": "bot account (dataset author's label)",
        "unit": "account", "evidential_weight": 1,
        "verdict": "Trusted. Large (37,438), real, independently collected API snapshot; audited; "
                   "vendor split agrees within 0.01. The only result earned on a large real dataset.",
        "quote": "RF accuracy 0.8538 (95% CI 0.8456-0.8616) vs majority baseline 0.6664; "
                 "MCC 0.6749; ROC-AUC 0.9231; PR-AUC 0.8798 (seed-42 held-out test, n = 7,474)",
        "comparable_quantitatively": True,
    },
    "instagram": {
        "platform": "instagram", "role": "supporting_small_sample",
        "label_concept": "fake / spammer profile (one annotator)",
        "unit": "account", "evidential_weight": 2,
        "verdict": "Defensible but narrow. 696 hand-labelled accounts, balanced 50/50, one crawl "
                   "(15-19 Mar 2019). No leakage. Quote the CV mean, not the lucky seed-42 split. "
                   "RF and LR are statistically indistinguishable (McNemar p = 1.0).",
        "quote": "RF accuracy 0.9239 +/- 0.0203 (5x10 CV inside train; 20-seed range 0.870-0.961); "
                 "ROC-AUC 0.9792 +/- 0.009; majority baseline about 0.50",
        "comparable_quantitatively": True,
    },
    "facebook": {
        "platform": "facebook", "role": "cautionary_artefact",
        "label_concept": "spam profile",
        "unit": "account", "evidential_weight": 3,
        "verdict": "Not believable as Facebook spam detection. The file is two label-sorted blocks; "
                   "the RF predicts the block (0.990) better than the label (0.947); one threshold "
                   "'#friends <= 126' reproduces the RF confusion matrix; 19 of 24 errors are the "
                   "23 rows whose label disagrees with their block; values look constructed.",
        "quote": "RF accuracy 0.9474 (seed 42) / 0.9673 +/- 0.0181 (CV) - reported as a failure of the "
                 "data, not a success of the model",
        "comparable_quantitatively": False,
    },
    "twitter_synthetic": {
        "platform": "twitter", "role": "negative_control",
        "label_concept": "bot label of a generated file (no signal)",
        "unit": "tweet (one per user)", "evidential_weight": 4,
        "verdict": "Negative control. Faker-generated: tweet text is exactly Faker's 971-word lorem "
                   "list; max |r| feature-label 0.0069. Every model sits on the baseline.",
        "quote": "RF accuracy 0.4945 vs baseline 0.501 (lift -0.0065), MCC -0.011, ROC-AUC 0.4925",
        "comparable_quantitatively": False,
    },
    "tiktok": {
        "platform": "tiktok", "role": "unlabelled_indicators",
        "label_concept": "none - no ground truth exists and none was created",
        "unit": "author (aggregated from 7,225 videos)", "evidential_weight": 5,
        "verdict": "No supervised result. 95.4% of authors have one video; popularity-filtered "
                   "sample; no profile/network data. Unsupervised indicator tiers only; model "
                   "transfer judged not defensible.",
        "quote": "No accuracy exists. Review tiers: 10 secondary_context, 828 weak_signal_only, "
                 "188 no_indicator_fired, 5,667 behaviour_not_assessable",
        "comparable_quantitatively": False,
    },
}


def load_inputs():
    metrics = {t: json.loads((common.OUT_METRICS / f"{t}.json").read_text()) for t in TRACKS}
    preds = {
        "twitter_real": pd.read_csv(common.OUT_PREDICTIONS / "twitter_real_scored.csv"),
        "twitter_synthetic": pd.read_csv(common.OUT_PREDICTIONS / "twitter_synthetic_scored.csv"),
        "instagram": pd.read_csv(common.OUT_PREDICTIONS / "instagram_scored.csv"),
        "facebook": pd.read_csv(common.OUT_PREDICTIONS / "facebook_scored.csv"),
        "tiktok": pd.read_csv(common.OUT_PREDICTIONS / "tiktok_author_profile.csv"),
    }
    # Normalise the three labelled schemas to (uid, source_ref, scope, label, rf_p, lr_p, rf_pred).
    norm = {}
    p = preds["twitter_real"]
    norm["twitter_real"] = pd.DataFrame({
        "account_uid": "x_" + p["id"].astype(str), "source_ref": p["id"].astype(str),
        "evaluation_scope": "test_holdout", "label": p["label"].astype(int),
        "rf_p": p["rf_probability_inauthentic"].astype(float),
        "lr_p": p["lr_probability_inauthentic"].astype(float),
        "rf_pred": p["rf_prediction"].astype(int)})
    p = preds["twitter_synthetic"]
    norm["twitter_synthetic"] = pd.DataFrame({
        "account_uid": "xs_" + p["user_id"].astype(str), "source_ref": p["user_id"].astype(str),
        "evaluation_scope": "test_holdout", "label": p["label"].astype(int),
        "rf_p": p["rf_probability_inauthentic"].astype(float),
        "lr_p": p["lr_probability_inauthentic"].astype(float),
        "rf_pred": p["rf_prediction"].astype(int)})
    for t, prefix in (("instagram", "ig_"), ("facebook", "fb_")):
        p = preds[t]
        ref = (p["source_file"] + ":" + p["source_row"].astype(str)) if t == "instagram" \
            else p["profile_id"].astype(str)
        norm[t] = pd.DataFrame({
            "account_uid": (prefix + p["account_key"].astype(str)) if t == "instagram"
            else p["account_key"].astype(str),
            "source_ref": ref, "evaluation_scope": p["score_source"], "label": p["label"].astype(int),
            "rf_p": p["rf_p_inauthentic"].astype(float), "lr_p": p["lr_p_inauthentic"].astype(float),
            "rf_pred": p["rf_prediction"].astype(int)})
    for t, d in norm.items():
        assert d["account_uid"].is_unique, f"{t}: duplicate account keys"
        assert d[["label", "rf_p"]].notna().all().all(), f"{t}: missing label or score"
    return metrics, preds, norm


# ==========================================================================
# 1. Cross-platform comparison by behavioural concept
# ==========================================================================
# Parent concepts: what a feature means behaviourally, independent of platform.
CONCEPTS = {
    "audience_size": "Ukuran audiens (followers / teman)",
    "outgoing_ties": "Ikatan keluar (akun yang diikuti)",
    "audience_balance": "Keseimbangan followers vs following",
    "lifetime_post_volume": "Volume unggahan seumur akun",
    "posting_rate": "Laju unggahan per hari",
    "account_age": "Umur akun",
    "profile_completeness": "Kelengkapan profil (bio, foto, lokasi, tautan)",
    "naming_pattern": "Pola nama/username (digit)",
    "engagement_given": "Interaksi yang DIBERIKAN (likes yang diberikan akun)",
    "engagement_received": "Interaksi yang DITERIMA (likes/komentar per post)",
    "link_sharing": "Berbagi tautan di unggahan",
    "group_membership": "Keanggotaan grup/komunitas",
    "tagging": "Tag & hashtag per unggahan",
    "content_mix": "Komposisi konten (foto/video)",
    "verification": "Penanda kepercayaan (verified)",
    "privacy_setting": "Pengaturan privasi (akun private)",
    "geolocation_setting": "Pengaturan geolokasi (geo_enabled)",
    "language": "Bahasa/locale",
}

# Univariate indicators measured on the raw files. Each maps to a parent concept.
# 'equivalence' says how closely the platform column matches the concept.
INDICATORS = [
    {"key": "audience_size", "concept": "audience_size",
     "map": {"twitter_real": ("followers_count", "equivalent", ""),
             "instagram": ("#followers", "equivalent", ""),
             "facebook": ("#friends", "partial", "Facebook friendship is mutual: it is audience and "
                                                 "outgoing tie at once")}},
    {"key": "outgoing_ties", "concept": "outgoing_ties",
     "map": {"twitter_real": ("friends_count", "equivalent", "X 'friends' = accounts followed"),
             "instagram": ("#follows", "equivalent", ""),
             "facebook": ("#following", "equivalent", "")}},
    {"key": "audience_balance", "concept": "audience_balance",
     "map": {"twitter_real": ("(followers+1)/(followers+friends+2)", "equivalent", ""),
             "instagram": ("(#followers+1)/(#followers+#follows+2)", "equivalent", ""),
             "facebook": ("(#friends+1)/(#friends+#following+2)", "partial",
                          "friends are mutual, so the ratio is not follower/following")}},
    {"key": "lifetime_post_volume", "concept": "lifetime_post_volume",
     "map": {"twitter_real": ("statuses_count", "equivalent",
                              "dropped from the X model (= rate x age), measured here only"),
             "instagram": ("#posts", "equivalent", ""),
             "facebook": ("#postshared", "partial", "'posts shared' - may count re-shares")}},
    {"key": "posting_rate", "concept": "posting_rate",
     "map": {"twitter_real": ("average_tweets_per_day", "equivalent", ""),
             "facebook": ("#postshared / age", "equivalent", "derived here; age is in days")}},
    {"key": "account_age", "concept": "account_age",
     "map": {"twitter_real": ("account_age_days", "equivalent", ""),
             "facebook": ("age", "equivalent", "account age in days per the dataset card")}},
    {"key": "bio_present", "concept": "profile_completeness",
     "map": {"twitter_real": ("description non-empty", "equivalent", ""),
             "instagram": ("description length > 0", "equivalent", "")}},
    {"key": "bio_length", "concept": "profile_completeness",
     "map": {"twitter_real": ("len(description)", "equivalent", ""),
             "instagram": ("description length", "equivalent", "")}},
    {"key": "no_profile_picture", "concept": "profile_completeness",
     "map": {"twitter_real": ("default_profile_image", "equivalent", "default avatar = no own picture"),
             "instagram": ("1 - profile pic", "equivalent", "")}},
    {"key": "external_link_in_bio", "concept": "profile_completeness",
     "map": {"instagram": ("external URL", "equivalent", "")}},
    {"key": "username_digit_share", "concept": "naming_pattern",
     "map": {"instagram": ("nums/length username", "equivalent",
                           "supplied by the dataset; NOT derived for X/TikTok because that would "
                           "mean using the handle (an identifier)")}},
    {"key": "likes_given", "concept": "engagement_given",
     "map": {"twitter_real": ("favourites_count", "equivalent",
                              "likes the account GAVE - not comparable with likes received")}},
    {"key": "likes_received_per_post", "concept": "engagement_received",
     "map": {"facebook": ("likes/post", "equivalent", "")}},
    {"key": "comments_received_per_post", "concept": "engagement_received",
     "map": {"facebook": ("avgcomment/post", "equivalent", "")}},
    {"key": "urls_shared", "concept": "link_sharing",
     "map": {"facebook": ("#urlshared", "equivalent", "")}},
    {"key": "communities_joined", "concept": "group_membership",
     "map": {"facebook": ("#community", "equivalent", "")}},
    {"key": "hashtags_per_post", "concept": "tagging",
     "map": {"facebook": ("#tags/post", "equivalent", "whole numbers in 100% of rows (8.4 of FB findings)")}},
    {"key": "verified", "concept": "verification",
     "map": {"twitter_real": ("verified", "equivalent", "")}},
    {"key": "private_account", "concept": "privacy_setting",
     "map": {"instagram": ("private", "equivalent", "")}},
]

# What TikTok has for each indicator (from the TikTok track's mapping); no label, so no direction.
TIKTOK_STATUS = {
    "audience_size": "missing", "outgoing_ties": "missing", "audience_balance": "missing",
    "lifetime_post_volume": "proxy only (video_count = viral videos captured in 6 weeks, median 1)",
    "posting_rate": "proxy only (videos_per_day defined for 309 authors = 4.6%, between viral videos)",
    "account_age": "missing", "bio_present": "missing", "bio_length": "missing (caption is not a bio)",
    "no_profile_picture": "missing", "external_link_in_bio": "missing",
    "username_digit_share": "excluded by rule (handle is an identifier)",
    "likes_given": "missing (TikTok likes are RECEIVED - opposite concept)",
    "likes_received_per_post": "present (avg_likes) but 5 orders of magnitude off Facebook's scale",
    "comments_received_per_post": "present (avg_comments) but on a viral-video scale",
    "urls_shared": "missing", "communities_joined": "missing",
    "hashtags_per_post": "present (avg_hashtag_count) - the only TikTok equivalent of a real labelled feature",
    "verified": "missing", "private_account": "missing",
}

# Model feature -> parent concept (for importance shares) and the LR coefficient that matches each
# univariate indicator (sign = +1, or -1 where the model feature is coded the other way round).
MODEL_FEATURE_CONCEPT = {
    "twitter_real": {
        "log_followers_count": "audience_size", "log_friends_count": "outgoing_ties",
        "follower_share": "audience_balance", "log_favourites_count": "engagement_given",
        "log_average_tweets_per_day": "posting_rate", "account_age_days": "account_age",
        "verified": "verification", "geo_enabled": "geolocation_setting",
        "default_profile": "profile_completeness", "default_profile_image": "profile_completeness",
        "has_description": "profile_completeness", "description_length": "profile_completeness",
        "has_location": "profile_completeness", "lang (all levels)": "language", "lang=*": "language",
    },
    "instagram": {
        "log1p_n_followers": "audience_size", "log1p_n_follows": "outgoing_ties",
        "log1p_n_posts": "lifetime_post_volume", "profile_pic": "profile_completeness",
        "description_length": "profile_completeness", "external_url": "profile_completeness",
        "fullname_words": "profile_completeness", "nums_length_username": "naming_pattern",
        "nums_length_fullname": "naming_pattern", "name_equals_username": "naming_pattern",
        "private": "privacy_setting",
    },
    "facebook": {
        "log1p_n_friends": "audience_size", "log1p_n_following": "outgoing_ties",
        "log1p_n_community": "group_membership", "account_age_days": "account_age",
        "log1p_n_posts_shared": "lifetime_post_volume", "log1p_n_photos_videos": "lifetime_post_volume",
        "frac_posts_with_photos_videos": "content_mix", "log1p_n_urls_shared": "link_sharing",
        "frac_posts_with_urls": "link_sharing", "log1p_avg_comments_per_post": "engagement_received",
        "avg_likes_per_post": "engagement_received", "avg_tags_per_post": "tagging",
        "avg_hashtags_per_post": "tagging",
    },
}
LR_FOR_INDICATOR = {
    "twitter_real": {"audience_size": ("log_followers_count", 1), "outgoing_ties": ("log_friends_count", 1),
                     "audience_balance": ("follower_share", 1),
                     "posting_rate": ("log_average_tweets_per_day", 1),
                     "account_age": ("account_age_days", 1), "bio_present": ("has_description", 1),
                     "bio_length": ("description_length", 1),
                     "no_profile_picture": ("default_profile_image", 1),
                     "likes_given": ("log_favourites_count", 1), "verified": ("verified", 1)},
    "instagram": {"audience_size": ("log1p_n_followers", 1), "outgoing_ties": ("log1p_n_follows", 1),
                  "lifetime_post_volume": ("log1p_n_posts", 1), "bio_length": ("description_length", 1),
                  "no_profile_picture": ("profile_pic", -1), "external_link_in_bio": ("external_url", 1),
                  "username_digit_share": ("nums_length_username", 1), "private_account": ("private", 1)},
    "facebook": {"audience_size": ("log1p_n_friends", 1), "outgoing_ties": ("log1p_n_following", 1),
                 "lifetime_post_volume": ("log1p_n_posts_shared", 1), "account_age": ("account_age_days", 1),
                 "likes_received_per_post": ("avg_likes_per_post", 1),
                 "comments_received_per_post": ("log1p_avg_comments_per_post", 1),
                 "urls_shared": ("log1p_n_urls_shared", 1), "communities_joined": ("log1p_n_community", 1),
                 "hashtags_per_post": ("avg_hashtags_per_post", 1)},
}
LABELLED_REAL = ["twitter_real", "instagram", "facebook"]


def concept_frames(spark):
    """Spark: read the three labelled raw files and compute the indicator columns."""
    from pyspark.sql import functions as F

    out = {}
    # --- X real (Treiman). multiLine parse, as in the X track; cross-checked against pandas.
    x = (spark.read.option("header", True).option("multiLine", True).option("escape", '"')
         .csv(str(RAW_X_REAL)))
    num = lambda c: F.col(c).cast("double")  # noqa: E731
    boolc = lambda c: F.when(F.col(c) == "True", 1.0).when(F.col(c) == "False", 0.0)  # noqa: E731
    desc = F.col("description")
    x = x.select(
        F.concat(F.lit("x_"), F.col("id")).alias("account_uid"),   # join key only, never a feature
        F.when(F.col("account_type") == "bot", 1).when(F.col("account_type") == "human", 0).alias("label"),
        boolc("geo_enabled").alias("geo_enabled"),
        num("followers_count").alias("audience_size"),
        num("friends_count").alias("outgoing_ties"),
        ((num("followers_count") + 1) / (num("followers_count") + num("friends_count") + 2))
        .alias("audience_balance"),
        num("statuses_count").alias("lifetime_post_volume"),
        num("average_tweets_per_day").alias("posting_rate"),
        num("account_age_days").alias("account_age"),
        # An empty bio arrives as NULL; length 0 / absent is its true value (X track rule).
        F.when(desc.isNotNull() & (F.trim(desc) != ""), 1.0).otherwise(0.0).alias("bio_present"),
        F.when(desc.isNotNull(), F.length(desc)).otherwise(0).cast("double").alias("bio_length"),
        boolc("default_profile_image").alias("no_profile_picture"),
        num("favourites_count").alias("likes_given"),
        boolc("verified").alias("verified"),
    )
    xp = x.toPandas()
    chk = pd.read_csv(RAW_X_REAL, usecols=["account_type"])
    assert len(xp) == len(chk) == 37438, f"X parse mismatch: spark {len(xp)} vs pandas {len(chk)}"
    assert int(xp["label"].sum()) == int((chk["account_type"] == "bot").sum())
    out["twitter_real"] = xp

    # --- Instagram (both vendor files)
    # Same provenance key as the Instagram track: file + 0-based row of a single-partition read.
    ig = None
    for path in RAW_IG:
        part = spark.read.csv(str(path), header=True, inferSchema=True)
        assert part.rdd.getNumPartitions() == 1, "row numbering assumes one partition"
        part = part.withColumn("account_uid", F.concat(
            F.lit(f"ig_{path.stem}_r"), F.lpad(F.monotonically_increasing_id().cast("string"), 4, "0")))
        ig = part if ig is None else ig.unionByName(part)
    ig = ig.select(
        "account_uid",
        F.col("fake").cast("int").alias("label"),
        F.col("`#followers`").cast("double").alias("audience_size"),
        F.col("`#follows`").cast("double").alias("outgoing_ties"),
        ((F.col("`#followers`") + 1) / (F.col("`#followers`") + F.col("`#follows`") + 2))
        .alias("audience_balance"),
        F.col("`#posts`").cast("double").alias("lifetime_post_volume"),
        (F.col("`description length`") > 0).cast("double").alias("bio_present"),
        F.col("`description length`").cast("double").alias("bio_length"),
        (1 - F.col("`profile pic`")).cast("double").alias("no_profile_picture"),
        F.col("`external URL`").cast("double").alias("external_link_in_bio"),
        F.col("`nums/length username`").cast("double").alias("username_digit_share"),
        F.col("private").cast("double").alias("private_account"),
    )
    out["instagram"] = ig.toPandas()
    assert len(out["instagram"]) == 696

    # --- Facebook
    fb = spark.read.csv(str(RAW_FB), header=True, inferSchema=True)
    fb = fb.select(
        F.concat(F.lit("fb_"), F.col("`profile id`").cast("string")).alias("account_uid"),
        F.col("Label").cast("int").alias("label"),
        F.col("`#friends`").cast("double").alias("audience_size"),
        F.col("`#following`").cast("double").alias("outgoing_ties"),
        ((F.col("`#friends`") + 1) / (F.col("`#friends`") + F.col("`#following`") + 2))
        .alias("audience_balance"),
        F.col("`#postshared`").cast("double").alias("lifetime_post_volume"),
        # age > 0 for all 600 rows; guard anyway so a zero age becomes NA, not infinity
        F.when(F.col("age") > 0, F.col("`#postshared`") / F.col("age")).alias("posting_rate"),
        F.col("age").cast("double").alias("account_age"),
        F.col("`likes/post`").cast("double").alias("likes_received_per_post"),
        F.col("`avgcomment/post`").cast("double").alias("comments_received_per_post"),
        F.col("`#urlshared`").cast("double").alias("urls_shared"),
        F.col("`#community`").cast("double").alias("communities_joined"),
        F.col("`#tags/post`").cast("double").alias("hashtags_per_post"),
    )
    out["facebook"] = fb.toPandas()
    assert len(out["facebook"]) == 600

    # --- X synthetic (negative control): only for the high-score profile of Worksheet Q2.
    xs = spark.read.option("header", True).option("escape", '"').csv(str(RAW_X_SYN))
    xs = xs.select(
        F.concat(F.lit("xs_"), F.col("`User ID`")).alias("account_uid"),
        F.col("`Bot Label`").cast("int").alias("label"),
        F.col("`Follower Count`").cast("double").alias("follower_count"),
        F.col("`Retweet Count`").cast("double").alias("retweet_count"),
        F.col("`Mention Count`").cast("double").alias("mention_count"),
        boolc("Verified").alias("verified"),
        # NULL Hashtags = the tweet had none, so 0 is the true count (X track rule)
        F.when(F.col("Hashtags").isNull(), 0.0)
        .otherwise(F.size(F.split(F.trim(F.col("Hashtags")), r"\s+")).cast("double")).alias("hashtag_count"),
    )
    out["twitter_synthetic"] = xs.toPandas()
    assert len(out["twitter_synthetic"]) == 50000
    return out


def auc_mw(y: np.ndarray, x: np.ndarray) -> float:
    """ROC-AUC of a raw value as a score for the inauthentic class (ties get half credit)."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    r = rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_auc(y, x, rng, n_boot=N_BOOT):
    pos = np.nonzero(y == 1)[0]
    neg = np.nonzero(y == 0)[0]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals[b] = auc_mw(y[idx], x[idx])
    return np.percentile(vals, [2.5, 97.5])


def lr_coef(metrics, track, feature):
    for row in metrics[track]["models"]["logistic_regression"]["feature_importance"]:
        if row["feature"] == feature:
            return row["coefficient"]
    return None


def concept_comparison(spark, metrics):
    frames = concept_frames(spark)
    rng = np.random.default_rng(SEED)
    rows = []
    for ind in INDICATORS:
        for track, (col_desc, equiv, note) in ind["map"].items():
            df = frames[track][["label", ind["key"]]].dropna()
            y = df["label"].to_numpy().astype(int)
            x = df[ind["key"]].to_numpy().astype(float)
            auc = auc_mw(y, x)
            lo, hi = boot_auc(y, x, rng)
            if lo > 0.5:
                direction = "higher_means_inauthentic"
            elif hi < 0.5:
                direction = "lower_means_inauthentic"
            else:
                direction = "no_clear_difference"
            lr = LR_FOR_INDICATOR.get(track, {}).get(ind["key"])
            coef = None
            if lr:
                c = lr_coef(metrics, track, lr[0])
                coef = None if c is None else c * lr[1]
            binary = set(np.unique(x)) <= {0.0, 1.0}
            rows.append({
                "indicator": ind["key"], "concept": ind["concept"], "dataset": track,
                "source_column": col_desc, "equivalence": equiv, "note": note,
                "n": int(len(y)), "n_inauthentic": int(y.sum()), "n_missing_dropped":
                    int(len(frames[track]) - len(df)),
                "is_binary": bool(binary),
                "median_authentic": r4(np.median(x[y == 0])) if not binary else None,
                "median_inauthentic": r4(np.median(x[y == 1])) if not binary else None,
                "share_authentic": r4(x[y == 0].mean()) if binary else None,
                "share_inauthentic": r4(x[y == 1].mean()) if binary else None,
                "auc": r4(auc), "auc_ci95": [r4(lo), r4(hi)],
                "cliffs_delta": r4(2 * auc - 1),
                "direction": direction,
                "lr_coefficient_per_sd_same_coding": r4(coef),
                "lr_direction": (None if coef is None else
                                 ("higher_means_inauthentic" if coef > 0 else "lower_means_inauthentic")),
                "facebook_caveat": track == "facebook",
            })
        log(f"  indicator {ind['key']}: " + ", ".join(
            f"{r['dataset']}={r['auc']}" for r in rows if r["indicator"] == ind["key"]))
    return rows, frames


def summarise_agreement(rows):
    """For each indicator measured on >= 2 labelled real platforms: do the directions agree?"""
    out = []
    by = {}
    for r in rows:
        by.setdefault(r["indicator"], []).append(r)
    for ind, rs in by.items():
        if len(rs) < 2:
            continue
        d = {r["dataset"]: r for r in rs}

        def sign(r):
            return {"higher_means_inauthentic": 1, "lower_means_inauthentic": -1}.get(r["direction"], 0)

        x_ig = None
        if "twitter_real" in d and "instagram" in d:
            sx, si = sign(d["twitter_real"]), sign(d["instagram"])
            x_ig = {
                "agree": bool(sx != 0 and sx == si),
                "twitter_real_delta": d["twitter_real"]["cliffs_delta"],
                "instagram_delta": d["instagram"]["cliffs_delta"],
                "abs_delta_ratio_ig_over_x": (r4(abs(d["instagram"]["cliffs_delta"]) /
                                                 abs(d["twitter_real"]["cliffs_delta"]))
                                              if d["twitter_real"]["cliffs_delta"] else None),
            }
        signs = {k: sign(v) for k, v in d.items()}
        real_trusted = {k: s for k, s in signs.items() if k != "facebook"}
        all_three = [s for s in signs.values()]
        out.append({
            "indicator": ind, "concept": rs[0]["concept"], "datasets": sorted(d),
            "directions": {k: v["direction"] for k, v in d.items()},
            "cliffs_delta": {k: v["cliffs_delta"] for k, v in d.items()},
            "agree_all_measured": bool(len(set(all_three)) == 1 and all_three[0] != 0),
            "agree_excluding_facebook": (bool(len(set(real_trusted.values())) == 1 and
                                              list(real_trusted.values())[0] != 0)
                                         if len(real_trusted) >= 2 else None),
            "facebook_disagrees_with_trusted": (
                bool("facebook" in signs and len(real_trusted) >= 1 and
                     signs["facebook"] != 0 and any(s != 0 and s != signs["facebook"]
                                                    for s in real_trusted.values()))),
            "x_vs_instagram": x_ig,
        })
    return out


def importance_by_concept(metrics):
    """Share of each RF's importance that sits on each behavioural concept."""
    out = {}
    for track in LABELLED_REAL:
        fmap = MODEL_FEATURE_CONCEPT[track]

        def concept_of(f):
            if f in fmap:
                return fmap[f]
            if f.startswith("lang="):
                return fmap["lang=*"]
            raise KeyError(f"{track}: no concept for feature {f}")

        rf = metrics[track]["models"]["random_forest"]
        imp = {}
        for row in rf["feature_importance"]:
            c = concept_of(row["feature"])
            imp[c] = imp.get(c, 0.0) + row["importance"]
        tot = sum(imp.values())
        imp = {k: v / tot for k, v in imp.items()}

        perm_raw = rf["permutation_importance"]
        perm_rows = perm_raw["features"] if isinstance(perm_raw, dict) else perm_raw
        perm = {}
        for row in perm_rows:
            if row.get("is_group"):
                continue
            c = concept_of(row["feature"])
            perm[c] = perm.get(c, 0.0) + max(0.0, row["roc_auc_drop_mean"])
        ptot = sum(perm.values())
        perm_share = {k: v / ptot for k, v in perm.items()} if ptot else {}
        out[track] = {
            "impurity_share": {k: r4(v) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])},
            "permutation_share": {k: r4(v) for k, v in sorted(perm_share.items(), key=lambda kv: -kv[1])},
            "permutation_note": "share of the summed positive test ROC-AUC drops; single-feature "
                                "permutation understates features that share information",
        }
    # Concepts both X and Instagram measure: how much of each model sits on that shared ground?
    shared = sorted(set(out["twitter_real"]["impurity_share"]) & set(out["instagram"]["impurity_share"]))
    for t in ("twitter_real", "instagram"):
        out[t]["share_on_concepts_shared_with_" + ("instagram" if t == "twitter_real" else "twitter_real")] = {
            "concepts": shared,
            "impurity": r4(sum(out[t]["impurity_share"].get(c, 0) for c in shared)),
            "permutation": r4(sum(out[t]["permutation_share"].get(c, 0) for c in shared)),
        }
    return out


# ==========================================================================
# 1b. Worksheet 3, question 2: what does a high-scoring account look like?
# ==========================================================================
PROFILE_COLUMNS = {
    "twitter_real": ["audience_size", "outgoing_ties", "audience_balance", "lifetime_post_volume",
                     "posting_rate", "account_age", "likes_given", "bio_present", "bio_length",
                     "no_profile_picture", "verified", "geo_enabled"],
    "instagram": ["audience_size", "outgoing_ties", "audience_balance", "lifetime_post_volume", "bio_present",
                  "bio_length", "no_profile_picture", "external_link_in_bio", "username_digit_share",
                  "private_account"],
    "facebook": ["audience_size", "outgoing_ties", "lifetime_post_volume", "posting_rate", "account_age",
                 "urls_shared", "communities_joined", "likes_received_per_post", "comments_received_per_post",
                 "hashtags_per_post"],
    "twitter_synthetic": ["follower_count", "retweet_count", "mention_count", "verified", "hashtag_count"],
}
TOP_FRACTION = 0.10


def high_score_profiles(frames, norm):
    """Top decile by out-of-sample RF score vs all scored accounts, per indicator.

    Answers Worksheet 3 Q2 ("Akun dengan karakteristik seperti apa yang memiliki kemungkinan tinggi
    diklasifikasikan sebagai bot/spam?") with feature values, not thresholds. Descriptive only.
    """
    out = {}
    for t, cols in PROFILE_COLUMNS.items():
        sc = norm[t][["account_uid", "label", "rf_p"]]
        d = sc.merge(frames[t][["account_uid"] + cols], on="account_uid", how="left", validate="1:1")
        assert sc["account_uid"].isin(frames[t]["account_uid"]).all(), f"{t}: scored account missing in raw"
        d = d.sort_values(["rf_p", "account_uid"], ascending=[False, True]).reset_index(drop=True)
        k = int(math.ceil(TOP_FRACTION * len(d)))
        top = d.iloc[:k]
        rows = []
        for c in cols:
            v_all, v_top = d[c].dropna(), top[c].dropna()
            binary = set(np.unique(v_all)) <= {0.0, 1.0}
            stat = (lambda v: float(v.mean())) if binary else (lambda v: float(v.median()))
            a, b = stat(v_all), stat(v_top)
            rows.append({
                "indicator": c, "statistic": "share" if binary else "median",
                "top_decile": r4(b), "all_scored": r4(a),
                "authentic_all": r4(stat(d.loc[d.label == 0, c].dropna())),
                "inauthentic_all": r4(stat(d.loc[d.label == 1, c].dropna())),
                "top_over_all": r4(b / a) if a else None,
                "top_decile_p25": None if binary else r4(v_top.quantile(0.25)),
                "top_decile_p75": None if binary else r4(v_top.quantile(0.75)),
            })
        out[t] = {
            "n_scored": int(len(d)), "top_fraction": TOP_FRACTION, "n_top": k,
            "score_cut_for_top_decile": r4(float(top["rf_p"].min())),
            "share_inauthentic_top_decile": r4(float(top["label"].mean())),
            "share_inauthentic_all": r4(float(d["label"].mean())),
            "n_authentic_in_top_decile": int((top["label"] == 0).sum()),
            "indicators": rows,
            "note": "descriptive: who the model ranks highest. The top-decile share is precision@10%; "
                    "the authentic accounts inside it are the ones a moderator must not presume guilty",
        }
        log(f"  Q2 profile {t}: top {k}/{len(d)} share inauthentic {out[t]['share_inauthentic_top_decile']} "
            f"(base {out[t]['share_inauthentic_all']})")
    return out


def evidence_table(metrics):
    """Per dataset: what can be quoted, ordered by evidential weight - deliberately NOT by accuracy."""
    rows = []
    for t in sorted(DATASET_INFO, key=lambda k: DATASET_INFO[k]["evidential_weight"]):
        info = DATASET_INFO[t]
        m = metrics[t]
        row = {"dataset": t, "platform": info["platform"], "role": info["role"],
               "evidential_weight_order": info["evidential_weight"],
               "label_concept": info["label_concept"], "unit": info["unit"],
               "verdict": info["verdict"], "number_to_quote": info["quote"],
               "comparable_quantitatively": info["comparable_quantitatively"]}
        if t != "tiktok":
            rf = m["models"]["random_forest"]["metrics"]
            lr = m["models"]["logistic_regression"]["metrics"]
            row.update({
                "n_rows": m["dataset"]["n_rows"], "positive_rate": m["class_balance"]["positive_rate"],
                "n_test": rf["n_test"],
                "seed42_rf": {k: rf.get(k) for k in ("accuracy", "balanced_accuracy", "precision_positive",
                                                     "recall_positive", "f1_positive", "mcc", "roc_auc",
                                                     "pr_auc")},
                "seed42_lr_accuracy": lr["accuracy"], "seed42_lr_roc_auc": lr.get("roc_auc"),
                "majority_baseline_accuracy": rf["baseline_majority_class"]["accuracy"],
                "rf_accuracy_lift": rf["baseline_majority_class"]["accuracy_lift_of_model"],
            })
            cv = m["models"]["random_forest"].get("cross_validation", {})
            if isinstance(cv, dict) and "summary" in cv:
                row["cv_rf"] = {k: cv["summary"][k] for k in ("accuracy", "mcc", "roc_auc")
                                if k in cv["summary"]}
        else:
            row.update({"n_rows": m["dataset"]["n_rows"], "n_authors": m["dataset"]["n_authors"],
                        "positive_rate": None, "seed42_rf": None})
        rows.append(row)
    return rows


# ==========================================================================
# 2. Risk levels 0-4
# ==========================================================================
LEVELS = [
    {"code": "0", "level": 0, "name_id": "Tidak ada indikasi",
     "name_concept_doc": "Authentic / Low Risk",
     "description_concept_doc": "Tidak menunjukkan indikasi kuat perilaku tidak autentik dari fitur yang tersedia.",
     "name_en": "No indication",
     "definition": "Model score below t1 (or, TikTok: behaviour assessed and no behavioural indicator "
                   "fired). NOT a statement that the account is authentic.",
     "required_evidence": "an out-of-sample model score (or an assessable behavioural indicator)",
     "suggested_action": "none - not queued", "reachable": True},
    {"code": "1", "level": 1, "name_id": "Indikasi lemah - pantau",
     "name_concept_doc": "Suspicious Automation",
     "description_concept_doc": "Ada indikasi otomatisasi atau pola aktivitas yang tidak biasa, tetapi bukti masih "
                                "terbatas.",
     "name_en": "Weak account-level indication - watch",
     "definition": "t1 <= score < t2. t1 maximises F2 (recall weighted 2x) on the calibration fold.",
     "required_evidence": "an out-of-sample model score with measured precision/recall, from a model "
                          "that passes the evidence gate",
     "suggested_action": "monitor / sample for QA; review only if capacity allows", "reachable": True},
    {"code": "2", "level": 2, "name_id": "Indikasi kuat tingkat akun - prioritas tinjauan manusia",
     "name_concept_doc": "Inauthentic / Spam",
     "description_concept_doc": "Indikasi perilaku tidak autentik atau spam lebih kuat.",
     "name_en": "Strong account-level indication - priority human review",
     "definition": "score >= t2. t2 maximises F0.5 (precision weighted 2x) on the calibration fold, "
                   "because every flag spends a moderator's time and may put a real person in the queue.",
     "required_evidence": "an out-of-sample model score whose precision at the cut has been measured "
                          "against labels (impossible for TikTok)",
     "suggested_action": "human review queue, ordered by score; never an automated sanction",
     "reachable": True},
    {"code": "3", "level": 3, "name_id": "Dugaan koordinasi antar-akun",
     "name_concept_doc": "Coordinated Manipulation",
     "description_concept_doc": "Ada indikasi pola koordinasi antar akun yang dapat didukung oleh data "
                                "jaringan/waktu/konten.",
     "name_en": "Suspected cross-account coordination",
     "definition": "Level 2 PLUS evidence that the account acts together with other accounts: "
                   "synchronised posting times, shared or near-duplicate content across accounts, "
                   "or shared infrastructure / network ties.",
     "required_evidence": "per-post timestamps and content for many accounts in the same window, and/or "
                          "an interaction or follow graph; platform-side device/IP signals",
     "suggested_action": "escalate to an investigation analyst (still human)", "reachable": False},
    {"code": "4", "level": 4, "name_id": "Amplifikasi terkoordinasi",
     "name_concept_doc": "High-Risk Coordinated Influence",
     "description_concept_doc": "Indikasi koordinasi dan amplifikasi kuat serta mempunyai potensi impact yang "
                                "lebih besar.",
     "name_en": "Coordinated amplification",
     "definition": "Level 3 PLUS evidence that the coordinated group amplifies a common target "
                   "(cross-account bursts boosting the same post, URL or hashtag) with measurable reach.",
     "required_evidence": "interaction event logs (who retweeted/shared/liked what, when), target "
                          "identifiers, and reach over time",
     "suggested_action": "escalate as a campaign investigation (still human)", "reachable": False},
    {"code": "NA", "level": None, "name_id": "Tidak dapat dinilai",
     "name_concept_doc": None,
     "description_concept_doc": "(tidak ada padanan di konsep §13; tier ditambahkan oleh track sintesis)",
     "name_en": "Not assessable",
     "definition": "No evidence can be computed (TikTok single-video authors), or the model has no "
                   "signal (evidence gate failed: X synthetic). Missing is not zero: this is NOT Level 0.",
     "required_evidence": "-", "suggested_action": "none - and do not report as 'clear'", "reachable": True},
]


def pr_curve(y, s):
    """Precision/recall/volume at every distinct cut 'score >= t', highest t first."""
    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order]
    tp, fp = np.cumsum(yy), np.cumsum(1 - yy)
    last = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]
    thr, TP, FP = ss[last], tp[last], fp[last]
    P = y.sum()
    prec = TP / (TP + FP)
    rec = TP / P if P else np.zeros_like(TP, dtype=float)
    return thr, prec, rec, (TP + FP) / len(y), TP, FP


def fbeta_cut(y, s, beta, floor=None):
    thr, p, r, vol, _, _ = pr_curve(y, s)
    b2 = beta * beta
    den = b2 * p + r
    f = np.where(den > 0, (1 + b2) * p * r / np.where(den > 0, den, 1), 0.0)
    if floor is not None:
        f = np.where(thr >= floor, f, -1.0)
    i = int(np.argmax(f))  # thr is descending, so ties resolve to the highest threshold
    return float(thr[i]), {"f": r4(f[i]), "precision": r4(p[i]), "recall": r4(r[i]), "volume": r4(vol[i])}


def derive_thresholds(y, s):
    t1, s1 = fbeta_cut(y, s, BETA_L1)
    t2, s2 = fbeta_cut(y, s, BETA_L2, floor=t1)
    return t1, t2, {"t1_on_calibration": s1, "t2_on_calibration": s2}


def assign(s, t1, t2):
    return np.where(s >= t2, 2, np.where(s >= t1, 1, 0))


def stratified_halves(y, seed=SEED):
    rng = np.random.RandomState(seed)
    fold = np.empty(len(y), dtype=int)
    for cls in (0, 1):
        idx = np.nonzero(y == cls)[0]
        idx = idx[rng.permutation(len(idx))]
        fold[idx] = np.arange(len(idx)) % 2
    return fold


def level_table(y, lvl):
    P, N, n = int(y.sum()), int(len(y) - y.sum()), len(y)
    per, cum = [], []
    for k in (0, 1, 2):
        m = lvl == k
        nk, pk = int(m.sum()), int(y[m].sum())
        per.append({"level": k, "n_accounts": nk, "volume_share": r4(nk / n),
                    "n_inauthentic": pk, "n_authentic": nk - pk,
                    "precision_in_level": r4(pk / nk) if nk else None,
                    "precision_ci95": wilson_ci(pk, nk),
                    "share_of_all_inauthentic": r4(pk / P) if P else None,
                    "share_of_all_authentic": r4((nk - pk) / N) if N else None})
    for k in (1, 2):
        m = lvl >= k
        nk, pk = int(m.sum()), int(y[m].sum())
        tpr = pk / P if P else 0.0
        fpr = (nk - pk) / N if N else 0.0
        adj = []
        for pi in PREVALENCES:
            flagged = pi * tpr + (1 - pi) * fpr
            adj.append({"assumed_prevalence": pi,
                        "precision": r4(pi * tpr / flagged) if flagged else None,
                        "flagged_per_10000_accounts": r4(10000 * flagged),
                        "real_people_flagged_per_10000": r4(10000 * (1 - pi) * fpr)})
        cum.append({"at_or_above_level": k, "n_flagged": nk, "volume_share": r4(nk / n),
                    "precision": r4(pk / nk) if nk else None, "precision_ci95": wilson_ci(pk, nk),
                    "recall": r4(tpr), "false_positive_rate": r4(fpr),
                    "real_people_flagged": nk - pk,
                    "prevalence_adjusted": adj})
    return per, cum


def risk_levels_labelled(norm):
    results, assigned = {}, {}
    for t in ("twitter_real", "instagram", "facebook", "twitter_synthetic"):
        d = norm[t]
        y = d["label"].to_numpy().astype(int)
        s = d["rf_p"].to_numpy().astype(float)
        fold = stratified_halves(y)
        lvl = np.empty(len(y), dtype=int)
        t1_used = np.empty(len(y))
        t2_used = np.empty(len(y))
        folds = {}
        for f in (0, 1):
            cal, eva = fold == f, fold != f
            t1, t2, cal_stats = derive_thresholds(y[cal], s[cal])
            lvl[eva] = assign(s[eva], t1, t2)
            t1_used[eva], t2_used[eva] = t1, t2
            folds["AB"[f]] = {"calibrated_on_fold": "AB"[f], "applied_to_fold": "AB"[1 - f],
                              "n_calibration": int(cal.sum()), "t1": r4(t1), "t2": r4(t2), **cal_stats}
        per, cum = level_table(y, lvl)
        # Evidence gate: levels are only issued if Level >= 2 is measurably better than picking
        # accounts at random. F-beta cuts on a no-signal model happily flag everything.
        base = float(y.mean())
        lo2 = cum[1]["precision_ci95"][0]
        gate = {"rule": "assign levels only if the Wilson 95% lower bound of cross-fitted precision at "
                        "Level >= 2 exceeds the base rate (the precision of flagging at random)",
                "level2_precision": cum[1]["precision"], "level2_precision_ci95_low": lo2,
                "base_rate": r4(base), "passed": bool(lo2 is not None and lo2 > base)}
        # Deployment thresholds (for scoring NEW accounts): fitted on all scored accounts.
        t1_all, t2_all, all_stats = derive_thresholds(y, s)
        rng = np.random.default_rng(SEED)
        boots = []
        for _ in range(N_BOOT_THRESH):
            idx = rng.integers(0, len(y), len(y))
            if y[idx].sum() in (0, len(idx)):
                continue
            a, b, _ = derive_thresholds(y[idx], s[idx])
            boots.append((a, b))
        boots = np.array(boots)
        # Score distributions by class, for the report and the figure.
        q = [0.05, 0.25, 0.5, 0.75, 0.95]
        dist = {cls: {f"p{int(qq * 100)}": r4(np.quantile(s[y == c], qq)) for qq in q}
                for cls, c in (("authentic", 0), ("inauthentic", 1))}
        results[t] = {
            "n_accounts": int(len(y)), "positive_rate": r4(y.mean()),
            "score_column": "rf_p_inauthentic (Random Forest, out-of-sample)",
            "scored_accounts": ("held-out test split only" if t.startswith("twitter")
                                else "all accounts: test holdout + CV out-of-fold (CV repeat 1)"),
            "cross_fitting": {"method": "stratified 2-fold, seed 42; thresholds derived on one half are "
                                        "applied to the other, so every reported level is out-of-sample "
                                        "for the threshold as well as for the model",
                              "folds": folds},
            "per_level": per, "cumulative": cum,
            "deployment_thresholds": {
                "t1": r4(t1_all), "t2": r4(t2_all), **all_stats,
                "bootstrap_ci95_t1": [r4(np.percentile(boots[:, 0], 2.5)), r4(np.percentile(boots[:, 0], 97.5))],
                "bootstrap_ci95_t2": [r4(np.percentile(boots[:, 1], 2.5)), r4(np.percentile(boots[:, 1], 97.5))],
                "note": "fitted on all scored accounts, for levelling NEW accounts; its expected "
                        "performance is the cross-fitted table, not a re-evaluation on the same rows"},
            "score_distribution": dist,
            "levels_3_4": "not assigned - no coordination or amplification evidence in this dataset",
            "evidence_gate": gate,
            "status": "levels_assigned" if gate["passed"] else
                      "levels_NOT_assigned (gate failed) - per_level/cumulative are a diagnostic of what "
                      "ungated cuts would have done",
        }
        assigned[t] = pd.DataFrame({"account_uid": d["account_uid"],
                                    "risk_level": lvl.astype(float) if gate["passed"] else np.nan,
                                    "risk_level_basis": "model_probability_crossfit" if gate["passed"]
                                    else "evidence_gate_failed_no_signal",
                                    "crossfit_fold": ["AB"[f] for f in fold],  # its own half; thresholds came from the other
                                    "t1_applied": t1_used, "t2_applied": t2_used})
        log(f"  {t}: t1/t2 folds A={folds['A']['t1']}/{folds['A']['t2']} "
            f"B={folds['B']['t1']}/{folds['B']['t2']}; L2 precision {per[2]['precision_in_level']} "
            f"(n={per[2]['n_accounts']}), recall>=L2 {cum[1]['recall']}; gate passed={gate['passed']}")
    return results, assigned


def risk_levels_tiktok(tt):
    """TikTok: no probability and no labels. Levels from behavioural indicators only, capped at 1."""
    assess = tt["behavioural_indicators_assessable"].fillna(0).astype(int)
    tier = tt["review_tier"]
    lvl = pd.Series([None] * len(tt), dtype=object)
    lvl[(assess >= 1)] = 0
    lvl[tier.isin(["secondary_context", "priority_review"])] = 1
    lvl[assess == 0] = None
    # sanity: every secondary/priority author is behaviourally assessable
    assert not (tier.isin(["secondary_context", "priority_review"]) & (assess == 0)).any()
    counts = {"NA": int(lvl.isna().sum()), "0": int((lvl == 0).sum()), "1": int((lvl == 1).sum()),
              "2": 0, "3": 0, "4": 0}
    cross = pd.crosstab(tier, lvl.fillna("NA").astype(str)).to_dict()
    return lvl, {
        "basis": "rule on the TikTok track's behavioural indicators (burst posting, near-duplicate "
                 "captions, mechanical regularity). Weak popularity-confounded signals (engagement "
                 "outlier, IsolationForest) never raise a level.",
        "mapping": {"NA": "behavioural_indicators_assessable == 0 (single-video authors) - includes "
                          "816 weak_signal_only authors whose behaviour cannot be assessed",
                    "0": "behaviour assessable, no behavioural indicator fired",
                    "1": "secondary_context (one of burst / caption duplication fired); priority_review "
                         "would also be capped here",
                    "2": "unreachable: no labels, so no precision can be measured for any cut",
                    "3-4": "unreachable: no graph; cross-author identical captions are trend tags (#fyp)"},
        "counts": counts, "tier_by_level": cross,
    }


COORDINATION_EVIDENCE = {
    # status: available / partial / absent
    "twitter_real": {
        "per_post_timestamps": ("absent", "profile snapshot only; no tweets"),
        "per_post_content": ("absent", "no tweet text"),
        "interaction_or_follow_graph": ("absent", "only follower/friend COUNTS, not who"),
        "cross_account_target_bursts": ("absent", "no retweet/share events"),
        "shared_infrastructure": ("absent", "platform-internal; never in public data"),
    },
    "twitter_synthetic": {
        "per_post_timestamps": ("partial", "one tweet per user, timestamps uniform 2020-2023 (generated)"),
        "per_post_content": ("partial", "Faker lorem words: identical words across users are BY "
                                        "CONSTRUCTION, so 'shared content' would be a false finding"),
        "interaction_or_follow_graph": ("absent", ""),
        "cross_account_target_bursts": ("absent", ""),
        "shared_infrastructure": ("absent", ""),
    },
    "instagram": {
        "per_post_timestamps": ("absent", "profile aggregates only"),
        "per_post_content": ("absent", ""),
        "interaction_or_follow_graph": ("absent", "counts only"),
        "cross_account_target_bursts": ("absent", ""),
        "shared_infrastructure": ("absent", ""),
    },
    "facebook": {
        "per_post_timestamps": ("absent", "profile aggregates only"),
        "per_post_content": ("absent", ""),
        "interaction_or_follow_graph": ("absent", "counts only"),
        "cross_account_target_bursts": ("absent", ""),
        "shared_infrastructure": ("absent", ""),
    },
    "tiktok": {
        "per_post_timestamps": ("partial", "per-video times exist, but 95.4% of authors have one video "
                                           "and only viral videos were captured"),
        "per_post_content": ("partial", "captions/hashtags/sounds exist; 29 captions shared across "
                                        "authors, 23 of them hashtag-only trend tags (#fyp)"),
        "interaction_or_follow_graph": ("absent", "no follower, duet, stitch or reply links"),
        "cross_account_target_bursts": ("absent", "no share/like events, only per-video totals"),
        "shared_infrastructure": ("absent", ""),
    },
}

DATA_NEEDED_FOR_L3_L4 = [
    "Timeline per akun: setiap post dengan timestamp, teks, URL, hashtag, dan referensi (retweet/quote/"
    "reply/duet/stitch) untuk BANYAK akun dalam jendela waktu yang sama - bukan snapshot profil.",
    "Graf interaksi atau follow (siapa me-retweet/membagikan/membalas siapa, siapa mengikuti siapa).",
    "Log event interaksi berstempel waktu (like/share/retweet) per target, agar burst lintas-akun ke "
    "target yang sama dan jangkauannya dapat diukur (syarat Level 4).",
    "Label atau ground truth di tingkat KAMPANYE/jaringan (mis. daftar takedown operasi terkoordinasi "
    "yang dipublikasikan platform, dengan provenance), bukan hanya label per akun.",
    "Kandidat dataset publik untuk dijajaki (verifikasi isinya dulu): Cresci-2017 versi lengkap (berisi "
    "tweet akun social spambot, bukan hanya file users yang ada di box ini) dan TwiBot-22 (tweet + graf "
    "relasi). Data X/Twitter real milik Treiman tidak cukup karena hanya berisi metadata profil.",
]


# ==========================================================================
# 3. Figures
# ==========================================================================
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e5e4e0", "#fcfcfb"
PLAT_COLOR = {"twitter_real": "#2a78d6", "instagram": "#eb6834", "facebook": "#1baf7a",
              "twitter_synthetic": "#8a8985"}
PLAT_NAME = {"twitter_real": "X (real, Treiman)", "instagram": "Instagram", "facebook": "Facebook (cautionary)",
             "twitter_synthetic": "X synthetic (negative control)"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_concept_direction(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [i["key"] for i in INDICATORS]
    labels = {
        "audience_size": "Audience size (followers / FB friends)", "outgoing_ties": "Accounts followed",
        "audience_balance": "Follower share  f/(f+following)", "lifetime_post_volume": "Lifetime posts",
        "posting_rate": "Posts per day", "account_age": "Account age (days)",
        "bio_present": "Has a bio", "bio_length": "Bio length", "no_profile_picture": "No own profile picture",
        "external_link_in_bio": "Link in bio", "username_digit_share": "Digits in username",
        "likes_given": "Likes GIVEN", "likes_received_per_post": "Likes received / post",
        "comments_received_per_post": "Comments received / post", "urls_shared": "URLs shared",
        "communities_joined": "Communities joined", "hashtags_per_post": "Hashtags / post",
        "verified": "Verified", "private_account": "Private account"}
    fig, ax = plt.subplots(figsize=(10, 8.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    offsets = {"twitter_real": 0.22, "instagram": 0.0, "facebook": -0.22}
    ys = {k: len(order) - 1 - i for i, k in enumerate(order)}
    n_shared = sum(1 for k in order if sum(r["indicator"] == k for r in rows) >= 2)
    ax.axhspan(len(order) - n_shared - 0.5, len(order) - 0.5, color="#f0efec", zorder=0)
    ax.text(0.985, len(order) - n_shared - 0.4, "above: measured on 2+ labelled platforms", fontsize=8,
            color=INK2, va="bottom", ha="right", transform=ax.get_yaxis_transform())
    for r in rows:
        y = ys[r["indicator"]] + offsets[r["dataset"]]
        c = PLAT_COLOR[r["dataset"]]
        lo, hi = r["auc_ci95"]
        ax.plot([lo, hi], [y, y], color=c, linewidth=2, solid_capstyle="round", zorder=2)
        fb = r["dataset"] == "facebook"
        ax.scatter([r["auc"]], [y], s=46, zorder=3, facecolor=SURFACE if fb else c,
                   edgecolor=c, linewidth=1.8)
    ax.axvline(0.5, color=INK2, linewidth=1, zorder=1)
    ax.set_yticks([ys[k] for k in order])
    ax.set_yticklabels([labels[k] for k in order], fontsize=8.5, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Univariate ROC-AUC of the raw value for the inauthentic class (95% bootstrap CI)",
                  fontsize=9, color=INK2)
    ax.text(0.25, 1.005, "<- higher value = more AUTHENTIC", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.text(0.75, 1.005, "higher value = more INAUTHENTIC ->", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=8.5, color=INK2)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="-", color=PLAT_COLOR[t],
                      markerfacecolor=SURFACE if t == "facebook" else PLAT_COLOR[t], label=PLAT_NAME[t])
               for t in LABELLED_REAL]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, frameon=False)
    ax.set_title("Which way does each behavioural concept point? (label concepts differ: X bot / "
                 "IG fake / FB spam)", fontsize=10.5, color=INK, loc="left", pad=20)
    fig.text(0.01, 0.005, "Facebook shown hollow: its directions describe two label-sorted file blocks, "
                          "not spammers. TikTok: no labels, so no direction can be measured for any concept.",
             fontsize=7.8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_importance(imp, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    concepts = [c for c in CONCEPTS if any(c in imp[t]["impurity_share"] for t in LABELLED_REAL)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.4), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    yy = np.arange(len(concepts))[::-1]
    for ax, t in zip(axes, LABELLED_REAL):
        _style(ax)
        perm = [imp[t]["permutation_share"].get(c, 0) or 0 for c in concepts]
        impu = [imp[t]["impurity_share"].get(c, 0) or 0 for c in concepts]
        ax.barh(yy, perm, height=0.62, hatch="////" if t == "facebook" else None,
                edgecolor=SURFACE if t != "facebook" else PLAT_COLOR[t], linewidth=0.6,
                facecolor=PLAT_COLOR[t] if t != "facebook" else SURFACE, label="permutation share (test)")
        ax.scatter(impu, yy, marker="|", s=160, color=INK, linewidth=1.6, zorder=3,
                   label="impurity share (train)")
        for y_, v, w in zip(yy, perm, impu):
            if v >= 0.05:
                ax.text(max(v, w) + 0.015, y_, f"{v:.0%}", va="center", fontsize=7.8, color=INK2)
        ax.set_title(PLAT_NAME[t], fontsize=10, color=INK, loc="left")
        ax.set_xlim(0, 0.75)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        absent = [c for c in concepts if c not in imp[t]["impurity_share"]]
        for c in absent:
            ax.text(0.01, yy[concepts.index(c)], "not in data", va="center", fontsize=7, color=MUTED)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels([c.replace("_", " ") for c in concepts], fontsize=8.5, color=INK)
    axes[0].legend(loc="lower right", fontsize=7.8, frameon=False)
    fig.suptitle("Share of each Random Forest's importance by behavioural concept", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_risk_levels(norm, risk, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tracks = ["twitter_real", "instagram", "facebook", "twitter_synthetic"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), dpi=150, gridspec_kw={"height_ratios": [1, 1.25]})
    fig.patch.set_facecolor(SURFACE)
    for j, t in enumerate(tracks):
        d = norm[t]
        y, s = d["label"].to_numpy(), d["rf_p"].to_numpy()
        dt = risk[t]["deployment_thresholds"]
        t1, t2 = dt["t1"], dt["t2"]
        ax = axes[0, j]
        _style(ax)
        bins = np.linspace(0, 1, 41)
        ax.hist(s[y == 0], bins=bins, density=True, alpha=0.85, color="#2a78d6", label="authentic (0)",
                edgecolor=SURFACE, linewidth=0.4)
        ax.hist(s[y == 1], bins=bins, density=True, alpha=0.65, color="#e34948", label="inauthentic (1)",
                edgecolor=SURFACE, linewidth=0.4)
        top = ax.get_ylim()[1]
        for tv, name, h in ((t1, "t1", 0.97), (t2, "t2", 0.80)):
            ax.axvline(tv, color=INK, linewidth=1.1, linestyle="--")
            ax.text(tv + 0.01, top * h, f"{name}={tv:.3f}", fontsize=7.5, color=INK, va="top",
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.6))
        ax.set_title(PLAT_NAME[t], fontsize=10, color=INK, loc="left")
        ax.set_yticks([])
        if j == 0:
            fig.legend(*ax.get_legend_handles_labels(), loc="upper right", ncol=2, fontsize=8,
                       frameon=False, bbox_to_anchor=(0.995, 0.955))
        ax2 = axes[1, j]
        _style(ax2)
        thr, p, r, vol, _, _ = pr_curve(y, s)
        ax2.axvspan(0, t1, color="#f6f5f2", zorder=0)
        ax2.axvspan(t1, t2, color="#e9e8e4", zorder=0)
        ax2.axvspan(t2, 1, color="#d9d8d3", zorder=0)
        ax2.plot(thr, p, color="#6250d6", linewidth=2, label="precision (score >= t)")
        ax2.plot(thr, r, color="#1baf7a", linewidth=2, linestyle="--", label="recall (score >= t)")
        ax2.plot(thr, vol, color=INK2, linewidth=1.2, linestyle=":", label="volume flagged")
        ax2.axhline(y.mean(), color=MUTED, linewidth=0.8)
        ax2.text(0.99, y.mean() + 0.015, "base rate", ha="right", fontsize=7, color=MUTED)
        per = risk[t]["per_level"]
        for k, (a, b) in enumerate(((0, t1), (t1, t2), (t2, 1))):
            pk = per[k]["precision_in_level"]
            ax2.text((a + b) / 2, 1.04, f"L{k}\n{per[k]['volume_share']:.0%} vol\n"
                     f"prec {pk:.2f}" if pk is not None else f"L{k}\nempty",
                     ha="center", va="bottom", fontsize=7, color=INK)
        if not risk[t]["evidence_gate"]["passed"]:
            ax2.text(0.5, 0.55, "EVIDENCE GATE FAILED\nlevels not assigned:\nungated cuts would flag "
                     f"{risk[t]['cumulative'][1]['volume_share']:.0%}\nat precision = base rate",
                     ha="center", va="center", fontsize=8.5, color="#d03b3b", fontweight="bold",
                     bbox=dict(facecolor=SURFACE, edgecolor="#d03b3b", boxstyle="round,pad=0.4"))
        ax2.set_ylim(0, 1.32)
        ax2.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("RF score threshold t", fontsize=8.5, color=INK2)
        if j == 0:
            ax2.legend(fontsize=7.3, frameon=False, loc="lower left")
    fig.suptitle("Risk levels 0-2 on out-of-sample RF scores: thresholds from the precision/recall "
                 "trade-off (t1 = max F2, t2 = max F0.5); level stats are cross-fitted",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, "Bands and dashed lines: deployment thresholds (fit on all scored accounts). Band "
                          "labels (volume, precision): cross-fitted, each account levelled by thresholds from the "
                          "other half. Levels 3-4 are not drawn: no dataset has coordination / amplification "
                          "evidence. TikTok has no score.", fontsize=7.8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ==========================================================================
# 4. Serving: unified account table, curated Parquet, MongoDB, Power BI
# ==========================================================================
LEVEL_NAME = {l["code"]: l["name_id"] for l in LEVELS}
ACTION = {"0": "none", "1": "monitor", "2": "human_review", "NA": "none_not_assessable"}


def unified_accounts(norm, assigned, tt, tt_lvl):
    frames = []
    for t in ("twitter_real", "twitter_synthetic", "instagram", "facebook"):
        d = norm[t].merge(assigned[t], on="account_uid", how="left", validate="1:1")
        d = d.sort_values("rf_p", ascending=False).reset_index(drop=True)
        frames.append(pd.DataFrame({
            "account_uid": d["account_uid"], "platform": DATASET_INFO[t]["platform"], "dataset": t,
            "dataset_role": DATASET_INFO[t]["role"], "source_ref": d["source_ref"],
            "evaluation_scope": d["evaluation_scope"], "label": d["label"].astype("Int64"),
            "label_concept": DATASET_INFO[t]["label_concept"],
            "rf_p_inauthentic": d["rf_p"].round(6), "lr_p_inauthentic": d["lr_p"].round(6),
            "rf_prediction_at_0_5": d["rf_pred"].astype("Int64"),
            "rank_in_dataset": np.arange(1, len(d) + 1),
            "risk_level": d["risk_level"].astype("Int64"),
            "risk_level_code": d["risk_level"].map(lambda v: "NA" if pd.isna(v) else str(int(v))),
            "risk_level_basis": d["risk_level_basis"],
            "crossfit_fold": d["crossfit_fold"],
            "t1_applied": d["t1_applied"].round(4), "t2_applied": d["t2_applied"].round(4),
            "tiktok_review_tier": None, "tiktok_video_count": pd.NA,
            "tiktok_behavioural_indicators_assessable": pd.NA,
            "ind_burst_posting": pd.NA, "ind_caption_duplication": pd.NA, "ind_regular_posting": pd.NA,
            "ind_engagement_outlier": pd.NA, "ind_iforest_outlier": pd.NA,
        }))
    code = tt_lvl.map(lambda v: "NA" if pd.isna(v) else str(int(v)))
    frames.append(pd.DataFrame({
        "account_uid": "tt_" + tt["author_key"].astype(str), "platform": "tiktok", "dataset": "tiktok",
        "dataset_role": DATASET_INFO["tiktok"]["role"], "source_ref": tt["author_key"].astype(str),
        "evaluation_scope": "unsupervised_all_authors", "label": pd.array([pd.NA] * len(tt), dtype="Int64"),
        "label_concept": DATASET_INFO["tiktok"]["label_concept"],
        "rf_p_inauthentic": np.nan, "lr_p_inauthentic": np.nan,
        "rf_prediction_at_0_5": pd.array([pd.NA] * len(tt), dtype="Int64"),
        "rank_in_dataset": pd.array([pd.NA] * len(tt), dtype="Int64"),
        "risk_level": pd.array([None if pd.isna(v) else int(v) for v in tt_lvl], dtype="Int64"),
        "risk_level_code": code,
        "risk_level_basis": np.where(code == "NA", "not_assessable", "tiktok_indicator_rule"),
        "crossfit_fold": None, "t1_applied": np.nan, "t2_applied": np.nan,
        "tiktok_review_tier": tt["review_tier"], "tiktok_video_count": tt["video_count"].astype("Int64"),
        "tiktok_behavioural_indicators_assessable":
            tt["behavioural_indicators_assessable"].astype("Int64"),
        "ind_burst_posting": tt["ind_burst_posting"].astype("Int64"),
        "ind_caption_duplication": tt["ind_caption_duplication"].astype("Int64"),
        "ind_regular_posting": tt["ind_regular_posting"].astype("Int64"),
        "ind_engagement_outlier": tt["ind_engagement_outlier"].astype("Int64"),
        "ind_iforest_outlier": tt["ind_iforest_outlier"].astype("Int64"),
    }))
    acc = pd.concat(frames, ignore_index=True)
    acc["risk_level_name"] = acc["risk_level_code"].map(LEVEL_NAME)
    acc["suggested_action"] = acc["risk_level_code"].map(ACTION)
    assert acc["account_uid"].is_unique
    assert acc["risk_level_name"].notna().all()
    return acc


def account_schema():
    from pyspark.sql import types as T
    S, D, L = T.StringType(), T.DoubleType(), T.LongType()
    cols = [("account_uid", S), ("platform", S), ("dataset", S), ("dataset_role", S), ("source_ref", S),
            ("evaluation_scope", S), ("label", L), ("label_concept", S), ("rf_p_inauthentic", D),
            ("lr_p_inauthentic", D), ("rf_prediction_at_0_5", L), ("rank_in_dataset", L), ("risk_level", L),
            ("risk_level_code", S), ("risk_level_name", S), ("risk_level_basis", S), ("suggested_action", S),
            ("crossfit_fold", S), ("t1_applied", D), ("t2_applied", D), ("tiktok_review_tier", S),
            ("tiktok_video_count", L), ("tiktok_behavioural_indicators_assessable", L),
            ("ind_burst_posting", L), ("ind_caption_duplication", L), ("ind_regular_posting", L),
            ("ind_engagement_outlier", L), ("ind_iforest_outlier", L)]
    return T.StructType([T.StructField(n, t, True) for n, t in cols]), [n for n, _ in cols]


def to_records(df, cols):
    """pandas -> list of tuples with real None for every missing value (missing is not zero)."""
    out = []
    for row in df[cols].itertuples(index=False, name=None):
        rec = []
        for v in row:
            if v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)):
                rec.append(None)
            elif isinstance(v, (np.integer,)):
                rec.append(int(v))
            elif isinstance(v, (np.floating,)):
                rec.append(float(v))
            else:
                rec.append(v)
        out.append(tuple(rec))
    return out


def mongo_settings():
    env = {}
    p = ROOT / ".env.mongo"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    env.update({k: v for k, v in os.environ.items() if k.startswith("MONGO")})
    user, pw = env.get("MONGO_INITDB_ROOT_USERNAME"), env.get("MONGO_INITDB_ROOT_PASSWORD")
    host, port = env.get("MONGO_HOST", "127.0.0.1"), env.get("MONGO_PORT", "27017")
    db = env.get("MONGO_DB", "socmed_bot_detection")
    uri = env.get("MONGO_URI") or (f"mongodb://{user}:{pw}@{host}:{port}/?authSource=admin" if user
                                   else f"mongodb://{host}:{port}/")
    redacted = uri.replace(pw, "<password>") if pw else uri
    return uri, db, redacted


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return r4(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if o is pd.NA:
        return None
    return o


def build_side_tables(metrics, concept_rows, imp, risk, tt_risk, profiles):
    """Flat tables shared by MongoDB and Power BI."""
    ds = []
    for t, info in DATASET_INFO.items():
        m = metrics[t]["dataset"]
        cb = metrics[t].get("class_balance") or {}
        ds.append({"dataset": t, "platform": info["platform"], "dataset_role": info["role"],
                   "evidential_weight_order": info["evidential_weight"], "source": m.get("source"),
                   "n_rows": m.get("n_rows"), "unit": info["unit"],
                   "n_authentic": cb.get("n_authentic"), "n_inauthentic": cb.get("n_inauthentic"),
                   "positive_rate": cb.get("positive_rate"), "label_concept": info["label_concept"],
                   "comparable_quantitatively": info["comparable_quantitatively"],
                   "verdict": info["verdict"], "number_to_quote": info["quote"]})
    mm = []
    for t in ("twitter_real", "twitter_synthetic", "instagram", "facebook"):
        for model, v in metrics[t]["models"].items():
            met = v["metrics"]
            for k in ("accuracy", "balanced_accuracy", "precision_positive", "recall_positive",
                      "f1_positive", "specificity", "mcc", "roc_auc", "pr_auc"):
                if k in met:
                    mm.append({"dataset": t, "model": model, "evaluation": "seed42_holdout_test",
                               "metric": k, "value": met[k], "n_eval": met["n_test"]})
            mm.append({"dataset": t, "model": model, "evaluation": "seed42_holdout_test",
                       "metric": "majority_baseline_accuracy",
                       "value": met["baseline_majority_class"]["accuracy"], "n_eval": met["n_test"]})
            cv = v.get("cross_validation")
            if isinstance(cv, dict) and "summary" in cv:
                for k, s in cv["summary"].items():
                    for stat in ("mean", "std"):
                        mm.append({"dataset": t, "model": model,
                                   "evaluation": f"cv_5x10_inside_train_{stat}", "metric": k,
                                   "value": s[stat], "n_eval": None})
    fi = []
    for t in ("twitter_real", "twitter_synthetic", "instagram", "facebook"):
        for model, v in metrics[t]["models"].items():
            for row in v["feature_importance"]:
                key = "importance" if "importance" in row else "coefficient"
                conc = None
                if t in MODEL_FEATURE_CONCEPT:
                    f = row["feature"]
                    conc = MODEL_FEATURE_CONCEPT[t].get(f) or ("language" if f.startswith("lang=") else None)
                fi.append({"dataset": t, "model": model, "feature": row["feature"], "concept": conc,
                           "importance_type": "rf_impurity_train" if key == "importance"
                           else "lr_coefficient_per_sd", "value": row[key], "rank": row["rank"]})
            pr = v.get("permutation_importance")
            prow = pr["features"] if isinstance(pr, dict) else (pr or [])
            for row in prow:
                if row.get("is_group"):
                    continue
                conc = None
                if t in MODEL_FEATURE_CONCEPT:
                    f = row["feature"]
                    conc = MODEL_FEATURE_CONCEPT[t].get(f) or ("language" if f.startswith("lang") else None)
                fi.append({"dataset": t, "model": model, "feature": row["feature"], "concept": conc,
                           "importance_type": "permutation_test_roc_auc_drop",
                           "value": row["roc_auc_drop_mean"], "rank": row["rank"]})
    rl = []
    for t, r in risk.items():
        for p in r["per_level"]:
            pre = "" if r["evidence_gate"]["passed"] else "diagnostic_ungated_"
            rl.append({"dataset": t, "level_code": str(p["level"]), "kind": pre + "per_level", **{
                k: p[k] for k in ("n_accounts", "volume_share", "n_inauthentic", "n_authentic",
                                  "precision_in_level", "share_of_all_inauthentic", "share_of_all_authentic")},
                "precision_ci95_low": p["precision_ci95"][0], "precision_ci95_high": p["precision_ci95"][1],
                "recall_at_or_above": None, "t1_deployment": r["deployment_thresholds"]["t1"],
                "t2_deployment": r["deployment_thresholds"]["t2"]})
        for c in r["cumulative"]:
            rl.append({"dataset": t, "level_code": f">={c['at_or_above_level']}", "kind": pre + "cumulative",
                       "n_accounts": c["n_flagged"], "volume_share": c["volume_share"],
                       "n_inauthentic": c["n_flagged"] - c["real_people_flagged"],
                       "n_authentic": c["real_people_flagged"], "precision_in_level": c["precision"],
                       "share_of_all_inauthentic": c["recall"], "share_of_all_authentic": c["false_positive_rate"],
                       "precision_ci95_low": c["precision_ci95"][0], "precision_ci95_high": c["precision_ci95"][1],
                       "recall_at_or_above": c["recall"], "t1_deployment": r["deployment_thresholds"]["t1"],
                       "t2_deployment": r["deployment_thresholds"]["t2"]})
        for code in ("3", "4"):
            rl.append({"dataset": t, "level_code": code, "kind": "unreachable", "n_accounts": 0})
    for code, n in tt_risk["counts"].items():
        rl.append({"dataset": "tiktok", "level_code": code, "kind": "indicator_rule" if code in ("0", "1")
                   else ("not_assessable" if code == "NA" else "unreachable"), "n_accounts": n,
                   "volume_share": r4(n / sum(tt_risk["counts"].values()))})
    lv = [{**{k: v for k, v in l.items()}, "level": l["level"]} for l in LEVELS]
    ce = []
    for t, ev in COORDINATION_EVIDENCE.items():
        for k, (status, note) in ev.items():
            ce.append({"dataset": t, "evidence_type": k, "status": status, "note": note})
    cc = []
    for r in concept_rows:
        cc.append({**{k: v for k, v in r.items() if k != "auc_ci95"},
                   "auc_ci95_low": r["auc_ci95"][0], "auc_ci95_high": r["auc_ci95"][1],
                   "tiktok_status": TIKTOK_STATUS.get(r["indicator"])})
    cs = []
    for t, v in imp.items():
        for kind in ("impurity_share", "permutation_share"):
            for c, val in v[kind].items():
                cs.append({"dataset": t, "concept": c, "concept_label_id": CONCEPTS.get(c),
                           "share_type": kind, "value": val})
    hp = []
    for t, v in profiles.items():
        for r in v["indicators"]:
            hp.append({"dataset": t, **r, "n_top": v["n_top"], "n_scored": v["n_scored"],
                       "share_inauthentic_top_decile": v["share_inauthentic_top_decile"],
                       "share_inauthentic_all": v["share_inauthentic_all"]})
    return {"dim_dataset": ds, "fact_model_metrics": mm, "fact_feature_importance": fi,
            "fact_high_score_profile": hp,
            "fact_risk_level_summary": rl, "dim_risk_level": lv, "fact_coordination_evidence": ce,
            "fact_concept_comparison": cc, "fact_importance_by_concept": cs}


def write_powerbi(acc, side):
    paths = {}
    p = OUT_POWERBI / "fact_account_scores.csv"
    acc.to_csv(p, index=False, encoding="utf-8")
    paths["fact_account_scores"] = p
    for name, rows in side.items():
        p = OUT_POWERBI / f"{name}.csv"
        df = pd.DataFrame(rows)
        for c in df.columns:  # counts stay integers even where some rows are null
            if (c.startswith("n_") or c in ("n", "rank")) and df[c].dtype.kind == "f":
                df[c] = df[c].astype("Int64")
        df.to_csv(p, index=False, encoding="utf-8")
        paths[name] = p
    n_rows = {"fact_account_scores": len(acc), **{k: len(v) for k, v in side.items()}}
    return {k: {"path": str(v.relative_to(ROOT)), "rows": n_rows[k]} for k, v in paths.items()}


def load_mongo(spark, acc_sdf, side, metrics, payload_extra, use_connector):
    from pymongo import ASCENDING, DESCENDING, MongoClient

    uri, dbname, redacted = mongo_settings()
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    db = client[dbname]
    write_path = None
    if use_connector:
        try:
            (acc_sdf.write.format("mongodb").mode("overwrite")
             .option("connection.uri", uri).option("database", dbname)
             .option("collection", "account_scores").save())
            write_path = f"spark -> mongo-spark-connector {MONGO_CONNECTOR.split(':')[-1]}"
        except Exception as e:  # pragma: no cover - fallback path
            log(f"  Spark connector write failed ({type(e).__name__}: {str(e)[:160]}); falling back to pymongo")
    if write_path is None:
        db.account_scores.drop()
        rows = [{k: v for k, v in r.asDict().items()} for r in acc_sdf.collect()]
        db.account_scores.insert_many(rows)
        write_path = "pymongo insert_many (Spark-collected rows)"
    db.account_scores.create_index([("platform", ASCENDING), ("risk_level", ASCENDING)])
    db.account_scores.create_index([("dataset", ASCENDING), ("rf_p_inauthentic", DESCENDING)])
    db.account_scores.create_index([("account_uid", ASCENDING)], unique=True)

    # Small, nested documents: pymongo.
    def reload(name, docs, indexes=()):
        db[name].drop()
        if docs:
            db[name].insert_many([jsonable(d) for d in docs])
        for ix in indexes:
            db[name].create_index(ix)

    metric_docs = []
    for t in TRACKS:
        m = metrics[t]
        doc = {"dataset": t, "platform": DATASET_INFO[t]["platform"], "role": DATASET_INFO[t]["role"],
               "verdict": DATASET_INFO[t]["verdict"], "number_to_quote": DATASET_INFO[t]["quote"],
               "dataset_info": m.get("dataset"), "class_balance": m.get("class_balance"),
               "split": m.get("split"), "limitations": m.get("limitations")}
        if t != "tiktok":
            doc["models"] = {k: {"params": v.get("params"), "metrics": v.get("metrics"),
                                 "cross_validation": v.get("cross_validation")}
                             for k, v in m["models"].items()}
        else:
            doc["review_tiers"] = m.get("review_tiers")
            doc["coordination_assessment"] = m.get("coordination_assessment")
        metric_docs.append(doc)
    reload("model_metrics", metric_docs, [[("dataset", ASCENDING)]])
    reload("datasets", side["dim_dataset"], [[("dataset", ASCENDING)]])
    reload("feature_importance", side["fact_feature_importance"],
           [[("dataset", ASCENDING), ("importance_type", ASCENDING), ("rank", ASCENDING)]])
    reload("risk_level_definitions", side["dim_risk_level"], [[("code", ASCENDING)]])
    reload("risk_level_summary", side["fact_risk_level_summary"], [[("dataset", ASCENDING)]])
    reload("concept_comparison", side["fact_concept_comparison"], [[("indicator", ASCENDING)]])
    reload("coordination_evidence", side["fact_coordination_evidence"], [[("dataset", ASCENDING)]])
    reload("high_score_profiles", side["fact_high_score_profile"], [[("dataset", ASCENDING)]])
    reload("cross_platform_summary", [payload_extra])
    return client, db, write_path, redacted, dbname


def verify_mongo(db, acc, risk):
    """Query MongoDB back and check it against what was computed in Python."""
    q = []

    def rec(name, query, result, check=None):
        q.append({"name": name, "query": query, "result": jsonable(result), "check": check})
        log(f"  [mongo] {name}: {json.dumps(jsonable(result))[:300]}" + (f"  -> {check}" if check else ""))

    counts = {c: db[c].count_documents({}) for c in sorted(db.list_collection_names())}
    rec("collection counts", "db[c].count_documents({}) for every collection", counts,
        "account_scores == %d: %s" % (len(acc), counts.get("account_scores") == len(acc)))
    assert counts["account_scores"] == len(acc)

    by_ds = list(db.account_scores.aggregate([
        {"$group": {"_id": "$dataset", "n": {"$sum": 1}}}, {"$sort": {"_id": 1}}]))
    expected = acc.groupby("dataset").size().to_dict()
    ok = all(expected[d["_id"]] == d["n"] for d in by_ds)
    rec("accounts per dataset", 'account_scores.aggregate([{$group:{_id:"$dataset", n:{$sum:1}}}])',
        by_ds, f"matches pandas: {ok}")
    assert ok

    lvl_q = [{"$match": {"label": {"$ne": None}}},
             {"$group": {"_id": {"dataset": "$dataset", "risk_level": "$risk_level"},
                         "n": {"$sum": 1}, "n_inauthentic": {"$sum": "$label"},
                         "precision": {"$avg": "$label"}}},
             {"$sort": {"_id.dataset": 1, "_id.risk_level": 1}}]
    lvl = list(db.account_scores.aggregate(lvl_q))
    mism = []
    for d in lvl:
        t, k = d["_id"]["dataset"], d["_id"]["risk_level"]
        if not risk[t]["evidence_gate"]["passed"]:
            if k is not None or d["n"] != risk[t]["n_accounts"]:
                mism.append((t, k))  # a gated dataset must be stored entirely as null
            continue
        exp = risk[t]["per_level"][k]
        if d["n"] != exp["n_accounts"] or d["n_inauthentic"] != exp["n_inauthentic"]:
            mism.append((t, k))
    rec("risk-level table recomputed inside MongoDB",
        'account_scores.aggregate([{$match:{label:{$ne:null}}}, {$group:{_id:{dataset:"$dataset", '
        'risk_level:"$risk_level"}, n:{$sum:1}, n_inauthentic:{$sum:"$label"}, precision:{$avg:"$label"}}}])',
        [{"dataset": d["_id"]["dataset"], "risk_level": d["_id"]["risk_level"], "n": d["n"],
          "n_inauthentic": d["n_inauthentic"], "precision": round(d["precision"], 4)} for d in lvl],
        f"identical to the Python cross-fitted table: {not mism}")
    assert not mism, mism

    top = list(db.account_scores.find(
        {"dataset": "twitter_real", "risk_level": 2},
        {"_id": 0, "account_uid": 1, "rf_p_inauthentic": 1, "label": 1, "suggested_action": 1}
    ).sort("rf_p_inauthentic", -1).limit(5))
    rec("top-5 X review queue", 'account_scores.find({dataset:"twitter_real", risk_level:2}, '
        '{account_uid:1, rf_p_inauthentic:1, label:1, suggested_action:1}).sort({rf_p_inauthentic:-1}).limit(5)',
        top, f"5 rows returned: {len(top) == 5}")
    assert len(top) == 5

    tt = list(db.account_scores.aggregate([
        {"$match": {"dataset": "tiktok"}},
        {"$group": {"_id": "$risk_level_code", "n": {"$sum": 1}}}, {"$sort": {"_id": 1}}]))
    rec("TikTok levels (NA kept as NA, not 0)",
        'account_scores.aggregate([{$match:{dataset:"tiktok"}}, {$group:{_id:"$risk_level_code", n:{$sum:1}}}])',
        tt)
    n_null = db.account_scores.count_documents({"dataset": "tiktok", "risk_level": None})
    n_na = int(((acc["dataset"] == "tiktok") & (acc["risk_level_code"] == "NA")).sum())
    rec("TikTok authors stored with risk_level = null",
        'account_scores.count_documents({dataset:"tiktok", risk_level:null})', n_null,
        f"equals the {n_na} NA authors: {n_null == n_na}")
    assert n_null == n_na

    xm = db.model_metrics.find_one({"dataset": "twitter_real"},
                                   {"_id": 0, "models.random_forest.metrics.accuracy": 1,
                                    "models.random_forest.metrics.roc_auc": 1,
                                    "models.random_forest.metrics.baseline_majority_class": 1})
    rec("X RF metrics", 'model_metrics.find_one({dataset:"twitter_real"}, {"models.random_forest.metrics'
        '.accuracy":1, ...})', xm,
        f"accuracy == 0.8538: {xm['models']['random_forest']['metrics']['accuracy'] == 0.8538}")

    names = list(db.risk_level_definitions.find({}, {"_id": 0, "code": 1, "name_id": 1, "name_concept_doc": 1})
                 .sort("code", 1))
    both = all(n.get("name_id") for n in names) and \
        all(n.get("name_concept_doc") for n in names if n["code"] != "NA")
    rec("level names: this report vs concept doc section 13",
        "risk_level_definitions.find({}, {code:1, name_id:1, name_concept_doc:1}).sort({code:1})", names,
        f"both names on levels 0-4, NA has no section-13 name: {both}")
    assert both and len(names) == 6
    unreach = list(db.risk_level_definitions.find({"reachable": False}, {"_id": 0, "code": 1, "name_en": 1}))
    rec("unreachable levels", "risk_level_definitions.find({reachable:false}, {code:1, name_en:1})", unreach,
        f"codes {[u['code'] for u in unreach]}")

    topf = list(db.feature_importance.find(
        {"model": "random_forest", "importance_type": "permutation_test_roc_auc_drop", "rank": {"$lte": 3}},
        {"_id": 0, "dataset": 1, "feature": 1, "concept": 1, "value": 1, "rank": 1}).sort([("dataset", 1),
                                                                                           ("rank", 1)]))
    rec("top-3 permutation features per dataset",
        'feature_importance.find({model:"random_forest", importance_type:"permutation_test_roc_auc_drop", '
        'rank:{$lte:3}}).sort({dataset:1, rank:1})', topf)
    return q, counts


# ==========================================================================
# main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mongo", action="store_true", help="analysis + CSV only")
    ap.add_argument("--no-connector", action="store_true", help="load MongoDB with pymongo only")
    args = ap.parse_args()
    LOG_PATH.write_text("")
    use_connector = not args.skip_mongo and not args.no_connector
    if use_connector:
        # The connector jar is resolved from Maven Central when the JVM starts.
        os.environ["PYSPARK_SUBMIT_ARGS"] = f"--packages {MONGO_CONNECTOR} pyspark-shell"
    spark = common.get_spark("synthesis", driver_memory="2g", cores=2)
    log(f"Spark {spark.version} up (connector requested: {use_connector})")

    metrics, preds, norm = load_inputs()
    log("inputs: " + ", ".join(f"{t}={len(d)}" for t, d in norm.items()) + f", tiktok={len(preds['tiktok'])}")

    # ---- 1. concept comparison
    log("1. cross-platform concept comparison")
    concept_rows, frames = concept_comparison(spark, metrics)
    agreement = summarise_agreement(concept_rows)
    # How separable is each FILE on its own indicators? If the accuracy ranking simply follows this,
    # the ranking measures the files, not the platforms.
    separability = {}
    for t in LABELLED_REAL:
        ds = [abs(r["cliffs_delta"]) for r in concept_rows if r["dataset"] == t]
        separability[t] = {"n_indicators": len(ds), "mean_abs_cliffs_delta": r4(float(np.mean(ds))),
                           "max_abs_cliffs_delta": r4(max(ds)),
                           "n_indicators_abs_delta_ge_0_6": int(sum(d >= 0.6 for d in ds))}
    log(f"  separability: {separability}")
    imp = importance_by_concept(metrics)
    evidence = evidence_table(metrics)
    profiles = high_score_profiles(frames, norm)

    # ---- 2. risk levels
    log("2. risk levels")
    risk, assigned = risk_levels_labelled(norm)
    tt = preds["tiktok"]
    tt_lvl, tt_risk = risk_levels_tiktok(tt)
    log(f"  tiktok levels: {tt_risk['counts']}")

    # ---- figures
    fig_paths = {
        "concept_direction": common.OUT_FIGURES / "synthesis_concept_direction.png",
        "importance_by_concept": common.OUT_FIGURES / "synthesis_importance_by_concept.png",
        "risk_levels": common.OUT_FIGURES / "synthesis_risk_levels.png",
    }
    fig_concept_direction(concept_rows, fig_paths["concept_direction"])
    fig_importance(imp, fig_paths["importance_by_concept"])
    fig_risk_levels(norm, risk, fig_paths["risk_levels"])
    log("figures written")

    # ---- 3/4. unified table -> curated parquet, Power BI, MongoDB
    acc = unified_accounts(norm, assigned, tt, tt_lvl)
    schema, cols = account_schema()
    acc_sdf = spark.createDataFrame(to_records(acc, cols), schema=schema)
    acc_sdf.write.mode("overwrite").parquet(str(CURATED_ACCOUNTS))
    back = spark.read.parquet(str(CURATED_ACCOUNTS))
    assert back.count() == len(acc)
    log(f"curated zone: {CURATED_ACCOUNTS.relative_to(ROOT)} ({len(acc)} rows)")
    acc = acc[cols]

    side = build_side_tables(metrics, concept_rows, imp, risk, tt_risk, profiles)
    powerbi = write_powerbi(acc, side)
    log("power bi tables: " + ", ".join(f"{k}={v['rows']}" for k, v in powerbi.items()))

    payload = {
        "platform": "cross_platform",
        "purpose": "synthesis of the four platform tracks: concept comparison, risk levels 0-4, serving",
        "target_user": "trust & safety moderator triaging accounts for HUMAN review; never an automated "
                       "banning or sanction system",
        "inputs": {t: f"outputs/metrics/{t}.json" for t in TRACKS},
        "comparison": {
            "not_a_ranking": "Rows are ordered by evidential weight, NOT by accuracy. Ranking by accuracy "
                             "(Facebook > Instagram > X) would invert the truth: the Facebook number is an "
                             "artefact of a label-sorted two-block file, and X is the only result earned on "
                             "a large, real, independently collected dataset.",
            "evidence_table": evidence,
            "quantitative_comparison_supported_for": ["twitter_real", "instagram"],
            "facebook": "reported as a cautionary case only",
            "tiktok": "no supervised result exists",
            "indicators": concept_rows,
            "agreement": agreement,
            "file_separability": separability,
            "worksheet_q2_high_score_profiles": profiles,
            "importance_by_concept": imp,
            "concept_labels_id": CONCEPTS,
            "tiktok_status_by_indicator": TIKTOK_STATUS,
            "method": {
                "univariate_auc": "ROC-AUC of the raw indicator value as a score for the inauthentic class "
                                  "(Mann-Whitney; ties half credit), whole labelled dataset, descriptive "
                                  "only (nothing is tuned on it). 95% CI = stratified bootstrap, "
                                  f"{N_BOOT} resamples, seed 42. Cliff's delta = 2*AUC - 1.",
                "direction_rule": "a direction is claimed only when the 95% CI excludes 0.5",
                "importance_share": "RF impurity importance (train) and positive permutation ROC-AUC drops "
                                    "(test), summed per concept and normalised to 1 per dataset",
                "engine": "raw files read and indicators computed in Spark; bootstrap statistics in numpy "
                          "on the collected (<= 37,438-row) indicator table",
            },
        },
        "risk_levels": {
            "source": "concept document section 13 (risk level 0-4 above the raw model probability). Each "
                      "definition carries the verbatim section-13 category and description "
                      "(name_concept_doc, description_concept_doc) next to the name used here (name_id).",
            "renaming_rationale": "Section 13's names (e.g. 'Inauthentic / Spam', 'Authentic / Low Risk') "
                                  "assert a conclusion about an account. Section 15 states that model output is "
                                  "a data-based indication, not absolute proof of the owner's intent, and the "
                                  "target user is a moderator who decides. The names here therefore describe "
                                  "strength of evidence and the action it warrants. Levels, order and meaning "
                                  "are unchanged; only the labels differ.",
            "na_tier": "Not in section 13. Needed because missing is not zero: TikTok authors with no "
                       "assessable behaviour (6,483) and every row of a dataset whose model fails the "
                       "evidence gate (X synthetic, 9,872) cannot be put in Level 0 'Authentic / Low Risk'.",
            "definitions": LEVELS,
            "threshold_method": {
                "t1": f"maximises F{BETA_L1:g} (recall weighted {BETA_L1:g}x precision) on the calibration fold",
                "t2": f"maximises F{BETA_L2:g} (precision weighted {1 / BETA_L2:g}x recall) on the calibration "
                      "fold, constrained to t2 >= t1",
                "why_beta": "a policy choice, stated so it can be changed: Level 1 costs nothing to the "
                            "account (watch only), so it favours recall; Level 2 puts a person in front of "
                            "a moderator, so it favours precision",
                "cross_fitting": "stratified 2-fold, seed 42; each account is levelled by thresholds derived "
                                 "on the other half (contract rule 6: never tune on the test set)",
                "evidence_gate": "levels are issued for a dataset only if the Wilson 95% lower bound of "
                                 "cross-fitted Level >= 2 precision exceeds the base rate; otherwise every "
                                 "account is NA (the synthetic X negative control fails this gate)",
                "prevalence_adjustment": "precision at assumed prevalences uses the cross-fitted TPR/FPR; "
                                         "the prevalences are illustrative, real prevalence is unknown",
            },
            "labelled": risk,
            "tiktok": tt_risk,
            "unreachable_levels": {
                "levels": [3, 4],
                "reason": "Levels 3 and 4 need coordination evidence (shared timing, shared content, "
                          "network ties) and amplification evidence (cross-account bursts on a common "
                          "target). None of the five datasets contains it; assigning them would be "
                          "fabrication (concept document section 15).",
                "evidence_matrix": {t: {k: {"status": s, "note": n} for k, (s, n) in ev.items()}
                                    for t, ev in COORDINATION_EVIDENCE.items()},
                "data_needed": DATA_NEEDED_FOR_L3_L4,
            },
        },
        "figures": {k: str(v.relative_to(ROOT)) for k, v in fig_paths.items()},
        "curated_zone": {"account_scores_parquet": str(CURATED_ACCOUNTS.relative_to(ROOT)), "rows": len(acc)},
        "powerbi_exports": powerbi,
    }

    if not args.skip_mongo:
        log("3. MongoDB load + round-trip verification")
        summary_doc = {"_id": "cross_platform_summary",
                       "headline": payload["comparison"]["not_a_ranking"],
                       "target_user": payload["target_user"],
                       "agreement": agreement, "importance_by_concept": imp,
                       "risk_level_tables": {t: {"per_level": r["per_level"], "cumulative": r["cumulative"],
                                                 "deployment_thresholds": r["deployment_thresholds"]}
                                             for t, r in risk.items()},
                       "tiktok_levels": tt_risk["counts"]}
        client, db, write_path, redacted, dbname = load_mongo(spark, acc_sdf, side, metrics, summary_doc,
                                                              use_connector)
        log(f"  account_scores written via: {write_path}")
        queries, counts = verify_mongo(db, acc, risk)
        payload["mongodb"] = {
            "connection_uri": redacted, "database": dbname, "container": "socmed-mongo (mongo:7.0)",
            "compose_file": "deploy/mongodb/docker-compose.yml", "credentials_file": ".env.mongo (chmod 600)",
            "account_scores_write_path": write_path, "collection_counts": counts,
            "verification_queries": queries,
        }
        client.close()
    else:
        payload["mongodb"] = {"skipped": True}

    path = common.save_metrics(jsonable(payload), "cross_platform_comparison")
    log(f"wrote {path.relative_to(ROOT)}")
    spark.stop()


if __name__ == "__main__":
    main()
