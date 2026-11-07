"""TikTok track: descriptive, unsupervised behavioural profiling.

Project "Penyayang Claude", Teknologi Big Data, SV IPB University.

This track differs from the X / Instagram / Facebook tracks. The TikTok data
(`haziqhalifi/tiktok-2025-dataset`) has NO label, and this pipeline does not
invent one. So no classifier is trained and no accuracy is reported. The output is:

  1. answers to the data questions the concept document left open
     (views vs plays, create_time vs posted_time vs fetch_time, how many
     videos each author has);
  2. an author-level behavioural profile table, plus the distribution of every
     engineered feature;
  3. unsupervised anomaly INDICATORS for human review. They are not bot labels,
     and each one has an explicit evidential weight;
  4. a test of whether a model trained on the labelled platforms can be
     transferred to TikTok (concept document section 5).

Spark does ingestion, cleaning, the window and pairwise features, and the
author aggregation. The author table (~6.7k rows) is then small, so the null-model
simulation, the robust z-scores, the IsolationForest (Spark MLlib has none),
the figures and the CSV are done in pandas / numpy / scikit-learn.

Input: data/landing/tiktok_videos/tiktok_merged_data_deduplicated.csv (config.yaml
`unlabelled_datasets`, downloaded by scripts/download_datasets.py). Section 6 also
reads the labelled platforms' landing files and outputs/metrics/*.json if present.

Run from the repo root (Java 17 on PATH or JAVA_HOME set):
    python src/tiktok_pipeline.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TRACK = "tiktok"
RAW_PATH = common.landing_path("tiktok_videos") / "tiktok_merged_data_deduplicated.csv"
SEED = common.RANDOM_SEED

# ---------------------------------------------------------------------------
# Thresholds. Every one is reported in the metrics JSON.
# ---------------------------------------------------------------------------
MIN_VIDEOS_MEAN_INTERVAL = 2      # 1 interval -> mean interval / rate defined
MIN_VIDEOS_DISPERSION = 3         # >=2 intervals -> std, CV, burstiness defined
MIN_VIDEOS_REGULARITY_FLAG = 5    # >=4 intervals before a regularity flag is allowed
REGULARITY_P_THRESHOLD = 0.01     # one-sided, vs a random (Poisson) posting null
NULL_SIMULATIONS = 200_000
BURST_WINDOW_SECONDS = 3600
BURST_MIN_VIDEOS = 3              # >=3 videos inside one 60-minute window
CAPTION_MIN_TOKENS = 3            # captions shorter than this are not compared
CAPTION_DUP_JACCARD = 0.8         # near-duplicate caption threshold
ROBUST_Z_THRESHOLD = 3.5          # Iglewicz & Hoaglin modified z-score cut-off
IFOREST_PARAMS = {"n_estimators": 300, "max_samples": 256,
                  "contamination": "auto", "random_state": SEED}
AUTHOR_KEY_SALT = "penyayang-claude/tiktok/v1"

# Bare "original sound" titles, as TikTok localises them. Every original
# sound is a different audio clip that happens to share one generic title, so
# two videos with the same generic title are NOT evidence of reused audio.
GENERIC_SOUND_TITLES = sorted({
    "original sound", "sonido original", "оригинальный звук", "som original",
    "son original", "originalton", "suono originale", "orijinal ses",
    "no sound", "الصوت الأصلي", "оригінальний звук", "原聲", "原声",
    "origineel geluid", "nhạc nền", "dźwięk oryginalny", "sunet original",
    "originalljud", "suara asli", "オリジナル楽曲", "오리지널 사운드",
    "เสียงต้นฉบับ", "originální zvuk", "eredeti hang", "originallyd",
    "alkuperäinen ääni", "πρωτότυπος ήχος", "צליל מקורי", "orihinal na tunog",
})

# Figure styling (reference palette from the dataviz skill; blue/orange pair
# validated for CVD separation). Gray is a neutral "context" tone, not a slot.
C_BLUE, C_ORANGE = "#2a78d6", "#eb6834"
C_GRAY, C_GRAY_LIGHT = "#8a8984", "#d9d8d3"
C_TEXT, C_TEXT2, C_GRID, C_SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
ORDINAL = {"equivalent": "#1c5cab", "proxy_not_equivalent": "#86b6ef",
           "excluded_by_rule": "#b9b8b2", "constant_on_tiktok": "#d9d8d3",
           "missing": "#efeeea"}


def r4(x):
    """Round to 4 dp, NaN/None -> None (JSON null), numpy -> python."""
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    if math.isnan(xf) or math.isinf(xf):
        return None
    return round(xf, 4)


def author_key(author_norm: str) -> str:
    """Pseudonymous, deterministic author key (same formula as the Spark sha2 step).

    This is pseudonymisation, not anonymisation. Whoever holds the raw file can
    re-link a key to a handle. That is intended, because a human moderator has to
    open the account to review it. But the shared deliverables never print a handle.
    """
    return hashlib.sha256(f"{AUTHOR_KEY_SALT}:{author_norm}".encode()).hexdigest()[:16]


# ===========================================================================
# 1. Spark: ingest + clean video level
# ===========================================================================
def load_videos(spark):
    from pyspark.sql import functions as F

    raw = (spark.read.option("header", True).option("multiLine", True)
           .option("escape", '"').option("inferSchema", False)
           .csv(str(RAW_PATH)))

    generic = F.array(*[F.lit(s) for s in GENERIC_SOUND_TITLES])
    ts_fmt = "yyyy-MM-dd HH:mm:ss"
    desc = F.coalesce(F.col("description"), F.lit(""))

    col_tags = F.transform(F.split(F.coalesce(F.col("hashtags"), F.lit("")), ","),
                           lambda x: F.lower(F.trim(x)))
    desc_tags = F.transform(
        F.regexp_extract_all(desc, F.lit(r"(?U)#([^\s#]+)"), 1), lambda x: F.lower(x))
    tags = F.array_distinct(F.filter(
        F.transform(F.concat(col_tags, desc_tags),
                    lambda x: F.regexp_replace(x, r"[.,!?;:…\"'()\[\]]+$", "")),
        lambda x: x != ""))

    caption_text = F.regexp_replace(
        F.regexp_replace(F.lower(desc), r"https?://\S+|www\.\S+", " "),
        r"(?U)[#@][^\s#@]+", " ")
    tokens = F.array_distinct(F.filter(F.split(caption_text, r"(?U)\W+"),
                                       lambda x: F.length(x) > 0))
    music_l = F.lower(F.trim(F.coalesce(F.col("music"), F.lit(""))))

    v = raw.select(
        F.col("video_id"),
        F.col("author").alias("author_raw"),
        F.lower(F.trim(F.regexp_replace("author", r"^@", ""))).alias("author_norm"),
        desc.alias("description"),
        F.col("likes").cast("double").alias("likes"),
        F.col("comments").cast("double").alias("comments"),
        F.col("shares").cast("double").alias("shares"),
        F.col("plays").cast("double").alias("plays_raw"),
        F.col("views").cast("double").alias("views_raw"),
        F.to_timestamp("create_time", ts_fmt).alias("create_ts"),
        F.to_timestamp("posted_time", ts_fmt).alias("posted_ts"),
        F.to_timestamp("fetch_time", ts_fmt).alias("fetch_ts"),
        tags.alias("tags"),
        tokens.alias("tokens"),
        music_l.alias("music_l"),
        F.col("video_url"),
    )
    v = (v
         .withColumn("plays", F.coalesce("plays_raw", "views_raw"))
         .withColumn("plays_source", F.when(F.col("plays_raw").isNotNull(), "plays")
                     .when(F.col("views_raw").isNotNull(), "views"))
         .withColumn("posted_at", F.coalesce("create_ts", "posted_ts"))
         .withColumn("time_source", F.when(F.col("create_ts").isNotNull(), "create_time")
                     .when(F.col("posted_ts").isNotNull(), "posted_time"))
         .withColumn("ts", F.unix_timestamp("posted_at"))
         .withColumn("desc_len", F.length("description"))
         .withColumn("hashtag_count", F.size("tags"))
         .withColumn("mention_count",
                     F.size(F.regexp_extract_all("description", F.lit(r"(?U)@([\w.]+)"), 1)))
         .withColumn("has_url", F.col("description").rlike(r"https?://|www\."))
         .withColumn("generic_sound", (F.col("music_l") == "") |
                     F.array_contains(generic, F.col("music_l")))
         .withColumn("author_key", F.substring(
             F.sha2(F.concat(F.lit(AUTHOR_KEY_SALT + ":"), F.col("author_norm")), 256), 1, 16)))
    return raw, v


# ===========================================================================
# 2. Spark: resolve the open data questions (computed, not asserted)
# ===========================================================================
def data_questions(raw, v) -> dict:
    from pyspark.sql import functions as F

    n = v.count()
    c = v.agg(
        F.sum(F.col("plays_raw").isNotNull().cast("int")).alias("plays_present"),
        F.sum(F.col("views_raw").isNotNull().cast("int")).alias("views_present"),
        F.sum((F.col("plays_raw").isNotNull() & F.col("views_raw").isNotNull()).cast("int")).alias("both"),
        F.sum((F.col("plays_raw").isNull() & F.col("views_raw").isNull()).cast("int")).alias("neither"),
        F.sum(F.col("create_ts").isNotNull().cast("int")).alias("create_present"),
        F.sum(F.col("posted_ts").isNotNull().cast("int")).alias("posted_present"),
        F.sum((F.col("create_ts").isNotNull() & F.col("posted_ts").isNotNull()).cast("int")).alias("time_both"),
        F.sum((F.col("create_ts").isNull() & F.col("posted_ts").isNull()).cast("int")).alias("time_neither"),
        F.sum(F.col("fetch_ts").isNotNull().cast("int")).alias("fetch_present"),
        F.countDistinct("fetch_ts").alias("fetch_distinct"),
        F.sum((F.col("views_raw").isNotNull() & F.col("posted_ts").isNotNull()
               & F.col("fetch_ts").isNotNull()).cast("int")).alias("second_pass_rows"),
        F.sum(F.col("author_raw").startswith("@").cast("int")).alias("at_prefixed"),
        F.countDistinct("video_id").alias("distinct_video_id"),
        F.countDistinct("video_url").alias("distinct_url"),
        F.countDistinct("author_raw").alias("authors_raw"),
        F.countDistinct("author_norm").alias("authors_norm"),
        F.min("posted_at").alias("min_posted"), F.max("posted_at").alias("max_posted"),
        F.min("plays").alias("min_plays"), F.min("likes").alias("min_likes"),
        F.expr("percentile_approx(plays, 0.5)").alias("median_plays"),
        F.sum((F.col("likes") > F.col("plays")).cast("int")).alias("likes_gt_plays"),
        F.sum((F.col("comments") == 0).cast("int")).alias("zero_comment_videos"),
        F.sum(F.col("generic_sound").cast("int")).alias("generic_sound_videos"),
        F.sum((F.second("posted_at") == 0).cast("int")).alias("exact_minute_videos"),
        F.sum((F.length("description") == 0).cast("int")).alias("empty_descriptions"),
        F.sum(F.col("has_url").cast("int")).alias("desc_with_url"),
        F.sum((F.col("mention_count") > 0).cast("int")).alias("desc_with_mention"),
    ).first().asDict()

    # Does the second-pass (@-prefixed) author namespace collide with the first?
    second = v.where(F.col("author_raw").startswith("@")).select("author_norm").distinct()
    first = v.where(~F.col("author_raw").startswith("@")).select("author_norm").distinct()
    collisions = second.join(first, "author_norm").count()

    # URL consistency: handle and id inside the URL must match the columns.
    url_mismatch = v.where(
        (F.lower(F.regexp_extract("video_url", r"tiktok\.com/@([^/]+)/video", 1)) != F.col("author_norm"))
        | (F.regexp_extract("video_url", r"/video/(\d+)", 1) != F.col("video_id"))).count()

    # Display rounding: TikTok shows "44.8K" / "1.2M", and the scraper stored those.
    rounding = {}
    for col in ("likes", "comments", "shares", "plays"):
        r = v.agg(
            F.sum((F.col(col) >= 1e4).cast("int")).alias("n10k"),
            F.sum(((F.col(col) >= 1e4) & (F.col(col) % 100 == 0)).cast("int")).alias("n10k_r"),
            F.sum((F.col(col) >= 1e6).cast("int")).alias("n1m"),
            F.sum(((F.col(col) >= 1e6) & (F.col(col) % 1e5 == 0)).cast("int")).alias("n1m_r"),
        ).first()
        rounding[col] = {
            "n_ge_10k": r["n10k"],
            "share_ge_10k_multiple_of_100": r4(r["n10k_r"] / r["n10k"]) if r["n10k"] else None,
            "n_ge_1m": r["n1m"],
            "share_ge_1m_multiple_of_100k": r4(r["n1m_r"] / r["n1m"]) if r["n1m"] else None,
        }

    # Hashtag source agreement (hashtags column vs '#' parsed from description).
    tag_agree = raw.select(
        F.when(F.col("hashtags").isNull(), 0).otherwise(F.size(F.split("hashtags", ","))).alias("n_col"),
        F.size(F.regexp_extract_all(F.coalesce(F.col("description"), F.lit("")),
                                    F.lit(r"(?U)#([^\s#]+)"), 1)).alias("n_desc"),
    ).agg(
        F.sum(((F.col("n_col") == 0) & (F.col("n_desc") > 0)).cast("int")).alias("col_empty_desc_has"),
        F.sum(((F.col("n_col") > 0) & (F.col("n_desc") == 0)).cast("int")).alias("desc_empty_col_has"),
        F.avg((F.col("n_col") == F.col("n_desc")).cast("double")).alias("same_count_share"),
    ).first().asDict()

    # Videos per author.
    per_author = v.groupBy("author_norm").count()
    dist_rows = per_author.groupBy("count").agg(F.count("*").alias("authors")).orderBy("count").collect()
    dist = {int(r["count"]): int(r["authors"]) for r in dist_rows}
    n_auth = sum(dist.values())

    def authors_ge(k):
        return sum(a for vc, a in dist.items() if vc >= k)

    def videos_ge(k):
        return sum(vc * a for vc, a in dist.items() if vc >= k)

    daily = (v.groupBy(F.to_date("posted_at").alias("d")).count().orderBy("d").collect())
    daily_counts = {str(r["d"]): int(r["count"]) for r in daily}
    last9 = sorted(daily_counts)[-9:]
    share_last9 = sum(daily_counts[d] for d in last9) / n

    exact_minute_share = c["exact_minute_videos"] / n
    return {
        "n_rows": n,
        "video_id_unique": c["distinct_video_id"] == n,
        "video_url_unique": c["distinct_url"] == n,
        "url_matches_author_and_video_id": url_mismatch == 0,
        "views_vs_plays": {
            "plays_present": c["plays_present"], "views_present": c["views_present"],
            "both_present": c["both"], "neither_present": c["neither"],
            "plays_null_rate": r4(1 - c["plays_present"] / n),
            "views_null_rate": r4(1 - c["views_present"] / n),
            "finding": ("Mutually exclusive: no row has both, no row has neither. "
                        f"The {c['views_present']} 'views' rows are a separate scrape pass (they are "
                        "also the only rows with posted_time and fetch_time, and their author "
                        "handles carry an '@' prefix). They cannot be compared numerically "
                        "because there is no overlap. On TikTok both names refer to the public "
                        "play counter."),
            "decision": ("Coalesce into one 'plays' column = coalesce(plays, views). Keep the "
                         "provenance in 'plays_source'. This is an assumption, and it affects "
                         f"{c['views_present']} rows ({c['views_present'] / n:.2%})."),
        },
        "create_time_vs_posted_time": {
            "create_time_present": c["create_present"], "posted_time_present": c["posted_present"],
            "both_present": c["time_both"], "neither_present": c["time_neither"],
            "create_time_null_rate": r4(1 - c["create_present"] / n),
            "posted_time_null_rate": r4(1 - c["posted_present"] / n),
            "finding": ("Mutually exclusive, exactly the same split as views/plays: posted_time "
                        "is the second pass's name for the publish timestamp."),
            "decision": "Coalesce into 'posted_at'; provenance in 'time_source'.",
        },
        "fetch_time": {
            "present": c["fetch_present"], "null_rate": r4(1 - c["fetch_present"] / n),
            "distinct_values": c["fetch_distinct"],
            "finding": ("Present only on the second-pass rows, with a single value (one batch "
                        "timestamp). The scrape time of the other 99.9% of rows is unknown."),
            "consequence": ("Video age at scrape time cannot be computed. A video posted the day "
                            "before the scrape and a video posted six weeks before it have had very "
                            "different exposure times, so raw engagement counts are not "
                            "exposure-normalised. Ratios to plays are less affected, but not immune."),
        },
        "author_namespace": {
            "authors_raw_distinct": c["authors_raw"],
            "authors_after_normalisation": c["authors_norm"],
            "at_prefixed_rows": c["at_prefixed"],
            "second_pass_authors_also_in_first_pass": collisions,
            "decision": ("Strip a leading '@', trim and lower-case. Without this, one author "
                         "appears as two."),
        },
        "videos_per_author": {
            "n_authors": n_auth,
            "distribution": dist,
            "authors_with_1_video": dist.get(1, 0),
            "share_authors_with_1_video": r4(dist.get(1, 0) / n_auth),
            "authors_ge_2": authors_ge(2), "authors_ge_3": authors_ge(3),
            "authors_ge_5": authors_ge(5),
            "share_authors_ge_3": r4(authors_ge(3) / n_auth),
            "share_authors_ge_5": r4(authors_ge(5) / n_auth),
            "videos_by_authors_ge_3": videos_ge(3),
            "max_videos_one_author": max(dist),
            "consequence": (
                "Per-author timing is undefined for most of the data: a single video has "
                "no interval, and two videos give one interval with no spread. Interval "
                f"mean needs >= {MIN_VIDEOS_MEAN_INTERVAL} videos, interval std/CV needs >= "
                f"{MIN_VIDEOS_DISPERSION}, and a regularity flag is only raised at >= "
                f"{MIN_VIDEOS_REGULARITY_FLAG}. Below those thresholds the features are "
                "left null (NA). They are not imputed."),
        },
        "sampling_frame": {
            "posted_at_min": str(c["min_posted"]), "posted_at_max": str(c["max_posted"]),
            "min_plays": c["min_plays"], "median_plays": c["median_plays"], "min_likes": c["min_likes"],
            "share_of_videos_in_last_9_days": r4(share_last9),
            "daily_counts": daily_counts,
            "finding": ("Every video has at least ~141k plays. This is a sample of popular / "
                        "trending videos, not of TikTok accounts in general. Low-reach accounts, "
                        "where most spam and bot-farm activity would sit, are absent by "
                        "construction. Posting dates are heavily skewed towards the scrape date."),
            "consequence": ("An author's 'videos' here are their videos that went viral in the "
                            "window, not all their uploads. Posting-interval features describe the "
                            "gaps between captured viral videos, so they are a thinned view of "
                            "posting cadence."),
        },
        "count_precision": {
            "by_column": rounding,
            "finding": ("Counts are display-rounded (e.g. 44.8K -> 44800, 1.2M -> 1200000). A "
                        "value >= 1M is only known to about +/-50k (~4% at 1.2M)."),
            "consequence": ("Engagement ratios carry a few percent of rounding noise. Equal "
                            "counts across videos are not evidence of duplication."),
        },
        "likes_greater_than_plays": c["likes_gt_plays"],
        "hashtag_source": {
            **{k: r4(val) for k, val in tag_agree.items()},
            "decision": ("Use the per-video union of the 'hashtags' column and '#tags' parsed "
                         "from the description, lower-cased and de-duplicated. Neither source is "
                         "complete on its own."),
        },
        "music_titles": {
            "generic_original_sound_videos": c["generic_sound_videos"],
            "share": r4(c["generic_sound_videos"] / n),
            "generic_titles_list": GENERIC_SOUND_TITLES,
            "decision": (f"Bare titles like 'original sound' ({len(GENERIC_SOUND_TITLES) - 1} "
                         "localisations, plus 'no sound') are placeholders: each one is a "
                         "different audio clip. Music repetition is therefore computed only over "
                         "specific (named) tracks."),
        },
        "zero_comment_videos": {
            "n": c["zero_comment_videos"],
            "decision": ("0 comments on a video with >140k plays most plausibly means comments "
                         "were switched off. The comment rate treats those videos as unknown, "
                         "not as zero."),
        },
        "exact_minute_timestamps": {
            "n": c["exact_minute_videos"], "share": r4(exact_minute_share),
            "expected_share_if_uniform": r4(1 / 60),
            "finding": ("Slightly more videos are published at second :00 than chance predicts. "
                        "This is consistent with scheduled posting, which TikTok offers as a "
                        "native creator feature. It is reported at population level only and is "
                        "not used as an indicator."),
        },
        "descriptions": {
            "empty": c["empty_descriptions"], "empty_share": r4(c["empty_descriptions"] / n),
            "with_url": c["desc_with_url"], "with_mention": c["desc_with_mention"],
            "decision": ("An empty description is a real value (no caption written), so its "
                         "length is 0, not missing."),
        },
    }


# ===========================================================================
# 3. Spark: author-level aggregation
# ===========================================================================
def within_author_pairs(v):
    """Every unordered pair of one author's videos (self-join; at most C(16,2)=120 per author).

    cap_j_raw: Jaccard of caption word sets (hashtags / mentions / URLs removed);
    min_tokens: the shorter caption's word count, so a caption-length floor can
    be applied afterwards; tag_j: Jaccard of hashtag sets.
    """
    from pyspark.sql import functions as F

    a = v.select("author_norm", F.col("video_id").alias("va"), F.col("tokens").alias("ta"),
                 F.col("tags").alias("ga"))
    b = v.select("author_norm", F.col("video_id").alias("vb"), F.col("tokens").alias("tb"),
                 F.col("tags").alias("gb"))

    def jacc(x, y):
        return F.size(F.array_intersect(x, y)) / F.size(F.array_union(x, y))

    return (a.join(b, "author_norm").where(F.col("va") < F.col("vb"))
            .withColumn("min_tokens", F.least(F.size("ta"), F.size("tb")))
            .withColumn("cap_j_raw", F.when(F.col("min_tokens") >= 1, jacc("ta", "tb")))
            .withColumn("tag_j", F.when((F.size("ga") >= 1) & (F.size("gb") >= 1), jacc("ga", "gb"))))


def threshold_sensitivity_spark(v) -> dict:
    """How the burst and duplication flag counts move when their thresholds move.

    Without ground truth the thresholds are judgement calls. This grid shows the
    reader how much of each count is the data and how much is the analyst's choice.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    windows = {"10min": 600, "60min": 3600, "24h": 86400}
    vv = v
    for name, secs in windows.items():
        wr = Window.partitionBy("author_norm").orderBy("ts").rangeBetween(0, secs)
        vv = vv.withColumn(f"w_{name}", F.count("*").over(wr))
    per_author = vv.groupBy("author_norm").agg(
        F.count("*").alias("n"), *[F.max(f"w_{k}").alias(k) for k in windows]).toPandas()
    burst = {}
    for k in windows:
        for m in (2, 3, 4):
            elig = per_author.n >= m
            burst[f">={m}_videos_within_{k}"] = {"eligible": int(elig.sum()),
                                                  "flagged": int((elig & (per_author[k] >= m)).sum())}

    pairs = within_author_pairs(v)
    ks = (1, 2, 3, 5)
    pa = pairs.groupBy("author_norm").agg(
        *[F.max(F.when(F.col("min_tokens") >= k, F.col("cap_j_raw"))).alias(f"max_k{k}") for k in ks]
    ).toPandas()
    dup = {}
    for k in ks:
        col = pa[f"max_k{k}"]
        for j in (0.6, 0.8, 1.0):
            dup[f"min_words_{k}__jaccard_{j}"] = {"eligible": int(col.notna().sum()),
                                                  "flagged": int((col >= j).sum())}
    return {"burst": burst, "caption_duplication": dup,
            "defaults": {"burst": f">={BURST_MIN_VIDEOS}_videos_within_60min",
                         "caption_duplication": f"min_words_{CAPTION_MIN_TOKENS}__jaccard_{CAPTION_DUP_JACCARD}"}}


def author_features(v):
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    w = Window.partitionBy("author_norm").orderBy("ts", "video_id")
    wr = Window.partitionBy("author_norm").orderBy("ts").rangeBetween(0, BURST_WINDOW_SECONDS)
    v = (v.withColumn("gap_s", (F.col("ts") - F.lag("ts").over(w)).cast("double"))
         .withColumn("n_in_window", F.count("*").over(wr)))

    pairs = (within_author_pairs(v)
             .withColumn("cap_j", F.when(F.col("min_tokens") >= CAPTION_MIN_TOKENS, F.col("cap_j_raw"))))
    pair_agg = pairs.groupBy("author_norm").agg(
        F.count("cap_j").alias("caption_pairs_compared"),
        F.avg("cap_j").alias("description_similarity_mean"),
        F.max("cap_j").alias("description_similarity_max"),
        F.count("tag_j").alias("hashtag_pairs_compared"),
        F.avg("tag_j").alias("hashtag_set_jaccard_mean"),
    )

    named = ~F.col("generic_sound")
    with_plays = F.col("plays").isNotNull()
    with_comments = with_plays & (F.col("comments") > 0)
    agg = v.groupBy("author_norm", "author_key").agg(
        F.count("*").alias("video_count"),
        F.min("posted_at").alias("first_posted_at"),
        F.max("posted_at").alias("last_posted_at"),
        F.countDistinct(F.to_date("posted_at")).alias("active_days"),
        F.avg("likes").alias("avg_likes"),
        F.avg("comments").alias("avg_comments"),
        F.avg("shares").alias("avg_shares"),
        F.avg("plays").alias("avg_plays"),
        F.sum(F.when(with_plays, F.col("likes"))).alias("_sum_likes"),
        F.sum(F.when(with_plays, F.col("shares"))).alias("_sum_shares"),
        F.sum(F.when(with_plays, F.col("plays"))).alias("_sum_plays"),
        F.sum(F.when(with_comments, F.col("comments"))).alias("_sum_comments_nz"),
        F.sum(F.when(with_comments, F.col("plays"))).alias("_sum_plays_nz"),
        F.sum((F.col("comments") == 0).cast("int")).alias("zero_comment_videos"),
        F.avg("desc_len").alias("avg_description_length"),
        F.avg((F.col("desc_len") == 0).cast("double")).alias("empty_description_share"),
        F.avg("hashtag_count").alias("avg_hashtag_count"),
        F.sum("hashtag_count").alias("_total_tags"),
        F.size(F.array_distinct(F.flatten(F.collect_list("tags")))).alias("_distinct_tags"),
        F.avg("mention_count").alias("avg_mention_count"),
        F.avg(F.col("generic_sound").cast("double")).alias("generic_sound_share"),
        F.sum(named.cast("int")).alias("_named_sounds"),
        F.countDistinct(F.when(named, F.col("music_l"))).alias("_distinct_named_sounds"),
        F.count("gap_s").alias("n_intervals"),
        F.avg("gap_s").alias("_gap_mean_s"),
        F.stddev_samp("gap_s").alias("_gap_std_s"),
        F.min("gap_s").alias("_gap_min_s"),
        F.max("n_in_window").alias("max_videos_in_60min"),
        F.sum(F.col("plays_source").eqNullSafe("views").cast("int")).alias("videos_from_second_pass"),
    ).join(pair_agg, "author_norm", "left")

    n = F.col("video_count")
    span_days = (F.unix_timestamp("last_posted_at") - F.unix_timestamp("first_posted_at")) / 86400.0
    agg = (agg
           .withColumn("active_span_days", F.when(n >= MIN_VIDEOS_MEAN_INTERVAL, span_days))
           .withColumn("videos_per_day", F.when(n >= MIN_VIDEOS_MEAN_INTERVAL,
                                                n / F.greatest(span_days, F.lit(1.0))))
           .withColumn("avg_posting_interval_hours",
                       F.when(n >= MIN_VIDEOS_MEAN_INTERVAL, F.col("_gap_mean_s") / 3600))
           .withColumn("posting_interval_std_hours",
                       F.when(n >= MIN_VIDEOS_DISPERSION, F.col("_gap_std_s") / 3600))
           .withColumn("min_posting_interval_minutes",
                       F.when(n >= MIN_VIDEOS_MEAN_INTERVAL, F.col("_gap_min_s") / 60))
           .withColumn("posting_interval_cv",
                       F.when((n >= MIN_VIDEOS_DISPERSION) & (F.col("_gap_mean_s") > 0),
                              F.col("_gap_std_s") / F.col("_gap_mean_s")))
           .withColumn("burstiness_coefficient",
                       F.when(F.col("posting_interval_cv").isNotNull(),
                              (F.col("posting_interval_cv") - 1) / (F.col("posting_interval_cv") + 1)))
           .withColumn("max_videos_in_60min", F.when(n >= MIN_VIDEOS_MEAN_INTERVAL,
                                                     F.col("max_videos_in_60min")))
           .withColumn("like_rate", F.col("_sum_likes") / F.col("_sum_plays"))
           .withColumn("share_rate", F.col("_sum_shares") / F.col("_sum_plays"))
           .withColumn("comment_rate", F.col("_sum_comments_nz") / F.col("_sum_plays_nz"))
           .withColumn("hashtag_repetition",
                       F.when((n >= 2) & (F.col("_total_tags") >= 2),
                              1 - F.col("_distinct_tags") / F.col("_total_tags")))
           .withColumn("music_repetition",
                       F.when(F.col("_named_sounds") >= 2,
                              1 - F.col("_distinct_named_sounds") / F.col("_named_sounds")))
           .withColumn("caption_pairs_compared", F.coalesce("caption_pairs_compared", F.lit(0)))
           .withColumn("hashtag_pairs_compared", F.coalesce("hashtag_pairs_compared", F.lit(0)))
           .withColumn("named_sound_videos", F.col("_named_sounds")))
    drop = [c for c in agg.columns if c.startswith("_")]
    return agg.drop(*drop)


def cross_author_shared_captions(v) -> dict:
    """Identical non-empty captions used by more than one author.

    Descriptive only. A shared caption is NOT evidence of coordination: it is
    usually a generic tag string or a TikTok programme tag.
    """
    from pyspark.sql import functions as F

    d = v.where(F.length(F.trim("description")) > 0).withColumn("cap", F.lower(F.trim("description")))
    g = d.groupBy("cap").agg(F.countDistinct("author_norm").alias("n_authors")).where("n_authors > 1")
    rows = g.orderBy(F.desc("n_authors")).collect()
    hashtag_only = [r for r in rows if all(t.startswith("#") for t in r["cap"].split())]
    return {
        "distinct_captions_shared_by_2plus_authors": len(rows),
        "authors_involved_upper_bound": int(sum(r["n_authors"] for r in rows)),
        "of_which_hashtag_only": len(hashtag_only),
        "top_hashtag_only_examples": [{"caption": r["cap"], "n_authors": r["n_authors"]}
                                      for r in hashtag_only[:6]],
    }


# ===========================================================================
# 4. pandas: indicators
# ===========================================================================
def poisson_cv_null(max_n: int, rng) -> dict[int, np.ndarray]:
    """CV of (n-1) inter-event gaps under a homogeneous Poisson (random) process.

    Conditional on n events in a window, the inner gaps are exchangeable with
    i.i.d. exponential gaps up to a common scale. CV is scale-free, so this is
    the exact null and does not depend on the window length.
    """
    out = {}
    for n in range(MIN_VIDEOS_DISPERSION, max_n + 1):
        g = rng.exponential(size=(NULL_SIMULATIONS, n - 1))
        out[n] = np.sort(g.std(axis=1, ddof=1) / g.mean(axis=1))
    return out


def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median()
    return 0.6745 * (x - med) / mad if mad else x * np.nan


def tri_state(flag: pd.Series, eligible: pd.Series) -> pd.Series:
    """1 = fired, 0 = assessed and did not fire, NA = not assessable."""
    out = pd.Series(pd.NA, index=flag.index, dtype="Int64")
    out[eligible] = flag[eligible].astype(int)
    return out


def build_indicators(p: pd.DataFrame, rng) -> tuple[pd.DataFrame, dict]:
    p = p.copy()
    diag: dict = {}

    # --- 1. Mechanical posting regularity (primary signal) ------------------
    max_n = int(p.video_count.max())
    null = poisson_cv_null(max_n, rng)
    has_cv = p.posting_interval_cv.notna()

    def pval(row):
        arr = null[int(row.video_count)]
        return np.searchsorted(arr, row.posting_interval_cv, side="right") / len(arr)

    p["regularity_p_value"] = np.nan
    p.loc[has_cv, "regularity_p_value"] = p[has_cv].apply(pval, axis=1)
    elig_reg = has_cv & (p.video_count >= MIN_VIDEOS_REGULARITY_FLAG)
    p["ind_regular_posting"] = tri_state(p.regularity_p_value < REGULARITY_P_THRESHOLD, elig_reg)
    sens = has_cv & (p.regularity_p_value < REGULARITY_P_THRESHOLD)
    diag["regularity_null"] = {
        "null_model": "homogeneous Poisson posting (i.i.d. exponential gaps), simulated",
        "simulations_per_n": NULL_SIMULATIONS, "seed": SEED,
        "cv_cutoff_at_p_threshold_by_n": {
            int(n): r4(np.quantile(arr, REGULARITY_P_THRESHOLD)) for n, arr in null.items()
            if n <= 16},
        "null_median_cv_by_n": {int(n): r4(np.median(arr)) for n, arr in null.items() if n <= 16},
        "expected_false_alarms_if_all_eligible_were_random": r4(elig_reg.sum() * REGULARITY_P_THRESHOLD),
        "sensitivity_if_threshold_lowered_to_3_videos": {
            "eligible": int(has_cv.sum()), "would_flag": int(sens.sum()),
            "expected_false_alarms_if_random": r4(has_cv.sum() * REGULARITY_P_THRESHOLD)},
        "observed_cv_min_among_eligible": r4(p.loc[elig_reg, "posting_interval_cv"].min()),
        "observed_cv_median_among_eligible": r4(p.loc[elig_reg, "posting_interval_cv"].median()),
        "observed_burstiness_median_among_eligible": r4(p.loc[elig_reg, "burstiness_coefficient"].median()),
        "share_eligible_burstier_than_random": r4((p.loc[elig_reg, "burstiness_coefficient"] > 0).mean()),
    }

    # --- 2. Burst posting (moderate) ---------------------------------------
    elig_burst = p.video_count >= BURST_MIN_VIDEOS
    burst = (p.max_videos_in_60min >= BURST_MIN_VIDEOS)
    p["ind_burst_posting"] = tri_state(burst.fillna(False), elig_burst)
    diag["burst"] = {"authors_with_two_videos_at_identical_second":
                     int((p.min_posting_interval_minutes == 0).sum())}

    # --- 3. Content duplication (moderate) ----------------------------------
    elig_dup = p.caption_pairs_compared >= 1
    dup = p.description_similarity_max >= CAPTION_DUP_JACCARD
    p["ind_caption_duplication"] = tri_state(dup.fillna(False), elig_dup)

    # --- 4. Engagement-ratio outlier (weak) ----------------------------------
    for rate in ("like_rate", "comment_rate", "share_rate"):
        ok = p[rate].notna() & (p[rate] > 0)
        p[f"z_log_{rate}"] = np.nan
        p.loc[ok, f"z_log_{rate}"] = robust_z(np.log10(p.loc[ok, rate]))
    zcols = ["z_log_like_rate", "z_log_comment_rate", "z_log_share_rate"]
    zmax = p[zcols].abs().max(axis=1, skipna=True)
    elig_eng = p[zcols].notna().any(axis=1)
    p["ind_engagement_outlier"] = tri_state(zmax > ROBUST_Z_THRESHOLD, elig_eng)
    fired_eng = p.ind_engagement_outlier == 1
    diag["engagement"] = {
        "fired_by_ratio": {c.replace("z_log_", ""): {
            "low_tail": int((p.loc[fired_eng, c] < -ROBUST_Z_THRESHOLD).sum()),
            "high_tail": int((p.loc[fired_eng, c] > ROBUST_Z_THRESHOLD).sum())} for c in zcols},
        "share_of_flags_with_video_count_1": r4((fired_eng & (p.video_count == 1)).sum() / max(fired_eng.sum(), 1)),
        "share_of_all_authors_with_video_count_1": r4((p.video_count == 1).mean()),
        "note": ("Author rates are ratios of sums over the author's videos, so averaging several "
                 "videos pulls extremes towards the middle. Single-video authors are therefore "
                 "over-represented among outliers for purely statistical reasons."),
    }

    # --- threshold sensitivity (pandas part) --------------------------------
    sens_reg = {}
    for min_n in (3, 4, 5):
        for pthr in (0.05, 0.01, 0.001):
            el = has_cv & (p.video_count >= min_n)
            sens_reg[f"min_videos_{min_n}__p_{pthr}"] = {
                "eligible": int(el.sum()), "flagged": int((el & (p.regularity_p_value < pthr)).sum()),
                "expected_false_alarms_if_random": r4(el.sum() * pthr)}
    sens_eng = {f"robust_z_{t}": int((zmax > t).sum()) for t in (3.0, 3.5, 4.0, 5.0)}
    diag["threshold_sensitivity_pandas"] = {
        "regularity": sens_reg, "engagement_outlier_flagged": sens_eng,
        "defaults": {"regularity": f"min_videos_{MIN_VIDEOS_REGULARITY_FLAG}__p_{REGULARITY_P_THRESHOLD}",
                     "engagement": f"robust_z_{ROBUST_Z_THRESHOLD}"}}

    # --- 5. IsolationForest (context only) ----------------------------------
    from sklearn.ensemble import IsolationForest

    from scipy.stats import spearmanr

    if_feats = {
        "log10_avg_plays": np.log10(p.avg_plays),
        "log10_like_rate": np.log10(p.like_rate),
        "log10_share_rate": np.log10(p.share_rate),
        "log1p_avg_description_length": np.log1p(p.avg_description_length),
        "avg_hashtag_count": p.avg_hashtag_count,
        "avg_mention_count": p.avg_mention_count,
    }
    X = pd.DataFrame(if_feats)
    if_ok = X.notna().all(axis=1) & np.isfinite(X).all(axis=1)
    iso = IsolationForest(**IFOREST_PARAMS).fit(X[if_ok])
    p["iforest_anomaly_score"] = np.nan
    p.loc[if_ok, "iforest_anomaly_score"] = -iso.score_samples(X[if_ok])
    pred = pd.Series(False, index=p.index)
    pred[if_ok] = iso.predict(X[if_ok]) == -1
    p["ind_iforest_outlier"] = tri_state(pred, if_ok)
    # What drives the forest? |Spearman| of score with each input, and how far the
    # flagged authors sit from the rest on each input (median, in the input's own units).
    drivers = []
    for name, col in X[if_ok].items():
        rho = spearmanr(p.loc[if_ok, "iforest_anomaly_score"], col).statistic
        fl = pred[if_ok]
        drivers.append({"feature": name, "spearman_with_score": r4(rho),
                        "median_flagged": r4(col[fl].median()), "median_not_flagged": r4(col[~fl].median()),
                        "p95_flagged": r4(col[fl].quantile(.95)), "p95_not_flagged": r4(col[~fl].quantile(.95))})
    drivers.sort(key=lambda d: abs(d["spearman_with_score"] or 0), reverse=True)
    diag["iforest"] = {"features": list(if_feats), "params": IFOREST_PARAMS,
                       "n_scored": int(if_ok.sum()),
                       "drivers": drivers,
                       "why_not_generic_sound_share": ("near-binary for single-video authors (0 or 1); "
                                                       "IsolationForest splits on it trivially, and a "
                                                       "named vs generic sound is a style choice, not an anomaly"),
                       "flag_rate_note": ("contamination='auto' uses the original paper's fixed score "
                                          "offset, so the flagged share is a property of the score "
                                          "distribution, not an estimate of bot prevalence"),
                       "why_not_comment_rate": ("comment_rate is NA for authors whose comments look "
                                                "switched off. IsolationForest cannot take NA, and "
                                                "imputing would invent data, so it is left out."),
                       "why_not_timing": ("Timing features exist for <5% of authors. Including them "
                                          "would require imputing the other 95%.")}

    # --- review tier: a rule, not a summed score ---------------------------
    def fired(x) -> bool:  # NA (not assessable) is never "fired" and never "clear"
        return pd.notna(x) and int(x) == 1

    def tier(r):
        reg, bur, dupl = r.ind_regular_posting, r.ind_burst_posting, r.ind_caption_duplication
        weak = fired(r.ind_engagement_outlier) or fired(r.ind_iforest_outlier)
        if fired(reg) or (fired(bur) and fired(dupl)):
            return "priority_review"
        if fired(bur) or fired(dupl):
            return "secondary_context"
        behavioural_assessable = any(pd.notna(x) for x in (reg, bur, dupl))
        if weak:
            return "weak_signal_only"
        return "no_indicator_fired" if behavioural_assessable else "behaviour_not_assessable"

    p["review_tier"] = p.apply(tier, axis=1)
    p["behavioural_indicators_assessable"] = (
        p[["ind_regular_posting", "ind_burst_posting", "ind_caption_duplication"]].notna().sum(axis=1))

    # Percentiles (within authors where the feature is defined), for reviewers.
    for col in ("avg_plays", "like_rate", "comment_rate", "share_rate", "posting_interval_cv"):
        p[f"pct_{col}"] = p[col].rank(pct=True)
    return p, diag


# ===========================================================================
# 5. Descriptive statistics
# ===========================================================================
FEATURES = {
    # name: (formula, min videos for defined, group)
    "video_count": ("number of videos by the author in this sample (NOT the account's total uploads)", 1, "volume"),
    "active_days": ("distinct calendar dates with >= 1 captured video", 1, "volume"),
    "active_span_days": ("(last_posted_at - first_posted_at) in days", 2, "timing"),
    "videos_per_day": ("video_count / max(active_span_days, 1)", 2, "timing"),
    "avg_posting_interval_hours": ("mean gap between consecutive captured videos", 2, "timing"),
    "min_posting_interval_minutes": ("smallest gap between consecutive captured videos", 2, "timing"),
    "posting_interval_std_hours": ("sample std of gaps", 3, "timing"),
    "posting_interval_cv": ("std / mean of gaps (0 = clockwork, ~1 = random, >1 = bursty)", 3, "timing"),
    "burstiness_coefficient": ("(CV - 1) / (CV + 1), Goh & Barabasi 2008; -1 regular, 0 random, +1 bursty", 3, "timing"),
    "max_videos_in_60min": ("most captured videos inside any 60-minute window", 2, "timing"),
    "avg_likes": ("mean likes per video", 1, "engagement"),
    "avg_comments": ("mean comments per video (0s included as reported)", 1, "engagement"),
    "avg_shares": ("mean shares per video", 1, "engagement"),
    "avg_plays": ("mean plays per video (plays coalesced with views)", 1, "engagement"),
    "like_rate": ("sum(likes) / sum(plays)", 1, "engagement"),
    "comment_rate": ("sum(comments) / sum(plays) over videos with comments > 0", 1, "engagement"),
    "share_rate": ("sum(shares) / sum(plays)", 1, "engagement"),
    "avg_description_length": ("mean caption length in characters (empty caption = 0)", 1, "content"),
    "empty_description_share": ("share of videos with no caption", 1, "content"),
    "avg_hashtag_count": ("mean distinct hashtags per video (column UNION caption)", 1, "content"),
    "avg_mention_count": ("mean @mentions per caption", 1, "content"),
    "hashtag_repetition": ("1 - distinct_tags / total_tags across the author's videos", 2, "repetition"),
    "hashtag_set_jaccard_mean": ("mean pairwise Jaccard of hashtag sets between the author's videos", 2, "repetition"),
    "description_similarity_mean": (f"mean pairwise Jaccard of caption word sets (captions >= {CAPTION_MIN_TOKENS} words, hashtags/mentions/URLs removed)", 2, "repetition"),
    "description_similarity_max": ("max of the above", 2, "repetition"),
    "generic_sound_share": ("share of videos whose sound is a bare 'original sound' placeholder", 1, "repetition"),
    "music_repetition": ("1 - distinct_named_tracks / named_track_videos (named tracks only)", 2, "repetition"),
}


def describe(p: pd.DataFrame) -> dict:
    out = {}
    n = len(p)
    for f, (formula, min_v, group) in FEATURES.items():
        s = pd.to_numeric(p[f], errors="coerce")
        d = s.dropna()
        out[f] = {
            "group": group, "formula": formula, "defined_when_video_count_ge": min_v,
            "n_defined": int(d.size), "n_undefined_NA": int(n - d.size),
            "share_defined": r4(d.size / n),
        }
        if d.size:
            q = d.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            out[f].update({"mean": r4(d.mean()), "std": r4(d.std()) if d.size > 1 else None,
                           "min": r4(d.min()), "p05": r4(q[0.05]), "p25": r4(q[0.25]),
                           "median": r4(q[0.5]), "p75": r4(q[0.75]), "p95": r4(q[0.95]),
                           "max": r4(d.max())})
    return out


# ===========================================================================
# 6. Cross-platform transfer assessment
# ===========================================================================
# Status vocabulary:
#   equivalent           same concept, same unit of analysis
#   proxy_not_equivalent something with a similar name exists, but the concept,
#                        unit or population differs, so a model would misread it
#   excluded_by_rule     derivable, but forbidden by the project rules
#   constant_on_tiktok   defined, but has one value for every TikTok row
#   missing              not in the TikTok data at all
CROSS_PLATFORM_MAP = [
    # Instagram (train.csv)
    ("instagram", "profile pic", "has a profile picture", None, "missing", "no profile metadata scraped"),
    ("instagram", "nums/length username", "digit ratio of handle", "derivable from author handle", "excluded_by_rule",
     "author is the grouping key; the project brief forbids using it as a predictor"),
    ("instagram", "fullname words", "display-name word count", None, "missing", "no display name"),
    ("instagram", "nums/length fullname", "digit ratio of display name", None, "missing", "no display name"),
    ("instagram", "name==username", "display name equals handle", None, "missing", "no display name"),
    ("instagram", "description length", "profile BIO length", "avg_description_length", "proxy_not_equivalent",
     "TikTok field is a per-video caption, not the account bio"),
    ("instagram", "external URL", "bio has external link", None, "missing", "no bio; 11 of 7225 captions contain a URL"),
    ("instagram", "private", "account is private", None, "constant_on_tiktok", "every scraped video is public by construction"),
    ("instagram", "#posts", "account's total posts", "video_count", "proxy_not_equivalent",
     "TikTok count is viral videos captured in a 6-week window (95% = 1), not lifetime uploads"),
    ("instagram", "#followers", "follower count", None, "missing", "not in the data"),
    ("instagram", "#follows", "following count", None, "missing", "not in the data"),
    # Facebook (Facebook Spam Dataset.csv)
    ("facebook", "#friends", "friend count", None, "missing", "not in the data"),
    ("facebook", "#following", "following count", None, "missing", "not in the data"),
    ("facebook", "#community", "groups / communities joined", None, "missing", "not in the data"),
    ("facebook", "age", "account age", None, "missing", "no account creation date; video create_time is not account age"),
    ("facebook", "#postshared", "account's total posts", "video_count", "proxy_not_equivalent", "in-sample viral videos, not lifetime posts"),
    ("facebook", "#urlshared", "posts containing URLs", None, "missing", "11 of 7225 captions contain a URL: effectively constant"),
    ("facebook", "#photos/videos", "media posts", "video_count", "proxy_not_equivalent", "same problem as #postshared"),
    ("facebook", "fpurls", "fraction of posts with URLs", None, "constant_on_tiktok", "~0 for every author"),
    ("facebook", "fpphotos/videos", "fraction of posts with media", None, "constant_on_tiktok", "every TikTok post is a video (= 1)"),
    ("facebook", "avgcomment/post", "comments per post", "avg_comments", "proxy_not_equivalent",
     "same concept, but source median ~0.46 vs TikTok median ~640: a different population (viral videos)"),
    ("facebook", "likes/post", "likes per post", "avg_likes", "proxy_not_equivalent",
     "same concept, but source median ~0.88 vs TikTok median ~100k: no shared scale"),
    ("facebook", "tags/post", "people tagged per post", "avg_mention_count", "proxy_not_equivalent",
     "@mentions in captions are the nearest analogue; source median ~39 vs TikTok ~0"),
    ("facebook", "#tags/post", "hashtags per post", "avg_hashtag_count", "equivalent",
     "same concept and unit (per post); source median ~15.5 vs TikTok ~4: overlapping range"),
    # X / Twitter, synthetic Kaggle file (bot_detection_data.csv): one tweet per user.
    # Superseded as the X headline by the real Treiman dataset below; kept because the
    # concept document's schema was written against it.
    ("twitter_synthetic", "Retweet Count", "re-shares of the post", "avg_shares", "proxy_not_equivalent",
     "source range 0-100 (uniform) vs TikTok 13 - 2.6M"),
    ("twitter_synthetic", "Mention Count", "mentions in the post", "avg_mention_count", "equivalent",
     "same concept, per post; but the source is uniform 0-5, which looks generated, vs TikTok mostly 0-2"),
    ("twitter_synthetic", "Follower Count", "follower count", None, "missing", "not in the data"),
    ("twitter_synthetic", "Verified", "verified badge", None, "missing", "not in the data"),
    ("twitter_synthetic", "Location", "profile location", None, "missing", "not in the data"),
    ("twitter_synthetic", "Created At", "post timestamp", "posted_at", "equivalent",
     "both are post timestamps, but the source has one post per user, so no per-user timing feature exists there"),
    ("twitter_synthetic", "Hashtags", "hashtags in the post", "avg_hashtag_count", "equivalent",
     "same concept (count per post), but the X/Twitter track found the source hashtag vocabulary is "
     "exactly Faker's lorem word list, so the source values are not real behaviour"),
    ("twitter_synthetic", "Tweet", "post text", "description", "proxy_not_equivalent",
     "text exists on both, but the source text is Faker lorem words (X/Twitter track check)"),
    # X / Twitter, real accounts (danieltreiman/twitter-human-bots-dataset): the X headline.
    ("twitter_real", "followers_count", "follower count", None, "missing", "not in the data"),
    ("twitter_real", "friends_count", "following count", None, "missing", "not in the data"),
    ("twitter_real", "favourites_count", "likes the account has GIVEN", None, "missing",
     "TikTok 'likes' are likes RECEIVED by a video, the opposite direction, so they must not be mapped"),
    ("twitter_real", "follower_friend_ratio", "followers / following", None, "missing", "neither side exists"),
    ("twitter_real", "average_tweets_per_day", "lifetime posting rate", "videos_per_day", "proxy_not_equivalent",
     "TikTok rate covers captured viral videos over the observed span and is defined for 4.6% of authors"),
    ("twitter_real", "account_age_days", "account age", None, "missing", "no account creation date"),
    ("twitter_real", "verified", "verified badge", None, "missing", "not in the data"),
    ("twitter_real", "default_profile", "profile left at defaults", None, "missing", "no profile metadata"),
    ("twitter_real", "default_profile_image", "default avatar", None, "missing", "no profile metadata"),
    ("twitter_real", "geo_enabled", "location services on", None, "missing", "not in the data"),
    ("twitter_real", "description", "bio present / bio length", "avg_description_length", "proxy_not_equivalent",
     "TikTok field is a per-video caption, not the account bio"),
    ("twitter_real", "location", "profile location set", None, "missing", "not in the data"),
    ("twitter_real", "lang", "account language setting", None, "missing",
     "caption language could be detected, but that is not the account setting"),
]
# Which mapping block a sibling metrics file belongs to.
SIBLING_PLATFORM_BY_STEM = {"instagram": "instagram", "facebook": "facebook",
                            "twitter_real": "twitter_real", "twitter": "twitter_synthetic"}

# Features the labelled platforms are expected to lean on (bot-detection literature).
# They are checked against the sibling tracks' real feature importances when those exist.
PROFILE_SIGNAL_FEATURES = ["#followers", "#follows", "#friends", "#following", "age",
                           "profile pic", "Follower Count", "Verified", "#posts"]


def empirical_scales() -> dict:
    """Compare value ranges of nominally matched features (read-only on raw files)."""
    out = {}
    try:
        fb = pd.read_csv(common.landing_path("facebook_spam") / "Facebook Spam Dataset.csv")
        for col in ("avgcomment/post", "likes/post", "tags/post", "#tags/post", "#postshared"):
            s = pd.to_numeric(fb[col], errors="coerce").dropna()
            out[f"facebook::{col}"] = {"median": r4(s.median()), "p95": r4(s.quantile(.95)), "max": r4(s.max())}
    except Exception as e:  # noqa: BLE001
        out["facebook_error"] = str(e)
    try:
        tw = pd.read_csv(common.landing_path("twitter_bot_detection") / "bot_detection_data.csv",
                         usecols=["User ID", "Retweet Count", "Mention Count", "Hashtags"])
        out["twitter::rows_per_user_id"] = r4(len(tw) / tw["User ID"].nunique())
        for col in ("Retweet Count", "Mention Count"):
            s = tw[col]
            out[f"twitter::{col}"] = {"median": r4(s.median()), "p95": r4(s.quantile(.95)),
                                      "max": r4(s.max())}
        tags_n = tw["Hashtags"].fillna("").str.split().str.len()
        out["twitter::hashtag_tokens_per_post"] = {"median": r4(tags_n.median()), "max": r4(tags_n.max())}
    except Exception as e:  # noqa: BLE001
        out["twitter_error"] = str(e)
    try:
        ig = pd.read_csv(common.landing_path("instagram_fake_spammer") / "train.csv")
        for col in ("#posts", "description length"):
            s = ig[col]
            out[f"instagram::{col}"] = {"median": r4(s.median()), "p95": r4(s.quantile(.95)), "max": r4(s.max())}
    except Exception as e:  # noqa: BLE001
        out["instagram_error"] = str(e)
    return out


def _feature_key(name: str) -> str:
    """Normalise raw ('#followers', 'nums/length username') and engineered
    ('log1p_n_followers', 'nums_length_username') feature names to one key."""
    import re
    s = name.lower().strip()
    s = s.split("=", 1)[0] if re.match(r"^[a-z_]+=[^=]", s) else s  # one-hot level 'lang=en', not 'name==username'
    s = re.sub(r"^(log1p_|log10_|log_)", "", s)
    s = s.replace("#", "n_").replace("==", "_eq_").replace("_per_", "_")
    s = re.sub(r"[^a-z0-9]", "", s)
    aliases = {
        # Instagram
        "nameequalsusername": "nameequsername", "hasprofilepic": "profilepic",
        "hasexternalurl": "externalurl", "isprivate": "private",
        # Facebook (engineered names used by the IG/FB track)
        "nurlsshared": "nurlshared", "fracpostswithurls": "fpurls", "npostsshared": "npostshared",
        "fracpostswithphotosvideos": "fpphotosvideos", "avghashtagspost": "ntagspost",
        "avgcommentspost": "avgcommentpost", "avglikespost": "likespost", "avgtagspost": "tagspost",
        # X / Twitter (synthetic + real)
        "hashtagcount": "hashtags", "nhashtags": "hashtags", "isverified": "verified",
        "followerfriendratiolog": "followerfriendratio", "hasdescription": "description",
        "haslocation": "location", "langcat": "lang", "langonehot": "lang",
    }
    return aliases.get(s, s)


def sibling_importances(mapping: pd.DataFrame) -> dict:
    """Read other tracks' metrics, if present, and check whether TikTok has their top RF features."""
    found = {}
    status_by_feature = {(r.source_platform, _feature_key(r.source_feature)): r.tiktok_status
                         for r in mapping.itertuples()}
    any_platform = {_feature_key(r.source_feature): r.tiktok_status for r in mapping.itertuples()}
    for path in sorted(common.OUT_METRICS.glob("*.json")):
        if path.stem == TRACK:
            continue
        try:
            m = json.loads(path.read_text())
            if "faker" in path.stem and "verdict" in m:
                found[path.stem] = {"verdict": m["verdict"]}
                continue
            fi = (m.get("models") or {}).get("random_forest", {}).get("feature_importance", []) or []
            if not fi:
                continue  # not a model bundle (or not finished yet)
            top = []
            plat = SIBLING_PLATFORM_BY_STEM.get(path.stem, str(m.get("platform") or path.stem).lower())

            def status_of(name: str) -> str:
                k = _feature_key(name)
                return (status_by_feature.get((plat, k)) or any_platform.get(k)
                        or "not_in_raw_mapping (engineered by that track; see findings doc)")

            all_rows = [{"feature": str(r.get("feature")), "importance": float(r.get("importance") or 0),
                         "tiktok_status": status_of(str(r.get("feature")))} for r in fi]
            total = sum(abs(r["importance"]) for r in all_rows) or 1.0
            mass = {}
            for r in all_rows:
                mass[r["tiktok_status"]] = mass.get(r["tiktok_status"], 0.0) + abs(r["importance"]) / total
            top = [{**r, "importance": r4(r["importance"])} for r in all_rows[:5]]
            found[path.stem] = {
                "platform": m.get("platform"), "mapping_block": plat,
                "dataset": (m.get("dataset") or {}).get("name"),
                "rf_top5": top,
                "top5_with_tiktok_equivalent": sum(t["tiktok_status"] == "equivalent" for t in top),
                "rf_importance_share_by_tiktok_status": {k: r4(v) for k, v in
                                                         sorted(mass.items(), key=lambda kv: -kv[1])},
            }
        except Exception as e:  # noqa: BLE001
            found[path.stem] = {"error": str(e)}
    return found


def cross_platform() -> tuple[dict, pd.DataFrame]:
    m = pd.DataFrame(CROSS_PLATFORM_MAP, columns=["source_platform", "source_feature", "concept",
                                                 "tiktok_counterpart", "tiktok_status", "note"])
    coverage = {}
    for plat, g in m.groupby("source_platform"):
        coverage[plat] = {
            "n_source_features": int(len(g)),
            **{k: int(v) for k, v in g.tiktok_status.value_counts().items()},
            "share_equivalent": r4((g.tiktok_status == "equivalent").mean()),
        }
    sib = sibling_importances(m)
    verdict = (
        "Transfer of a trained model to TikTok is NOT defensible with these data. "
        "Instagram: 0 of 11 features have a TikTok equivalent. Facebook: 1 of 13 (hashtags "
        "per post). X/Twitter, real accounts (the X headline dataset): 0 of 13. The superseded "
        "synthetic X file has 3 of 8 (mentions, hashtags, post time), but the X/Twitter track "
        "showed its text and hashtags are Faker-generated, so those matches carry no real "
        "behaviour. The profile and network features that the labelled models actually rank "
        "highest (followers, following/friends, favourites, community, account age, profile "
        "completeness, verified) are entirely absent from TikTok. The few nominal matches sit "
        "on scales up to five orders of magnitude apart, because the TikTok sample contains "
        "only viral videos. Above all, TikTok has no labels, so a transferred score could "
        "never be validated: its precision on TikTok would be unknown, and the 'audit "
        "suspiciously high scores' rule could not be applied.")
    narrow = {
        "considered": ("A 'narrow transfer' on the features TikTok shares with a labelled source: "
                       "hashtags per post (Facebook, real) plus mentions and hashtags per post "
                       "(synthetic X only)."),
        "rejected_because": [
            "Two content-count features cannot carry a bot model: on the labelled data they "
            "are not where the signal is (profile and network features are).",
            "The labelled sources disagree with each other on even these features: Facebook "
            "'tags/post' median ~39 vs synthetic-X mentions 0-5.",
            "Within-platform rank normalisation would align the scales, but it assumes a "
            "bot's percentile means the same thing on every platform. That cannot be tested "
            "without TikTok labels.",
            "No TikTok ground truth means no way to measure the transferred model, and an "
            "unmeasured classifier must not feed a moderation queue.",
        ],
    }
    needed = [
        "Account-level metadata per author: follower_count, following_count, total video "
        "count, account creation date (account age), verified flag, bio text, avatar present.",
        "A population sample that is not popularity-filtered, e.g. accounts reached via "
        "hashtag/search crawl, including low-reach accounts.",
        "A labelled TikTok subset built by documented human annotation (written criteria, "
        "two annotators, inter-annotator agreement reported), or an external ground-truth "
        "source such as platform takedown lists, with provenance stated. Never labels "
        "inferred from the indicators in this track, which would be circular.",
        "Per-video scrape timestamps, so engagement can be normalised by exposure time.",
    ]
    records = [{k: (val if isinstance(val, str) else None) for k, val in rec.items()}
               for rec in m.to_dict(orient="records")]  # pandas-3 NaN -> JSON null
    out = {
        "feature_mapping": records,
        "coverage_by_platform": coverage,
        "empirical_scale_comparison": empirical_scales(),
        "sibling_track_top_features": sib or "no sibling metrics available at run time",
        "profile_and_network_features_missing_on_tiktok": PROFILE_SIGNAL_FEATURES,
        "verdict": verdict,
        "narrow_transfer": narrow,
        "what_transferable": ("The method transfers: author-level aggregation, explicit NA "
                              "handling, indicator-not-label framing, and the human-review "
                              "output. The trained models do not."),
        "what_would_be_needed": needed,
    }
    return out, m


# ===========================================================================
# 7. Figures
# ===========================================================================
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": C_SURFACE, "axes.facecolor": C_SURFACE, "savefig.facecolor": C_SURFACE,
        "axes.edgecolor": C_GRID, "axes.linewidth": 0.8, "axes.grid": True,
        "grid.color": C_GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "text.color": C_TEXT, "axes.labelcolor": C_TEXT2, "xtick.color": C_TEXT2,
        "ytick.color": C_TEXT2, "axes.titlesize": 10, "axes.titleweight": "regular",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "font.family": "DejaVu Sans", "legend.frameon": False, "legend.fontsize": 8,
    })
    return plt


def fig_videos_per_author(dq: dict, path: Path):
    plt = _style()
    dist = dq["videos_per_author"]["distribution"]
    ks, vs = list(dist.keys()), list(dist.values())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(ks, vs, width=0.6, color=C_BLUE)
    ax.set_yscale("log")
    ax.set_xlabel("captured videos per author")
    ax.set_ylabel("authors (log scale)")
    ax.set_xticks(ks)
    for k, val in zip(ks, vs):
        ax.text(k, val * 1.15, f"{val:,}", ha="center", va="bottom", fontsize=7, color=C_TEXT2)
    n_auth = dq["videos_per_author"]["n_authors"]
    lines = []
    for thr, lab in ((MIN_VIDEOS_MEAN_INTERVAL, "mean posting interval defined"),
                     (MIN_VIDEOS_DISPERSION, "interval spread (std, CV) defined"),
                     (MIN_VIDEOS_REGULARITY_FLAG, "regularity flag allowed")):
        n_ge = sum(v for k, v in dist.items() if k >= thr)
        ax.axvline(thr - 0.5, color=C_GRAY, linewidth=0.8)
        ax.text(thr - 0.45, 20000, f">= {thr}", fontsize=7.5, color=C_TEXT2, va="top")
        lines.append(f">= {thr} videos: {n_ge:,} authors ({n_ge / n_auth:.1%}), {lab}")
    ax.text(0.98, 0.80, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=C_TEXT2, linespacing=1.6)
    ax.set_ylim(0.7, 30000)
    ax.set_title(f"{dist.get(1, 0) / n_auth:.1%} of {n_auth:,} authors have exactly one video: "
                 "per-author timing is undefined for them", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_feature_distributions(p: pd.DataFrame, path: Path):
    plt = _style()
    logx = {"avg_likes", "avg_comments", "avg_shares", "avg_plays", "like_rate", "comment_rate",
            "share_rate", "avg_posting_interval_hours", "posting_interval_std_hours",
            "min_posting_interval_minutes", "avg_description_length", "videos_per_day",
            "active_span_days"}
    feats = [f for f in FEATURES if f not in ("video_count",)]
    ncol = 5
    nrow = math.ceil(len(feats) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.5 * nrow))
    n = len(p)
    for ax, f in zip(axes.flat, feats):
        s = pd.to_numeric(p[f], errors="coerce").dropna()
        if f in logx:
            s = s[s > 0]
            bins = np.logspace(np.log10(s.min()), np.log10(s.max()), 30) if len(s) > 1 else 10
            ax.set_xscale("log")
        else:
            bins = 30
        ax.hist(s, bins=bins, color=C_BLUE, edgecolor=C_SURFACE, linewidth=0.5)
        share = len(pd.to_numeric(p[f], errors="coerce").dropna()) / n
        ax.set_title(f"{f}\ndefined for {share:.1%} of authors", fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(feats):]:
        ax.axis("off")
    fig.suptitle("Author-level feature distributions (NA = undefined, not plotted; never imputed)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


INDICATOR_META = {
    "ind_regular_posting": ("Mechanical posting regularity", "primary"),
    "ind_burst_posting": ("Burst posting (>=3 in 60 min)", "moderate"),
    "ind_caption_duplication": ("Near-duplicate captions", "moderate"),
    "ind_engagement_outlier": ("Engagement-ratio outlier", "weak"),
    "ind_iforest_outlier": ("IsolationForest outlier", "context only"),
}


def fig_indicator_coverage(p: pd.DataFrame, path: Path):
    plt = _style()
    rows = list(INDICATOR_META)
    n = len(p)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, col in enumerate(rows):
        s = p[col]
        fired = int((s == 1).sum())
        clear = int((s == 0).sum())
        na = int(s.isna().sum())
        left = 0
        for val, colr in ((fired, C_ORANGE), (clear, C_BLUE), (na, C_GRAY_LIGHT)):
            ax.barh(i, val, left=left, height=0.55, color=colr, edgecolor=C_SURFACE, linewidth=1.5)
            left += val
        name, w = INDICATOR_META[col]
        ax.text(n * 1.01, i, f"fired {fired:,} · assessed {fired + clear:,} · NA {na:,}",
                va="center", fontsize=7.5, color=C_TEXT2)
        rows[i] = f"{name}\n[{w}]"
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, n * 1.42)
    ax.set_xticks([0, 2000, 4000, 6000, n])
    ax.set_xticklabels(["0", "2,000", "4,000", "6,000", f"{n:,}"])
    ax.set_xlabel(f"authors (N = {n:,})")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_ORANGE, label="fired"),
                       Patch(color=C_BLUE, label="assessed, did not fire"),
                       Patch(color=C_GRAY_LIGHT, label="not assessable (NA)")],
              loc="lower right", ncol=3, bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Who can each indicator even assess? The strongest signal reaches the fewest authors",
                 loc="left", pad=22)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_regularity_null(p: pd.DataFrame, diag: dict, path: Path):
    plt = _style()
    cut = diag["regularity_null"]["cv_cutoff_at_p_threshold_by_n"]
    med = diag["regularity_null"]["null_median_cv_by_n"]
    d = p[p.posting_interval_cv.notna()]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ns = sorted(int(k) for k in cut)
    ax.plot(ns, [med[n] for n in ns], color=C_GRAY, linewidth=2, label="random-posting null: median CV")
    ax.plot(ns, [cut[n] for n in ns], color=C_ORANGE, linewidth=2,
            label=f"random-posting null: p = {REGULARITY_P_THRESHOLD} cut-off (below = 'regular')")
    rng = np.random.default_rng(SEED)
    jit = rng.uniform(-0.18, 0.18, len(d))
    eligible = d.video_count >= MIN_VIDEOS_REGULARITY_FLAG
    ax.scatter(d.video_count[~eligible] + jit[~eligible], d.posting_interval_cv[~eligible], s=22,
               color=C_GRAY_LIGHT, edgecolor=C_SURFACE, linewidth=1,
               label=f"authors with {MIN_VIDEOS_DISPERSION}-{MIN_VIDEOS_REGULARITY_FLAG - 1} videos (computed, too few to flag)")
    ax.scatter(d.video_count[eligible] + jit[eligible], d.posting_interval_cv[eligible], s=30,
               color=C_BLUE, edgecolor=C_SURFACE, linewidth=1,
               label=f"authors with >= {MIN_VIDEOS_REGULARITY_FLAG} videos (eligible)")
    ax.axvline(MIN_VIDEOS_REGULARITY_FLAG - 0.5, color=C_GRID, linewidth=1)
    ax.set_xlabel("captured videos per author")
    ax.set_ylabel("posting-interval CV (std / mean)")
    ax.set_xticks(ns)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2, fontsize=7)
    nflag = int((p.ind_regular_posting == 1).sum())
    nelig = int(p.ind_regular_posting.notna().sum())
    ax.set_title(f"Posting regularity vs a random-posting null: {nflag} of {nelig} eligible authors "
                 "fall below the cut-off", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_engagement(p: pd.DataFrame, path: Path):
    plt = _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    d = p[p.like_rate.notna()]
    fl = d.ind_engagement_outlier == 1
    ax1.scatter(d.avg_plays[~fl], d.like_rate[~fl], s=6, color=C_GRAY_LIGHT, linewidth=0,
                label="not flagged")
    ax1.scatter(d.avg_plays[fl], d.like_rate[fl], s=16, color=C_ORANGE, edgecolor=C_SURFACE,
                linewidth=0.8, label="engagement-ratio outlier (any of like/comment/share)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("average plays per video (log)")
    ax1.set_ylabel("like rate = likes / plays (log)")
    ax1.legend(loc="lower left", fontsize=7)
    ax1.set_title("Like rate vs reach, per author", loc="left")
    dec = pd.qcut(d.avg_plays.rank(method="first"), 10, labels=False)
    rate = d.groupby(dec).ind_engagement_outlier.apply(lambda s: (s == 1).mean())
    ax2.bar(rate.index + 1, rate.values * 100, width=0.6, color=C_BLUE)
    ax2.set_xticks(range(1, 11))
    ax2.set_xlabel("reach decile (1 = least plays, 10 = most)")
    ax2.set_ylabel("% of authors flagged")
    ax2.set_title("Flag rate by reach decile", loc="left")
    for x, y in zip(rate.index + 1, rate.values * 100):
        ax2.text(x, y + 0.1, f"{y:.1f}", ha="center", fontsize=7, color=C_TEXT2)
    fig.suptitle("Engagement outliers are the weakest indicator: a viral human looks like an outlier too",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_cross_platform(m: pd.DataFrame, path: Path):
    plt = _style()
    plats = [("instagram", "Instagram"), ("facebook", "Facebook"),
             ("twitter_real", "X real (Treiman, X headline)"),
             ("twitter_synthetic", "X synthetic (Kaggle, Faker)")]
    fig, axes = plt.subplots(1, 4, figsize=(21, 6.2))
    label = {"equivalent": "equivalent", "proxy_not_equivalent": "proxy (not equivalent)",
             "excluded_by_rule": "excluded by rule", "constant_on_tiktok": "constant on TikTok",
             "missing": "missing"}
    for ax, (key, title) in zip(axes, plats):
        g = m[m.source_platform == key].reset_index(drop=True)
        for i, r in g.iterrows():
            ax.add_patch(plt.Rectangle((0, i), 1, 0.9, color=ORDINAL[r.tiktok_status], linewidth=0))
            ink = "#ffffff" if r.tiktok_status == "equivalent" else C_TEXT
            ax.text(0.03, i + 0.45, r.source_feature, va="center", fontsize=8.5, color=ink, weight="bold")
            tk = r.tiktok_counterpart if isinstance(r.tiktok_counterpart, str) else "no counterpart"
            ax.text(0.97, i + 0.45, f"{label[r.tiktok_status]}\n{tk}", va="center", ha="right",
                    fontsize=7, color=ink)
        ax.set_xlim(0, 1)
        ax.set_ylim(len(g), -0.1)
        ax.axis("off")
        n_eq = int((g.tiktok_status == "equivalent").sum())
        ax.set_title(f"{title}\n{n_eq} of {len(g)} features have a TikTok equivalent", loc="left", fontsize=10)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=ORDINAL[k], label=label[k]) for k in ORDINAL],
               loc="lower left", ncol=5, bbox_to_anchor=(0.01, 0.0))
    fig.suptitle("Labelled-platform features vs what the TikTok data actually contains",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ===========================================================================
# 8. Main
# ===========================================================================
def main():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(SEED)
    # cores=3 as in the reference run (the old analysis default), so partitioning is unchanged.
    spark = common.get_spark("tiktok_pipeline", driver_memory="2g", cores=3)
    spark.conf.set("spark.sql.session.timeZone", "UTC")  # treat naive timestamps literally, no DST shifts
    try:
        raw, v = load_videos(spark)
        v = v.cache()
        dq = data_questions(raw, v)
        assert dq["n_rows"] == 7225, f"unexpected row count {dq['n_rows']} (CSV parsing?)"
        shared = cross_author_shared_captions(v)
        sens_spark = threshold_sensitivity_spark(v)
        authors = author_features(v)
        p = authors.toPandas()

        # Curated layer for the group (local only; contains handles, so it lives in data/).
        (v.select("video_id", "author_key", "posted_at", "time_source", "likes", "comments",
                  "shares", "plays", "plays_source", "desc_len", "hashtag_count", "mention_count",
                  "generic_sound")
         .write.mode("overwrite").parquet(str(common.DATA_CURATED / "tiktok_videos_clean.parquet")))
        (authors.select("author_key", "author_norm").toPandas()
         .to_csv(common.DATA_CURATED / "tiktok_author_key_map.csv", index=False))
    finally:
        spark.stop()

    # Check that the Spark sha2 key matches the documented python function.
    probe = p.iloc[0]
    assert author_key(probe.author_norm) == probe.author_key, "author_key mismatch"
    p = p.drop(columns=["author_norm"]).sort_values("author_key").reset_index(drop=True)

    p, diag = build_indicators(p, rng)
    stats = describe(p)

    # ---- audits ----------------------------------------------------------
    n_auth = len(p)
    ind_cols = list(INDICATOR_META)
    eng_fl = p.ind_engagement_outlier == 1
    if_fl = p.ind_iforest_outlier == 1
    top_decile = p.avg_plays >= p.avg_plays.quantile(0.9)
    log_plays = np.log10(p.avg_plays)
    ok = p.iforest_anomaly_score.notna()
    rho_if, _ = spearmanr(p.loc[ok, "iforest_anomaly_score"], log_plays[ok])
    zl = p.z_log_like_rate.notna()
    rho_like, _ = spearmanr(p.loc[zl, "like_rate"], log_plays[zl])
    popularity_audit = {
        "spearman_like_rate_vs_log_avg_plays": r4(rho_like),
        "spearman_iforest_score_vs_log_avg_plays": r4(rho_if),
        "share_of_engagement_flags_in_top_reach_decile": r4((eng_fl & top_decile).sum() / max(eng_fl.sum(), 1)),
        "share_of_iforest_flags_in_top_reach_decile": r4((if_fl & top_decile).sum() / max(if_fl.sum(), 1)),
        "expected_share_if_unrelated_to_reach": 0.1,
        "engagement_and_iforest_overlap_jaccard": r4((eng_fl & if_fl).sum() / max((eng_fl | if_fl).sum(), 1)),
        "share_of_iforest_flags_with_video_count_1": r4((if_fl & (p.video_count == 1)).sum() / max(if_fl.sum(), 1)),
    }
    decile = pd.qcut(p.avg_plays.rank(method="first"), 10, labels=False) + 1
    popularity_audit["engagement_flag_rate_by_reach_decile"] = {
        int(d): r4(r) for d, r in eng_fl.groupby(decile).mean().items()}
    popularity_audit["iforest_flag_rate_by_reach_decile"] = {
        int(d): r4(r) for d, r in if_fl.groupby(decile).mean().items()}
    overlap = {a: {b: int(((p[a] == 1) & (p[b] == 1)).sum()) for b in ind_cols} for a in ind_cols}

    indicators = {}
    meta = {
        "ind_regular_posting": {
            "measures": ("posting-interval CV significantly below what random posting produces "
                         f"(one-sided p < {REGULARITY_P_THRESHOLD} under a simulated Poisson null)"),
            "eligible_rule": f"video_count >= {MIN_VIDEOS_REGULARITY_FLAG}",
            "evidential_weight": "primary",
            "why": ("Clockwork timing is the one behaviour in this data that points at automation "
                    "specifically and not at popularity."),
            "benign_explanations": ["TikTok's native post scheduler", "social-media managers / brand accounts",
                                    "disciplined creators with a fixed daily slot"],
            "blind_spots": ("Only viral videos are captured, so a perfectly periodic poster whose "
                            "videos go viral at random shows irregular gaps. A non-flag is NOT "
                            "evidence of human posting."),
        },
        "ind_burst_posting": {
            "measures": f">= {BURST_MIN_VIDEOS} captured videos inside one {BURST_WINDOW_SECONDS // 60}-minute window",
            "eligible_rule": f"video_count >= {BURST_MIN_VIDEOS}",
            "evidential_weight": "moderate",
            "why": "Rapid-fire uploading is a common automation footprint.",
            "benign_explanations": ["batch-uploading a multi-part series", "re-uploading after an edit",
                                    "event / live coverage"],
            "blind_spots": "Only bursts in which several videos went viral are visible.",
        },
        "ind_caption_duplication": {
            "measures": (f"two of the author's captions (each >= {CAPTION_MIN_TOKENS} words, hashtags "
                         f"removed) with word-set Jaccard >= {CAPTION_DUP_JACCARD}"),
            "eligible_rule": f">= 2 videos with captions of >= {CAPTION_MIN_TOKENS} words",
            "evidential_weight": "moderate",
            "why": "Copy-paste captions are a template / content-farm footprint.",
            "benign_explanations": ["numbered series ('part 18 ...')", "creator's fixed sign-off or call to action",
                                    "brand campaign copy"],
            "blind_spots": "Within-author only. Cross-account copying is not assessed (see coordination).",
        },
        "ind_engagement_outlier": {
            "measures": (f"|robust z| > {ROBUST_Z_THRESHOLD} on log like-, comment- or share-rate vs "
                         "all authors"),
            "eligible_rule": "plays known (all authors; comment rate NA when comments look disabled)",
            "evidential_weight": "weak",
            "why": ("Bought likes or views can distort ratios, but so can virality, niche "
                    "audiences, giveaways and comment-bait. A high like rate is as likely to be a "
                    "popular human."),
            "benign_explanations": ["viral reach to casual viewers lowers like rate",
                                    "loyal fan base raises it", "giveaway / 'comment to win' posts",
                                    "display-rounding at small counts"],
            "blind_spots": "Two-sided. The direction alone means nothing.",
        },
        "ind_iforest_outlier": {
            "measures": "IsolationForest (contamination='auto') on engagement and caption features",
            "eligible_rule": "all authors with finite inputs",
            "evidential_weight": "context only",
            "why": ("Shown so the group can see what a naive multivariate anomaly model finds: "
                    "mostly scale and caption-style extremes, not automation."),
            "benign_explanations": ["very viral or very niche authors", "long-caption storytellers",
                                    "hashtag-heavy but human posting"],
            "blind_spots": "Cannot see timing (undefined for 95% of authors).",
        },
    }
    for col, m in meta.items():
        s = p[col]
        fired, assessed = int((s == 1).sum()), int(s.notna().sum())
        indicators[col] = {
            **m,
            "n_assessed": assessed, "n_not_assessable_NA": int(n_auth - assessed),
            "share_of_authors_assessable": r4(assessed / n_auth),
            "n_fired": fired,
            "fire_rate_among_assessed": r4(fired / assessed) if assessed else None,
            "fire_rate_among_all_authors": r4(fired / n_auth),
        }

    tiers = p.review_tier.value_counts().to_dict()
    xp, mapping = cross_platform()

    # ---- outputs -----------------------------------------------------------
    figs = {
        "videos_per_author": common.OUT_FIGURES / "tiktok_videos_per_author.png",
        "feature_distributions": common.OUT_FIGURES / "tiktok_feature_distributions.png",
        "indicator_coverage": common.OUT_FIGURES / "tiktok_indicator_coverage.png",
        "regularity_vs_null": common.OUT_FIGURES / "tiktok_regularity_vs_null.png",
        "engagement_vs_reach": common.OUT_FIGURES / "tiktok_engagement_vs_reach.png",
        "cross_platform_feature_overlap": common.OUT_FIGURES / "tiktok_cross_platform_feature_overlap.png",
    }
    fig_videos_per_author(dq, figs["videos_per_author"])
    fig_feature_distributions(p, figs["feature_distributions"])
    fig_indicator_coverage(p, figs["indicator_coverage"])
    fig_regularity_null(p, diag, figs["regularity_vs_null"])
    fig_engagement(p, figs["engagement_vs_reach"])
    fig_cross_platform(mapping, figs["cross_platform_feature_overlap"])

    profile_cols = (["author_key", "video_count", "first_posted_at", "last_posted_at",
                     "videos_from_second_pass", "zero_comment_videos", "named_sound_videos",
                     "n_intervals", "caption_pairs_compared", "hashtag_pairs_compared"]
                    + [f for f in FEATURES if f != "video_count"]
                    + ["regularity_p_value", "z_log_like_rate", "z_log_comment_rate", "z_log_share_rate",
                       "iforest_anomaly_score", "pct_avg_plays", "pct_like_rate", "pct_comment_rate",
                       "pct_share_rate", "pct_posting_interval_cv"]
                    + ind_cols + ["behavioural_indicators_assessable", "review_tier"])
    out = p[profile_cols].copy()
    for c in out.select_dtypes(include="float").columns:
        out[c] = out[c].round(4)
    csv_path = common.OUT_PREDICTIONS / "tiktok_author_profile.csv"
    out.to_csv(csv_path, index=False)

    payload = {
        "platform": "tiktok",
        "dataset": {
            "name": "TikTok 2025 dataset (merged, deduplicated)",
            "source": "kaggle: haziqhalifi/tiktok-2025-dataset",
            "file": str(RAW_PATH.relative_to(common.PROJECT_ROOT)),
            "n_rows": dq["n_rows"], "unit_in_file": "video", "unit_of_analysis": "author",
            "n_authors": n_auth,
            "label_col": None, "positive_means": None,
            "ground_truth": ("NONE. No label exists and none was created. Nothing in this "
                             "file may be read as 'bot' or 'not bot'."),
        },
        "class_balance": None,
        "split": None,
        "models": {"note": ("No supervised model was trained. Without ground truth, precision, "
                            "recall, F1, MCC, ROC-AUC and PR-AUC are undefined, so the contract's "
                            "supervised metric block is replaced by the descriptive and "
                            "indicator sections below.")},
        "data_questions": dq,
        "features": {
            "used": list(FEATURES),
            "dropped": {
                "video_id": "identifier (contract rule 1)",
                "author": "grouping key + identifier: replaced by a pseudonymous author_key, never a feature",
                "video_url": "identifier (contains author handle and video id)",
                "fetch_time": "99.9% missing; single batch value where present",
                "views": "merged into plays (mutually exclusive columns)",
                "posted_time": "merged into posted_at (mutually exclusive columns)",
                "music": "used only through generic_sound_share and music_repetition, never as raw text",
                "description / hashtags": "used only through engineered length / count / similarity features",
            },
            "engineered": {f: v[0] for f, v in FEATURES.items()},
            "na_policy": ("A feature that is mathematically undefined for an author (e.g. interval "
                          "spread with < 3 videos) is NA. No feature is imputed. Indicators are "
                          "tri-state: 1 fired, 0 assessed and did not fire, NA not assessable. "
                          "NA is never treated as 0."),
            "thresholds": {
                "MIN_VIDEOS_MEAN_INTERVAL": MIN_VIDEOS_MEAN_INTERVAL,
                "MIN_VIDEOS_DISPERSION": MIN_VIDEOS_DISPERSION,
                "MIN_VIDEOS_REGULARITY_FLAG": MIN_VIDEOS_REGULARITY_FLAG,
                "REGULARITY_P_THRESHOLD": REGULARITY_P_THRESHOLD,
                "BURST_WINDOW_SECONDS": BURST_WINDOW_SECONDS, "BURST_MIN_VIDEOS": BURST_MIN_VIDEOS,
                "CAPTION_MIN_TOKENS": CAPTION_MIN_TOKENS, "CAPTION_DUP_JACCARD": CAPTION_DUP_JACCARD,
                "ROBUST_Z_THRESHOLD": ROBUST_Z_THRESHOLD,
            },
        },
        "descriptive_statistics": stats,
        "indicators": indicators,
        "indicator_diagnostics": diag,
        "threshold_sensitivity": {
            "purpose": ("Every threshold here is a judgement call that no ground truth can "
                        "calibrate. These grids show how far each flag count depends on the "
                        "choice. Defaults were fixed before the results were inspected."),
            **sens_spark,
            **diag["threshold_sensitivity_pandas"],
        },
        "indicator_cooccurrence": overlap,
        "popularity_audit": popularity_audit,
        "review_tiers": {
            "counts": {k: int(v) for k, v in tiers.items()},
            "rules": {
                "priority_review": "regular-posting fired, OR burst AND caption-duplication both fired",
                "secondary_context": "exactly one of burst / caption-duplication fired",
                "weak_signal_only": "only engagement-outlier and/or IsolationForest fired: do not queue on this alone",
                "no_indicator_fired": ">= 1 behavioural indicator assessable and none fired (NOT 'authentic')",
                "behaviour_not_assessable": "no behavioural indicator assessable (single-video authors) and no weak signal",
            },
            "why_not_a_score": ("The indicators differ in evidential weight and in who they can "
                                "assess. Adding them up would let several weak, popularity-driven "
                                "signals outrank one strong one, and would make a 5-indicator "
                                "count look more precise than the data allows."),
        },
        "coordination_assessment": {
            "claim_possible": False,
            "reason": ("The data has no follower / following graph, no reply or duet / stitch links, "
                       "no shared device or network signal. Shared hashtags, sounds and captions "
                       "are normal on TikTok (trends, platform programme tags), so they do not "
                       "show that accounts are linked."),
            "cross_author_identical_captions": shared,
        },
        "cross_platform": xp,
        "leakage_audit": {
            "checks_run": [
                "identifiers (video_id, author, video_url) excluded from every feature and model input",
                "video_id and video_url uniqueness (dedup claim of the file name verified)",
                "video_url consistent with author and video_id columns",
                "author namespace collision between scrape passes (the '@' prefix)",
                "likes > plays impossibility check",
                "display-rounding of counts quantified",
                "IsolationForest score and engagement flags tested for dependence on reach (popularity confound)",
                "Spark sha2 author_key matches the documented python author_key()",
            ],
            "findings": [
                f"video_id unique: {dq['video_id_unique']}; video_url unique: {dq['video_url_unique']}",
                f"{dq['author_namespace']['second_pass_authors_also_in_first_pass']} author(s) appear in "
                "both scrape passes under '@handle' and 'handle'; merged by normalisation",
                f"likes > plays rows: {dq['likes_greater_than_plays']}",
                (f"IsolationForest score vs log reach: Spearman {popularity_audit['spearman_iforest_score_vs_log_avg_plays']}; "
                 f"{popularity_audit['share_of_engagement_flags_in_top_reach_decile']:.0%} of engagement flags "
                 "fall in the top reach decile (10% expected if unrelated)"),
                "No supervised leakage audit applies: there is no label to leak.",
            ],
        },
        "outputs": {
            "author_profile_csv": str(csv_path.relative_to(common.PROJECT_ROOT)),
            "figures": {k: str(v.relative_to(common.PROJECT_ROOT)) for k, v in figs.items()},
            "curated_local_only": ["data/curated/tiktok_videos_clean.parquet",
                                   "data/curated/tiktok_author_key_map.csv (contains handles: do not share)"],
        },
        "limitations": [
            "NO GROUND TRUTH. Every flag is an indicator for human review, never a bot label. The "
            "share of bots in this population is unknown, so the precision of any flag is unknown.",
            "Bot does not mean harmful, and an indicator firing does not mean bot.",
            "Popularity-filtered sample: every video has >= ~141k plays, and low-reach accounts are absent.",
            f"{dq['videos_per_author']['share_authors_with_1_video']:.1%} of authors have one video, so the "
            "strongest signal (timing) is undefined for them.",
            "Posting intervals are between captured viral videos, not all uploads: they are thinned.",
            "No follower / following / account-age / profile data: the features that matter on the "
            "labelled platforms do not exist here.",
            "Counts are display-rounded (about 3 significant figures).",
            "Engagement is not normalised for exposure time (scrape time unknown for 99.9% of rows).",
            "Timestamp timezone is not documented. Values are treated as naive UTC, so hour-of-day "
            "features are not interpreted.",
            "Coordination cannot be assessed (no links between accounts in the data).",
            "views/plays and create_time/posted_time are assumed to be the same quantity under two "
            "names. This affects 7 rows and cannot be verified, because the columns never co-occur.",
        ],
    }
    mpath = common.save_metrics(payload, TRACK)

    print(f"rows={dq['n_rows']} authors={n_auth}")
    print("indicators:", {k: (v["n_fired"], v["n_assessed"]) for k, v in indicators.items()})
    print("tiers:", tiers)
    print("popularity audit:", popularity_audit)
    print(f"wrote {mpath}\nwrote {csv_path}")
    for f in figs.values():
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
