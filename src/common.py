"""Utilitas bersama: konfigurasi, SparkSession, dan standardisasi skema.

`standardize()` dipakai oleh jalur BATCH (process.py) dan jalur STREAMING
(stream.py) sehingga fitur yang dilihat model saat training dan saat
scoring identik (mencegah training-serving skew).
"""
import re
from pathlib import Path

import yaml
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

ROOT = Path(__file__).resolve().parents[1]

POSITIVE_DEFAULT = ["1", "true", "bot", "fake", "spam", "spammer", "yes", "y", "malicious"]
LABEL_CANDIDATES = ["bot_label", "label", "is_fake", "isfake", "fake", "class",
                    "account_type", "is_bot", "bot", "spam", "is_spam", "target"]
USER_COLS = ["username", "screen_name", "user_name"]


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path_of(cfg: dict, key: str, *parts) -> Path:
    return ROOT.joinpath(cfg["paths"][key], *parts)


def report_dir(cfg: dict, ds_key: str) -> Path:
    d = ROOT / cfg["paths"]["reports"] / ds_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def snake(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip())
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", s)
    return re.sub(r"_+", "_", s).strip("_").lower()


def get_spark(name: str = "bigdata-bot-detection") -> SparkSession:
    spark = (SparkSession.builder.master("local[*]").appName(name)
             .config("spark.driver.memory", "2g")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.ui.enabled", "false")
             .config("spark.sql.session.timeZone", "UTC")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def detect_label(columns, configured=None):
    cols = [snake(c) for c in columns]
    if configured:
        c = snake(configured)
        if c not in cols:
            raise KeyError(f"Kolom label '{configured}' tidak ada. Kolom tersedia: {cols}")
        return c
    for cand in LABEL_CANDIDATES:
        if cand in cols:
            return cand
    return None


def standardize(df: DataFrame, ds: dict, keep_label: bool = True) -> DataFrame:
    """Rapikan nama kolom, petakan label -> 0/1, turunkan fitur teks/waktu,
    ubah boolean -> int, dan buang kolom identitas/teks mentah dari fitur."""
    df = df.toDF(*[snake(c) for c in df.columns])
    cols = df.columns
    keep = lambda names: [snake(c) for c in (names or []) if snake(c) in cols]
    id_cols, text_cols = keep(ds.get("id_columns")), keep(ds.get("text_columns"))
    tag_cols, dt_cols = keep(ds.get("hashtag_columns")), keep(ds.get("datetime_columns"))
    label_col = detect_label(cols, ds.get("label_column"))
    positives = [str(v).lower() for v in (ds.get("positive_values") or POSITIVE_DEFAULT)]

    out = df
    user_col = next((c for c in USER_COLS if c in cols), None)
    rid = F.col(id_cols[0]).cast("string") if id_cols else F.monotonically_increasing_id().cast("string")
    out = out.withColumn("record_id", rid)
    if user_col:
        out = out.withColumn("username_ref", F.col(user_col).cast("string"))
    if label_col and keep_label:
        raw = F.lower(F.col(label_col).cast("string"))
        out = out.withColumn("label", F.when(F.col(label_col).isNull(), F.lit(None).cast("int"))
                             .when(raw.isin(positives), F.lit(1)).otherwise(F.lit(0)))
    for c in text_cols:
        out = (out.withColumn(f"{c}_len", F.length(F.col(c)))
                  .withColumn(f"{c}_words", F.size(F.split(F.trim(F.col(c)), r"\s+"))))
    for c in tag_cols:
        out = out.withColumn(f"{c}_n", F.when(F.col(c).isNull(), 0)
                             .otherwise(F.length(F.col(c)) - F.length(F.regexp_replace(F.col(c), "#", ""))))
    for i, c in enumerate(dt_cols):
        ts = F.to_timestamp(F.col(c))
        if i == 0:
            out = out.withColumn("event_ts", ts)
        out = out.withColumn(f"{c}_hour", F.hour(ts)).withColumn(f"{c}_dow", F.dayofweek(ts))
    for fld in out.schema.fields:
        if isinstance(fld.dataType, T.BooleanType):
            out = out.withColumn(fld.name, F.col(fld.name).cast("int"))
    # kolom label asli dibuang, KECUALI bila namanya sudah 'label' (kolom turunan menimpanya di tempat)
    drop = set(id_cols + text_cols + tag_cols + dt_cols + ([label_col] if label_col and label_col != "label" else []) +
               ([user_col] if user_col else []))
    return out.drop(*[c for c in drop if c in out.columns])
