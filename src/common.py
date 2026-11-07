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


def landing_path(key: str, cfg: dict | None = None) -> Path:
    """Folder landing satu dataset, data/landing/<key>/ (diisi scripts/download_datasets.py).
    Dibaca oleh kedua jalur: pipeline Rafi (ingest.py) dan pipeline analisis."""
    return path_of(cfg or load_config(), "landing", key)


# ---- Zona jalur analisis (src/*_pipeline.py, synthesis.py), diambil dari config.yaml `paths`.
# Sengaja terpisah dari data/bronze|silver|gold/<key> supaya tidak menimpa keluaran pipeline Rafi.
_PATHS = load_config()["paths"]
PROJECT_ROOT = ROOT
DATA_LANDING = ROOT / _PATHS["landing"]
DATA_CURATED = ROOT / _PATHS["curated"]
OUTPUTS = ROOT / _PATHS["outputs"]
OUT_METRICS = OUTPUTS / "metrics"          # ringkasan per track (JSON, tanpa data per-akun)
OUT_FIGURES = OUTPUTS / "figures"
OUT_MODELS = OUTPUTS / "models"
OUT_PREDICTIONS = OUTPUTS / "predictions"  # skor per akun -> .gitignore
OUT_POWERBI = OUTPUTS / "powerbi"          # ekspor CSV Power BI -> .gitignore

for _d in (DATA_CURATED, OUT_METRICS, OUT_FIGURES, OUT_MODELS, OUT_PREDICTIONS):
    _d.mkdir(parents=True, exist_ok=True)


def snake(name: str) -> str:
    # '#' di awal nama berarti "jumlah" dan dipertahankan sebagai 'n_': '#tags/post' -> 'n_tags_post'.
    # Tanpa ini '#tags/post' dan 'tags/post' (keduanya ada di dataset Facebook) sama-sama menjadi
    # 'tags_post' dan Spark gagal dengan AMBIGUOUS_REFERENCE.
    s = re.sub(r"^#", "n_", str(name).strip())
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", s)
    return re.sub(r"_+", "_", s).strip("_").lower()


def snake_all(names) -> list:
    """snake() untuk seluruh kolom sekaligus, dengan jaminan hasilnya unik.

    Jaring pengaman untuk dataset yang belum pernah dilihat: bila dua nama masih menjadi sama,
    nama berikutnya diberi akhiran _1, _2, ... (nama kosong menjadi 'col')."""
    out, used = [], set()
    for n in names:
        base = snake(n) or "col"
        s, i = base, 0
        while s in used:
            i += 1
            s = f"{base}_{i}"
        used.add(s)
        out.append(s)
    return out


def spark_master(cores=None) -> str:
    """'local[*]' (semua core) bila cores=None, selain itu 'local[<cores>]'."""
    return "local[*]" if cores is None else f"local[{int(cores)}]"


def get_spark(name: str = "bigdata-bot-detection", driver_memory: str = "2g", cores=None) -> SparkSession:
    """SparkSession lokal untuk semua skrip di src/.

    Dua gaya pemanggilan sama-sama didukung:
      get_spark("ingest")                                   # pipeline batch/streaming
      get_spark("twitter_pipeline", driver_memory="2g", cores=3)   # pipeline analisis
    cores=None memakai semua core (local[*]). Jalur analisis menyebut cores secara eksplisit karena
    jumlah partisi ikut menentukan hasil split acak ber-seed (sampleBy).
    Java diambil dari lingkungan (JAVA_HOME / java di PATH); tidak ada path yang di-hard-code.
    """
    spark = (SparkSession.builder.master(spark_master(cores)).appName(name)
             .config("spark.driver.memory", driver_memory)
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.ui.enabled", "false")
             .config("spark.ui.showConsoleProgress", "false")
             .config("spark.sql.session.timeZone", "UTC")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ---- Jalur analisis: split, nama fitur, evaluasi, keluaran ---------------------------------
# Sama untuk setiap track agar metrik dan split dapat dibandingkan lintas platform.
RANDOM_SEED = 42
TEST_FRACTION = 0.20


def stratified_split(df, label_col: str = "label",
                     test_fraction: float = TEST_FRACTION,
                     seed: int = RANDOM_SEED):
    """Stratified train/test split that preserves duplicate rows.

    Uses an anti-join on a surrogate row id rather than DataFrame.subtract(),
    because subtract() de-duplicates and would silently drop repeated rows.
    """
    tagged = df.withColumn("_row_id", F.monotonically_increasing_id()).cache()
    labels = [r[0] for r in tagged.select(label_col).distinct().collect()]
    fractions = {lab: 1.0 - test_fraction for lab in labels}

    train = tagged.sampleBy(label_col, fractions=fractions, seed=seed).cache()
    test = tagged.join(train.select("_row_id"), on="_row_id", how="left_anti")
    return train.drop("_row_id"), test.drop("_row_id")


def assembled_feature_names(df, features_col: str = "features") -> list[str]:
    """Recover column names from a VectorAssembler output, in vector order.

    Handles one-hot / indexed expansions, which a plain inputCols list does not.
    """
    attrs = df.schema[features_col].metadata["ml_attr"]["attrs"]
    indexed = []
    for kind in ("numeric", "binary", "nominal"):
        for a in attrs.get(kind, []):
            indexed.append((a["idx"], a["name"]))
    return [name for _, name in sorted(indexed)]


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
    df = df.toDF(*snake_all(df.columns))
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
