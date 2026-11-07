"""Helper jalur analisis di common.py: get_spark, path, split, evaluasi, keluaran, MongoDB."""
import json
import math

import pytest

import common


def test_spark_master():
    assert common.spark_master(None) == "local[*]"
    assert common.spark_master(3) == "local[3]"


def test_get_spark_supports_both_call_styles(spark):
    pipeline_style = common.get_spark("ingest")                              # gaya Rafi
    analysis_style = common.get_spark("twitter_pipeline", driver_memory="512m", cores=1)  # gaya analisis
    assert pipeline_style is analysis_style is spark      # getOrCreate memakai sesi yang sama
    assert spark.conf.get("spark.sql.session.timeZone") == "UTC"
    assert spark.conf.get("spark.sql.shuffle.partitions") == "8"


def test_paths_come_from_config_and_avoid_pipeline_zones():
    cfg = common.load_config()
    assert common.landing_path("facebook_spam") == common.ROOT / cfg["paths"]["landing"] / "facebook_spam"
    assert common.DATA_CURATED == common.ROOT / cfg["paths"]["curated"]
    assert common.OUT_METRICS == common.ROOT / cfg["paths"]["outputs"] / "metrics"
    rafi_zones = [common.ROOT / cfg["paths"][k] for k in ("bronze", "silver", "gold")]
    for p in (common.DATA_CURATED, common.OUT_METRICS, common.OUT_FIGURES, common.OUT_PREDICTIONS):
        assert not any(p == z or z in p.parents for z in rafi_zones), p


def test_stratified_split_keeps_duplicate_rows_and_label_ratio(spark):
    rows = [(1.0, 0)] * 60 + [(float(i), 0) for i in range(2, 142)] + [(float(-i), 1) for i in range(100)]
    df = spark.createDataFrame(rows, ["x", "label"])            # 300 baris, 60 duplikat persis, 1/3 positif
    train, test = common.stratified_split(df, "label", test_fraction=0.2, seed=42)
    assert train.count() + test.count() == 300                   # tidak ada baris yang hilang
    dup = train.filter("x = 1.0").count() + test.filter("x = 1.0").count()
    assert dup == 60                                             # subtract() akan menyisakan 1
    for part in (train, test):
        n, pos = part.count(), part.filter("label = 1").count()
        assert abs(pos / n - 1 / 3) < 0.1
    assert 0.1 < test.count() / 300 < 0.3


def test_evaluate_binary_on_toy_predictions(spark):
    from pyspark.ml.linalg import Vectors
    # (label, prediksi, skor kelas 1): TP=3, FP=1, TN=4, FN=2
    data = [(1, 1, .9), (1, 1, .8), (1, 1, .7), (0, 1, .65), (1, 0, .4), (1, 0, .3),
            (0, 0, .35), (0, 0, .2), (0, 0, .1), (0, 0, .05)]
    df = spark.createDataFrame([(y, float(p), Vectors.dense([1 - s, s])) for y, p, s in data],
                               ["label", "prediction", "probability"])
    m = common.evaluate_binary(df)
    assert m["confusion_matrix"] == {"true_negative": 4, "false_positive": 1,
                                     "false_negative": 2, "true_positive": 3}
    assert m["accuracy"] == 0.7 and m["precision_positive"] == 0.75 and m["recall_positive"] == 0.6
    assert m["specificity"] == 0.8 and m["balanced_accuracy"] == 0.7
    assert m["mcc"] == round(10 / math.sqrt(4 * 5 * 5 * 6), 4)
    pos = [s for y, _, s in data if y == 1]
    neg = [s for y, _, s in data if y == 0]
    auc = sum(p > n for p in pos for n in neg) / (len(pos) * len(neg))   # Mann-Whitney, tanpa ikatan
    assert m["roc_auc"] == pytest.approx(auc, abs=1e-4)
    assert m["baseline_majority_class"] == {"predicts": 1, "accuracy": 0.5, "accuracy_lift_of_model": 0.2}


def test_assembled_feature_names_and_importance_table(spark):
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.linalg import Vectors
    df = spark.createDataFrame([(1.0, 2.0)], ["a", "b"])
    out = VectorAssembler(inputCols=["b", "a"], outputCol="features").transform(df)
    assert common.assembled_feature_names(out) == ["b", "a"]

    class RF:
        featureImportances = Vectors.dense([0.2, 0.8])

    class LR:
        coefficients = Vectors.dense([-3.0, 1.0])

    assert [r["feature"] for r in common.importance_table(RF(), ["b", "a"])] == ["a", "b"]
    lr = common.importance_table(LR(), ["b", "a"], top_n=1)
    assert lr == [{"feature": "b", "coefficient": -3.0, "rank": 1}]


def test_save_metrics_writes_json(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "OUT_METRICS", tmp_path)
    path = common.save_metrics({"accuracy": 0.5}, "toy")
    payload = json.loads(path.read_text())
    assert path == tmp_path / "toy.json"
    assert payload["accuracy"] == 0.5 and payload["_track"] == "toy" and "_generated_at" in payload


def test_mongo_uri_and_redaction():
    assert common.mongo_uri({"MONGO_URI": "mongodb://x:1/"}) == "mongodb://x:1/"
    built = common.mongo_uri({"MONGO_INITDB_ROOT_USERNAME": "admin", "MONGO_INITDB_ROOT_PASSWORD": "p@ss",
                              "MONGO_PORT": "27018"})
    assert built == "mongodb://admin:p%40ss@127.0.0.1:27018/?authSource=admin"
    assert common.mongo_uri({}) == "mongodb://127.0.0.1:27017/"
    assert common.redact_uri(built) == "mongodb://admin:<password>@127.0.0.1:27018/?authSource=admin"
    assert common.redact_uri("mongodb://127.0.0.1:27017/") == "mongodb://127.0.0.1:27017/"


def test_load_env_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# komentar\nMONGO_DB=from_file\nexport KLP4_TEST_X='a=b'\n\n")
    monkeypatch.setenv("MONGO_DB", "from_env")
    monkeypatch.delenv("KLP4_TEST_X", raising=False)
    assert common.load_env(env) == ["MONGO_DB", "KLP4_TEST_X"]
    import os
    assert os.environ["MONGO_DB"] == "from_env" and os.environ["KLP4_TEST_X"] == "a=b"
    monkeypatch.delenv("KLP4_TEST_X")


def test_get_mongo_uses_environment_without_connecting(monkeypatch):
    pytest.importorskip("pymongo")
    monkeypatch.setenv("MONGO_URI", "mongodb://127.0.0.1:1/")      # port tak terpakai: tidak ada koneksi
    monkeypatch.setenv("MONGO_DB", "klp4_test")
    client, db = common.get_mongo(timeout_ms=100)
    assert db.name == "klp4_test"
    client.close()
