"""Kolom identitas tidak boleh menjadi fitur model (aturan metode #1)."""
import pytest

from common import load_config, snake, standardize

CFG = load_config()


def test_config_lists_the_known_identifiers():
    ds = CFG["datasets"]
    assert "profile id" in ds["facebook_spam"]["id_columns"]
    assert ds["facebook_spam"]["label_column"] == "Label"
    assert {"id", "screen_name", "c0", "split"} <= set(ds["twitter_human_bots"]["id_columns"])
    assert ds["twitter_human_bots"]["id_columns"][0] == "id"   # kolom pertama = record_id
    assert ds["instagram_fake_spammer"]["label_column"] == "fake"
    assert {"User ID", "Username"} <= set(ds["twitter_bot_detection"]["id_columns"])


@pytest.mark.parametrize("key", list(CFG["datasets"]))
def test_label_column_is_never_an_id_column(key):
    ds = CFG["datasets"][key]
    ids = {snake(c) for c in ds["id_columns"]}
    assert ds["label_column"] is None or snake(ds["label_column"]) not in ids


def test_standardize_drops_facebook_profile_id(spark):
    df = spark.createDataFrame([(1, 10, 0.5, 0), (2, 20, 0.7, 1)],
                               ["profile id", "#friends", "tags/post", "Label"])
    out = standardize(df, CFG["datasets"]["facebook_spam"])
    assert "profile_id" not in out.columns
    assert {"n_friends", "tags_post", "label", "record_id"} <= set(out.columns)
    rows = {r["record_id"]: r["label"] for r in out.collect()}
    assert rows == {"1": 0, "2": 1}          # profile id tetap ada sebagai record_id, bukan fitur


def test_standardize_drops_treiman_index_and_split(spark):
    df = spark.createDataFrame(
        [(0, 101, "alice", 2, 50, "human"), (1, 102, "bot_b", 0, 9000, "bot")],
        ["_c0", "id", "screen_name", "split", "followers_count", "account_type"])
    out = standardize(df, CFG["datasets"]["twitter_human_bots"])
    for col in ("c0", "id", "screen_name", "split", "account_type"):
        assert col not in out.columns, col
    assert {"followers_count", "label", "record_id"} <= set(out.columns)
    assert sorted(r["label"] for r in out.collect()) == [0, 1]
