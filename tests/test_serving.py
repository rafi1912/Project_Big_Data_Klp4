"""Lapisan penyajian tanpa MongoDB: konversi dokumen, isi loader, pemindai identitas, level demo."""
import importlib.util
import math

import numpy as np
import pandas as pd
import pytest

import common
import load_mongo

pytest.importorskip("scipy")


def _script(name):
    spec = importlib.util.spec_from_file_location(name, common.ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_clean_turns_missing_into_null_and_numpy_into_python():
    assert load_mongo.clean(float("nan")) is None and load_mongo.clean(pd.NA) is None
    assert load_mongo.clean(pd.NaT) is None and load_mongo.clean(np.float64("inf")) is None
    v = load_mongo.clean({"a": np.int64(3), "b": [np.float32(0.5), None], 1: pd.Timestamp("2024-01-02")})
    assert v == {"a": 3, "b": [0.5, None], "1": pd.Timestamp("2024-01-02").to_pydatetime()}
    assert type(v["a"]) is int and not isinstance(v["b"][0], np.generic)


def test_loader_documents_for_the_committed_demo_dataset():
    docs = load_mongo.dataset_documents(common.load_config(), "twitter_bot_detection")
    assert set(docs) == set(load_mongo.COLLECTIONS)
    assert docs["pipeline_dq"][0]["dataset"] == "twitter_bot_detection"
    assert docs["pipeline_metrics"][0]["best_model"]
    assert len(docs["pipeline_roc"]) > 10
    eda = docs["pipeline_eda_sample"]
    assert eda and not ({"record_id", "username_ref"} & set(eda[0]))     # EDA tanpa identitas
    for coll, rows in docs.items():
        for row in rows[:200]:
            assert row["dataset"] == "twitter_bot_detection", coll
            assert not any(isinstance(x, float) and math.isnan(x) for x in row.values()), coll


def test_identifier_scan_catches_known_patterns():
    find = _script("export_demo_snapshot").find_identifiers
    assert find('{"auc": 0.8538, "n_rows": 37438, "note": "profile pic share 0.3"}') == {}
    assert "account_uid" in find('"x_1000000001"')
    assert "long_numeric_id" in find("user 900000001 flagged")
    assert "instagram_row_handle" in find("duplicate group test_r0020")
    assert "facebook_profile_id_list" in find("profile ids [255, 257, 300]")
    assert "per_account_field" in find('{"source_ref": "abc"}')
    assert "tiktok_author_key" in find("author 3f2a9c0d1e4b5a6f")


def test_demo_levels_follow_the_evidence_gate():
    seed = _script("seed_demo")
    rng = np.random.default_rng(0)
    y = (rng.random(600) < 0.4).astype(int)
    noise = rng.random(600)
    lvl, gate = seed.demo_levels(y, noise)
    assert gate["passed"] is False and np.isnan(lvl).all()              # tanpa sinyal -> semua NA
    lvl, gate = seed.demo_levels(y, np.clip(0.6 * y + 0.4 * noise, 0, 1))
    assert gate["passed"] is True and set(np.unique(lvl)) <= {0.0, 1.0, 2.0}


def test_demo_accounts_on_committed_synthetic_predictions():
    seed = _script("seed_demo")
    pred = pd.read_parquet(common.path_of(common.load_config(), "gold", "twitter_bot_detection",
                                          "test_predictions.parquet"))
    docs, gate = seed.demo_accounts(pred)
    assert len(docs) == len(pred) and gate["passed"] is False
    assert {d["risk_level_code"] for d in docs} == {"NA"}
    assert all(d["dataset"] == "twitter_bot_detection_demo" and d["label"] in (0, 1) for d in docs[:50])
