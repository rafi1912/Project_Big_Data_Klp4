"""Helper murni track Instagram/Facebook yang bisa diuji tanpa dataset."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")
import facebook_pipeline as fb  # noqa: E402
import profile_tracks_utils as U  # noqa: E402


def test_model_params_records_every_setting():
    rf = U.model_params("random_forest", weighted=False)
    assert rf["numTrees"] == 200 and rf["seed"] == 42 and rf["tuned"] is False and rf["weightCol"] is None
    lr = U.model_params("logistic_regression", weighted=True)
    assert lr["regParam"] == 0.01 and lr["weightCol"].startswith("weight (balanced")
    assert lr["feature_scaling"].startswith("StandardScaler(")


def test_balanced_weights_use_training_rows_only():
    pdf = pd.DataFrame({"label": [0, 0, 0, 1]})
    w = U.add_balanced_weights(pdf)["weight"].tolist()
    assert w == [4 / 6, 4 / 6, 4 / 6, 2.0]                     # n / (2 * n_kelas)
    assert "weight" not in pdf.columns                           # input tidak diubah


def test_threshold_metrics_and_mcnemar():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    a = np.array([1, 1, 0, 0, 0, 0, 1, 1])
    m = U._threshold_metrics(y, a)
    assert m["accuracy"] == 0.75 and m["precision_positive"] == 0.75 and m["recall_positive"] == 0.75
    b = np.array([1, 0, 0, 0, 1, 0, 1, 1])
    t = U.mcnemar_exact(y, a, b)
    assert t["rf_right_lr_wrong"] == 2 and t["lr_right_rf_wrong"] == 0 and 0 < t["p_value_exact"] <= 1


def test_wilson_and_r4():
    assert U.wilson_ci(0, 0) == [0.0, 0.0]
    lo, hi = U.wilson_ci(5, 10)
    assert lo < 0.5 < hi
    assert U.r4({"a": [np.float64(0.123456), np.int64(3)], "b": 1.000049}) == {"a": [0.1235, 3], "b": 1.0}


def test_bootstrap_ci_is_reproducible():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(40), np.ones(40)].astype(int)
    s = np.clip(y * 0.4 + rng.random(80) * 0.6, 0, 1)
    pdf = pd.DataFrame({"label": y, "prediction": (s >= 0.5).astype(int), "p_inauthentic": s})
    a, b = U.bootstrap_ci(pdf, n_boot=200), U.bootstrap_ci(pdf, n_boot=200)
    assert a == b                                                # seed tetap -> hasil sama
    lo, hi = a["roc_auc"]
    assert 0.5 < lo <= hi <= 1.0


def test_facebook_one_rule_picks_threshold_on_train():
    train = pd.DataFrame({"n_friends": [1, 2, 3, 10, 11, 12], "label": [0, 0, 0, 1, 1, 1]})
    test = pd.DataFrame({"n_friends": [2, 9, 13, 0], "label": [0, 1, 1, 0]})
    r = fb.one_rule(train, test, "n_friends")
    assert r["_dir"] == ">=" and r["_thr"] == 6.5
    assert r["train_balanced_accuracy"] == 1.0 and r["test_accuracy"] == 1.0
