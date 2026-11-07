"""Helper murni track X/Twitter yang bisa diuji tanpa dataset."""
import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
import twitter_faker_check as fc  # noqa: E402
import twitter_pipeline as tp  # noqa: E402


def test_wilson_ci_brackets_the_proportion():
    lo, hi = tp.wilson_ci(80, 100)
    assert lo < 0.8 < hi
    assert (lo, hi) == (0.7112, 0.8666)


def test_auc_null_band_is_centred_on_half():
    lo, hi = tp.auc_null_band(100, 100)
    assert lo < 0.5 < hi and math.isclose(0.5 - lo, hi - 0.5, abs_tol=1e-4)


def test_single_feature_auc_is_direction_free():
    y = np.array([0] * 10 + [1] * 10)
    x = np.arange(20, dtype=float)
    assert tp.single_feature_auc(y, x) == 1.0
    assert tp.single_feature_auc(y, -x) == 1.0                  # arah terbalik tetap 1.0
    assert math.isnan(tp.single_feature_auc(y[:5], x[:5]))       # terlalu sedikit data


def test_review_queue_precision_at_top_fraction():
    y = np.array([1] * 10 + [0] * 90)
    p = np.linspace(1, 0, 100)                                   # semua bot di skor tertinggi
    q = {r["top_fraction"]: r for r in tp.review_queue(y, p)}
    assert q[0.10]["k_accounts"] == 10 and q[0.10]["precision_at_k"] == 1.0
    assert q[0.10]["recall_at_k"] == 1.0 and q[0.05]["recall_at_k"] == 0.5
    assert q[0.20]["base_rate"] == 0.1


LOREM = {"alpha", "beta", "gamma"}


def test_faker_check_detects_closed_word_list():
    d = pd.DataFrame({"Tweet": ["Alpha beta.", "gamma alpha"], "Hashtags": ["beta", None],
                      "Location": ["Port Smith", "Lake Town"]})
    out = fc.compare(d, LOREM, prefixes={"Port", "Lake"}, suffixes={"town"})
    assert out["tweet_vocabulary_size"] == 3 and out["tweet_words_in_faker_list"] == 3
    assert out["tweet_words_not_in_faker_list"] == []
    assert out["verdict"].startswith("Tweet and Hashtags vocabularies are exactly")
    assert out["location_share_with_faker_city_prefix"] == 1.0


def test_faker_check_flags_foreign_words():
    d = pd.DataFrame({"Tweet": ["alpha beta gamma delta"], "Hashtags": [""], "Location": ["Bogor"]})
    out = fc.compare(d, LOREM, prefixes=set(), suffixes=set())
    assert out["tweet_words_not_in_faker_list"] == ["delta"]
    assert out["verdict"] == "vocabulary differs from Faker's list"
    assert out["location_share_with_prefix_or_suffix"] == 0.0
