"""Lapisan level risiko 0-4 (synthesis.py): ambang, cross-fitting, evidence gate, TikTok.

Nama level menggambarkan kekuatan bukti, bukan vonis; level 3-4 butuh bukti koordinasi
yang tidak ada di dataset mana pun, jadi tidak pernah diberikan."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
import synthesis as syn  # noqa: E402


@pytest.fixture(autouse=True)
def _quiet_log(tmp_path, monkeypatch):
    monkeypatch.setattr(syn, "LOG_PATH", tmp_path / "synthesis.log")


def test_assign_maps_scores_to_levels():
    s = np.array([0.05, 0.3, 0.6, 0.95])
    assert syn.assign(s, 0.3, 0.6).tolist() == [0, 1, 2, 2]     # batas bawah inklusif


def test_fbeta_cut_prefers_recall_for_l1_and_precision_for_l2():
    y = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    s = np.linspace(1.0, 0.1, 10)
    t1, t2, stats = syn.derive_thresholds(y, s)
    assert t2 >= t1                                              # t2 dibatasi >= t1
    assert stats["t1_on_calibration"]["recall"] == 1.0           # F2: jaring lebar
    assert stats["t2_on_calibration"]["precision"] == 1.0        # F0.5: presisi dulu
    assert syn.assign(s, t1, t2).tolist() == [2, 2, 2, 1, 1, 0, 0, 0, 0, 0]


def test_stratified_halves_split_each_class_evenly_and_reproducibly():
    y = np.array([0] * 7 + [1] * 5)
    f = syn.stratified_halves(y)
    assert np.array_equal(f, syn.stratified_halves(y))           # seed 42
    for cls in (0, 1):
        counts = np.bincount(f[y == cls], minlength=2)
        assert abs(counts[0] - counts[1]) <= 1


def test_level_table_counts_add_up():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    lvl = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    per, cum = syn.level_table(y, lvl)
    assert [r["n_accounts"] for r in per] == [2, 2, 4]
    assert per[2]["n_inauthentic"] == 3 and per[2]["precision_in_level"] == 0.75
    assert cum[1]["at_or_above_level"] == 2 and cum[1]["recall"] == 0.75
    assert cum[0]["n_flagged"] == 6 and cum[0]["real_people_flagged"] == 2


def _scores(n, signal, seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    s = np.clip(0.5 * signal * y + rng.random(n) * (1 - 0.5 * signal), 0, 1)
    return pd.DataFrame({"account_uid": [f"a{i}" for i in range(n)], "label": y, "rf_p": s})


def test_evidence_gate_withholds_levels_from_a_model_without_signal():
    norm = {"twitter_real": _scores(400, 1.0, 1), "instagram": _scores(400, 1.0, 2),
            "facebook": _scores(400, 1.0, 3), "twitter_synthetic": _scores(400, 0.0, 4)}
    results, assigned = syn.risk_levels_labelled(norm)
    assert results["twitter_real"]["evidence_gate"]["passed"] is True
    assert assigned["twitter_real"]["risk_level"].notna().all()
    assert results["twitter_synthetic"]["evidence_gate"]["passed"] is False
    assert assigned["twitter_synthetic"]["risk_level"].isna().all()      # NA, bukan Level 0
    assert (assigned["twitter_synthetic"]["risk_level_basis"] == "evidence_gate_failed_no_signal").all()
    for t, r in results.items():
        a, b = r["cross_fitting"]["folds"]["A"], r["cross_fitting"]["folds"]["B"]
        assert a["applied_to_fold"] == "B" and b["applied_to_fold"] == "A"   # tidak pernah menilai diri sendiri


def test_tiktok_levels_cap_at_one_and_keep_na():
    tt = pd.DataFrame({"behavioural_indicators_assessable": [0, 1, 1, 1, None],
                       "review_tier": ["weak_signal_only", "none", "secondary_context",
                                       "priority_review", "none"]})
    lvl, info = syn.risk_levels_tiktok(tt)
    assert pd.isna(lvl[0]) and pd.isna(lvl[4])                   # tidak dapat dinilai
    assert lvl[1] == 0 and lvl[2] == 1 and lvl[3] == 1           # dibatasi di level 1
    assert info["counts"] == {"NA": 2, "0": 1, "1": 2, "2": 0, "3": 0, "4": 0}


def test_level_definitions_and_actions():
    codes = [lv["code"] for lv in syn.LEVELS]
    assert codes == ["0", "1", "2", "3", "4", "NA"]
    assert {lv["code"] for lv in syn.LEVELS if not lv["reachable"]} == {"3", "4"}
    assert syn.ACTION == {"0": "none", "1": "monitor", "2": "human_review", "NA": "none_not_assessable"}
    assert all("sanction" not in a and "ban" not in a for a in syn.ACTION.values())
    assert syn.wilson_ci(0, 0) == [None, None]
