"""TikTok: konfigurasi dataset tanpa label, target unduhan, dan helper murni pipeline."""
import importlib.util

import numpy as np
import pandas as pd

import common
import tiktok_pipeline as tp

ROOT = common.ROOT


def _download_module():
    spec = importlib.util.spec_from_file_location("download_datasets", ROOT / "scripts" / "download_datasets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)            # aman: unduhan hanya terjadi di main()
    return mod


def test_tiktok_is_unlabelled_and_outside_the_labelled_pipeline():
    cfg = common.load_config()
    assert "tiktok_videos" in cfg["unlabelled_datasets"]
    assert "tiktok_videos" not in cfg["datasets"]          # ingest/process/train tidak menyentuhnya
    tt = cfg["unlabelled_datasets"]["tiktok_videos"]
    assert tt["kaggle"] == "haziqhalifi/tiktok-2025-dataset" and tt["platform"] == "TikTok"
    assert "label_column" not in tt                          # tidak ada label yang dibuat-buat
    assert common.landing_path("tiktok_videos") == ROOT / "data" / "landing" / "tiktok_videos"


def test_download_targets_cover_labelled_and_unlabelled():
    dl = _download_module()
    cfg = common.load_config()
    keys = set(dl.targets(cfg))
    assert keys == set(cfg["datasets"]) | set(cfg["unlabelled_datasets"])
    assert list(dl.targets(cfg, "tiktok_videos")) == ["tiktok_videos"]
    assert dl.targets(cfg, "tidak_ada") == {}



def test_author_key_is_deterministic_pseudonym():
    k = tp.author_key("some_handle")
    assert k == tp.author_key("some_handle") and k != tp.author_key("other_handle")
    assert len(k) == 16 and int(k, 16) >= 0 and "some_handle" not in k


def test_robust_z_and_constant_series():
    z = tp.robust_z(pd.Series([1.0, 2.0, 3.0, 4.0, 100.0]))
    assert z.iloc[2] == 0 and z.iloc[-1] > 3.5
    assert tp.robust_z(pd.Series([5.0, 5.0, 5.0])).isna().all()       # MAD = 0 -> tidak dapat dinilai


def test_tri_state_keeps_not_assessable_as_na():
    flag = pd.Series([True, False, True])
    eligible = pd.Series([True, True, False])
    out = tp.tri_state(flag, eligible)
    assert out.iloc[0] == 1 and out.iloc[1] == 0 and pd.isna(out.iloc[2])   # missing is not zero


def test_poisson_null_is_sorted_per_video_count():
    null = tp.poisson_cv_null(tp.MIN_VIDEOS_DISPERSION + 2, np.random.default_rng(42))
    assert set(null) == set(range(tp.MIN_VIDEOS_DISPERSION, tp.MIN_VIDEOS_DISPERSION + 3))
    for arr in null.values():
        assert len(arr) == tp.NULL_SIMULATIONS and np.all(np.diff(arr) >= 0)
