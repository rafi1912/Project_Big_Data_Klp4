"""config.yaml harus bisa dibaca dan setiap dataset punya kunci yang dipakai pipeline."""
import pytest

from common import load_config

REQUIRED_DATASET_KEYS = {"platform", "kaggle", "label_column", "positive_values", "id_columns",
                         "text_columns", "hashtag_columns", "datetime_columns", "note"}
LIST_KEYS = ("id_columns", "text_columns", "hashtag_columns", "datetime_columns")


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_config_has_top_level_sections(cfg):
    assert {"paths", "datasets", "modeling"} <= set(cfg)
    for key in ("landing", "bronze", "silver", "gold", "models", "reports"):
        assert key in cfg["paths"], key


@pytest.mark.parametrize("key", list(load_config()["datasets"]))
def test_every_dataset_has_required_keys(cfg, key):
    ds = cfg["datasets"][key]
    missing = REQUIRED_DATASET_KEYS - set(ds)
    assert not missing, f"{key} tidak punya kunci {sorted(missing)}"
    assert "/" in ds["kaggle"], f"{key}: slug Kaggle harus berbentuk pemilik/nama"
    for k in LIST_KEYS:
        assert isinstance(ds[k], list), f"{key}.{k} harus list (boleh kosong)"


def test_split_fractions_sum_to_one(cfg):
    split = cfg["modeling"]["split"]
    assert len(split) == 3
    assert sum(split) == pytest.approx(1.0)
