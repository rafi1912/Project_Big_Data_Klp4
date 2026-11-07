"""Setiap halaman dashboard dijalankan headless (streamlit AppTest) terhadap database demo.

Butuh MongoDB yang terjangkau (MONGO_URI, atau .env + compose); bila tidak ada, test dilewati.
CI menyediakan service container mongo:7.0. Memakai database sementara yang dihapus setelahnya,
jadi database lain di server yang sama tidak tersentuh.
"""
import importlib.util
import os

import pytest

import common

pytest.importorskip("streamlit")
pytest.importorskip("scipy")
APP = str(common.ROOT / "app" / "streamlit_app.py")
TEST_DB = "klp4_test_dashboard"
PAGES = ["Arsitektur & Data Lake", "Kualitas Data", "EDA", "Model & Evaluasi", "Streaming (simulasi)",
         "Antrean Review Moderator", "Level Risiko", "Perbandingan Lintas Platform"]


def _seed_module():
    spec = importlib.util.spec_from_file_location("seed_demo", common.ROOT / "scripts" / "seed_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def demo_db():
    from pymongo import MongoClient

    with pytest.MonkeyPatch.context() as mp:
        for k, v in {k: v for k, v in os.environ.items() if k.startswith("MONGO")}.items():
            mp.setenv(k, v)                      # dipulihkan otomatis setelah modul selesai
        common.load_env()
        if not (os.environ.get("MONGO_URI") or os.environ.get("MONGO_INITDB_ROOT_USERNAME")):
            # tanpa konfigurasi eksplisit jangan menyentuh MongoDB apa pun yang kebetulan ada di 27017
            pytest.skip("MongoDB belum dikonfigurasi: set MONGO_URI atau isi .env (lihat .env.example)")
        mp.setenv("MONGO_DB", TEST_DB)
        client = MongoClient(common.mongo_uri(), serverSelectionTimeoutMS=1500)
        try:
            client[TEST_DB].list_collection_names()          # butuh autentikasi, bukan sekadar ping
        except Exception as e:
            pytest.skip(f"MongoDB tidak terjangkau/ditolak ({type(e).__name__}): cek MONGO_URI atau .env")
        client.drop_database(TEST_DB)
        counts = _seed_module().seed(client[TEST_DB])
        yield counts
        client.drop_database(TEST_DB)
        client.close()


def _run(page=None):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=60).run()
    if page:
        at.sidebar.radio[0].set_value(page).run()
    return at


def test_demo_seed_contains_every_dashboard_collection(demo_db):
    for coll in ("pipeline_dq", "pipeline_metrics", "pipeline_eda_sample", "stream_scores", "account_scores",
                 "risk_level_summary", "risk_level_definitions", "datasets", "concept_comparison", "model_metrics"):
        assert demo_db[coll] > 0, coll


def test_sidebar_lists_all_pages(demo_db):
    assert _run().sidebar.radio[0].options == PAGES


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_from_mongodb(demo_db, page):
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]
    assert at.title and at.title[0].value
    assert not any("MongoDB tidak dapat dihubungi" in e.value for e in at.error)


def test_unreachable_mongodb_stops_with_start_command(demo_db, monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb://127.0.0.1:1/")
    at = _run()
    assert not at.exception
    assert len(at.error) == 1 and "docker compose -f deploy/mongodb/docker-compose.yml" in at.error[0].value
    assert not at.title                          # berhenti: tidak ada halaman yang digambar dari file
