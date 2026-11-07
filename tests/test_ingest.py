"""ingest.py: entri manifest hanya dianggap sudah di-ingest bila datanya (`_batch_id`) ada di bronze."""
import json

import pytest

import ingest

KEY = "toy"


class _KeepAlive:
    """Sesi Spark fixture untuk ingest.main(), yang memanggil spark.stop() di akhir."""

    def __init__(self, spark):
        self._spark = spark

    def __getattr__(self, name):
        return getattr(self._spark, name)

    def stop(self):
        pass


@pytest.fixture
def toy(tmp_path, spark, monkeypatch):
    cfg = {"paths": {k: str(tmp_path / k) for k in ("landing", "bronze", "reports")},
           "datasets": {KEY: {"platform": "Toy"}}}
    monkeypatch.setattr(ingest, "load_config", lambda: cfg)
    monkeypatch.setattr(ingest, "get_spark", lambda name: _KeepAlive(spark))
    land = tmp_path / "landing" / KEY
    land.mkdir(parents=True)

    def add_file(name, rows):
        p = land / name
        p.write_text("Account ID,#posts,Label\n" + "".join(f"{i},{i * 3},{i % 2}\n" for i in rows))
        return p

    def manifest():
        p = tmp_path / "reports" / KEY / "ingest_manifest.json"
        return json.loads(p.read_text()) if p.exists() else {"files": []}

    def write_manifest(m):
        p = tmp_path / "reports" / KEY / "ingest_manifest.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(m))

    def bronze_rows():
        return spark.read.parquet(str(tmp_path / "bronze" / KEY)).count()

    return add_file, manifest, write_manifest, bronze_rows


def _entry(path, batch_id, rows):
    return {"name": path.name, "sha256": ingest.sha256(path), "rows": rows, "batch_id": batch_id,
            "size_bytes": path.stat().st_size}


def test_committed_manifest_without_bronze_is_ingested(toy, capsys):
    add_file, manifest, write_manifest, bronze_rows = toy
    f = add_file("data.csv", range(5))
    write_manifest({"files": [_entry(f, "20200101T000000", 5)]})   # mis. manifest demo yang ikut ter-commit
    ingest.main(KEY)
    out = capsys.readouterr().out
    assert "1 entri manifest tanpa data di bronze (data.csv)" in out
    assert bronze_rows() == 5
    [e] = manifest()["files"]
    assert e["name"] == "data.csv" and e["batch_id"] != "20200101T000000"


def test_entry_whose_batch_is_in_bronze_is_skipped(toy, capsys):
    add_file, manifest, write_manifest, bronze_rows = toy
    add_file("data.csv", range(5))
    ingest.main(KEY)
    first = manifest()["files"]
    capsys.readouterr()
    ingest.main(KEY)                                                 # idempoten
    out = capsys.readouterr().out
    assert "sudah pernah di-ingest -> dilewati" in out and "tanpa data di bronze" not in out
    assert bronze_rows() == 5 and manifest()["files"] == first


def test_entry_missing_from_bronze_is_reingested_and_replaced(toy, capsys):
    add_file, manifest, write_manifest, bronze_rows = toy
    add_file("a.csv", range(4))
    ingest.main(KEY)
    [kept] = manifest()["files"]
    b = add_file("b.csv", range(10, 13))
    m = manifest()
    m["files"].append(_entry(b, "19990101T000000", 3))               # tercatat, tetapi tidak ada di bronze
    write_manifest(m)
    capsys.readouterr()
    ingest.main(KEY)
    out = capsys.readouterr().out
    assert "1 entri manifest tanpa data di bronze (b.csv)" in out
    assert "a.csv: sudah pernah di-ingest -> dilewati" in out
    files = {e["name"]: e for e in manifest()["files"]}
    assert set(files) == {"a.csv", "b.csv"} and files["a.csv"] == kept
    assert files["b.csv"]["batch_id"] != "19990101T000000"
    assert bronze_rows() == 7
