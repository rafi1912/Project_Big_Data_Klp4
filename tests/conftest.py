"""Fixture bersama untuk test.

`spark` dibuat sekali per sesi pytest dan sengaja kecil (local[1], 512m, UI mati)
supaya test cepat dan tidak berebut memori dengan layanan lain di mesin yang sama.
Test yang memakainya otomatis dilewati bila PySpark/Java tidak tersedia.
"""
import pytest


@pytest.fixture(scope="session")
def spark():
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    session = (SparkSession.builder.master("local[1]").appName("pytest")
               .config("spark.driver.memory", "512m")
               .config("spark.ui.enabled", "false")
               .config("spark.ui.showConsoleProgress", "false")
               .config("spark.sql.shuffle.partitions", "1")
               .config("spark.sql.session.timeZone", "UTC")
               .getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
