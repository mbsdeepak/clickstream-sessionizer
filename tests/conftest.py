"""Shared pytest fixtures.

A single module-scoped local SparkSession is reused across the whole test run --
spinning up a JVM per test is slow, and Spark is happy with one session.
"""

from __future__ import annotations

import os
import sys

import pytest

# Make the src/ package importable when running `pytest` from the repo root
# without an editable install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Pin Spark's Python workers to this exact interpreter (see spark_session.py)
# so tests never hit PYTHON_VERSION_MISMATCH from a stray system python.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession  # noqa: E402


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    session = (
        SparkSession.builder.master("local[2]")
        .appName("clickstream-sessionizer-tests")
        # Tiny data in tests -> keep shuffles tiny too.
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
