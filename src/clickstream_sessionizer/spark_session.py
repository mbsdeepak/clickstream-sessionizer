"""SparkSession builder.

Centralises session creation so every entry point (pipeline, streaming, tests)
gets the same sensible defaults: ``local[*]`` master, a modest fixed shuffle
partition count for laptop-scale data, and Adaptive Query Execution enabled.
"""

from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession

from .config import Config


def _pin_worker_python() -> None:
    """Ensure Spark's Python workers use the SAME interpreter as the driver.

    Without this, Spark launches workers with whatever ``python3`` is first on
    PATH (often a different minor version), which fails with
    ``PYTHON_VERSION_MISMATCH``. Pinning both to ``sys.executable`` guarantees
    the driver and workers agree.
    """
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def build_spark(config: Config, *, app_suffix: str | None = None) -> SparkSession:
    """Create (or fetch) a configured SparkSession.

    Args:
        config: Loaded project configuration.
        app_suffix: Optional suffix appended to the app name, handy for
            distinguishing the streaming job from the batch job in the UI.

    Returns:
        A ready-to-use SparkSession.
    """
    _pin_worker_python()

    app_name = config.spark.app_name
    if app_suffix:
        app_name = f"{app_name}-{app_suffix}"

    builder = (
        SparkSession.builder.master(config.spark.master)
        .appName(app_name)
        # A cluster derives shuffle partitions from data volume; on a laptop a
        # fixed, small number keeps tiny shuffles from spawning 200 empty tasks.
        .config("spark.sql.shuffle.partitions", config.spark.shuffle_partitions)
        # Adaptive Query Execution: coalesces post-shuffle partitions and can
        # split skewed shuffle partitions automatically. We STILL demonstrate
        # manual salting in skew.py because AQE only rescues shuffle joins/aggs,
        # not every skew pattern, and interviewers expect the manual technique.
        .config("spark.sql.adaptive.enabled", str(config.spark.adaptive_enabled).lower())
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        # Deterministic, timezone-stable timestamp handling across machines.
        .config("spark.sql.session.timeZone", "UTC")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
