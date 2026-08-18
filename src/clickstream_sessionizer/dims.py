"""Dimension tables.

Small reference tables that enrich the fact stream. Because they are tiny
(tens of rows) we join them with an explicit ``broadcast`` hint: Spark ships the
whole dimension to every executor and does a map-side (broadcast hash) join,
avoiding a shuffle of the large fact table entirely. This is the single most
impactful join optimisation for star-schema workloads.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import broadcast

from .config import Config

# (code, human name, region) -- a realistic country dimension.
_COUNTRY_ROWS = [
    ("US", "United States", "Americas"),
    ("CA", "Canada", "Americas"),
    ("BR", "Brazil", "Americas"),
    ("GB", "United Kingdom", "EMEA"),
    ("DE", "Germany", "EMEA"),
    ("FR", "France", "EMEA"),
    ("NG", "Nigeria", "EMEA"),
    ("IN", "India", "APAC"),
    ("JP", "Japan", "APAC"),
    ("AU", "Australia", "APAC"),
]

# (device, form factor class) -- used to bucket UX metrics.
_DEVICE_ROWS = [
    ("desktop", "large_screen"),
    ("tablet", "medium_screen"),
    ("mobile", "small_screen"),
]


def country_dim(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        _COUNTRY_ROWS, ["country", "country_name", "region"]
    )


def device_dim(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        _DEVICE_ROWS, ["device", "form_factor"]
    )


def enrich_with_dims(spark: SparkSession, events: DataFrame, config: Config) -> DataFrame:
    """Left-join the country + device dimensions using broadcast joins.

    ``broadcast(dim)`` forces a broadcast hash join. Even though AQE can pick
    broadcast automatically, the explicit hint documents intent and guarantees
    the plan regardless of stale/absent table statistics.
    """
    countries = country_dim(spark)
    devices = device_dim(spark)

    return (
        events.join(broadcast(countries), on="country", how="left")
        .join(broadcast(devices), on="device", how="left")
    )
