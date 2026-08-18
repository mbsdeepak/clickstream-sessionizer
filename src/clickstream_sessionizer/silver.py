"""Silver layer: clean + enrich bronze.

Transformations applied here (the "conformed" zone):
    * Drop rows with no usable key (null user_id / timestamp / event_type).
    * Filter out bot traffic by user-agent (analytics should measure humans).
    * De-duplicate exact repeat events (at-least-once producers create these).
    * Normalise a couple of fields.
    * Enrich with the country + device dimensions via broadcast joins.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .bronze import read_bronze
from .config import Config
from .dims import enrich_with_dims

# Substrings that mark a user-agent as automated traffic.
_BOT_MARKERS = ["bot", "crawler", "spider", "python-requests", "ahrefs", "curl", "wget"]


def _is_bot_expr():
    ua = F.lower(F.coalesce(F.col("user_agent"), F.lit("")))
    cond = F.lit(False)
    for marker in _BOT_MARKERS:
        cond = cond | ua.contains(marker)
    return cond


def build_silver(spark: SparkSession, bronze: DataFrame, config: Config) -> DataFrame:
    """Clean, dedupe, filter bots, and enrich."""
    cleaned = (
        bronze
        # Rows missing a natural key can't be sessionized -> discard.
        .filter(
            F.col("user_id").isNotNull()
            & F.col("event_timestamp").isNotNull()
            & F.col("event_type").isNotNull()
        )
        # Tag then drop bot traffic.
        .withColumn("is_bot", _is_bot_expr())
        .filter(~F.col("is_bot"))
        .drop("is_bot")
        # Normalise categorical fields.
        .withColumn("event_type", F.lower(F.trim(F.col("event_type"))))
        .withColumn("device", F.lower(F.trim(F.col("device"))))
        .withColumn("country", F.upper(F.trim(F.col("country"))))
    )

    # Exact-duplicate removal. dropDuplicates on the business identity of an
    # event; event_id alone would also work but this is robust if the producer
    # reuses ids.
    deduped = cleaned.dropDuplicates(
        ["user_id", "event_timestamp", "event_type", "path"]
    )

    enriched = enrich_with_dims(spark, deduped, config)
    return enriched


def write_silver(df: DataFrame, config: Config) -> None:
    (
        df.write.mode("overwrite")
        .partitionBy(*config.partition_cols)
        .parquet(config.paths.silver)
    )


def read_silver(spark: SparkSession, config: Config) -> DataFrame:
    return spark.read.parquet(config.paths.silver)


def run(spark: SparkSession, config: Config) -> int:
    bronze = read_bronze(spark, config)
    silver = build_silver(spark, bronze, config)
    write_silver(silver, config)
    return read_silver(spark, config).count()
