"""Bronze layer: raw JSON -> typed Parquet.

The bronze zone is a faithful, typed copy of the source with ingestion metadata
added. We apply an explicit schema (schema-on-read) rather than letting Spark
infer it: inference triggers a full extra pass over the data and can guess types
inconsistently between runs. Explicit schemas are the norm in production lakes.

No business filtering happens here -- bronze is deliberately "everything that
arrived", so we can always reprocess silver/gold without re-ingesting.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from .config import Config

# Explicit source schema. event_timestamp is read as string then cast, which is
# more robust than trusting JSON timestamp inference across locales.
RAW_SCHEMA = StructType(
    [
        StructField("event_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("event_timestamp", StringType(), True),
        StructField("event_type", StringType(), True),
        StructField("path", StringType(), True),
        StructField("referrer", StringType(), True),
        StructField("device", StringType(), True),
        StructField("country", StringType(), True),
        StructField("user_agent", StringType(), True),
    ]
)


def build_bronze(spark: SparkSession, config: Config) -> DataFrame:
    """Read raw JSON.gz and produce the typed bronze DataFrame."""
    raw = (
        spark.read.schema(RAW_SCHEMA)
        # recursiveFileLookup lets us read across the event_date=… partition
        # folders while still reconstructing the partition column ourselves.
        .option("recursiveFileLookup", "true")
        .json(config.paths.raw)
    )

    bronze = (
        raw
        # Cast the string timestamp to a real TimestampType once, up front.
        .withColumn("event_timestamp", F.to_timestamp("event_timestamp"))
        # Derive the partition column from the event time (source of truth),
        # not the folder name -- guards against mislabelled input folders.
        .withColumn("event_date", F.to_date("event_timestamp"))
        # Ingestion lineage: when did we load this, and from where.
        .withColumn("ingest_ts", F.current_timestamp())
        .withColumn("source_file", F.input_file_name())
    )
    return bronze


def write_bronze(df: DataFrame, config: Config) -> None:
    """Persist bronze as Parquet partitioned by event_date."""
    (
        df.write.mode("overwrite")
        .partitionBy(*config.partition_cols)
        .parquet(config.paths.bronze)
    )


def read_bronze(spark: SparkSession, config: Config) -> DataFrame:
    return spark.read.parquet(config.paths.bronze)


def run(spark: SparkSession, config: Config) -> int:
    """Full bronze step. Returns row count written."""
    bronze = build_bronze(spark, config)
    write_bronze(bronze, config)
    return read_bronze(spark, config).count()
