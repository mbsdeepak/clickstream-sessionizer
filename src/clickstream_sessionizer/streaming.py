"""Structured Streaming sessionization.

The batch pipeline sessionizes with window functions; the streaming variant uses
Spark's built-in ``session_window``, which maintains session state across
micro-batches and merges events that fall within the inactivity gap.

Key streaming concepts demonstrated:
    * File source streaming over the raw zone (each raw file = a micro-batch).
    * Event-time ``withWatermark`` so Spark can bound state and finalise
      sessions once they are older than the watermark (late data past the
      watermark is dropped).
    * ``session_window(timestamp, gap)`` for stateful, gap-based sessions.
    * ``availableNow`` trigger so the job drains all currently-available input
      and stops -- ideal for a reproducible demo / CI, while the same code runs
      continuously in production by swapping the trigger.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.streaming import StreamingQuery

from .bronze import RAW_SCHEMA
from .config import Config


def stream_events(spark: SparkSession, config: Config) -> DataFrame:
    """Read the raw zone as a stream (one micro-batch per new file)."""
    return (
        spark.readStream.schema(RAW_SCHEMA)
        .option("recursiveFileLookup", "true")
        # maxFilesPerTrigger keeps micro-batches small and demonstrable.
        .option("maxFilesPerTrigger", 1)
        .json(config.paths.raw)
        .withColumn("event_timestamp", F.to_timestamp("event_timestamp"))
        .filter(F.col("user_id").isNotNull() & F.col("event_timestamp").isNotNull())
    )


def sessionize_stream(events: DataFrame, config: Config) -> DataFrame:
    """Aggregate the event stream into sessions using ``session_window``.

    The watermark tells Spark how long to wait for late events before it can
    emit and evict a session's state. We set it to the session gap plus a
    margin so genuinely in-gap late events still merge.
    """
    gap = f"{config.session_gap_minutes} minutes"
    watermark = f"{config.session_gap_minutes * 2} minutes"

    return (
        events.withWatermark("event_timestamp", watermark)
        .groupBy(
            F.col("user_id"),
            F.session_window(F.col("event_timestamp"), gap),
        )
        .agg(
            F.count(F.lit(1)).alias("num_events"),
            F.sum(F.when(F.col("event_type") == "page_view", 1).otherwise(0)).alias(
                "num_pageviews"
            ),
            F.max(F.when(F.col("event_type") == "purchase", True).otherwise(False)).alias(
                "converted"
            ),
        )
        .select(
            F.col("user_id"),
            F.col("session_window.start").alias("start_ts"),
            F.col("session_window.end").alias("end_ts"),
            (
                F.unix_timestamp("session_window.end")
                - F.unix_timestamp("session_window.start")
            ).alias("duration_sec"),
            "num_events",
            "num_pageviews",
            "converted",
        )
    )


def run(
    spark: SparkSession,
    config: Config,
    *,
    sink: str = "console",
    await_termination: bool = True,
) -> StreamingQuery:
    """Start the streaming sessionization job.

    Args:
        sink: ``"console"`` (prints each batch) or ``"parquet"`` (writes to the
            sessions path with a checkpoint).
        await_termination: If True, block until the ``availableNow`` trigger has
            drained all input, then return. Set False to manage the query
            yourself (e.g. in tests).

    Returns:
        The (possibly-completed) StreamingQuery handle.
    """
    events = stream_events(spark, config)
    sessions = sessionize_stream(events, config)

    writer = (
        sessions.writeStream
        # session_window requires append/complete/update; append emits a session
        # only once it is finalised past the watermark.
        .outputMode("append")
        .option("checkpointLocation", f"{config.paths.checkpoints}/streaming")
        # availableNow: process everything currently available, then stop.
        .trigger(availableNow=True)
    )

    if sink == "parquet":
        writer = writer.format("parquet").option(
            "path", f"{config.paths.sessions}_streaming"
        )
    else:
        writer = writer.format("console").option("truncate", "false").option(
            "numRows", 20
        )

    query = writer.start()
    if await_termination:
        query.awaitTermination()
    return query
