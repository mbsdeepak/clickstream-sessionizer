"""Core: gap-based sessionization with window functions.

A "session" is a maximal run of a single user's events where consecutive events
are no further apart than the inactivity gap (default 30 min). This is the
canonical web-analytics definition and the heart of this project.

Algorithm (all in SQL/window functions, so it runs distributed):

    1. Order each user's events by time:   W = partitionBy(user).orderBy(ts)
    2. Look back one event:                prev_ts = lag(ts) over W
    3. Flag a session boundary when the gap exceeds the threshold, OR when
       there is no previous event (first event of the user):
                                           is_new = gap > threshold  (1/0)
    4. Cumulative sum of the boundary flags gives a per-user session index:
                                           sess_idx = sum(is_new) over W-running
    5. session_id = hash(user_id, sess_idx) -> globally unique, stable id.

Steps 1-4 are exactly how you'd sessionize on a real cluster. The only shuffle
is the window's partitionBy(user_id); ``skew.py`` shows how to keep that shuffle
from being dominated by a few hot users.

Finally we aggregate the per-event, session-tagged frame into one row per
session with the metrics an analyst actually wants (duration, funnel flags,
entry/exit path, bounce).
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from .config import Config
from .silver import read_silver


def add_session_ids(events: DataFrame, gap_seconds: int) -> DataFrame:
    """Tag each event with a ``session_id`` using the gap algorithm.

    Args:
        events: Silver-level events; must contain ``user_id`` and
            ``event_timestamp`` (TimestampType).
        gap_seconds: Inactivity gap that closes a session.

    Returns:
        The input frame plus a ``session_id`` column.
    """
    # Ordered window over each user's timeline.
    w_ordered = Window.partitionBy("user_id").orderBy("event_timestamp")
    # Running window (unbounded preceding -> current) for the cumulative sum.
    w_running = (
        Window.partitionBy("user_id")
        .orderBy("event_timestamp")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )

    prev_ts = F.lag("event_timestamp").over(w_ordered)
    gap_secs = F.unix_timestamp("event_timestamp") - F.unix_timestamp(prev_ts)

    tagged = (
        events
        # A boundary opens a new session: first event (prev is null) OR the gap
        # since the previous event exceeds the threshold.
        .withColumn(
            "is_session_start",
            F.when(prev_ts.isNull() | (gap_secs > gap_seconds), 1).otherwise(0),
        )
        # Cumulative count of boundaries = this user's session index (1, 2, ...).
        .withColumn("session_index", F.sum("is_session_start").over(w_running))
        # Stable, collision-resistant global id from (user, index).
        .withColumn(
            "session_id",
            F.sha2(F.concat_ws("|", F.col("user_id"), F.col("session_index").cast("string")), 256),
        )
        .drop("is_session_start")
    )
    return tagged


def build_sessions(events: DataFrame, config: Config) -> DataFrame:
    """Aggregate session-tagged events into one row per session.

    Output columns:
        session_id, user_id, start_ts, end_ts, duration_sec, num_events,
        num_pageviews, entry_path, exit_path, converted, bounce, event_date.
    """
    tagged = add_session_ids(events, config.session_gap_seconds)

    # first/last by time within the session need an ordered window; we grab the
    # entry/exit path with argmin/argmax on the timestamp via struct ordering.
    w_sess = Window.partitionBy("session_id")

    with_edges = (
        tagged
        # struct(ts, path) so min()/max() pick the path at the earliest/latest ts.
        .withColumn(
            "entry_path",
            F.min(F.struct("event_timestamp", "path")).over(w_sess)["path"],
        )
        .withColumn(
            "exit_path",
            F.max(F.struct("event_timestamp", "path")).over(w_sess)["path"],
        )
    )

    sessions = (
        with_edges.groupBy("session_id", "user_id")
        .agg(
            F.min("event_timestamp").alias("start_ts"),
            F.max("event_timestamp").alias("end_ts"),
            F.count(F.lit(1)).alias("num_events"),
            F.sum(F.when(F.col("event_type") == "page_view", 1).otherwise(0)).alias(
                "num_pageviews"
            ),
            F.max(F.when(F.col("event_type") == "purchase", True).otherwise(False)).alias(
                "converted"
            ),
            F.first("entry_path", ignorenulls=True).alias("entry_path"),
            F.first("exit_path", ignorenulls=True).alias("exit_path"),
        )
        .withColumn(
            "duration_sec",
            (F.unix_timestamp("end_ts") - F.unix_timestamp("start_ts")).cast("long"),
        )
        # A "bounce" is a single-event session -- the user left immediately.
        .withColumn("bounce", F.col("num_events") == 1)
        # Partition sessions by their start date for pruning downstream.
        .withColumn("event_date", F.to_date("start_ts"))
    )
    return sessions


def write_sessions(df: DataFrame, config: Config) -> None:
    (
        df.write.mode("overwrite")
        .partitionBy(*config.partition_cols)
        .parquet(config.paths.sessions)
    )


def read_sessions(spark: SparkSession, config: Config) -> DataFrame:
    return spark.read.parquet(config.paths.sessions)


def run(spark: SparkSession, config: Config) -> int:
    silver = read_silver(spark, config)
    sessions = build_sessions(silver, config)
    write_sessions(sessions, config)
    return read_sessions(spark, config).count()
