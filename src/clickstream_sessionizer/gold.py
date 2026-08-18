"""Gold layer: business analytics.

Consumes the sessions table (and silver events for the funnel) and produces the
aggregate tables an analyst / dashboard would query:

    * funnel           -- view -> cart -> purchase counts and conversion rates
    * daily_sessions   -- sessions/day, avg duration, bounce rate, conversions
    * top_paths        -- most common entry paths by session volume
    * returning_users  -- returning-user (retention proxy) metrics

Each is written as its own partition-free Parquet table under data/gold/<name>.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .config import Config
from .sessionize import read_sessions
from .silver import read_silver


def funnel(events: DataFrame) -> DataFrame:
    """View -> add_to_cart -> purchase funnel with absolute counts and rates."""
    counts = events.groupBy("event_type").agg(F.count(F.lit(1)).alias("events"))
    # Pull the three funnel stages into scalars so we can compute rates.
    row = {r["event_type"]: r["events"] for r in counts.collect()}
    views = row.get("page_view", 0)
    carts = row.get("add_to_cart", 0)
    purchases = row.get("purchase", 0)

    spark = events.sparkSession
    data = [
        ("1_view", views, 1.0),
        ("2_add_to_cart", carts, (carts / views) if views else 0.0),
        ("3_purchase", purchases, (purchases / views) if views else 0.0),
    ]
    return spark.createDataFrame(data, ["stage", "count", "rate_from_view"])


def daily_sessions(sessions: DataFrame) -> DataFrame:
    """Per-day session KPIs."""
    return (
        sessions.groupBy("event_date")
        .agg(
            F.count(F.lit(1)).alias("num_sessions"),
            F.countDistinct("user_id").alias("distinct_users"),
            F.round(F.avg("duration_sec"), 1).alias("avg_duration_sec"),
            F.round(F.avg(F.col("bounce").cast("int")), 4).alias("bounce_rate"),
            F.round(F.avg(F.col("converted").cast("int")), 4).alias("conversion_rate"),
        )
        .orderBy("event_date")
    )


def top_paths(sessions: DataFrame, limit: int = 15) -> DataFrame:
    """Most common session entry paths by volume."""
    return (
        sessions.groupBy("entry_path")
        .agg(F.count(F.lit(1)).alias("sessions"))
        .orderBy(F.desc("sessions"))
        .limit(limit)
    )


def returning_users(sessions: DataFrame) -> DataFrame:
    """Retention proxy: how many users have >1 session, and their share."""
    per_user = sessions.groupBy("user_id").agg(
        F.count(F.lit(1)).alias("sessions_per_user")
    )
    total_users = per_user.count()
    returning = per_user.filter(F.col("sessions_per_user") > 1).count()
    spark = sessions.sparkSession
    rate = (returning / total_users) if total_users else 0.0
    return spark.createDataFrame(
        [(total_users, returning, round(rate, 4))],
        ["total_users", "returning_users", "returning_rate"],
    )


def _write(df: DataFrame, config: Config, name: str) -> None:
    df.write.mode("overwrite").parquet(f"{config.paths.gold}/{name}")


def run(spark: SparkSession, config: Config) -> dict[str, int]:
    """Build and persist all gold tables. Returns row counts per table."""
    events = read_silver(spark, config)
    sessions = read_sessions(spark, config)

    tables = {
        "funnel": funnel(events),
        "daily_sessions": daily_sessions(sessions),
        "top_paths": top_paths(sessions),
        "returning_users": returning_users(sessions),
    }
    counts: dict[str, int] = {}
    for name, df in tables.items():
        df = df.cache()
        _write(df, config, name)
        counts[name] = df.count()
    return counts
