"""CLI entry point orchestrating the medallion pipeline.

Usage::

    python -m clickstream_sessionizer.pipeline gen-data
    python -m clickstream_sessionizer.pipeline bronze
    python -m clickstream_sessionizer.pipeline silver
    python -m clickstream_sessionizer.pipeline sessions
    python -m clickstream_sessionizer.pipeline gold
    python -m clickstream_sessionizer.pipeline all       # full batch pipeline
    python -m clickstream_sessionizer.pipeline stream     # streaming variant
"""

from __future__ import annotations

import argparse
import sys

from . import bronze, gold, sessionize, silver, skew, streaming
from .config import Config, load_config
from .generate_events import generate
from .spark_session import build_spark


def _banner(title: str) -> None:
    print(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


def cmd_gen_data(config: Config) -> None:
    _banner("Generating synthetic clickstream")
    counts = generate(config)
    total = sum(counts.values())
    for day, n in counts.items():
        print(f"  {day}: {n:>8,} events")
    print(f"  ----\n  total: {total:,} raw events written to {config.paths.raw}")


def cmd_bronze(config: Config) -> None:
    _banner("BRONZE: raw JSON -> typed Parquet")
    spark = build_spark(config)
    n = bronze.run(spark, config)
    print(f"  bronze rows: {n:,}")


def cmd_silver(config: Config) -> None:
    _banner("SILVER: clean, dedupe, filter bots, enrich (broadcast join)")
    spark = build_spark(config)
    n = silver.run(spark, config)
    print(f"  silver rows: {n:,}")


def cmd_sessions(config: Config) -> None:
    _banner("SESSIONIZE: gap-based sessions via window functions")
    spark = build_spark(config)
    n = sessionize.run(spark, config)
    print(f"  sessions: {n:,}")


def cmd_gold(config: Config) -> None:
    _banner("GOLD: funnel, daily KPIs, top paths, retention")
    spark = build_spark(config)
    counts = gold.run(spark, config)
    for name, n in counts.items():
        print(f"  gold/{name}: {n:,} rows")


def cmd_all(config: Config) -> None:
    """Full batch pipeline in one SparkSession, with a summary + samples."""
    cmd_gen_data(config)

    spark = build_spark(config)

    _banner("BRONZE: raw JSON -> typed Parquet")
    n_bronze = bronze.run(spark, config)
    print(f"  bronze rows: {n_bronze:,}")

    _banner("SILVER: clean, dedupe, filter bots, enrich (broadcast join)")
    n_silver = silver.run(spark, config)
    print(f"  silver rows: {n_silver:,}")

    _banner("SKEW: hot-user skew + salted aggregation (naive vs salted)")
    silver_df = silver.read_silver(spark, config)
    print("  top hot users by event volume:")
    skew.key_skew_report(silver_df, top_n=5).show(truncate=False)
    metrics = skew.demo(silver_df, config)
    print(f"  skew metrics: {metrics}")

    _banner("SESSIONIZE: gap-based sessions via window functions")
    n_sessions = sessionize.run(spark, config)
    print(f"  sessions: {n_sessions:,}")
    print("  sample sessions:")
    sessionize.read_sessions(spark, config).select(
        "user_id", "start_ts", "end_ts", "duration_sec",
        "num_events", "num_pageviews", "converted", "bounce",
    ).show(8, truncate=False)

    _banner("GOLD: funnel, daily KPIs, top paths, retention")
    counts = gold.run(spark, config)
    for name, n in counts.items():
        print(f"  gold/{name}: {n:,} rows")

    print("\n  --- funnel ---")
    spark.read.parquet(f"{config.paths.gold}/funnel").orderBy("stage").show(truncate=False)
    print("  --- daily_sessions ---")
    spark.read.parquet(f"{config.paths.gold}/daily_sessions").show(truncate=False)
    print("  --- returning_users ---")
    spark.read.parquet(f"{config.paths.gold}/returning_users").show(truncate=False)

    _banner("PIPELINE SUMMARY")
    print(f"  bronze : {n_bronze:,} rows")
    print(f"  silver : {n_silver:,} rows")
    print(f"  sessions: {n_sessions:,} rows")
    print(f"  gold tables: {list(counts.keys())}")
    print("  status : OK\n")


def cmd_stream(config: Config) -> None:
    _banner("STREAMING: session_window sessionization (availableNow)")
    spark = build_spark(config, app_suffix="stream")
    streaming.run(spark, config, sink="console", await_termination=True)
    print("  streaming job drained all available input and stopped. OK")


COMMANDS = {
    "gen-data": cmd_gen_data,
    "bronze": cmd_bronze,
    "silver": cmd_silver,
    "sessions": cmd_sessions,
    "gold": cmd_gold,
    "all": cmd_all,
    "stream": cmd_stream,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="clickstream-sessionizer",
        description="PySpark clickstream sessionization lakehouse (medallion).",
    )
    parser.add_argument("command", choices=sorted(COMMANDS.keys()))
    parser.add_argument(
        "--config", default=None, help="Path to config.yaml (default: conf/config.yaml)"
    )
    parser.add_argument(
        "--events", type=int, default=None,
        help="Override generator num_events (gen-data / all).",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.events is not None:
        # Rebuild the frozen generator dataclass with the override.
        from dataclasses import replace
        config = replace(config, generator=replace(config.generator, num_events=args.events))

    COMMANDS[args.command](config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
