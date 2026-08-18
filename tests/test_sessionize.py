"""Tests for the core gap-based sessionization logic."""

from __future__ import annotations

from datetime import datetime

from pyspark.sql import functions as F

from clickstream_sessionizer.config import (
    Config, Generator, Paths, Salting, SparkConf,
)
from clickstream_sessionizer.sessionize import add_session_ids, build_sessions


def _config(gap_minutes: int = 30) -> Config:
    """Minimal config for unit tests (paths unused by the pure functions)."""
    return Config(
        session_gap_minutes=gap_minutes,
        paths=Paths("r", "b", "s", "se", "g", "c"),
        partition_cols=["event_date"],
        salting=Salting(factor=4, hot_key_threshold=10),
        generator=Generator(100, 10, 1, 0.3, 1, 1),
        spark=SparkConf("local[2]", "t", 2, True),
    )


def _ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def test_gap_splits_into_two_sessions(spark):
    """Two events 40 min apart (> 30 min gap) must be in different sessions."""
    rows = [
        ("u1", _ts("2026-08-01 10:00:00"), "page_view", "/a"),
        ("u1", _ts("2026-08-01 10:05:00"), "click", "/b"),      # +5m  -> same
        ("u1", _ts("2026-08-01 10:45:00"), "page_view", "/c"),  # +40m -> new
    ]
    df = spark.createDataFrame(rows, ["user_id", "event_timestamp", "event_type", "path"])

    tagged = add_session_ids(df, gap_seconds=30 * 60)
    distinct_sessions = tagged.select("session_id").distinct().count()
    assert distinct_sessions == 2

    # First two events share a session; third is on its own.
    idx = {r["path"]: r["session_index"] for r in tagged.collect()}
    assert idx["/a"] == idx["/b"] == 1
    assert idx["/c"] == 2


def test_boundary_exactly_at_gap_stays_same_session(spark):
    """A gap of exactly the threshold is NOT a new session (strictly greater)."""
    rows = [
        ("u1", _ts("2026-08-01 10:00:00"), "page_view", "/a"),
        ("u1", _ts("2026-08-01 10:30:00"), "page_view", "/b"),  # exactly +30m
    ]
    df = spark.createDataFrame(rows, ["user_id", "event_timestamp", "event_type", "path"])
    tagged = add_session_ids(df, gap_seconds=30 * 60)
    assert tagged.select("session_id").distinct().count() == 1


def test_users_are_independent(spark):
    """Sessionization must be per-user; interleaved users don't merge."""
    rows = [
        ("u1", _ts("2026-08-01 10:00:00"), "page_view", "/a"),
        ("u2", _ts("2026-08-01 10:01:00"), "page_view", "/x"),
        ("u1", _ts("2026-08-01 10:02:00"), "page_view", "/b"),
    ]
    df = spark.createDataFrame(rows, ["user_id", "event_timestamp", "event_type", "path"])
    tagged = add_session_ids(df, gap_seconds=30 * 60)
    # u1 -> 1 session, u2 -> 1 session => 2 distinct sessions total.
    assert tagged.select("session_id").distinct().count() == 2


def test_session_metrics_bounce_and_convert(spark):
    """build_sessions computes duration, bounce, convert, entry/exit correctly."""
    rows = [
        # Session A for u1: 3 events, ends in a purchase (converted, not bounce).
        ("u1", _ts("2026-08-01 09:00:00"), "page_view", "/home"),
        ("u1", _ts("2026-08-01 09:02:00"), "add_to_cart", "/cart"),
        ("u1", _ts("2026-08-01 09:05:00"), "purchase", "/checkout"),
        # Session B for u2: single event => bounce, not converted.
        ("u2", _ts("2026-08-01 09:00:00"), "page_view", "/landing"),
    ]
    df = spark.createDataFrame(rows, ["user_id", "event_timestamp", "event_type", "path"])
    sessions = build_sessions(df, _config(30)).collect()
    by_user = {r["user_id"]: r for r in sessions}

    a = by_user["u1"]
    assert a["num_events"] == 3
    assert a["num_pageviews"] == 1
    assert a["duration_sec"] == 300  # 5 minutes
    assert a["converted"] is True
    assert a["bounce"] is False
    assert a["entry_path"] == "/home"
    assert a["exit_path"] == "/checkout"

    b = by_user["u2"]
    assert b["num_events"] == 1
    assert b["bounce"] is True
    assert b["converted"] is False
    assert b["duration_sec"] == 0


def test_multiple_sessions_per_user_counted(spark):
    """A user with two gap-separated bursts yields two session rows."""
    rows = [
        ("u1", _ts("2026-08-01 08:00:00"), "page_view", "/a"),
        ("u1", _ts("2026-08-01 08:10:00"), "page_view", "/b"),
        ("u1", _ts("2026-08-01 12:00:00"), "page_view", "/c"),  # hours later
    ]
    df = spark.createDataFrame(rows, ["user_id", "event_timestamp", "event_type", "path"])
    sessions = build_sessions(df, _config(30))
    assert sessions.filter(F.col("user_id") == "u1").count() == 2
