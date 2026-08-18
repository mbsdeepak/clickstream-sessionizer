"""Tests for silver-layer cleaning: bot filtering and de-duplication."""

from __future__ import annotations

from datetime import datetime

from clickstream_sessionizer.config import (
    Config, Generator, Paths, Salting, SparkConf,
)
from clickstream_sessionizer.silver import build_silver


def _config() -> Config:
    return Config(
        session_gap_minutes=30,
        paths=Paths("r", "b", "s", "se", "g", "c"),
        partition_cols=["event_date"],
        salting=Salting(factor=4, hot_key_threshold=10),
        generator=Generator(100, 10, 1, 0.3, 1, 1),
        spark=SparkConf("local[2]", "t", 2, True),
    )


def _ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


_COLS = [
    "event_id", "user_id", "event_timestamp", "event_type", "path",
    "referrer", "device", "country", "user_agent", "event_date",
    "ingest_ts", "source_file",
]


def _row(eid, uid, ts, etype, path, ua):
    return (eid, uid, _ts(ts), etype, path, "direct", "mobile", "us", ua,
            _ts(ts).date(), _ts(ts), "file://x")


def test_bots_are_filtered(spark):
    rows = [
        _row("1", "u1", "2026-08-01 10:00:00", "page_view", "/a",
             "Mozilla/5.0 (Macintosh) Safari/17"),
        _row("2", "bot", "2026-08-01 10:00:00", "page_view", "/a",
             "Googlebot/2.1 (+http://www.google.com/bot.html)"),
        _row("3", "req", "2026-08-01 10:00:00", "page_view", "/a",
             "python-requests/2.31.0"),
    ]
    df = spark.createDataFrame(rows, _COLS)
    out = build_silver(spark, df, _config())
    users = {r["user_id"] for r in out.collect()}
    assert users == {"u1"}  # both bots removed


def test_null_key_rows_dropped(spark):
    rows = [
        _row("1", "u1", "2026-08-01 10:00:00", "page_view", "/a",
             "Mozilla/5.0 Safari"),
        _row("2", None, "2026-08-01 10:00:00", "page_view", "/a",
             "Mozilla/5.0 Safari"),
    ]
    df = spark.createDataFrame(rows, _COLS)
    out = build_silver(spark, df, _config())
    assert out.count() == 1


def test_exact_duplicates_removed(spark):
    dup = _row("1", "u1", "2026-08-01 10:00:00", "page_view", "/a",
               "Mozilla/5.0 Safari")
    rows = [
        dup,
        dup,  # exact duplicate event
        _row("2", "u1", "2026-08-01 10:01:00", "click", "/b",
             "Mozilla/5.0 Safari"),
    ]
    df = spark.createDataFrame(rows, _COLS)
    out = build_silver(spark, df, _config())
    assert out.count() == 2  # one copy of the duplicate + the click


def test_enrichment_columns_present(spark):
    """Broadcast-join enrichment adds country_name/region/form_factor."""
    rows = [
        _row("1", "u1", "2026-08-01 10:00:00", "page_view", "/a",
             "Mozilla/5.0 Safari"),
    ]
    df = spark.createDataFrame(rows, _COLS)
    out = build_silver(spark, df, _config())
    row = out.first()
    assert row["country_name"] == "United States"
    assert row["region"] == "Americas"
    assert row["form_factor"] == "small_screen"
