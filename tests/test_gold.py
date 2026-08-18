"""Tests for gold-layer analytics on tiny fixtures."""

from __future__ import annotations

from datetime import datetime

from clickstream_sessionizer.gold import funnel, returning_users
from clickstream_sessionizer.skew import naive_counts, salted_counts
from clickstream_sessionizer.config import (
    Config, Generator, Paths, Salting, SparkConf,
)


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


def test_funnel_counts_and_rates(spark):
    rows = [
        ("u1", "page_view"), ("u1", "page_view"), ("u2", "page_view"),
        ("u3", "page_view"), ("u1", "add_to_cart"), ("u2", "add_to_cart"),
        ("u1", "purchase"),
    ]
    df = spark.createDataFrame(rows, ["user_id", "event_type"])
    out = {r["stage"]: r for r in funnel(df).collect()}

    assert out["1_view"]["count"] == 4
    assert out["2_add_to_cart"]["count"] == 2
    assert out["3_purchase"]["count"] == 1
    assert abs(out["2_add_to_cart"]["rate_from_view"] - 0.5) < 1e-9
    assert abs(out["3_purchase"]["rate_from_view"] - 0.25) < 1e-9


def test_returning_users(spark):
    # sessions frame: u1 has 2 sessions, u2 has 1.
    rows = [("s1", "u1"), ("s2", "u1"), ("s3", "u2")]
    df = spark.createDataFrame(rows, ["session_id", "user_id"])
    out = returning_users(df).first()
    assert out["total_users"] == 2
    assert out["returning_users"] == 1
    assert abs(out["returning_rate"] - 0.5) < 1e-9


def test_salted_counts_match_naive(spark):
    """The salting optimisation must not change the answer."""
    rows = [("hot", "page_view")] * 50 + [("cold", "click")] * 3
    df = spark.createDataFrame(rows, ["user_id", "event_type"])

    naive = {r["user_id"]: r["event_count"] for r in naive_counts(df).collect()}
    salted = {r["user_id"]: r["event_count"] for r in salted_counts(df, _config()).collect()}
    assert naive == salted
    assert naive["hot"] == 50
    assert naive["cold"] == 3
