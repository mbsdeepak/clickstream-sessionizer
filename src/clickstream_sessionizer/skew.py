"""Data-skew demonstration + handling via key salting.

The problem: a few hot users (bots, power users) own a huge share of events.
Any aggregation keyed by ``user_id`` sends all of one user's rows to a single
task, so a handful of straggler tasks dominate the stage runtime while the rest
sit idle -- classic partition skew.

The fix demonstrated here (salting):
    1. Add a random salt bucket to each row:  salt = rand() % factor
    2. Aggregate on (user_id, salt) -- the hot user's rows now spread across
       ``factor`` tasks instead of one (partial aggregation).
    3. Re-aggregate the partials on user_id alone (tiny second shuffle).

For a *count* the two-phase combine is exact. The same salt-then-recombine
pattern also rescues skewed JOINs (salt the big side, explode the small side).

``spark.sql.adaptive.skewJoin.enabled`` handles skewed shuffle *joins*
automatically in Spark 3+/4, but salting is the portable, explicit technique
interviewers ask about and applies to aggregations AQE won't split.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .config import Config


def key_skew_report(events: DataFrame, top_n: int = 10) -> DataFrame:
    """Show the skew: event count per user, most frequent first."""
    return (
        events.groupBy("user_id")
        .agg(F.count(F.lit(1)).alias("event_count"))
        .orderBy(F.desc("event_count"))
        .limit(top_n)
    )


def naive_counts(events: DataFrame) -> DataFrame:
    """The straightforward (skew-prone) per-user aggregation."""
    return events.groupBy("user_id").agg(F.count(F.lit(1)).alias("event_count"))


def salted_counts(events: DataFrame, config: Config) -> DataFrame:
    """Skew-resistant per-user aggregation using two-phase salted counting.

    Produces identical results to :func:`naive_counts` but distributes the work
    for hot keys across ``salting.factor`` tasks.
    """
    factor = config.salting.factor

    # Phase 1: partial counts keyed by (user_id, salt). The salt spreads each
    # user's rows across `factor` reduce tasks instead of collapsing to one.
    salted = events.withColumn(
        "salt", (F.rand(seed=7) * factor).cast("int")
    )
    partial = salted.groupBy("user_id", "salt").agg(
        F.count(F.lit(1)).alias("partial_count")
    )

    # Phase 2: combine the partials back per user. This shuffle is small -- at
    # most `factor` rows per user -- so no single task is overwhelmed.
    final = partial.groupBy("user_id").agg(
        F.sum("partial_count").alias("event_count")
    )
    return final


def demo(events: DataFrame, config: Config) -> dict[str, int]:
    """Run both strategies, prove they agree, and return quick diagnostics.

    Returns a small dict of metrics (kept lightweight so it can be printed in
    the pipeline summary without dumping whole DataFrames).
    """
    naive = naive_counts(events)
    salted = salted_counts(events, config)

    # Correctness check: the salted total must equal the naive total.
    naive_total = naive.agg(F.sum("event_count")).first()[0]
    salted_total = salted.agg(F.sum("event_count")).first()[0]

    top = key_skew_report(events, top_n=1).first()
    return {
        "naive_total_events": int(naive_total),
        "salted_total_events": int(salted_total),
        "totals_match": int(naive_total == salted_total),
        "hottest_user_events": int(top["event_count"]) if top else 0,
        "salt_factor": config.salting.factor,
    }
