"""Clickstream Sessionizer.

A PySpark lakehouse that ingests raw clickstream events and turns them into
sessions and business analytics using a bronze -> silver -> gold medallion
architecture.

Highlights:
    * Gap-based sessionization implemented with window functions.
    * Data-skew handling via key salting for hot users.
    * Broadcast joins against a small dimension table.
    * Partitioned Parquet output for efficient pruning.
    * A Structured Streaming variant using ``session_window``.
"""

__version__ = "0.1.0"
