"""Configuration loading.

Reads ``conf/config.yaml`` into a set of frozen dataclasses so the rest of the
codebase gets typed, autocomplete-friendly access instead of poking at raw
dicts. Relative paths in the YAML are resolved against the project root so the
pipeline behaves the same regardless of the current working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Project root = two levels up from this file: src/clickstream_sessionizer/ -> root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "conf" / "config.yaml"


@dataclass(frozen=True)
class Paths:
    raw: str
    bronze: str
    silver: str
    sessions: str
    gold: str
    checkpoints: str


@dataclass(frozen=True)
class Salting:
    factor: int
    hot_key_threshold: int


@dataclass(frozen=True)
class Generator:
    num_events: int
    num_users: int
    num_hot_users: int
    hot_user_share: float
    num_days: int
    seed: int


@dataclass(frozen=True)
class SparkConf:
    master: str
    app_name: str
    shuffle_partitions: int
    adaptive_enabled: bool


@dataclass(frozen=True)
class Config:
    session_gap_minutes: int
    paths: Paths
    partition_cols: list[str]
    salting: Salting
    generator: Generator
    spark: SparkConf

    @property
    def session_gap_seconds(self) -> int:
        return self.session_gap_minutes * 60


def _resolve(path_str: str, root: Path) -> str:
    """Make a possibly-relative config path absolute against the project root."""
    p = Path(path_str)
    return str(p if p.is_absolute() else (root / p))


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load and validate configuration from YAML.

    Args:
        path: Optional path to a config file. Defaults to ``conf/config.yaml``.

    Returns:
        A fully-populated, frozen :class:`Config`.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    root = PROJECT_ROOT
    paths_raw = raw["paths"]
    paths = Paths(**{k: _resolve(v, root) for k, v in paths_raw.items()})

    return Config(
        session_gap_minutes=int(raw["session_gap_minutes"]),
        paths=paths,
        partition_cols=list(raw["partition_cols"]),
        salting=Salting(**raw["salting"]),
        generator=Generator(**raw["generator"]),
        spark=SparkConf(**raw["spark"]),
    )
