"""Synthetic clickstream generator.

Produces realistic raw events and writes them as gzipped JSON partitioned by
``event_date`` -- exactly the shape a real ingestion pipeline would land in the
raw zone of a lake.

Realism baked in on purpose so the downstream layers have something to do:
    * A handful of HOT users (power users / crawlers) own a large share of
      events, creating the key skew that ``skew.py`` later handles.
    * Events arrive in temporal *bursts* per user so gap-based sessionization
      produces meaningful multi-session histories.
    * A mix of event types with a plausible funnel (many views, few purchases).
    * Some bot user-agents so ``silver.py`` has something to filter.
    * Occasional nulls / duplicate events so cleaning is non-trivial.

The generator is driver-side Python (deterministic, seeded) rather than Spark
so the sample data is fully reproducible; for genuinely huge volumes you would
swap this for a distributed generator, but the *schema* is the contract that
matters and it is identical either way.
"""

from __future__ import annotations

import gzip
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from .config import Config

EVENT_TYPES = ["page_view", "search", "click", "add_to_cart", "purchase"]
# Weights encode a funnel: lots of views, fewer clicks, few purchases.
EVENT_WEIGHTS = [0.55, 0.12, 0.20, 0.09, 0.04]

PATHS = [
    "/", "/home", "/search", "/category/shoes", "/category/electronics",
    "/product/1234", "/product/5678", "/product/9012", "/cart", "/checkout",
    "/account", "/deals", "/blog/spark-tips",
]
REFERRERS = [
    "https://www.google.com/", "https://www.bing.com/", "https://t.co/",
    "https://news.ycombinator.com/", "direct", "https://www.reddit.com/",
]
DEVICES = ["desktop", "mobile", "tablet"]
DEVICE_WEIGHTS = [0.45, 0.48, 0.07]
COUNTRIES = ["US", "IN", "GB", "DE", "BR", "JP", "CA", "AU", "FR", "NG"]

HUMAN_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/17.4",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4) Mobile/15E148",
    "Mozilla/5.0 (Linux; Android 14) Chrome/124.0 Mobile",
]
BOT_AGENTS = [
    "Googlebot/2.1 (+http://www.google.com/bot.html)",
    "bingbot/2.0 (+http://www.bing.com/bingbot.htm)",
    "python-requests/2.31.0",
    "AhrefsBot/7.0",
]


def _weighted_choice(rng: random.Random, items: list[str], weights: list[float]) -> str:
    return rng.choices(items, weights=weights, k=1)[0]


def _iter_events(config: Config) -> Iterator[dict]:
    """Yield synthetic event dicts.

    Each user gets a country/device "profile" and a set of session bursts. Hot
    users get many more bursts, concentrating rows on their keys.
    """
    g = config.generator
    rng = random.Random(g.seed)

    hot_user_ids = [f"u_hot_{i}" for i in range(g.num_hot_users)]
    normal_user_ids = [f"u_{i}" for i in range(g.num_users)]

    # Stable per-user profile so a user looks consistent across sessions.
    def profile(uid: str) -> dict:
        pr = random.Random(hash(uid) & 0xFFFFFFFF)
        return {
            "country": pr.choice(COUNTRIES),
            "device": _weighted_choice(pr, DEVICES, DEVICE_WEIGHTS),
            "is_bot": uid.startswith("u_hot_") and pr.random() < 0.4,
        }

    hot_target = int(g.num_events * g.hot_user_share)
    normal_target = g.num_events - hot_target
    start_day = datetime(2026, 8, 1, tzinfo=timezone.utc)

    def emit_user_events(uid: str, budget: int) -> Iterator[dict]:
        """Emit ~budget events for one user across several session bursts."""
        prof = profile(uid)
        produced = 0
        while produced < budget:
            day_offset = rng.randint(0, g.num_days - 1)
            # Burst start somewhere in the day.
            burst_start = start_day + timedelta(
                days=day_offset,
                hours=rng.randint(0, 23),
                minutes=rng.randint(0, 59),
                seconds=rng.randint(0, 59),
            )
            burst_len = rng.randint(1, 12)
            ts = burst_start
            saw_cart = False
            for _ in range(burst_len):
                if produced >= budget:
                    break
                etype = _weighted_choice(rng, EVENT_TYPES, EVENT_WEIGHTS)
                # Make the funnel coherent: purchase implies a cart earlier.
                if etype == "add_to_cart":
                    saw_cart = True
                if etype == "purchase" and not saw_cart:
                    etype = "add_to_cart"
                    saw_cart = True

                event = {
                    "event_id": f"{uid}-{produced}-{rng.getrandbits(32)}",
                    "user_id": uid,
                    # ISO-8601 with Z; bronze parses this schema-on-read.
                    "event_timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "event_type": etype,
                    "path": rng.choice(PATHS),
                    "referrer": rng.choice(REFERRERS),
                    "device": prof["device"],
                    "country": prof["country"],
                    "user_agent": (
                        rng.choice(BOT_AGENTS) if prof["is_bot"]
                        else rng.choice(HUMAN_AGENTS)
                    ),
                }

                # Inject a little dirtiness for silver.py to clean:
                # ~0.5% missing user_id, ~1% duplicate events.
                if rng.random() < 0.005:
                    event["user_id"] = None
                yield event
                if rng.random() < 0.01:
                    yield dict(event)  # exact duplicate row

                produced += 1
                # Intra-burst gaps are short (seconds/minutes) so they stay in
                # one session; the gap BETWEEN bursts is large by construction.
                ts += timedelta(seconds=rng.randint(3, 240))

    # Hot users.
    per_hot = max(1, hot_target // max(1, len(hot_user_ids)))
    for uid in hot_user_ids:
        yield from emit_user_events(uid, per_hot)

    # Normal users: pick a random subset large enough to hit the budget.
    produced_normal = 0
    while produced_normal < normal_target:
        uid = rng.choice(normal_user_ids)
        budget = rng.randint(1, 30)
        before = produced_normal
        for ev in emit_user_events(uid, budget):
            yield ev
            produced_normal += 1
            if produced_normal >= normal_target:
                break
        if produced_normal == before:  # safety against zero-progress
            produced_normal += 1


def generate(config: Config) -> dict[str, int]:
    """Generate raw events and write gzipped JSON partitioned by event_date.

    Returns:
        Mapping of ``event_date`` partition -> number of rows written.
    """
    raw_dir = Path(config.paths.raw)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Bucket rows by date partition, writing one gz file per partition.
    buffers: dict[str, list[str]] = {}
    for ev in _iter_events(config):
        event_date = ev["event_timestamp"][:10]  # YYYY-MM-DD
        buffers.setdefault(event_date, []).append(json.dumps(ev))

    counts: dict[str, int] = {}
    for event_date, lines in sorted(buffers.items()):
        part_dir = raw_dir / f"event_date={event_date}"
        part_dir.mkdir(parents=True, exist_ok=True)
        out_path = part_dir / "events.json.gz"
        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        counts[event_date] = len(lines)

    return counts
