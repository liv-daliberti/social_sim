#!/usr/bin/env python3
"""Select the experiment subset from a raw Gamma API JSONL dump.

Reads data/raw_markets/markets_YYYY-MM-DD.jsonl and applies filters to
produce a curated set of ~50-100 politics/economics/geopolitics markets
suitable for the structured world-model consistency experiment.

Filters applied (in order):
  1. Binary markets only (Yes/No outcomes)
  2. Active and unresolved
  3. Category/keyword match (politics, elections, economics, geopolitics, …)
  4. Resolution window: 14–180 days from today
  5. Non-trivial yes_price: 0.10 < price < 0.90
  6. Minimum volume: >= $1,000 USD
  7. Dedup: max 1 market per event_id (keep highest-volume)

Output: data/selected_markets/selected_YYYY-MM-DD.jsonl
        data/selected_markets/selected_YYYY-MM-DD.manifest.json

Usage (from exp1_prospective/):
    python fetch_markets/select_markets.py
    python fetch_markets/select_markets.py --input data/raw_markets/markets_2026-06-05.jsonl
    python fetch_markets/select_markets.py --min-volume 5000 --max-days 90
    python fetch_markets/select_markets.py --top 100
    python fetch_markets/select_markets.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_RAW_DIR = _ROOT / "data" / "raw_markets"
_SEL_DIR = _ROOT / "data" / "selected_markets"

# ── topic filter ───────────────────────────────────────────────────────────────
# Categories that pass without keyword check (case-insensitive substring match)
_PASS_CATEGORIES: tuple[str, ...] = (
    "politics",
    "government",
    "elections",
    "election",
    "economics",
    "economy",
    "geopolitics",
    "science",
    "technology",
    "artificial intelligence",
    "ai",
    "climate",
    "environment",
    "international",
    "global",
    "foreign policy",
    "diplomacy",
    "military",
    "defense",
    "trade",
    "finance",
    "monetary",
    "fiscal",
    "regulation",
    "law",
    "legal",
    "court",
    "judiciary",
    "health",
    "medicine",
    "pandemic",
)

# Keywords that, if found in the question or event title, pass the market
_PASS_KEYWORDS: tuple[str, ...] = (
    # Political actors / processes
    "election", "vote", "ballot", "poll", "candidate", "president", "senator",
    "congress", "parliament", "prime minister", "chancellor", "cabinet",
    "administration", "white house", "supreme court", "federal", "legislation",
    "bill", "referendum", "impeach", "resign", "appoint", "confirmation",
    # Geopolitics / international
    "war", "ceasefire", "sanction", "treaty", "nato", "un ", "united nations",
    "invasion", "occupation", "troop", "military", "ukraine", "russia", "china",
    "taiwan", "iran", "israel", "gaza", "middle east", "europe", "eu ", "g7",
    "g20", "brics", "trade war", "tariff", "embargo",
    # Economics / finance
    "gdp", "inflation", "recession", "interest rate", "federal reserve", "fed ",
    "ecb", "imf", "world bank", "unemployment", "cpi", "ppi", "jobs report",
    "debt ceiling", "budget", "fiscal", "deficit", "surplus", "tax",
    "market crash", "stock market", "s&p", "nasdaq", "dow jones",
    # AI / tech / science
    "artificial intelligence", " ai ", "chatgpt", "openai", "deepmind",
    "climate change", "carbon", "emission", "temperature", "nuclear",
    "space", "nasa", "spacex", "drug approval", "fda", "vaccine", "trial",
)

# Categories that are explicitly excluded (no keyword redemption)
_BLOCK_CATEGORIES: tuple[str, ...] = (
    "crypto",
    "sports",
    "entertainment",
    "celebrity",
    "pop culture",
    "music",
    "movies",
    "tv",
    "gaming",
    "esports",
)

_BLOCK_KEYWORDS: tuple[str, ...] = (
    "bitcoin", "ethereum", "btc", "eth", "solana", "doge", "shib",
    "nba", "nfl", "mlb", "nhl", "nba", "soccer", "football", "basketball",
    "tennis", "golf", "formula 1", "f1 ", "ufc", "boxing",
    "oscar", "grammy", "emmy", "tony award", "box office",
    "taylor swift", "beyoncé", "kanye",
)


def _norm(s: str | None) -> str:
    return (s or "").lower()


def _passes_topic(r: dict) -> bool:
    cat = _norm(r.get("category"))
    q   = _norm(r.get("question"))
    title = _norm(r.get("event_title"))
    text  = f"{cat} {q} {title}"

    # Hard block
    for blk in _BLOCK_CATEGORIES:
        if blk in cat:
            return False
    for blk in _BLOCK_KEYWORDS:
        if blk in text:
            return False

    # Pass by category
    for pc in _PASS_CATEGORIES:
        if pc in cat:
            return True

    # Pass by keyword in question or title
    for kw in _PASS_KEYWORDS:
        if kw in q or kw in title:
            return True

    return False


# ── filters ────────────────────────────────────────────────────────────────────

def _to_epoch(v: str | None) -> float | None:
    if not v:
        return None
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def is_binary(r: dict) -> bool:
    outcomes = r.get("outcomes") or []
    return len(outcomes) == 2


def is_active_unresolved(r: dict) -> bool:
    return (
        r.get("active")   is not False
        and r.get("closed")   is not True
        and r.get("resolved") is not True
    )


def days_to_resolution(r: dict, now_ts: float) -> float | None:
    end = _to_epoch(r.get("end_time"))
    if end is None:
        return None
    return (end - now_ts) / 86400


def apply_filters(
    records: list[dict],
    *,
    min_days: int,
    max_days: int,
    min_price: float,
    max_price: float,
    min_volume: float,
    now_ts: float,
) -> tuple[list[dict], dict[str, int]]:
    counts: dict[str, int] = {
        "total":      len(records),
        "binary":     0,
        "active":     0,
        "topic":      0,
        "days":       0,
        "price":      0,
        "volume":     0,
    }
    passed: list[dict] = []

    for r in records:
        if not is_binary(r):
            continue
        counts["binary"] += 1

        if not is_active_unresolved(r):
            continue
        counts["active"] += 1

        if not _passes_topic(r):
            continue
        counts["topic"] += 1

        dtd = days_to_resolution(r, now_ts)
        if dtd is None or not (min_days <= dtd <= max_days):
            continue
        counts["days"] += 1

        yp = r.get("yes_price")
        if yp is None or not (min_price < yp < max_price):
            continue
        counts["price"] += 1

        vol = r.get("volume_usd") or 0.0
        if vol < min_volume:
            continue
        counts["volume"] += 1

        passed.append(r)

    return passed, counts


def dedup_by_event(records: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for r in records:
        eid = r.get("event_id") or r["market_id"]
        vol = r.get("volume_usd") or 0.0
        if eid not in best or vol > (best[eid].get("volume_usd") or 0.0):
            best[eid] = r
    return list(best.values())


# ── helpers ────────────────────────────────────────────────────────────────────

def _latest_markets_jsonl(raw_dir: Path) -> Path | None:
    for f in sorted(raw_dir.glob("markets_*.jsonl"), reverse=True):
        if "price_history" not in f.name:
            return f
    return None


def _enrich(r: dict, now_ts: float, fetch_date: str) -> dict:
    dtd = days_to_resolution(r, now_ts)
    out = dict(r)
    out["task_id"]             = f"pm_{r['market_id']}_{fetch_date}"
    out["days_to_resolution"]  = round(dtd, 1) if dtd is not None else None
    out["selected_at"]         = datetime.now(timezone.utc).isoformat()
    return out


# ── cli ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select experiment subset from raw Polymarket JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input",      type=Path,  default=None,
                    help="Raw markets JSONL. Defaults to latest in data/raw_markets/.")
    ap.add_argument("--out",        type=Path,  default=_SEL_DIR,
                    help="Output directory.")
    ap.add_argument("--min-days",   type=int,   default=14,
                    help="Min days to resolution (inclusive).")
    ap.add_argument("--max-days",   type=int,   default=180,
                    help="Max days to resolution (inclusive).")
    ap.add_argument("--min-price",  type=float, default=0.10,
                    help="Min yes_price (exclusive).")
    ap.add_argument("--max-price",  type=float, default=0.90,
                    help="Max yes_price (exclusive).")
    ap.add_argument("--min-volume", type=float, default=1_000.0,
                    help="Min lifetime volume in USD.")
    ap.add_argument("--top",        type=int,   default=0,
                    help="Keep top-N by volume after all filters (0 = keep all).")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Print stats but don't write output files.")
    args = ap.parse_args()

    input_path = args.input or _latest_markets_jsonl(_RAW_DIR)
    if input_path is None or not input_path.exists():
        print(f"No markets JSONL found in {_RAW_DIR}.  Run fetch_markets.py first.")
        sys.exit(1)

    print(f"Reading  {input_path}")
    with input_path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"  {len(records):,} total records")

    now_ts     = datetime.now(timezone.utc).timestamp()
    fetch_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    passed, counts = apply_filters(
        records,
        min_days=args.min_days,
        max_days=args.max_days,
        min_price=args.min_price,
        max_price=args.max_price,
        min_volume=args.min_volume,
        now_ts=now_ts,
    )

    print(f"\nFilter funnel:")
    print(f"  total input          : {counts['total']:>6,}")
    print(f"  binary               : {counts['binary']:>6,}")
    print(f"  active+unresolved    : {counts['active']:>6,}")
    print(f"  topic match          : {counts['topic']:>6,}")
    print(f"  resolution {args.min_days}–{args.max_days}d    : {counts['days']:>6,}")
    print(f"  price {args.min_price:.2f}–{args.max_price:.2f}         : {counts['price']:>6,}")
    print(f"  volume >= ${args.min_volume:,.0f}    : {counts['volume']:>6,}")

    deduped = dedup_by_event(passed)
    print(f"  after event dedup    : {len(deduped):>6,}")

    # Sort by volume descending
    deduped.sort(key=lambda r: r.get("volume_usd") or 0.0, reverse=True)

    if args.top > 0 and len(deduped) > args.top:
        deduped = deduped[: args.top]
        print(f"  top {args.top} by volume      : {len(deduped):>6,}")

    # Category breakdown
    cats = Counter((r.get("category") or "unknown").lower() for r in deduped)
    print(f"\nCategory breakdown:")
    for cat, cnt in cats.most_common(20):
        bar = "▪" * min(30, max(1, cnt))
        print(f"  {cat:<35s} {cnt:>4,}  {bar}")

    # Price + volume stats
    vols = sorted(r.get("volume_usd") or 0.0 for r in deduped)
    prices = [r.get("yes_price") for r in deduped if r.get("yes_price") is not None]
    if vols:
        n = len(vols)
        print(f"\nVolume stats (USD):")
        print(f"  min    ${vols[0]:>12,.0f}")
        print(f"  median ${vols[n//2]:>12,.0f}")
        print(f"  p90    ${vols[int(n*0.9)]:>12,.0f}")
        print(f"  max    ${vols[-1]:>12,.0f}")
    if prices:
        print(f"\nYes-price stats:")
        print(f"  min    {min(prices):.3f}")
        print(f"  mean   {sum(prices)/len(prices):.3f}")
        print(f"  max    {max(prices):.3f}")

    # Sample
    print(f"\nSample (top 20 by volume):")
    for r in deduped[:20]:
        dtd = days_to_resolution(r, now_ts)
        vol = r.get("volume_usd") or 0.0
        yp  = r.get("yes_price") or 0.0
        print(
            f"  [{yp:.2f}] ${vol:>11,.0f}  {dtd:>4.0f}d  {r['question'][:70]}"
        )

    if args.dry_run:
        print(f"\n  --dry-run: skipping write ({len(deduped)} markets would be saved)")
        return

    # Enrich and write
    enriched = [_enrich(r, now_ts, fetch_date) for r in deduped]

    args.out.mkdir(parents=True, exist_ok=True)
    out_path  = args.out / f"selected_{fetch_date}.jsonl"
    mani_path = args.out / f"selected_{fetch_date}.manifest.json"

    with out_path.open("w") as f:
        for r in enriched:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "selected_at":   datetime.now(timezone.utc).isoformat(),
        "source_file":   str(input_path),
        "filters": {
            "min_days_to_resolution": args.min_days,
            "max_days_to_resolution": args.max_days,
            "min_yes_price":          args.min_price,
            "max_yes_price":          args.max_price,
            "min_volume_usd":         args.min_volume,
        },
        "funnel":        counts,
        "after_dedup":   len(deduped),
        "total_selected": len(enriched),
    }
    with mani_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(enriched):,} records → {out_path}")
    print(f"Wrote manifest         → {mani_path}")


if __name__ == "__main__":
    main()
