#!/usr/bin/env python3
"""Fetch ALL active, open Polymarket markets from the Gamma API.

Paginates through every page of active events, flattens nested markets,
extracts key fields (including all volume windows and price changes), and
writes one JSON record per market to a JSONL file.  A live counter updates
in-place on every page so you can watch the totals accumulate.

Output: data/raw_markets/markets_YYYY-MM-DD.jsonl
        data/raw_markets/markets_YYYY-MM-DD.manifest.json

Usage (from exp1_prospective/):
    python fetch_markets/fetch_markets.py
    python fetch_markets/fetch_markets.py --out data/raw_markets/
    python fetch_markets/fetch_markets.py --page-limit 200 --timeout 20
    python fetch_markets/fetch_markets.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── self-contained: import vendored clients from this folder ───────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from gamma_api import GammaApiError, fetch_events_keyset_page, fetch_events_page  # noqa: E402

# ── default paths ──────────────────────────────────────────────────────────────
_ROOT = _HERE.parent
_DEFAULT_OUT = _ROOT / "data" / "raw_markets"


# ── field helpers ──────────────────────────────────────────────────────────────

def _coerce_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_json_list(v: Any) -> list | None:
    if v is None:
        return None
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            return None
    return None


def _pick(d: dict, *keys: str) -> Any:
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def _parse_ts(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, (int, float)):
        ts = int(v)
        if ts > 1_000_000_000_000:
            ts //= 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def extract_market(market: dict, event: dict, fetched_at: str) -> dict:
    """Flatten one Gamma market + its parent event into a flat record."""
    outcomes = _parse_json_list(_pick(market, "outcomes", "outcome")) or []
    outcome_prices_raw = _parse_json_list(_pick(market, "outcomePrices", "outcome_prices")) or []
    outcome_prices = [_coerce_float(p) for p in outcome_prices_raw]

    # Polymarket convention: Yes is index 0
    yes_price = outcome_prices[0] if outcome_prices else None

    # Volume — prefer CLOB variants when present
    volume     = _coerce_float(_pick(market, "volumeClob",    "volumeNum",    "volume"))
    volume_24h = _coerce_float(_pick(market, "volume24hrClob","volume24hr",   "volume24h"))
    volume_1wk = _coerce_float(_pick(market, "volume1wkClob", "volume1wk"))
    volume_1mo = _coerce_float(_pick(market, "volume1moClob", "volume1mo"))
    volume_1yr = _coerce_float(_pick(market, "volume1yrClob", "volume1yr"))
    liquidity  = _coerce_float(_pick(market, "liquidityClob", "liquidity",    "liquidityNum"))

    # Price snapshots + rolling changes
    last_trade_price = _coerce_float(_pick(market, "lastTradePrice"))
    best_ask         = _coerce_float(_pick(market, "bestAsk"))
    best_bid         = _coerce_float(_pick(market, "bestBid"))
    spread           = _coerce_float(market.get("spread"))
    price_change_1d  = _coerce_float(_pick(market, "oneDayPriceChange"))
    price_change_1wk = _coerce_float(_pick(market, "oneWeekPriceChange"))
    price_change_1mo = _coerce_float(_pick(market, "oneMonthPriceChange"))
    price_change_1yr = _coerce_float(_pick(market, "oneYearPriceChange"))

    # CLOB token IDs — needed for price-history API (Yes = index 0)
    clob_raw = _parse_json_list(_pick(market, "clobTokenIds", "clob_token_ids")) or []
    clob_token_ids = [str(t) for t in clob_raw if t is not None]

    tags = event.get("tags") if isinstance(event.get("tags"), list) else []
    tag_names = [
        t.get("name") or t.get("label") or t.get("slug") or str(t)
        for t in tags if isinstance(t, dict)
    ]
    category = _pick(event, "category", "topic", "tag") or (tag_names[0] if tag_names else None)

    return {
        # identifiers
        "market_id":   str(_pick(market, "id",   "market_id",  "marketId")  or ""),
        "event_id":    str(_pick(event,  "id",   "event_id",   "eventId")   or ""),
        "event_slug":  str(_pick(event,  "slug", "event_slug", "eventSlug") or ""),
        "market_slug": str(_pick(market, "slug", "market_slug","marketSlug") or ""),
        # content
        "event_title": str(_pick(event,  "title", "question", "name") or ""),
        "question":    str(_pick(market, "question", "title",  "name") or ""),
        "description": str(_pick(market, "description", "details") or ""),
        "rules":       str(_pick(market, "rules", "additionalInfo") or ""),
        "category":    str(category) if category else None,
        "tags":        tag_names,
        # outcomes & current price
        "outcomes":         outcomes,
        "outcome_prices":   outcome_prices,
        "yes_price":        yes_price,
        "last_trade_price": last_trade_price,
        "best_bid":         best_bid,
        "best_ask":         best_ask,
        "spread":           spread,
        # rolling price changes (signed delta vs. now)
        "price_change_1d":  price_change_1d,
        "price_change_1wk": price_change_1wk,
        "price_change_1mo": price_change_1mo,
        "price_change_1yr": price_change_1yr,
        # volume — rolling windows in USD
        "volume_usd":     volume,
        "volume_24h_usd": volume_24h,
        "volume_1wk_usd": volume_1wk,
        "volume_1mo_usd": volume_1mo,
        "volume_1yr_usd": volume_1yr,
        "liquidity_usd":  liquidity,
        # needed by fetch_price_history.py
        "clob_token_ids": clob_token_ids,
        # status flags
        "active":   market.get("active"),
        "closed":   market.get("closed"),
        "resolved": market.get("resolved"),
        # timing
        "open_time":  _parse_ts(_pick(market, "startTime",  "startDate")),
        "end_time":   _parse_ts(_pick(market, "endTime",    "endDate", "closeTime")),
        "created_at": _parse_ts(_pick(market, "createdAt",  "created_at")),
        "updated_at": _parse_ts(_pick(market, "updatedAt",  "updated_at")),
        "fetched_at": fetched_at,
    }


def is_binary(r: dict) -> bool:
    return len(r.get("outcomes") or []) == 2


def is_active_unresolved(r: dict) -> bool:
    return (
        r.get("active")   is not False
        and r.get("closed")   is not True
        and r.get("resolved") is not True
    )


# ── live progress ──────────────────────────────────────────────────────────────

class _Progress:
    def __init__(self) -> None:
        self.pages   = 0
        self.events  = 0
        self.markets = 0
        self.binary  = 0
        self._start  = time.time()

    def update(self, new_events: int, new_markets: int, new_binary: int) -> None:
        self.pages   += 1
        self.events  += new_events
        self.markets += new_markets
        self.binary  += new_binary
        elapsed = time.time() - self._start
        rate = self.markets / max(elapsed, 0.1)
        sys.stdout.write(
            f"\r  page {self.pages:>4}"
            f"  |  events {self.events:>5}"
            f"  |  markets {self.markets:>6}"
            f"  |  active+binary {self.binary:>5}"
            f"  |  {elapsed:>5.1f}s"
            f"  |  {rate:>6.0f} mkts/s   "
        )
        sys.stdout.flush()

    def done(self) -> None:
        elapsed = time.time() - self._start
        sys.stdout.write("\n")
        sys.stdout.flush()
        print(
            f"\n  Finished in {elapsed:.1f}s — "
            f"{self.markets:,} markets  |  "
            f"{self.events:,} events  |  "
            f"{self.pages} pages  |  "
            f"{self.binary:,} active+binary"
        )


# ── core fetch ─────────────────────────────────────────────────────────────────

def fetch_all(
    *,
    max_pages: int = 5000,
    page_limit: int = 100,
    timeout: float = 15.0,
) -> tuple[list[dict], dict]:
    """Paginate the Gamma API to exhaustion.  Returns (records, stats)."""
    fetched_at = datetime.now(timezone.utc).isoformat()
    progress   = _Progress()
    records:   list[dict] = []
    seen:      set[str]   = set()
    cursor:    str | None = None
    page_num   = 0
    use_keyset = True

    print(f"Fetching ALL active Polymarket markets  (max_pages={max_pages}, page_limit={page_limit})\n")

    while page_num < max_pages:
        try:
            if use_keyset:
                events, next_cursor = fetch_events_keyset_page(
                    limit=page_limit, after_cursor=cursor,
                    active=True, closed=False, timeout=timeout,
                )
            else:
                events = fetch_events_page(
                    active=True, closed=False,
                    limit=page_limit, offset=page_num * page_limit, timeout=timeout,
                )
                next_cursor = None
        except GammaApiError as exc:
            if use_keyset and page_num == 0:
                print(f"\n  keyset error ({exc}), falling back to offset pagination")
                use_keyset = False
                continue
            print(f"\n  API error on page {page_num}: {exc} — stopping")
            break

        if not events:
            break

        new_events = len(events)
        new_markets = new_binary = 0

        for event in events:
            for market in (event.get("markets") or []):
                if not isinstance(market, dict):
                    continue
                rec = extract_market(market, event, fetched_at)
                mid = rec["market_id"]
                if not mid or mid in seen:
                    continue
                seen.add(mid)
                records.append(rec)
                new_markets += 1
                if is_active_unresolved(rec) and is_binary(rec):
                    new_binary += 1

        progress.update(new_events, new_markets, new_binary)
        page_num += 1

        if use_keyset:
            if not next_cursor:
                break
            cursor = next_cursor
        elif len(events) < page_limit:
            break

    progress.done()

    stats = {
        "fetched_at":    fetched_at,
        "pages_fetched": page_num,
        "events_seen":   progress.events,
        "total_markets": len(records),
        "active_binary": sum(1 for r in records if is_active_unresolved(r) and is_binary(r)),
        "page_limit":    page_limit,
        "max_pages":     max_pages,
    }
    return records, stats


# ── output ─────────────────────────────────────────────────────────────────────

def _pct(lst: list[float], p: float) -> float:
    if not lst:
        return 0.0
    return lst[max(0, min(len(lst) - 1, int(len(lst) * p)))]


def write(records: list[dict], stats: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    jsonl = out_dir / f"markets_{date}.jsonl"
    mani  = out_dir / f"markets_{date}.manifest.json"
    with jsonl.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with mani.open("w") as f:
        json.dump(stats, f, indent=2)
    return jsonl, mani


def print_summary(records: list[dict]) -> None:
    q = [r for r in records if is_active_unresolved(r) and is_binary(r)]
    print(f"Summary")
    print(f"  Total records       : {len(records):>7,}")
    print(f"  Active + binary     : {len(q):>7,}")

    vols = sorted(r.get("volume_usd") or 0.0 for r in q)
    if vols:
        print(f"\nVolume (active+binary, USD):")
        for lbl, p in [("min","min"),("p10",0.10),("p25",0.25),("median",0.50),
                       ("p75",0.75),("p90",0.90),("max","max")]:
            v = vols[0] if p == "min" else vols[-1] if p == "max" else _pct(vols, p)
            print(f"  {lbl:<8}: ${v:>14,.0f}")
        print(f"  total   : ${sum(vols):>14,.0f}")
        print(f"\n  vol > 0     : {sum(1 for v in vols if v > 0):>7,}")
        print(f"  vol >= $1k  : {sum(1 for v in vols if v >= 1_000):>7,}")
        print(f"  vol >= $10k : {sum(1 for v in vols if v >= 10_000):>7,}")
        print(f"  vol >= $100k: {sum(1 for v in vols if v >= 100_000):>7,}")

    cats = Counter((r.get("category") or "unknown").lower() for r in q)
    print(f"\nTop categories (active+binary):")
    for cat, cnt in cats.most_common(15):
        bar = "▪" * min(40, cnt // max(1, len(q) // 40))
        print(f"  {cat:<30s} {cnt:>5,}  {bar}")

    sample = [r for r in q if (r.get("volume_usd") or 0) >= 10_000][:8]
    if sample:
        print(f"\nSample (vol >= $10k):")
        for r in sample:
            vol = r.get("volume_usd") or 0
            p   = r.get("yes_price")
            print(f"  [{p:.2f}] ${vol:>11,.0f}  {r['question'][:75]}")


# ── cli ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fetch all active Polymarket markets from the Gamma API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--max-pages",  type=int,   default=5000,        help="Hard cap on API pages.")
    ap.add_argument("--page-limit", type=int,   default=100,         help="Events per request.")
    ap.add_argument("--timeout",    type=float, default=15.0,        help="Per-request timeout (s).")
    ap.add_argument("--out",        type=Path,  default=_DEFAULT_OUT,help="Output directory.")
    ap.add_argument("--dry-run",    action="store_true",             help="Fetch but don't write.")
    args = ap.parse_args()

    records, stats = fetch_all(
        max_pages=args.max_pages,
        page_limit=args.page_limit,
        timeout=args.timeout,
    )

    print()
    print_summary(records)

    if args.dry_run:
        print("\n  --dry-run: skipping file write")
        return

    jsonl, mani = write(records, stats, args.out)
    print(f"\nWrote {len(records):,} records → {jsonl}")
    print(f"Wrote manifest        → {mani}")


if __name__ == "__main__":
    main()
