#!/usr/bin/env python3
"""Fetch day-by-day price history for a set of markets from the CLOB API.

Reads a markets JSONL (fetch_markets.py output or a filtered subset) and
fetches the full daily price series for each market's Yes token.

IMPORTANT — run on a filtered subset, not all 61k markets:
  The CLOB API is behind Cloudflare and rate-limits aggressive bulk fetches.
  This script uses a serial approach with a configurable delay (~1s between
  requests).  For ~100 markets at 1s delay it finishes in under 15 minutes.
  Run fetch_markets.py first, then filter with select_markets.py, then run
  this on the filtered ~100 markets.

Window limit:
  The CLOB API rejects windows longer than ~28 days.  Long-lived markets are
  chunked automatically and merged.

Volume note:
  True per-day volume is not exposed by any public Polymarket API.
  This script preserves the rolling volume windows from the Gamma snapshot
  (volume_24h_usd, volume_1wk_usd, volume_1mo_usd, volume_1yr_usd).

Output: data/raw_markets/price_history_YYYY-MM-DD.jsonl
  One record per market:
    market_id, clob_token_yes, question, open_time, end_time,
    price_history: [{date, ts, price}],   ← daily Yes-token price from CLOB
    volume_windows: {total, 1d, 1wk, 1mo, 1yr},
    n_price_points, fetched_at

Usage (from exp1_prospective/):
    python fetch_markets/fetch_price_history.py --input data/selected_markets/selected_2026-06-05.jsonl
    python fetch_markets/fetch_price_history.py --input data/raw_markets/markets_2026-06-05.jsonl --only-with-volume 1000000
    python fetch_markets/fetch_price_history.py --fidelity 60   # hourly instead of daily
    python fetch_markets/fetch_price_history.py --delay 2.0     # slower if getting rate-limited
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── self-contained: import vendored client from this folder ────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from price_history_api import PriceHistoryError, fetch_price_history  # noqa: E402

_ROOT = _HERE.parent
_DEFAULT_OUT = _ROOT / "data" / "raw_markets"

# CLOB hard limit is 15 days (1,296,000s); use 14 to stay safely under it
_MAX_WINDOW_DAYS = 14


# ── helpers ────────────────────────────────────────────────────────────────────

def _latest_markets_jsonl(out_dir: Path) -> Path | None:
    """Find the most recent markets JSONL that isn't a price_history file."""
    for f in sorted(out_dir.glob("markets_*.jsonl"), reverse=True):
        if "price_history" not in f.name:
            return f
    return None


def _to_epoch(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except (ValueError, TypeError):
        return None


def _epoch_to_date(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


# ── per-market fetch ───────────────────────────────────────────────────────────

def _fetch_one(market: dict, *, fidelity_minutes: int, timeout: float, delay: float) -> dict | None:
    """Fetch price history for one market, chunking if needed.  Returns None if no token."""
    token_ids = market.get("clob_token_ids") or []
    if not token_ids:
        return None
    yes_token = str(token_ids[0])

    open_ts  = _to_epoch(market.get("open_time"))
    end_ts   = _to_epoch(market.get("end_time"))
    now_ts   = int(datetime.now(timezone.utc).timestamp())

    start_ts     = open_ts if open_ts else now_ts - 365 * 86400
    fetch_end_ts = min(end_ts, now_ts) if end_ts else now_ts

    if fetch_end_ts <= start_ts:
        return None

    chunk_secs = _MAX_WINDOW_DAYS * 86400
    all_points: list[dict] = []
    chunk_start = start_ts

    while chunk_start < fetch_end_ts:
        chunk_end = min(chunk_start + chunk_secs, fetch_end_ts)
        try:
            pts = fetch_price_history(
                token_id=yes_token,
                start_ts=chunk_start,
                end_ts=chunk_end,
                fidelity_minutes=fidelity_minutes,
                timeout=timeout,
            )
            all_points.extend(pts)
        except (PriceHistoryError, Exception):
            pass  # partial data still useful; failed chunks just leave gaps
        chunk_start = chunk_end + 1
        if chunk_start < fetch_end_ts:
            time.sleep(delay)

    # deduplicate + sort
    seen: set[int] = set()
    unique: list[dict] = []
    for p in all_points:
        ts = p.get("ts")
        if ts is not None and ts not in seen:
            seen.add(ts)
            unique.append(p)
    unique.sort(key=lambda x: x["ts"])

    price_history = [
        {"date": _epoch_to_date(p["ts"]), "ts": p["ts"], "price": p["price"]}
        for p in unique
        if p.get("ts") is not None
    ]

    return {
        "market_id":        market["market_id"],
        "clob_token_yes":   yes_token,
        "question":         market.get("question", ""),
        "category":         market.get("category"),
        "open_time":        market.get("open_time"),
        "end_time":         market.get("end_time"),
        "yes_price":        market.get("yes_price"),
        "last_trade_price": market.get("last_trade_price"),
        # daily price series (from CLOB)
        "price_history":    price_history,
        "n_price_points":   len(price_history),
        # rolling volume windows from Gamma snapshot (best available)
        "volume_windows": {
            "total": market.get("volume_usd"),
            "1d":    market.get("volume_24h_usd"),
            "1wk":   market.get("volume_1wk_usd"),
            "1mo":   market.get("volume_1mo_usd"),
            "1yr":   market.get("volume_1yr_usd"),
        },
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


# ── progress bar ───────────────────────────────────────────────────────────────

class _Progress:
    def __init__(self, total: int) -> None:
        self.total   = total
        self.done    = self.ok = self.skipped = self.empty = self.errors = 0
        self._start  = time.time()

    def record(self, status: str) -> None:
        self.done += 1
        if   status == "ok":    self.ok      += 1
        elif status == "skip":  self.skipped += 1
        elif status == "empty": self.empty   += 1
        else:                   self.errors  += 1
        elapsed = time.time() - self._start
        rate    = self.done / max(elapsed, 0.1)
        eta_s   = (self.total - self.done) / max(rate, 0.01)
        eta_str = f"{eta_s:.0f}s" if eta_s < 3600 else f"{eta_s/60:.1f}m"
        pct     = 100 * self.done / max(self.total, 1)
        filled  = int(20 * self.done / max(self.total, 1))
        bar     = "█" * filled + "░" * (20 - filled)
        sys.stdout.write(
            f"\r  [{bar}] {self.done:>5}/{self.total}"
            f"  ok {self.ok:>5}  skip {self.skipped:>4}"
            f"  empty {self.empty:>4}  err {self.errors:>4}"
            f"  {pct:>5.1f}%  ETA {eta_str}   "
        )
        sys.stdout.flush()

    def done_message(self) -> None:
        elapsed = time.time() - self._start
        sys.stdout.write("\n")
        sys.stdout.flush()
        print(
            f"\n  Finished in {elapsed:.1f}s — "
            f"{self.ok:,} with price history  |  "
            f"{self.skipped:,} no token  |  "
            f"{self.empty:,} empty  |  "
            f"{self.errors:,} errors"
        )


# ── cli ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fetch CLOB daily price history for a set of Polymarket markets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input",    type=Path, default=None,
                    help="Markets JSONL to process.  Defaults to latest in --out dir.")
    ap.add_argument("--out",      type=Path, default=_DEFAULT_OUT,
                    help="Output directory (also searched for default --input).")
    ap.add_argument("--fidelity", type=int,  default=1440,
                    help="CLOB granularity in minutes (1440=daily, 60=hourly).")
    ap.add_argument("--delay",    type=float, default=1.0,
                    help="Seconds between requests.  Increase to 2.0+ if you get 403s.")
    ap.add_argument("--timeout",  type=float, default=15.0,
                    help="Per-request HTTP timeout (s).")
    ap.add_argument("--only-with-volume", type=float, default=0.0,
                    help="Skip markets with lifetime volume below this USD threshold.")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Fetch and print stats but don't write output files.")
    args = ap.parse_args()

    input_path = args.input or _latest_markets_jsonl(args.out)
    if input_path is None or not input_path.exists():
        print(f"No markets JSONL found in {args.out}.  Run fetch_markets.py first.")
        sys.exit(1)

    print(f"Reading  {input_path}")
    with input_path.open() as f:
        markets = [json.loads(line) for line in f if line.strip()]

    if args.only_with_volume > 0:
        before = len(markets)
        markets = [m for m in markets if (m.get("volume_usd") or 0) >= args.only_with_volume]
        print(f"  volume >= ${args.only_with_volume:,.0f}: {before:,} → {len(markets):,} markets")

    with_token    = [m for m in markets if m.get("clob_token_ids")]
    without_token = [m for m in markets if not m.get("clob_token_ids")]

    print(
        f"\nFetching price history  "
        f"(fidelity={args.fidelity}min  delay={args.delay}s  serial)\n"
        f"  {len(with_token):,} have CLOB token  |  "
        f"{len(without_token):,} without (skipped)\n"
    )

    progress = _Progress(len(markets))
    results: list[dict] = []

    for _ in without_token:
        progress.record("skip")

    for i, market in enumerate(with_token):
        result = _fetch_one(market, fidelity_minutes=args.fidelity,
                            timeout=args.timeout, delay=args.delay)
        if result is None:
            progress.record("error")
        elif result["n_price_points"] == 0:
            progress.record("empty")
            results.append(result)
        else:
            progress.record("ok")
            results.append(result)
        if i < len(with_token) - 1:
            time.sleep(args.delay)

    progress.done_message()

    ok = [r for r in results if r["n_price_points"] > 0]
    if ok:
        pts = sorted(r["n_price_points"] for r in ok)
        n   = len(pts)
        print(f"Price-point counts (markets with data):")
        print(f"  min {pts[0]}  median {pts[n//2]}  p90 {pts[int(n*0.9)]}  max {pts[-1]}")
        print(f"\nSample (first 5):")
        for r in ok[:5]:
            first = r["price_history"][0]  if r["price_history"] else {}
            last  = r["price_history"][-1] if r["price_history"] else {}
            vol   = (r["volume_windows"] or {}).get("total") or 0
            print(
                f"  {r['question'][:62]:<62}"
                f"  {first.get('date','?')}→{last.get('date','?')}"
                f"  ({r['n_price_points']}pts)"
                f"  ${vol:>10,.0f}"
            )

    if args.dry_run:
        print("\n  --dry-run: skipping write")
        return

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    args.out.mkdir(parents=True, exist_ok=True)
    out_path  = args.out / f"price_history_{date_str}.jsonl"
    mani_path = args.out / f"price_history_{date_str}.manifest.json"

    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "fetched_at":       datetime.now(timezone.utc).isoformat(),
        "source_file":      str(input_path),
        "fidelity_minutes": args.fidelity,
        "total_input":      len(markets),
        "with_price_data":  len(ok),
        "empty":            progress.empty,
        "skipped_no_token": progress.skipped,
        "errors":           progress.errors,
    }
    with mani_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(results):,} records → {out_path}")
    print(f"Wrote manifest       → {mani_path}")


if __name__ == "__main__":
    main()
