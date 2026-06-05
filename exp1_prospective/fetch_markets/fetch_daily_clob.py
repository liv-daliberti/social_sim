#!/usr/bin/env python3
"""Fetch daily CLOB price history and daily volume for tracked markets.

Reads a markets JSONL (typically diverse_YYYY-MM-DD.jsonl or
selected_YYYY-MM-DD.jsonl) and for each market:

  1. Fetches the full daily Yes-token price series from CLOB /prices-history
     (same chunked approach as fetch_price_history.py).
  2. Records today's CLOB 24h trade volume from the Gamma API snapshot
     (field: volume_24h_usd, sourced from volume24hrClob in the Gamma feed).
     The CLOB /trades endpoint requires authentication and is not publicly
     accessible; Gamma proxies the same rolling 24h CLOB volume, so recording
     this daily at a fixed time gives per-day CLOB volume.
  3. Merges today's volume into a persistent accumulator so every prior day's
     volume is also preserved across daily runs.

The source JSONL is *enriched in-place* with two new fields:
  price_history:  [{date, ts, price}, ...]   — full CLOB daily close series
  volume_history: [{date, volume_usd}, ...]  — accumulated daily CLOB volume

A separate per-day snapshot is written to data/daily_tracking/ as well.

Usage (from exp1_prospective/):
    python fetch_markets/fetch_daily_clob.py
    python fetch_markets/fetch_daily_clob.py --input data/selected_markets/diverse_2026-06-05.jsonl
    python fetch_markets/fetch_daily_clob.py --skip-volume   # price-only, faster
    python fetch_markets/fetch_daily_clob.py --delay 2.0     # slower if getting 403s
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from price_history_api import PriceHistoryError, fetch_price_history  # noqa: E402

_ROOT        = _HERE.parent
_SEL_DIR     = _ROOT / "data" / "selected_markets"
_TRACK_DIR   = _ROOT / "data" / "daily_tracking"
_HISTORY_FILE = _TRACK_DIR / "history.jsonl"  # persistent per-market accumulator

_MAX_WINDOW_DAYS = 14  # CLOB hard limit is 15 days; use 14 to stay safely under


# ── helpers ────────────────────────────────────────────────────────────────────

def _latest_diverse(sel_dir: Path) -> Path | None:
    for f in sorted(sel_dir.glob("diverse_*.jsonl"), reverse=True):
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


# ── price history fetch (mirrors fetch_price_history._fetch_one) ──────────────

def _fetch_price_history(
    market: dict,
    *,
    fidelity_minutes: int,
    timeout: float,
    delay: float,
) -> list[dict]:
    token_ids = market.get("clob_token_ids") or []
    if not token_ids:
        return []
    yes_token = str(token_ids[0])

    open_ts      = _to_epoch(market.get("open_time"))
    end_ts       = _to_epoch(market.get("end_time"))
    now_ts       = int(datetime.now(timezone.utc).timestamp())
    start_ts     = open_ts if open_ts else now_ts - 365 * 86400
    fetch_end_ts = min(end_ts, now_ts) if end_ts else now_ts

    if fetch_end_ts <= start_ts:
        return []

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
            pass
        chunk_start = chunk_end + 1
        if chunk_start < fetch_end_ts:
            time.sleep(delay)

    seen: set[int] = set()
    unique: list[dict] = []
    for p in all_points:
        ts = p.get("ts")
        if ts is not None and ts not in seen:
            seen.add(ts)
            unique.append(p)
    unique.sort(key=lambda x: x["ts"])

    return [
        {"date": _epoch_to_date(p["ts"]), "ts": p["ts"], "price": p["price"]}
        for p in unique
        if p.get("ts") is not None
    ]


# ── persistent volume history (market_id → [{date, volume_usd}]) ──────────────

def _load_volume_history(path: Path) -> dict[str, list[dict]]:
    """Load accumulated volume history from JSONL.  Returns {market_id: [...]}."""
    history: dict[str, list[dict]] = {}
    if not path.exists():
        return history
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                mid = rec.get("market_id")
                if mid:
                    history[mid] = rec.get("volume_history", [])
            except json.JSONDecodeError:
                pass
    return history


def _save_volume_history(history: dict[str, list[dict]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for market_id, vol_hist in history.items():
            f.write(json.dumps({"market_id": market_id, "volume_history": vol_hist}) + "\n")


def _merge_volume(existing: list[dict], date_str: str, volume_usd: float) -> list[dict]:
    """Append or update today's volume entry, deduplicating by date."""
    updated = [e for e in existing if e.get("date") != date_str]
    updated.append({"date": date_str, "volume_usd": round(volume_usd, 4)})
    updated.sort(key=lambda x: x.get("date", ""))
    return updated


# ── progress ──────────────────────────────────────────────────────────────────

class _Progress:
    def __init__(self, total: int) -> None:
        self.total  = total
        self.done   = self.ok = self.price_ok = self.vol_ok = self.errors = 0
        self._start = time.time()

    def record(self, *, price_ok: bool, vol_ok: bool) -> None:
        self.done += 1
        if price_ok or vol_ok:
            self.ok += 1
        if price_ok:
            self.price_ok += 1
        if vol_ok:
            self.vol_ok += 1
        else:
            self.errors += 1
        elapsed = time.time() - self._start
        rate    = self.done / max(elapsed, 0.1)
        eta_s   = (self.total - self.done) / max(rate, 0.01)
        eta_str = f"{eta_s:.0f}s" if eta_s < 3600 else f"{eta_s/60:.1f}m"
        pct     = 100 * self.done / max(self.total, 1)
        filled  = int(20 * self.done / max(self.total, 1))
        bar     = "█" * filled + "░" * (20 - filled)
        sys.stdout.write(
            f"\r  [{bar}] {self.done:>4}/{self.total}"
            f"  price {self.price_ok:>4}  vol {self.vol_ok:>4}"
            f"  err {self.errors:>3}"
            f"  {pct:>5.1f}%  ETA {eta_str}   "
        )
        sys.stdout.flush()

    def done_message(self) -> None:
        elapsed = time.time() - self._start
        sys.stdout.write("\n")
        sys.stdout.flush()
        print(
            f"\n  Finished in {elapsed:.1f}s — "
            f"{self.price_ok}/{self.total} with price history  |  "
            f"{self.vol_ok}/{self.total} with today's volume"
        )


# ── cli ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Enrich diverse/selected JSONL with CLOB price history and daily volume.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input",       type=Path, default=None,
                    help="Markets JSONL (diverse or selected). Defaults to latest diverse in --sel-dir.")
    ap.add_argument("--sel-dir",     type=Path, default=_SEL_DIR)
    ap.add_argument("--track-dir",   type=Path, default=_TRACK_DIR)
    ap.add_argument("--fidelity",    type=int,  default=1440,
                    help="Price-history granularity in minutes (1440 = daily).")
    ap.add_argument("--delay",       type=float, default=1.0,
                    help="Seconds between CLOB requests.")
    ap.add_argument("--timeout",     type=float, default=15.0)
    ap.add_argument("--skip-volume", action="store_true",
                    help="Skip volume fetch (prices only).")
    ap.add_argument("--dry-run",     action="store_true",
                    help="Fetch and print but don't write any files.")
    args = ap.parse_args()

    input_path = args.input or _latest_diverse(args.sel_dir)
    if input_path is None or not input_path.exists():
        print(f"No diverse JSONL found in {args.sel_dir}.  Run diversify_markets.py first.")
        sys.exit(1)

    print(f"Reading  {input_path}")
    with input_path.open() as f:
        markets = [json.loads(line) for line in f if line.strip()]
    print(f"  {len(markets):,} markets")

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Load persistent volume history
    history_path = args.track_dir / "history.jsonl"
    vol_history  = _load_volume_history(history_path)
    print(f"  {len(vol_history):,} markets in existing volume history")

    print(
        f"\nFetching CLOB data  "
        f"(fidelity={args.fidelity}min  delay={args.delay}s  date={today_str})\n"
    )

    progress  = _Progress(len(markets))
    enriched: list[dict] = []
    snapshot: list[dict] = []

    with_token    = [m for m in markets if m.get("clob_token_ids")]
    without_token = [m for m in markets if not m.get("clob_token_ids")]

    for m in without_token:
        rec = dict(m)
        rec["price_history"]  = []
        # Still record today's volume_24h_usd even without a CLOB token
        existing_vol = vol_history.get(m["market_id"], [])
        if not args.skip_volume:
            today_vol    = float(m.get("volume_24h_usd") or 0.0)
            existing_vol = _merge_volume(existing_vol, today_str, today_vol)
        vol_history[m["market_id"]] = existing_vol
        rec["volume_history"] = existing_vol
        enriched.append(rec)
        progress.record(price_ok=False, vol_ok=not args.skip_volume)

    for i, market in enumerate(with_token):
        mid = market["market_id"]

        # ── price history from CLOB /prices-history ────────────────────────────
        price_hist = _fetch_price_history(
            market,
            fidelity_minutes=args.fidelity,
            timeout=args.timeout,
            delay=args.delay,
        )
        price_ok = len(price_hist) > 0

        if i < len(with_token) - 1:
            time.sleep(args.delay)

        # ── today's volume: Gamma-proxied CLOB 24h volume (volume24hrClob) ─────
        # The CLOB /trades endpoint requires auth; Gamma exposes the same
        # rolling 24h CLOB volume as volume_24h_usd.  Recording this daily at
        # a fixed time gives per-day CLOB trade volume.
        existing_vol = vol_history.get(mid, [])
        today_vol    = 0.0
        if not args.skip_volume:
            today_vol    = float(market.get("volume_24h_usd") or 0.0)
            existing_vol = _merge_volume(existing_vol, today_str, today_vol)
        vol_history[mid] = existing_vol

        rec = dict(market)
        rec["price_history"]  = price_hist
        rec["volume_history"] = existing_vol
        enriched.append(rec)

        snapshot.append({
            "market_id":   mid,
            "date":        today_str,
            "price":       price_hist[-1]["price"] if price_hist else None,
            "volume_usd":  today_vol if not args.skip_volume else None,
            "n_price_pts": len(price_hist),
        })

        progress.record(price_ok=price_ok, vol_ok=True)

        if i < len(with_token) - 1:
            time.sleep(args.delay)

    progress.done_message()

    if args.dry_run:
        print("  --dry-run: skipping writes")
        return

    # ── write enriched source JSONL (in-place) ─────────────────────────────────
    with input_path.open("w") as f:
        for rec in enriched:
            f.write(json.dumps(rec) + "\n")
    print(f"Enriched {len(enriched):,} records → {input_path}")

    # ── persist volume history ──────────────────────────────────────────────────
    if not args.skip_volume:
        _save_volume_history(vol_history, history_path)
        print(f"Updated volume history → {history_path}")

    # ── daily snapshot ─────────────────────────────────────────────────────────
    args.track_dir.mkdir(parents=True, exist_ok=True)
    snap_path = args.track_dir / f"snapshot_{today_str}.jsonl"
    with snap_path.open("w") as f:
        for rec in snapshot:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote daily snapshot   → {snap_path}")


if __name__ == "__main__":
    main()
