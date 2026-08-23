#!/usr/bin/env python3
"""Sampled per-event market-completeness audit.

Draws N random event ids from the archive DB, fetches those events fresh from
Gamma (batched id queries), and verifies that every market Gamma lists for the
event exists in polymarket_markets — plus field agreement (closed flag,
outcome_prices presence for closed markets).

Statistical guarantee: with N=2000 and zero misses, per-event market-miss rate
is < 0.15% at 95% confidence.

Run under pmpython:
  pmpython sample_event_market_audit.py --db-url URL [--n 2000] [--report OUT.json]
"""
import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

import psycopg

HOST = "https://gamma-api.polymarket.com"


def fetch_events(ids, retries=6, timeout=30):
    qs = "&".join(f"id={i}" for i in ids)
    url = f"{HOST}/events?{qs}&limit={max(100, len(ids))}"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "archive-sample-audit"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read())
                return payload if isinstance(payload, list) else []
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(30, 0.5 * 2 ** attempt))
    raise RuntimeError(f"fetch failed: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--recent-n", type=int, default=500, help="extra sample from gap window")
    ap.add_argument("--seed", type=int, default=20260711)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with psycopg.connect(args.db_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT event_id FROM polymarket_events")
        all_ids = [r[0] for r in cur.fetchall()]
        cur.execute(
            "SELECT DISTINCT event_id FROM polymarket_markets "
            "WHERE created_time >= '2026-05-01' AND event_id IS NOT NULL"
        )
        recent_ids = [r[0] for r in cur.fetchall()]
    sample = rng.sample(all_ids, min(args.n, len(all_ids)))
    sample += rng.sample(recent_ids, min(args.recent_n, len(recent_ids)))
    sample = list(dict.fromkeys(sample))
    print(f"sampling {len(sample)} events ({len(all_ids)} total, {len(recent_ids)} gap-window)", flush=True)

    missing_markets = []
    closed_flag_mismatch = []
    closed_missing_prices = []
    events_not_on_gamma = 0
    markets_checked = 0

    with psycopg.connect(args.db_url) as conn, conn.cursor() as cur:
        for i in range(0, len(sample), 100):
            batch = sample[i : i + 100]
            gamma_events = fetch_events(batch)
            got = {int(e["id"]) for e in gamma_events}
            events_not_on_gamma += len([b for b in batch if int(b) not in got])
            for ev in gamma_events:
                for gm in ev.get("markets") or []:
                    mid = int(gm["id"])
                    markets_checked += 1
                    cur.execute(
                        "SELECT closed, outcome_prices FROM polymarket_markets WHERE market_id=%s",
                        (mid,),
                    )
                    row = cur.fetchone()
                    if row is None:
                        missing_markets.append({"event_id": ev["id"], "market_id": mid})
                        continue
                    db_closed, db_prices = row
                    g_closed = bool(gm.get("closed"))
                    if g_closed != bool(db_closed):
                        closed_flag_mismatch.append({"market_id": mid, "gamma": g_closed, "db": bool(db_closed)})
                    if g_closed and gm.get("outcomePrices") and not db_prices:
                        closed_missing_prices.append({"market_id": mid})
            if (i // 100) % 5 == 0:
                print(
                    f"progress {i + len(batch)}/{len(sample)}: markets_checked={markets_checked} "
                    f"missing={len(missing_markets)} closed_mismatch={len(closed_flag_mismatch)}",
                    flush=True,
                )
            time.sleep(0.05)

    report = {
        "events_sampled": len(sample),
        "events_not_on_gamma_anymore": events_not_on_gamma,
        "markets_checked": markets_checked,
        "markets_missing_in_db": len(missing_markets),
        "missing_examples": missing_markets[:20],
        "closed_flag_mismatches": len(closed_flag_mismatch),
        "closed_flag_examples": closed_flag_mismatch[:20],
        "closed_missing_outcome_prices": len(closed_missing_prices),
        "verdict": "PASS" if not missing_markets else "FAIL",
    }
    out = json.dumps(report, indent=2)
    if args.report:
        Path(args.report).write_text(out + "\n")
    print(out)
    return 1 if missing_markets else 0


if __name__ == "__main__":
    sys.exit(main())
