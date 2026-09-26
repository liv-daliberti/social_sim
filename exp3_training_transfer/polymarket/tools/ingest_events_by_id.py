#!/usr/bin/env python3
"""Targeted ingest: fetch specific Gamma event ids (batched) and upsert them
into polymarket_archive using the pipeline's own mapping/upsert helpers.

Run under the pipeline container python (pmpython) from any cwd:
  pmpython ingest_events_by_id.py --db-url URL --id-file missing_in_db.txt
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path("/n/fs/similarity/kalshi/agentic_forecasting")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import psycopg  # noqa: E402
from refill_polymarket_closed_history import (  # noqa: E402
    _event_payload,
    _market_payload,
    _upsert_events_and_markets,
    split_gamma_event_markets,
)

HOST = "https://gamma-api.polymarket.com"


def fetch_batch(ids, retries=6, timeout=30):
    qs = "&".join(f"id={i}" for i in ids)
    url = f"{HOST}/events?{qs}&limit={max(100, len(ids))}"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "archive-gap-ingest"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read())
                return payload if isinstance(payload, list) else []
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(30, 0.5 * 2 ** attempt))
    raise RuntimeError(f"batch fetch failed: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", required=True)
    ap.add_argument("--id-file", required=True)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=0.05)
    args = ap.parse_args()

    ids = [line.strip() for line in open(args.id_file) if line.strip().isdigit()]
    print(f"ids to ingest: {len(ids)}", flush=True)

    fetched = upserted_events = upserted_markets = 0
    returned_ids = set()
    with psycopg.connect(args.db_url) as conn:
        for i in range(0, len(ids), args.batch_size):
            batch = ids[i : i + args.batch_size]
            rows = fetch_batch(batch)
            fetched += len(rows)
            event_payloads = {}
            market_payloads = []
            for raw in rows:
                returned_ids.add(str(raw.get("id")))
                mapped_event, mapped_markets = split_gamma_event_markets(raw)
                event_id = mapped_event.get("event_id")
                if event_id is None:
                    continue
                event_payloads[event_id] = _event_payload(mapped_event)
                for m in mapped_markets:
                    if m.get("market_id") is None:
                        continue
                    market_payloads.append(_market_payload(m))
            if event_payloads or market_payloads:
                _upsert_events_and_markets(
                    conn,
                    event_payloads_by_id=event_payloads,
                    market_payloads=market_payloads,
                )
            upserted_events += len(event_payloads)
            upserted_markets += len(market_payloads)
            if (i // args.batch_size) % 10 == 0:
                print(
                    f"progress: {i + len(batch)}/{len(ids)} fetched={fetched} "
                    f"events={upserted_events} markets={upserted_markets}",
                    flush=True,
                )
            time.sleep(args.sleep)

    not_returned = [i for i in ids if i not in returned_ids]
    print(
        json.dumps(
            {
                "requested": len(ids),
                "returned_by_gamma": len(returned_ids),
                "not_returned": len(not_returned),
                "event_upserts": upserted_events,
                "market_upserts": upserted_markets,
            },
            indent=2,
        ),
        flush=True,
    )
    if not_returned:
        out = Path(args.id_file).with_suffix(".not_returned.txt")
        out.write_text("\n".join(not_returned) + "\n")
        print(f"ids gamma no longer serves written to {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
