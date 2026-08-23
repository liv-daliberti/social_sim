#!/usr/bin/env python3
"""Enumerate ALL Gamma event ids and market ids via the keyset endpoints.

Stdlib-only (urllib), resumable (state json + append-only id files).
Produces sorted unique id lists for exact diffing against the archive DB.

Feeds walked:
  events:  /events/keyset  closed=true  -> events_closed.ids
           /events/keyset  closed=false -> events_open.ids
  markets: /markets/keyset (no filter)  -> markets_all.ids

Usage:
  gamma_keyset_enumerate.py --out-dir OUT [--feeds events_closed,events_open,markets_all]
                            [--limit 100] [--sleep 0.05] [--max-pages 0]

Run AFTER the closed-history walk (or during the candle phase - candles hit
clob.polymarket.com, so there is no rate-limit contention with gamma).
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

HOST = "https://gamma-api.polymarket.com"

FEEDS = {
    "events_closed": {"path": "/events/keyset", "params": {"closed": "true"}, "items_key": "events"},
    "events_open": {"path": "/events/keyset", "params": {"closed": "false"}, "items_key": "events"},
    "markets_all": {"path": "/markets/keyset", "params": {}, "items_key": "markets"},
}


def fetch(url, retries=6, timeout=30):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "archive-completeness-audit"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(30, 0.5 * 2 ** attempt))
    raise RuntimeError(f"fetch failed after {retries} tries: {url}: {last}")


def walk(feed_name, spec, out_dir, limit, sleep, max_pages):
    state_path = os.path.join(out_dir, f"{feed_name}.state.json")
    ids_path = os.path.join(out_dir, f"{feed_name}.ids")
    state = {"cursor": None, "pages": 0, "done": False}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
    if state.get("done"):
        print(f"[{feed_name}] already done ({state['pages']} pages)", flush=True)
        return
    mode = "a" if os.path.exists(ids_path) else "w"
    with open(ids_path, mode) as out:
        while True:
            if max_pages and state["pages"] >= max_pages:
                break
            params = dict(spec["params"], limit=limit)
            if state["cursor"]:
                params["after_cursor"] = state["cursor"]
            url = f"{HOST}{spec['path']}?{urllib.parse.urlencode(params)}"
            payload = fetch(url)
            items = payload.get(spec["items_key"]) or []
            next_cursor = payload.get("next_cursor")
            for it in items:
                iid = it.get("id")
                if iid is not None:
                    out.write(f"{iid}\n")
            out.flush()
            state["pages"] += 1
            state["cursor"] = str(next_cursor) if next_cursor else None
            if state["pages"] % 100 == 0:
                print(f"[{feed_name}] pages={state['pages']} ids~={state['pages']*limit}", flush=True)
                with open(state_path, "w") as f:
                    json.dump(state, f)
            if not next_cursor or not items:
                state["done"] = True
                break
            time.sleep(sleep)
    with open(state_path, "w") as f:
        json.dump(state, f)
    # sort -u the id file
    with open(ids_path) as f:
        ids = sorted({int(line) for line in f if line.strip().isdigit()})
    with open(ids_path + ".sorted", "w") as f:
        f.write("\n".join(map(str, ids)) + "\n")
    print(f"[{feed_name}] DONE pages={state['pages']} unique_ids={len(ids)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--feeds", default="events_closed,events_open,markets_all")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--max-pages", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for feed in args.feeds.split(","):
        feed = feed.strip()
        if feed not in FEEDS:
            print(f"unknown feed {feed}", file=sys.stderr)
            return 1
        walk(feed, FEEDS[feed], args.out_dir, args.limit, args.sleep, args.max_pages)
    return 0


if __name__ == "__main__":
    sys.exit(main())
