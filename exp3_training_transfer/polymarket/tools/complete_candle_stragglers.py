#!/usr/bin/env python3
"""Complete dense-daily candles for markets the stock backfill can never plan.

Root cause: Gamma metadata for retroactively-listed markets has start_time >
end_time (endDate backdated to the event time). backfill_polymarket_daily_candles_all
silently skips any token whose clamped window has end < start, so these markets
stay candle-less forever even when CLOB holds their full trade history.

This tool:
  1. selects tokens that are strictly incomplete per the OFFICIAL definition
     (daily rows < GREATEST(1, expected_days over [start, min(end, now)]))
  2. fetches CLOB prices-history with interval=max (window-independent)
  3. repairs the densification window to
        [min(nominal_start, first_trade_day), max(nominal_end_clamped, last_trade_day)]
     (clamped to now; if no trades and the nominal window is invalid, uses the
     single nominal-start day so the series is minimally dense and honest)
  4. reuses the pipeline's own _build_dense_daily_candles + _upsert_daily_candles
     (same empty/carry-forward semantics, same ON CONFLICT upsert)

Usage (run under pmpython):
  complete_candle_stragglers.py --db-url URL --shard-index N --shard-count M
                                [--limit 0] [--sleep 0.05] [--dry-run]
"""
import argparse
import json
import logging
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path("/n/fs/similarity/kalshi/agentic_forecasting")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import psycopg  # noqa: E402
from backfill_polymarket_daily_candles_all import (  # noqa: E402
    TokenPlan,
    _build_dense_daily_candles,
    _upsert_daily_candles,
)

LOGGER = logging.getLogger("complete_candle_stragglers")
CLOB = "https://clob.polymarket.com"

INCOMPLETE_TOKENS_SQL = """
WITH tok AS (
  SELECT
    m.market_id,
    m.event_id,
    m.condition_id,
    token.token_id,
    outcome.outcome,
    COALESCE(m.start_time, e.start_time, m.created_time, e.created_time, m.end_time, e.end_time) AS nom_start,
    COALESCE(m.end_time, e.end_time) AS nom_end,
    ROW_NUMBER() OVER (ORDER BY m.market_id, token.ord) - 1 AS rn
  FROM polymarket_markets m
  LEFT JOIN polymarket_events e ON e.event_id = m.event_id
  JOIN LATERAL jsonb_array_elements_text(COALESCE(m.clob_token_ids, '[]'::jsonb))
    WITH ORDINALITY AS token(token_id, ord) ON TRUE
  LEFT JOIN LATERAL jsonb_array_elements_text(COALESCE(m.outcomes, '[]'::jsonb))
    WITH ORDINALITY AS outcome(outcome, ord) ON outcome.ord = token.ord
  WHERE m.clob_token_ids IS NOT NULL
    AND jsonb_typeof(m.clob_token_ids) = 'array'
    AND COALESCE(m.start_time, e.start_time, m.created_time, e.created_time, m.end_time, e.end_time) IS NOT NULL
),
have AS (
  SELECT token_id, COUNT(*) AS have_days
  FROM polymarket_market_candles
  WHERE period_interval_minutes = 1440
  GROUP BY token_id
),
counted AS (
  SELECT
    tok.*,
    GREATEST(
      1,
      (
        DATE_TRUNC('day', LEAST(COALESCE(tok.nom_end, NOW()), NOW()))::date
        - DATE_TRUNC('day', tok.nom_start)::date
      ) + 1
    ) AS expected_days,
    COALESCE(have.have_days, 0) AS have_days
  FROM tok LEFT JOIN have ON have.token_id = tok.token_id
)
SELECT market_id, event_id, condition_id, token_id, outcome,
       nom_start, nom_end, expected_days, have_days
FROM counted
WHERE have_days < expected_days AND rn %% %s = %s
"""


def fetch_history_max(token_id: str, retries: int = 6, timeout: int = 45):
    url = f"{CLOB}/prices-history?" + urllib.parse.urlencode(
        {"market": token_id, "interval": "max", "fidelity": 1440}
    )
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "candle-straggler-completer"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read())
            hist = payload.get("history") if isinstance(payload, dict) else None
            return [{"ts": p.get("t"), "price": p.get("p")} for p in (hist or [])], None
        except urllib.error.HTTPError as e:
            if e.code in (404, 422):
                return [], f"http{e.code}"
            last = f"http{e.code}"
            time.sleep(min(30, 0.5 * 2 ** attempt))
        except Exception as e:  # noqa: BLE001
            last = str(e)[:60]
            time.sleep(min(30, 0.5 * 2 ** attempt))
    return None, last


def repaired_window(nom_start, nom_end, points, now):
    day = timedelta(days=1)
    start = nom_start
    end = min(nom_end, now) if nom_end else now
    if points:
        ts = [datetime.fromtimestamp(int(p["ts"]), tz=timezone.utc) for p in points if p.get("ts")]
        if ts:
            start = min(start, min(ts))
            end = max(end, max(ts))
    end = min(end, now)
    if end < start:
        end = min(start + day, now)
    if end < start:
        start = end
    return start, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    now = datetime.now(timezone.utc)
    with psycopg.connect(args.db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(INCOMPLETE_TOKENS_SQL, (args.shard_count, args.shard_index))
            rows = cur.fetchall()
        if args.limit:
            rows = rows[: args.limit]
        LOGGER.info("shard %s/%s: %s incomplete tokens", args.shard_index, args.shard_count, len(rows))

        done = written = empty = errors = 0
        for (market_id, event_id, condition_id, token_id, outcome,
             nom_start, nom_end, expected_days, have_days) in rows:
            nom_start = nom_start.astimezone(timezone.utc)
            nom_end = nom_end.astimezone(timezone.utc) if nom_end else None
            points, err = fetch_history_max(str(token_id))
            if points is None:
                errors += 1
                LOGGER.warning("fetch failed token=%s err=%s", token_id, err)
                continue
            w_start, w_end = repaired_window(nom_start, nom_end, points, now)
            plan = TokenPlan(
                token_id=str(token_id),
                condition_id=condition_id,
                outcome=outcome,
                event_id=event_id,
                market_id=market_id,
                window_start=w_start,
                window_end=w_end,
            )
            candles = _build_dense_daily_candles(plan, points)
            if not args.dry_run and candles:
                written += _upsert_daily_candles(conn, candles)
            if not points:
                empty += 1
            done += 1
            if done % args.log_every == 0:
                LOGGER.info(
                    "progress shard=%s %s/%s written=%s empty_tokens=%s errors=%s",
                    args.shard_index, done, len(rows), written, empty, errors,
                )
            time.sleep(args.sleep)

    print(json.dumps({
        "shard": args.shard_index,
        "tokens": len(rows),
        "processed": done,
        "candles_written": written,
        "empty_tokens": empty,
        "errors": errors,
    }))


if __name__ == "__main__":
    main()
