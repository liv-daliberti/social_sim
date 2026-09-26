#!/usr/bin/env python3
"""Streaming validator for polymarket_full_market_dataset_*.json exports (schema v2).

Reads the export line-by-line (header block, then one compact JSON row per line)
and verifies everything checkable without the source DB:
  - file parses completely; header row_count / markets_with_candles match reality
  - no duplicate market_id / condition_id
  - row ordering matches header row_ordering (open_time asc nulls last)
  - required top-level keys present in every row
  - candle integrity: per-row counts vs candle_summary.count, sortedness,
    interval inventory, global end_period_ts range
  - availability flags consistent with row contents
  - resolution / closed / event coverage stats

Usage: validate_export.py EXPORT.json [--report OUT.json] [--summary SUMMARY.json]
"""
import argparse
import json
import sys
from collections import Counter

try:
    import orjson as fastjson

    def loads(b):
        return fastjson.loads(b)
except ImportError:  # pragma: no cover
    def loads(b):
        return json.loads(b)

REQUIRED_ROW_KEYS = (
    "availability",
    "candle_summary",
    "candles",
    "event",
    "ids",
    "market",
    "trade_metrics",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_path")
    ap.add_argument("--report", default=None)
    ap.add_argument("--summary", default=None, help="optional .summary.json to cross-check")
    ap.add_argument("--progress-every", type=int, default=100_000)
    args = ap.parse_args()

    issues = Counter()
    examples = {}

    def issue(kind, detail=None):
        issues[kind] += 1
        if detail is not None and kind not in examples:
            examples[kind] = detail

    header_lines = []
    n_rows = 0
    n_with_candles = 0
    n_avail_has_candles = 0
    n_avail_has_resolution = 0
    n_avail_has_memory_case = 0
    n_resolution_obj = 0
    resolution_methods = Counter()
    n_closed = 0
    n_resolved = 0
    n_active = 0
    n_event_present = 0
    total_candle_rows = 0
    n_candles_nonnull_close = 0
    interval_inventory = Counter()
    market_ids = set()
    condition_ids = set()
    min_created, max_created = None, None
    min_candle_ts, max_candle_ts = None, None
    prev_open = None
    seen_null_open = False
    missing_key_counts = Counter()

    state = "header"
    with open(args.export_path, "rb") as f:
        for raw in f:
            if state == "header":
                s = raw.strip()
                if s == b'"rows": [':
                    state = "rows"
                    continue
                header_lines.append(raw)
                continue
            s = raw.strip()
            if not s:
                continue
            if s in (b"]", b"],"):
                state = "tail"
                continue
            if state == "tail":
                if s != b"}":
                    issue("unexpected_content_after_rows", s[:80].decode("utf-8", "replace"))
                continue
            if s.endswith(b","):
                s = s[:-1]
            try:
                row = loads(s)
            except Exception as e:
                issue("row_parse_error", f"row ~{n_rows}: {e}")
                continue
            n_rows += 1
            if args.progress_every and n_rows % args.progress_every == 0:
                print(f"... {n_rows:,} rows", file=sys.stderr, flush=True)

            for k in REQUIRED_ROW_KEYS:
                if k not in row:
                    missing_key_counts[k] += 1

            ids = row.get("ids") or {}
            mid = ids.get("market_id")
            cid = ids.get("condition_id")
            if mid is None:
                issue("null_market_id", f"row {n_rows}")
            elif mid in market_ids:
                issue("duplicate_market_id", mid)
            else:
                market_ids.add(mid)
            if cid:
                if cid in condition_ids:
                    issue("duplicate_condition_id", cid)
                else:
                    condition_ids.add(cid)
            else:
                issue("null_condition_id", f"row {n_rows} market_id={mid}")

            market = row.get("market") or {}
            times = row.get("times") or {}
            if market.get("closed"):
                n_closed += 1
            if market.get("resolved"):
                n_resolved += 1
            if market.get("active"):
                n_active += 1
            ct = times.get("market_created_time")
            if ct:
                if min_created is None or ct < min_created:
                    min_created = ct
                if max_created is None or ct > max_created:
                    max_created = ct

            event = row.get("event") or {}
            if event.get("title") or event.get("slug"):
                n_event_present += 1

            ot = times.get("open_time")
            if ot is None:
                seen_null_open = True
            else:
                if seen_null_open:
                    issue("nonnull_open_time_after_null", f"row {n_rows} market_id={mid}")
                if prev_open is not None and ot < prev_open:
                    issue("open_time_order_violation", f"row {n_rows}: {ot} < {prev_open}")
                prev_open = ot

            candles = row.get("candles") or []
            summary = row.get("candle_summary") or {}
            avail = row.get("availability") or {}
            if candles:
                n_with_candles += 1
            if avail.get("has_candles"):
                n_avail_has_candles += 1
            if bool(candles) != bool(avail.get("has_candles")):
                issue("has_candles_flag_mismatch", f"market_id={mid}")
            if avail.get("has_resolution"):
                n_avail_has_resolution += 1
            if avail.get("has_market_memory_case"):
                n_avail_has_memory_case += 1
            res = row.get("resolution") or {}
            if res.get("winner_index") is not None or res.get("winner_yes") is not None:
                n_resolution_obj += 1
                resolution_methods[res.get("resolution_method") or "?"] += 1
            if bool(avail.get("has_resolution")) != (
                res.get("winner_index") is not None
                or res.get("winner_yes") is not None
                or res.get("strict_settlement_yes") is not None
            ):
                issue("has_resolution_flag_mismatch", f"market_id={mid}")

            sc = summary.get("count")
            if sc is not None and sc != len(candles):
                issue("candle_summary_count_mismatch", f"market_id={mid}: summary={sc} actual={len(candles)}")
            total_candle_rows += len(candles)
            prev_ts_by_token = {}
            per_row_intervals = set()
            for c in candles:
                ts = c.get("end_period_ts")
                tok = (c.get("token_id"), c.get("period_interval_minutes"))
                if ts:
                    if min_candle_ts is None or ts < min_candle_ts:
                        min_candle_ts = ts
                    if max_candle_ts is None or ts > max_candle_ts:
                        max_candle_ts = ts
                    prev_ts = prev_ts_by_token.get(tok)
                    if prev_ts is not None and ts < prev_ts:
                        issue("candle_ts_order_violation", f"market_id={mid} token={tok[0][:20] if tok[0] else None}")
                    elif prev_ts is not None and ts == prev_ts:
                        issue("duplicate_candle_ts", f"market_id={mid}")
                    prev_ts_by_token[tok] = ts
                if c.get("close") is not None:
                    n_candles_nonnull_close += 1
                iv = c.get("period_interval_minutes")
                if iv is not None:
                    per_row_intervals.add(iv)
            for iv in per_row_intervals:
                interval_inventory[iv] += 1

    header_text = b"".join(header_lines).strip()
    if header_text.endswith(b","):
        header_text = header_text[:-1]
    header = loads(header_text + b"}")

    checks = {
        "header_row_count": header.get("row_count"),
        "actual_rows": n_rows,
        "row_count_match": header.get("row_count") == n_rows,
        "header_markets_with_candles": header.get("markets_with_candles"),
        "actual_markets_with_candles": n_with_candles,
        "markets_with_candles_match": header.get("markets_with_candles") == n_with_candles,
        "unique_market_ids": len(market_ids),
        "unique_condition_ids": len(condition_ids),
    }
    report = {
        "export_path": args.export_path,
        "generated_at": header.get("generated_at"),
        "row_schema_version": header.get("row_schema_version"),
        "checks": checks,
        "issues": dict(issues),
        "issue_examples": examples,
        "missing_required_keys": dict(missing_key_counts),
        "stats": {
            "closed": n_closed,
            "resolved": n_resolved,
            "active": n_active,
            "event_present": n_event_present,
            "availability_has_candles": n_avail_has_candles,
            "availability_has_resolution": n_avail_has_resolution,
            "availability_has_memory_case": n_avail_has_memory_case,
            "resolution_object_present": n_resolution_obj,
            "resolution_methods": dict(resolution_methods),
            "total_candle_rows": total_candle_rows,
            "candles_nonnull_close": n_candles_nonnull_close,
            "interval_inventory_markets": {str(k): v for k, v in sorted(interval_inventory.items())},
            "created_time_range": [min_created, max_created],
            "candle_end_ts_range": [min_candle_ts, max_candle_ts],
        },
    }
    if args.summary:
        with open(args.summary, "rb") as sf:
            summ = loads(sf.read())
        report["summary_cross_check"] = {
            "summary_row_count": summ.get("row_count"),
            "summary_matches_actual": summ.get("row_count") == n_rows,
        }

    hard_fail = (
        not checks["row_count_match"]
        or not checks["markets_with_candles_match"]
        or issues.get("row_parse_error", 0) > 0
        or issues.get("duplicate_market_id", 0) > 0
        or issues.get("duplicate_condition_id", 0) > 0
        or issues.get("open_time_order_violation", 0) > 0
        or sum(missing_key_counts.values()) > 0
    )
    report["verdict"] = "FAIL" if hard_fail else "PASS"

    out = json.dumps(report, indent=2, default=str)
    if args.report:
        with open(args.report, "w") as rf:
            rf.write(out + "\n")
    print(out)
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
