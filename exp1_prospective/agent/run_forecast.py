#!/usr/bin/env python3
"""Step 2: Run structured world-model forecast agent on diverse markets.

Reads data/selected_markets/diverse_YYYY-MM-DD.jsonl, calls the Azure AI
Foundry agent k times per market in a multi-turn conversation (event model →
evidence search → synthesis → structured JSON forecast), and writes results
to data/initial_forecasts/forecasts_YYYY-MM-DD.jsonl.

Each output record contains a top-level yes_prob (mean across k runs) and a
k_runs list with the full per-run conversation and structured forecast.

Usage (from exp1_prospective/):
    export AZURE_AI_API_KEY=<your-key>
    python agent/run_forecast.py
    python agent/run_forecast.py --input data/selected_markets/diverse_2026-06-05.jsonl
    python agent/run_forecast.py --n 10 --k 5     # pilot: first 10 markets, 5 runs each
    python agent/run_forecast.py --no-third-turn   # faster 3-turn mode
    python agent/run_forecast.py --delay 5 --verbose
    python agent/run_forecast.py --dry-run         # print prompts, no API calls

The script resumes automatically: if an output file already exists, markets
already forecasted (by task_id) are skipped.
"""

from __future__ import annotations

import argparse
import json
import fcntl
import math
import statistics
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── self-contained: add agent/ dir to path for local imports ──────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from forecast_agent import ForecastRecord, make_openai_client, forecast_market, DEFAULT_MODEL, AGENT_NAME, AGENT_VERSION

_ROOT     = _HERE.parent
_SEL_DIR  = _ROOT / "data" / "selected_markets"
_OUT_DIR  = _ROOT / "data" / "initial_forecasts"


# ── helpers ────────────────────────────────────────────────────────────────────

def _latest_diverse(sel_dir: Path) -> Path | None:
    for f in sorted(sel_dir.glob("diverse_*.jsonl"), reverse=True):
        return f
    return None


def _load_done(out_path: Path, k: int) -> set[str]:
    """Return task_ids that have all k runs written.

    Each task may have multiple lines (one per completed run); the last line
    for a given task_id is the authoritative record.  A task is considered
    done only when its latest record has len(k_runs) == k.
    """
    if not out_path.exists():
        return set()
    latest: dict[str, dict] = {}
    with out_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                tid = rec["task_id"]
                latest[tid] = rec
            except (json.JSONDecodeError, KeyError):
                pass
    return {tid for tid, rec in latest.items() if len(rec.get("k_runs", [])) >= k}


def _build_record(market: dict, runs: list[dict], k: int) -> dict:
    """Assemble the aggregate record for a market given completed runs so far."""
    probs = [r["yes_prob"] for r in runs if r["yes_prob"] is not None]
    median_prob = statistics.median(probs) if probs else None
    std_prob    = statistics.stdev(probs) if len(probs) > 1 else 0.0
    first = runs[0] if runs else {}
    return {
        "task_id":             market.get("task_id", f"pm_{market['market_id']}"),
        "market_id":           market["market_id"],
        "question":            market.get("question", ""),
        "description":         market.get("description", ""),
        "yes_price_market":    market.get("yes_price"),
        "days_to_resolution":  market.get("days_to_resolution"),
        "category":            market.get("category"),
        "k":                   k,
        "k_done":              len(runs),
        "k_runs":              runs,
        "yes_prob":            round(median_prob, 4) if median_prob is not None else None,
        "yes_prob_std":        round(std_prob, 4),
        "yes_prob_runs":       probs,
        # backwards-compat top-level fields (first run's data)
        "turns":               first.get("turns", []),
        "structured_forecast": first.get("structured_forecast"),
        "parse_error":         first.get("parse_error"),
        "error":               first.get("error"),
        "n_turns":             first.get("n_turns", 0),
        "total_input_tokens":  sum(r.get("total_input_tokens", 0) for r in runs),
        "total_output_tokens": sum(r.get("total_output_tokens", 0) for r in runs),
        "forecast_at":         datetime.now(timezone.utc).isoformat(),
    }


def _run_k_times(
    market: dict,
    *,
    client,
    model: str,
    k: int,
    do_third_turn: bool,
    verbose: bool,
    run_delay: float = 1.0,
    write_fn=None,
) -> dict:
    """Run forecast_market k times.

    After each completed run, calls write_fn(record_dict) so callers can
    flush incremental progress to disk immediately.  Returns the final
    aggregated record.
    """
    runs: list[dict] = []

    for run_id in range(k):
        if verbose and k > 1:
            print(f"    [run {run_id + 1}/{k}]")
        rec: ForecastRecord = forecast_market(
            market,
            client=client,
            model=model,
            do_third_turn=do_third_turn,
            verbose=verbose,
        )
        d = rec.to_dict()
        runs.append({
            "run_id":              run_id,
            "turns":               d["turns"],
            "structured_forecast": d["structured_forecast"],
            "yes_prob":            d["yes_prob"],
            "parse_error":         d["parse_error"],
            "error":               d["error"],
            "n_turns":             d["n_turns"],
            "total_input_tokens":  d["total_input_tokens"],
            "total_output_tokens": d["total_output_tokens"],
            "forecast_at":         d["forecast_at"],
        })

        # incremental write after every completed run
        if write_fn is not None:
            write_fn(_build_record(market, runs, k))

        if run_id < k - 1:
            time.sleep(run_delay)

    return _build_record(market, runs, k)


class _Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.done = self.ok = self.errors = self.skipped = 0
        self._start = time.time()

    def record(self, status: str) -> None:
        self.done += 1
        if   status == "ok":   self.ok      += 1
        elif status == "skip": self.skipped  += 1
        else:                  self.errors   += 1
        elapsed = time.time() - self._start
        rate    = self.done / max(elapsed, 0.1)
        eta_s   = (self.total - self.done) / max(rate, 1e-6)
        eta_str = f"{eta_s/60:.1f}m" if eta_s > 90 else f"{eta_s:.0f}s"
        pct     = 100 * self.done / max(self.total, 1)
        filled  = int(30 * self.done / max(self.total, 1))
        bar     = "█" * filled + "░" * (30 - filled)
        sys.stdout.write(
            f"\r  [{bar}] {self.done}/{self.total}"
            f"  ok {self.ok}  err {self.errors}  skip {self.skipped}"
            f"  {pct:.1f}%  ETA {eta_str}   "
        )
        sys.stdout.flush()

    def done_line(self) -> None:
        elapsed = time.time() - self._start
        sys.stdout.write("\n")
        print(
            f"\n  Done in {elapsed/60:.1f}m — "
            f"{self.ok} forecasted  {self.errors} errors  {self.skipped} skipped"
        )


# ── cli ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run structured world-model forecast agent on diverse Polymarket markets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input",          type=Path, default=None,
                    help="Markets JSONL to forecast.  Defaults to latest diverse_*.jsonl.")
    ap.add_argument("--out",            type=Path, default=_OUT_DIR,
                    help="Output directory.")
    ap.add_argument("--n",              type=int,  default=0,
                    help="Stop after N markets (0 = all).  Useful for pilots.")
    ap.add_argument("--k",              type=int,  default=5,
                    help="Number of independent forecast runs per market.")
    ap.add_argument("--delay",          type=float, default=3.0,
                    help="Seconds between markets (avoid rate limits).")
    ap.add_argument("--run-delay",      type=float, default=1.0,
                    help="Seconds between runs within a single market.")
    ap.add_argument("--model",          type=str,  default=DEFAULT_MODEL,
                    help="Model deployment name in Azure AI Foundry.")
    ap.add_argument("--api-key",        type=str,  default=None,
                    help="Azure AI API key (overrides AZURE_AI_API_KEY env var).")
    ap.add_argument("--endpoint",       type=str,  default=None,
                    help="Azure agent endpoint base URL.")
    ap.add_argument("--no-third-turn",  action="store_true",
                    help="Skip the optional 3rd evidence-deepening turn (faster).")
    ap.add_argument("--verbose",        action="store_true",
                    help="Print per-turn progress.")
    ap.add_argument("--dry-run",        action="store_true",
                    help="Print the first market's prompts without making API calls.")
    args = ap.parse_args()

    # ── resolve input ──────────────────────────────────────────────────────────
    input_path = args.input or _latest_diverse(_SEL_DIR)
    if input_path is None or not input_path.exists():
        print(f"No diverse JSONL found.  Run diversify_markets.py first.")
        sys.exit(1)

    with input_path.open() as f:
        markets = [json.loads(line) for line in f if line.strip()]

    print(f"Input:  {input_path}  ({len(markets)} markets)")
    print(f"k={args.k} runs per market")

    # ── output path ────────────────────────────────────────────────────────────
    date_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path  = args.out / f"forecasts_{date_str}.jsonl"
    mani_path = args.out / f"forecasts_{date_str}.manifest.json"
    args.out.mkdir(parents=True, exist_ok=True)

    done_ids = _load_done(out_path, args.k)
    if done_ids:
        print(f"Resuming: {len(done_ids)} already done, skipping those task_ids")

    # ── filter + cap ───────────────────────────────────────────────────────────
    pending = [m for m in markets if m.get("task_id") not in done_ids]
    if args.n > 0:
        pending = pending[:args.n]

    print(f"Pending: {len(pending)} markets to forecast")
    print(f"Output: {out_path}\n")

    # ── dry run ────────────────────────────────────────────────────────────────
    if args.dry_run:
        from prompts import SYSTEM_PROMPT, TURN1_TEMPLATE, TURN2_TEMPLATE, TURN3_TEMPLATE, FINAL_TURN_TEMPLATE
        m = pending[0] if pending else markets[0]
        print("=" * 70)
        print("SYSTEM PROMPT:\n" + SYSTEM_PROMPT)
        print("\n" + "=" * 70)
        print("TURN 1:\n" + TURN1_TEMPLATE.format(
            question=m.get("question",""), description=m.get("description","")[:300],
            yes_price=m.get("yes_price",0.5), days_to_resolution=m.get("days_to_resolution",0),
            category=m.get("category","unknown"),
        ))
        print("\n" + "=" * 70)
        print("TURN 2:\n" + TURN2_TEMPLATE)
        print("\n" + "=" * 70)
        print("TURN 3:\n" + TURN3_TEMPLATE)
        print("\n" + "=" * 70)
        print("FINAL:\n" + FINAL_TURN_TEMPLATE)
        return

    # ── build client ───────────────────────────────────────────────────────────
    client = make_openai_client(api_key=args.api_key, endpoint=args.endpoint)

    # ── run ────────────────────────────────────────────────────────────────────
    progress = _Progress(len(pending))
    n_ok = n_err = 0

    with out_path.open("a") as out_f:
        for i, market in enumerate(pending):
            q = market.get("question", "")[:60]
            if args.verbose:
                print(f"\n[{i+1}/{len(pending)}] {q}")

            def _write(rec_dict, _f=out_f):
                line = json.dumps(rec_dict) + "\n"
                fcntl.flock(_f, fcntl.LOCK_EX)
                try:
                    _f.write(line)
                    _f.flush()
                finally:
                    fcntl.flock(_f, fcntl.LOCK_UN)

            try:
                out_rec = _run_k_times(
                    market,
                    client=client,
                    model=args.model,
                    k=args.k,
                    do_third_turn=not args.no_third_turn,
                    verbose=args.verbose,
                    run_delay=args.run_delay,
                    write_fn=_write,
                )
                # final record already written by write_fn on last run

                if out_rec.get("error"):
                    progress.record("error")
                    n_err += 1
                else:
                    progress.record("ok")
                    n_ok += 1

                if not args.verbose:
                    yp = out_rec.get("yes_prob")
                    runs_str = ", ".join(f"{p:.0%}" for p in (out_rec.get("yes_prob_runs") or []))
                    label = f"  median={yp:.0%} [{runs_str}]" if yp is not None else "  parse_err"
                    sys.stdout.write(f"  {q[:45]:<45}{label}\n")

            except Exception as exc:
                err_rec = {
                    "task_id":          market.get("task_id", "?"),
                    "market_id":        market.get("market_id", "?"),
                    "question":         market.get("question", ""),
                    "description":      market.get("description", ""),
                    "yes_price_market": market.get("yes_price"),
                    "days_to_resolution": market.get("days_to_resolution"),
                    "category":         market.get("category"),
                    "k":                args.k,
                    "k_runs":           [],
                    "yes_prob":         None,
                    "yes_prob_runs":    [],
                    "error":            str(exc),
                    "forecast_at":      datetime.now(timezone.utc).isoformat(),
                }
                out_f.write(json.dumps(err_rec) + "\n")
                out_f.flush()
                progress.record("error")
                n_err += 1
                if args.verbose:
                    print(f"    ERROR: {exc}")

            if i < len(pending) - 1:
                time.sleep(args.delay)

    progress.done_line()

    # ── manifest ───────────────────────────────────────────────────────────────
    manifest = {
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "source_file": str(input_path),
        "model":       args.model,
        "k":           args.k,
        "n_markets":   len(pending),
        "n_ok":        n_ok,
        "n_errors":    n_err,
        "third_turn":  not args.no_third_turn,
    }
    with mani_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest → {mani_path}")


if __name__ == "__main__":
    main()
