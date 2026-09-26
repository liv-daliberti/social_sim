#!/usr/bin/env python3
"""Add additional forecast runs to an existing local-model forecasts file.

Reads forecasts_{slug}_{date}.jsonl (which already has k_current runs per market)
and runs (target_k - k_current) additional runs per market, then writes updated
records (with all target_k runs merged) back to the same file.

Usage:
    python agent/topup_local_forecast.py --model qwen2.5:7b --target-k 5
    python agent/topup_local_forecast.py --model llama3.1:8b --target-k 5
    python agent/topup_local_forecast.py --model qwen2.5:7b --target-k 5 --input forecasts_qwen2.5-7b_2026-06-10.jsonl
"""

from __future__ import annotations

import argparse
import fcntl
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_ENV_FILE = _HERE / ".env"
if _ENV_FILE.exists():
    import os
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from local_agent import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_ENDPOINT,
    ForecastRecord,
    WebSearchTool,
    forecast_market_local,
    make_local_client,
)


_ROOT    = _HERE.parent
_OUT_DIR = _ROOT / "data" / "initial_forecasts"


def _model_slug(model: str) -> str:
    return model.replace(":", "-").replace("/", "-")


def _load_latest(path: Path) -> dict[str, dict]:
    """Return latest record per task_id from a JSONL file."""
    latest: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                latest[rec["task_id"]] = rec
            except (json.JSONDecodeError, KeyError):
                pass
    return latest


def _build_record(market_rec: dict, all_runs: list[dict], target_k: int, model: str) -> dict:
    probs    = [r["yes_prob"] for r in all_runs if r.get("yes_prob") is not None]
    median_p = statistics.median(probs) if probs else None
    std_p    = statistics.stdev(probs) if len(probs) > 1 else 0.0
    first    = all_runs[0] if all_runs else {}
    return {
        "task_id":             market_rec["task_id"],
        "market_id":           market_rec["market_id"],
        "question":            market_rec.get("question", ""),
        "description":         market_rec.get("description", ""),
        "yes_price_market":    market_rec.get("yes_price_market"),
        "days_to_resolution":  market_rec.get("days_to_resolution"),
        "category":            market_rec.get("category"),
        "k":                   target_k,
        "k_done":              len(all_runs),
        "k_runs":              all_runs,
        "yes_prob":            round(median_p, 4) if median_p is not None else None,
        "yes_prob_std":        round(std_p, 4),
        "yes_prob_runs":       probs,
        "backend":             "local-ollama",
        "model":               model,
        "turns":               first.get("turns", []),
        "structured_forecast": first.get("structured_forecast"),
        "parse_error":         first.get("parse_error"),
        "error":               first.get("error"),
        "n_turns":             first.get("n_turns", 0),
        "total_input_tokens":  sum(r.get("total_input_tokens", 0) for r in all_runs),
        "total_output_tokens": sum(r.get("total_output_tokens", 0) for r in all_runs),
        "forecast_at":         datetime.now(timezone.utc).isoformat(),
    }


class _Progress:
    def __init__(self, total: int) -> None:
        self.total  = total
        self.done   = self.ok = self.errors = self.skipped = 0
        self._start = time.time()

    def record(self, status: str) -> None:
        self.done += 1
        if   status == "ok":   self.ok     += 1
        elif status == "skip": self.skipped += 1
        else:                  self.errors  += 1
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
            f"{self.ok} topped-up  {self.errors} errors  {self.skipped} skipped"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Top up local model forecast runs to target-k.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model",     type=str, default=DEFAULT_MODEL,
                    help="Ollama model tag (e.g. qwen2.5:7b, llama3.1:8b).")
    ap.add_argument("--endpoint",  type=str, default=DEFAULT_OLLAMA_ENDPOINT)
    ap.add_argument("--target-k",  type=int, default=5,
                    help="Target total runs per market.")
    ap.add_argument("--input",     type=Path, default=None,
                    help="Specific JSONL file to top up. Defaults to latest forecasts_{slug}_*.jsonl.")
    ap.add_argument("--out-dir",   type=Path, default=_OUT_DIR)
    ap.add_argument("--delay",     type=float, default=1.0)
    ap.add_argument("--max-search-calls", type=int, default=6)
    ap.add_argument("--max-tokens",       type=int, default=1500)
    ap.add_argument("--temperature",      type=float, default=0.1)
    ap.add_argument("--timeout",          type=float, default=600.0,
                    help="Per-request timeout in seconds. Use 3600+ for 70b models.")
    ap.add_argument("--search-backend",   type=str, default="auto",
                    choices=["auto", "tavily", "duckduckgo"])
    ap.add_argument("--no-third-turn",    action="store_true")
    ap.add_argument("--verbose",          action="store_true")
    ap.add_argument("--n",                type=int, default=0,
                    help="Stop after N markets (0 = all). For testing.")
    args = ap.parse_args()

    slug = _model_slug(args.model)

    if args.input:
        in_path = args.input if args.input.is_absolute() else args.out_dir / args.input
    else:
        candidates = sorted(args.out_dir.glob(f"forecasts_{slug}_*.jsonl"), reverse=True)
        candidates = [c for c in candidates if "old" not in c.name]
        if not candidates:
            print(f"No forecasts file found for {slug} in {args.out_dir}")
            sys.exit(1)
        in_path = candidates[0]

    print(f"Input:    {in_path}")
    print(f"Model:    {args.model}")
    print(f"Target k: {args.target_k}")

    records = _load_latest(in_path)
    print(f"Loaded:   {len(records)} records")

    to_topup = {
        tid: rec for tid, rec in records.items()
        if len(rec.get("k_runs", [])) < args.target_k
    }
    already_done = len(records) - len(to_topup)
    print(f"Already at k={args.target_k}: {already_done}  Need top-up: {len(to_topup)}")

    if not to_topup:
        print("Nothing to do.")
        return

    pending = list(to_topup.values())
    if args.n > 0:
        pending = pending[:args.n]
    print(f"Processing: {len(pending)} markets\n")

    search_tool = WebSearchTool(backend=args.search_backend)
    print(f"Search backend: {search_tool.active_backend}")
    client = make_local_client(endpoint=args.endpoint, timeout=args.timeout)

    progress = _Progress(len(pending))

    with in_path.open("a") as out_f:
        def _write(rec_dict):
            line = json.dumps(rec_dict) + "\n"
            fcntl.flock(out_f, fcntl.LOCK_EX)
            try:
                out_f.write(line)
                out_f.flush()
            finally:
                fcntl.flock(out_f, fcntl.LOCK_UN)

        for i, existing_rec in enumerate(pending):
            tid       = existing_rec["task_id"]
            q         = existing_rec.get("question", "")[:60]
            cur_runs  = existing_rec.get("k_runs", [])
            n_needed  = args.target_k - len(cur_runs)

            if args.verbose:
                print(f"\n[{i+1}/{len(pending)}] {q}  (adding {n_needed} runs)")

            market = {
                "task_id":            existing_rec["task_id"],
                "market_id":          existing_rec["market_id"],
                "question":           existing_rec.get("question", ""),
                "description":        existing_rec.get("description", ""),
                "yes_price":          existing_rec.get("yes_price_market"),
                "days_to_resolution": existing_rec.get("days_to_resolution"),
                "category":           existing_rec.get("category"),
            }

            new_runs: list[dict] = []
            try:
                for run_id in range(n_needed):
                    if args.verbose:
                        print(f"    [new run {run_id + 1}/{n_needed}]")
                    rec: ForecastRecord = forecast_market_local(
                        market,
                        client=client,
                        model=args.model,
                        search_tool=search_tool,
                        do_third_turn=not args.no_third_turn,
                        max_search_calls=args.max_search_calls,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        verbose=args.verbose,
                    )
                    d = rec.to_dict()
                    # Assign run_ids continuing from existing
                    new_runs.append({
                        "run_id":              len(cur_runs) + run_id,
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

                    all_runs = cur_runs + new_runs
                    _write(_build_record(existing_rec, all_runs, args.target_k, args.model))

                    if run_id < n_needed - 1:
                        time.sleep(0.5)

                all_runs = cur_runs + new_runs
                out_rec  = _build_record(existing_rec, all_runs, args.target_k, args.model)

                if out_rec.get("error"):
                    progress.record("error")
                else:
                    progress.record("ok")

                if not args.verbose:
                    yp       = out_rec.get("yes_prob")
                    runs_str = ", ".join(f"{p:.0%}" for p in (out_rec.get("yes_prob_runs") or []))
                    label    = f"  median={yp:.0%} [{runs_str}]" if yp is not None else "  parse_err"
                    sys.stdout.write(f"  {q[:45]:<45}{label}\n")

            except Exception as exc:
                progress.record("error")
                if args.verbose:
                    print(f"    ERROR: {exc}")
                else:
                    sys.stdout.write(f"  {q[:45]:<45}  ERROR: {exc}\n")

            if i < len(pending) - 1:
                time.sleep(args.delay)

    progress.done_line()


if __name__ == "__main__":
    main()
