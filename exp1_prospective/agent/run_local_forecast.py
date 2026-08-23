#!/usr/bin/env python3
"""Step 2b: Run structured world-model forecast on local models (Qwen / Llama).

Reads data/selected_markets/diverse_YYYY-MM-DD.jsonl, calls a locally-served
Ollama model (default: qwen2.5:7b) via its OpenAI-compatible API, and writes
results to data/initial_forecasts/forecasts_{model-slug}_{YYYY-MM-DD}.jsonl.

Web search is handled by an explicit ReAct tool-calling loop using Tavily
(set TAVILY_API_KEY) or DuckDuckGo (free fallback) — unlike the frontier
agents which delegate search to Azure AI Foundry.

Output schema is identical to run_forecast.py plus a top-level "backend" field.
The script resumes automatically: markets whose task_id already appears in the
output file with k_done >= k are skipped.

Prerequisites:
    ollama pull qwen2.5:7b          # or llama3.1:8b
    pip install tavily-python duckduckgo-search
    export TAVILY_API_KEY=<key>     # optional; DDG used if absent

Usage (from exp1_prospective/):
    python agent/run_local_forecast.py
    python agent/run_local_forecast.py --model llama3.1:8b
    python agent/run_local_forecast.py --model qwen2.5:7b --n 5 --k 3
    python agent/run_local_forecast.py --no-third-turn --search-backend duckduckgo
    python agent/run_local_forecast.py --dry-run
    python agent/run_local_forecast.py --verbose
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

from local_agent import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_ENDPOINT,
    ForecastRecord,
    WebSearchTool,
    forecast_market_local,
    make_local_client,
)
from prompts import SYSTEM_PROMPT, TURN1_TEMPLATE, TURN2_TEMPLATE, TURN3_TEMPLATE, FINAL_TURN_TEMPLATE

_ROOT    = _HERE.parent
_SEL_DIR = _ROOT / "data" / "selected_markets"
_OUT_DIR = _ROOT / "data" / "initial_forecasts"


# ── helpers ────────────────────────────────────────────────────────────────────

def _latest_diverse(sel_dir: Path) -> Path | None:
    for f in sorted(sel_dir.glob("diverse_*.jsonl"), reverse=True):
        return f
    return None


def _model_slug(model: str) -> str:
    """qwen2.5:7b → qwen2.5-7b"""
    return model.replace(":", "-").replace("/", "-")


def _load_done(out_path: Path, k: int) -> set[str]:
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


def _build_record(market: dict, runs: list[dict], k: int, model: str) -> dict:
    probs      = [r["yes_prob"] for r in runs if r["yes_prob"] is not None]
    median_p   = statistics.median(probs) if probs else None
    std_p      = statistics.stdev(probs) if len(probs) > 1 else 0.0
    first      = runs[0] if runs else {}
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
        "yes_prob":            round(median_p, 4) if median_p is not None else None,
        "yes_prob_std":        round(std_p, 4),
        "yes_prob_runs":       probs,
        "backend":             "local-ollama",
        "model":               model,
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
    search_tool: WebSearchTool,
    k: int,
    do_third_turn: bool,
    max_search_calls: int,
    max_tokens: int,
    temperature: float,
    verbose: bool,
    run_delay: float = 1.0,
    write_fn=None,
) -> dict:
    runs: list[dict] = []

    for run_id in range(k):
        if verbose and k > 1:
            print(f"    [run {run_id + 1}/{k}]")
        rec: ForecastRecord = forecast_market_local(
            market,
            client=client,
            model=model,
            search_tool=search_tool,
            do_third_turn=do_third_turn,
            max_search_calls=max_search_calls,
            max_tokens=max_tokens,
            temperature=temperature,
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

        if write_fn is not None:
            write_fn(_build_record(market, runs, k, model))

        if run_id < k - 1:
            time.sleep(run_delay)

    return _build_record(market, runs, k, model)


class _Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.done = self.ok = self.errors = self.skipped = 0
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
            f"{self.ok} forecasted  {self.errors} errors  {self.skipped} skipped"
        )


# ── cli ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run structured world-model forecast on local Ollama models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input",            type=Path,  default=None,
                    help="Markets JSONL.  Defaults to latest diverse_*.jsonl.")
    ap.add_argument("--out",              type=Path,  default=_OUT_DIR,
                    help="Output directory.")
    ap.add_argument("--out-file",         type=Path,  default=None,
                    help="Explicit output .jsonl path (overrides --out + auto-naming).")
    ap.add_argument("--model",            type=str,   default=DEFAULT_MODEL,
                    help="Ollama model tag (e.g. qwen2.5:7b, llama3.1:8b).")
    ap.add_argument("--endpoint",         type=str,   default=DEFAULT_OLLAMA_ENDPOINT,
                    help="Ollama OpenAI-compatible API base URL.")
    ap.add_argument("--n",                type=int,   default=0,
                    help="Stop after N markets (0 = all).  Useful for pilots.")
    ap.add_argument("--k",                type=int,   default=3,
                    help="Independent forecast runs per market.")
    ap.add_argument("--delay",            type=float, default=1.0,
                    help="Seconds between markets.")
    ap.add_argument("--run-delay",        type=float, default=0.5,
                    help="Seconds between runs within a market.")
    ap.add_argument("--max-search-calls", type=int,   default=6,
                    help="Max web_search tool calls per turn-loop pass.")
    ap.add_argument("--max-tokens",       type=int,   default=1500,
                    help="Max tokens per model call.")
    ap.add_argument("--temperature",      type=float, default=0.1)
    ap.add_argument("--timeout",          type=float, default=600.0,
                    help="Per-request timeout in seconds. Use 3600+ for 70b models.")
    ap.add_argument("--search-backend",   type=str,   default="auto",
                    choices=["auto", "tavily", "duckduckgo"],
                    help="Web search backend.")
    ap.add_argument("--no-third-turn",    action="store_true",
                    help="Skip the optional red/blue-team turn (faster).")
    ap.add_argument("--verbose",          action="store_true",
                    help="Print per-turn progress.")
    ap.add_argument("--dry-run",          action="store_true",
                    help="Print prompts for the first market without calling the API.")
    args = ap.parse_args()

    # ── resolve input ──────────────────────────────────────────────────────────
    input_path = args.input or _latest_diverse(_SEL_DIR)
    if input_path is None or not input_path.exists():
        print("No diverse JSONL found.  Run select_markets.py first.")
        sys.exit(1)

    with input_path.open() as f:
        markets = [json.loads(line) for line in f if line.strip()]

    slug     = _model_slug(args.model)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if args.out_file:
        out_path  = args.out_file
        mani_path = args.out_file.with_name(args.out_file.stem + ".manifest.json")
    else:
        out_path  = args.out / f"forecasts_{slug}_{date_str}.jsonl"
        mani_path = args.out / f"forecasts_{slug}_{date_str}.manifest.json"

    print(f"Input:   {input_path}  ({len(markets)} markets)")
    print(f"Model:   {args.model}  (via Ollama at {args.endpoint})")
    print(f"Search:  {args.search_backend}")
    print(f"k={args.k} runs per market  max_search_calls={args.max_search_calls}")

    done_ids = _load_done(out_path, args.k)
    if done_ids:
        print(f"Resuming: {len(done_ids)} already done")

    pending = [m for m in markets if m.get("task_id") not in done_ids]
    if args.n > 0:
        pending = pending[: args.n]

    print(f"Pending: {len(pending)} markets")
    print(f"Output:  {out_path}\n")

    # ── dry run ────────────────────────────────────────────────────────────────
    if args.dry_run:
        m    = pending[0] if pending else markets[0]
        days = m.get("days_to_resolution", 0) or 0
        print("=" * 70)
        print("SYSTEM:\n" + SYSTEM_PROMPT)
        print("\n" + "=" * 70)
        print("TURN 1:\n" + TURN1_TEMPLATE.format(
            question=m.get("question", ""), description=m.get("description", "")[:300],
            yes_price=m.get("yes_price", 0.5), days_to_resolution=days,
            category=m.get("category", "unknown"),
        ))
        print("\n" + "=" * 70)
        print("TURN 2:\n" + TURN2_TEMPLATE.format(days_to_resolution=days))
        print("\n" + "=" * 70)
        print("TURN 3:\n" + TURN3_TEMPLATE)
        print("\n" + "=" * 70)
        print("FINAL:\n" + FINAL_TURN_TEMPLATE)
        return

    # ── initialise search tool and client ─────────────────────────────────────
    search_tool = WebSearchTool(backend=args.search_backend)
    print(f"Search backend active: {search_tool.active_backend}")

    client = make_local_client(endpoint=args.endpoint, timeout=args.timeout)
    args.out.mkdir(parents=True, exist_ok=True)

    def _write_manifest(n_ok: int = 0, n_errors: int = 0, done: bool = False) -> None:
        manifest = {
            "created_at":    datetime.now(timezone.utc).isoformat(),
            "source_file":   str(input_path),
            "model":         args.model,
            "backend":       "local-ollama",
            "search_backend": search_tool.active_backend,
            "k":             args.k,
            "n_markets":     len(pending),
            "n_ok":          n_ok,
            "n_errors":      n_errors,
            "third_turn":    not args.no_third_turn,
            "status":        "complete" if done else "running",
        }
        with mani_path.open("w") as mf:
            json.dump(manifest, mf, indent=2)

    _write_manifest()

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
                    search_tool=search_tool,
                    k=args.k,
                    do_third_turn=not args.no_third_turn,
                    max_search_calls=args.max_search_calls,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    verbose=args.verbose,
                    run_delay=args.run_delay,
                    write_fn=_write,
                )

                if out_rec.get("error"):
                    progress.record("error")
                    n_err += 1
                else:
                    progress.record("ok")
                    n_ok += 1

                if not args.verbose:
                    yp       = out_rec.get("yes_prob")
                    runs_str = ", ".join(f"{p:.0%}" for p in (out_rec.get("yes_prob_runs") or []))
                    label    = f"  median={yp:.0%} [{runs_str}]" if yp is not None else "  parse_err"
                    sys.stdout.write(f"  {q[:45]:<45}{label}\n")

            except Exception as exc:
                err_rec = {
                    "task_id":            market.get("task_id", "?"),
                    "market_id":          market.get("market_id", "?"),
                    "question":           market.get("question", ""),
                    "description":        market.get("description", ""),
                    "yes_price_market":   market.get("yes_price"),
                    "days_to_resolution": market.get("days_to_resolution"),
                    "category":           market.get("category"),
                    "backend":            "local-ollama",
                    "model":              args.model,
                    "k":                  args.k,
                    "k_runs":             [],
                    "yes_prob":           None,
                    "yes_prob_runs":      [],
                    "error":              str(exc),
                    "forecast_at":        datetime.now(timezone.utc).isoformat(),
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
    _write_manifest(n_ok=n_ok, n_errors=n_err, done=True)
    print(f"Manifest → {mani_path}")


if __name__ == "__main__":
    main()
