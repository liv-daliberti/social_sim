#!/usr/bin/env python3
"""Step 4 (local models): Inject counterfactual evidence into local-model forecasts.

For each market × each k-run × each counterfactual packet:
  - Takes the structured forecast from that specific k-run as context
  - Appends the CF evidence text as "NEW EVIDENCE"
  - Makes a single-turn call (no tools) and asks for an updated JSON forecast

This mirrors claude_updated_forecast.py and updated_forecast.py exactly, but
uses a plain Ollama client instead of the Azure AI Foundry agent endpoint.

Usage (from exp1_prospective/):
    python agent/local_updated_forecast.py --model qwen2.5:7b
    python agent/local_updated_forecast.py --model llama3.1:8b
    python agent/local_updated_forecast.py --model qwen2.5:7b --n 1 --k 2
    python agent/local_updated_forecast.py --model qwen2.5:7b --cf-direction pro_H1
    python agent/local_updated_forecast.py --model qwen2.5:7b --dry-run
    python agent/local_updated_forecast.py --model qwen2.5:7b --verbose
"""

from __future__ import annotations

import argparse
import json
import re
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
    _call_with_retry,
    make_local_client,
)
from forecast_agent import _parse_json_forecast

_ROOT   = _HERE.parent
_FC_DIR = _ROOT / "data" / "initial_forecasts"
_CF_DIR = _ROOT / "data" / "counterfactuals"
_UF_DIR = _ROOT / "data" / "updated_forecasts"


# ── injection prompt (identical to updated_forecast.py / claude_updated_forecast.py) ──

_UPDATE_PROMPT = """\
You previously produced this structured forecast for a binary prediction market:

QUESTION: {question}

INITIAL FORECAST:
{initial_sf_json}

---
NEW EVIDENCE (received after your initial research):

{evidence_text}
---

Using ONLY the information above — your prior structured forecast plus this new \
evidence — update your world model and produce a revised forecast.

Rules:
- DO NOT search the web, use any tools, or look up any external information.
  Reason entirely from your existing forecast and the new evidence above.
- Update H1 and H2 posterior_probability to reflect what the new evidence implies.
- Update the supporting_evidence or contradicting_evidence list for the affected \
hypothesis to include a brief note about the new evidence.
- Update yes_prob (must equal H1 posterior_probability).
- Update rationale to explain concisely what changed and why.
- Keep all other fields (key_actors, key_mechanisms, etc.) unchanged.
- Output ONLY a valid JSON object in the exact same structure as the initial \
forecast above.  No markdown fences, no commentary before or after."""


# ── helpers ────────────────────────────────────────────────────────────────────

def _model_slug(model: str) -> str:
    return model.replace(":", "-").replace("/", "-")


def _latest_forecast_file(fc_dir: Path, slug: str) -> Path | None:
    for f in sorted(fc_dir.glob(f"forecasts_{slug}_*.jsonl"), reverse=True):
        return f
    return None


def _latest_cf_file(cf_dir: Path) -> Path | None:
    for f in sorted(cf_dir.glob("counterfactuals_*.jsonl"), reverse=True):
        return f
    return None


def _norm_tid(tid: str) -> str:
    """Strip trailing _YYYY-MM-DD so forecasts and CFs from different run dates match."""
    return re.sub(r"_\d{4}-\d{2}-\d{2}$", "", tid)


def _load_forecasts(path: Path) -> dict[str, dict]:
    """Return {norm_task_id: record} using the last record per task_id."""
    records: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                records[_norm_tid(rec["task_id"])] = rec
            except (json.JSONDecodeError, KeyError):
                pass
    return records


def _load_counterfactuals(path: Path) -> dict[str, list[dict]]:
    """Return {norm_task_id: [cf_packet, ...]}."""
    cfs: dict[str, list[dict]] = {}
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                tid = _norm_tid(rec["task_id"])
                cfs.setdefault(tid, []).append(rec)
            except (json.JSONDecodeError, KeyError):
                pass
    return cfs


def _load_done(out_path: Path) -> set[str]:
    """Return set of (task_id, run_id, cf_id) strings already written."""
    done: set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                key = f"{rec['task_id']}|{rec.get('initial_run_id', rec.get('run_id', 0))}|{rec.get('cf_id', '')}"
                done.add(key)
            except (json.JSONDecodeError, KeyError):
                pass
    return done


# ── single update call ─────────────────────────────────────────────────────────

def _do_update(
    question: str,
    initial_sf: dict,
    evidence_text: str,
    *,
    client,
    model: str,
    max_tokens: int,
    verbose: bool,
) -> tuple[dict | None, str | None, str]:
    prompt = _UPDATE_PROMPT.format(
        question       = question,
        initial_sf_json = json.dumps(initial_sf, indent=2),
        evidence_text  = evidence_text,
    )
    response = _call_with_retry(
        client.chat.completions.create,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    text   = (response.choices[0].message.content or "").strip()
    parsed, err = _parse_json_forecast(text)
    return parsed, err, text


# ── cli ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inject counterfactual evidence into local-model forecasts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model",        type=str,  default=DEFAULT_MODEL,
                    help="Ollama model tag (must match the initial forecast file).")
    ap.add_argument("--endpoint",     type=str,  default=DEFAULT_OLLAMA_ENDPOINT)
    ap.add_argument("--forecasts",    type=Path, default=None,
                    help="Initial forecasts JSONL.  Defaults to latest for --model.")
    ap.add_argument("--counterfactuals", type=Path, default=None,
                    help="Counterfactuals JSONL.  Defaults to latest.")
    ap.add_argument("--out",          type=Path, default=_UF_DIR)
    ap.add_argument("--out-file",     type=Path, default=None,
                    help="Explicit output path; loads done-keys from it to resume. "
                         "Defaults to updated_{slug}_{today}.jsonl in --out.")
    ap.add_argument("--timeout",      type=float, default=600.0,
                    help="Per-request timeout in seconds. Use 3600+ for 70b models.")
    ap.add_argument("--n",            type=int,  default=0,
                    help="Max markets to process (0 = all).")
    ap.add_argument("--k",            type=int,  default=0,
                    help="Max k-runs per market (0 = all available).")
    ap.add_argument("--cf-direction", type=str,  default=None,
                    choices=["pro_H1", "anti_H1"],
                    help="Process only CFs with this direction label.")
    ap.add_argument("--max-tokens",   type=int,  default=1500)
    ap.add_argument("--delay",        type=float, default=0.5,
                    help="Seconds between API calls.")
    ap.add_argument("--dry-run",      action="store_true")
    ap.add_argument("--verbose",      action="store_true")
    args = ap.parse_args()

    slug = _model_slug(args.model)

    # ── resolve files ──────────────────────────────────────────────────────────
    fc_path = args.forecasts or _latest_forecast_file(_FC_DIR, slug)
    if fc_path is None or not fc_path.exists():
        print(f"No forecast file found for model '{slug}'. Run run_local_forecast.py first.")
        sys.exit(1)

    cf_path = args.counterfactuals or _latest_cf_file(_CF_DIR)
    if cf_path is None or not cf_path.exists():
        print("No counterfactuals file found. Run build_counterfactuals.py first.")
        sys.exit(1)

    forecasts = _load_forecasts(fc_path)
    cf_map    = _load_counterfactuals(cf_path)

    date_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if args.out_file:
        out_path = args.out_file
    else:
        # Auto-resume: if an existing updated_{slug}_*.jsonl exists, append to the
        # latest one (so _load_done can skip already-finished markets).
        existing = sorted(args.out.glob(f"updated_{slug}_*.jsonl"), reverse=True)
        out_path = existing[0] if existing else args.out / f"updated_{slug}_{date_str}.jsonl"
    args.out.mkdir(parents=True, exist_ok=True)

    done_keys = _load_done(out_path)
    print(f"Forecasts:       {fc_path}  ({len(forecasts)} markets)")
    print(f"Counterfactuals: {cf_path}")
    print(f"Output:          {out_path}")
    print(f"Already done:    {len(done_keys)} records")

    if args.dry_run:
        tid  = next(iter(forecasts))
        rec  = forecasts[tid]
        cfs  = cf_map.get(tid, [])
        cf   = cfs[0] if cfs else {"cf_id": "example", "evidence_text": "[evidence]", "direction": "pro_H1"}
        sf   = rec.get("structured_forecast") or rec.get("k_runs", [{}])[0].get("structured_forecast") or {}
        print(_UPDATE_PROMPT.format(
            question=rec.get("question", ""),
            initial_sf_json=json.dumps(sf, indent=2)[:800],
            evidence_text=cf.get("evidence_text", ""),
        ))
        return

    client = make_local_client(endpoint=args.endpoint, timeout=args.timeout)

    n_written = n_err = 0
    task_ids  = list(forecasts.keys())
    if args.n > 0:
        task_ids = task_ids[: args.n]

    with out_path.open("a") as out_f:
        for task_id in task_ids:
            fc_rec  = forecasts[task_id]
            cfs     = cf_map.get(task_id, [])
            if not cfs:
                continue

            if args.cf_direction:
                cfs = [c for c in cfs if c.get("direction") == args.cf_direction]

            k_runs  = fc_rec.get("k_runs", [])
            if args.k > 0:
                k_runs = k_runs[: args.k]

            question = fc_rec.get("question", "")

            for run in k_runs:
                run_id = run.get("run_id", 0)
                initial_sf = run.get("structured_forecast")
                if initial_sf is None:
                    continue

                for cf in cfs:
                    cf_id = cf.get("cf_id", "")
                    key   = f"{task_id}|{run_id}|{cf_id}"
                    if key in done_keys:
                        continue

                    if args.verbose:
                        print(f"  {task_id} run={run_id} cf={cf_id} dir={cf.get('direction','?')}")

                    try:
                        parsed, err, raw_text = _do_update(
                            question,
                            initial_sf,
                            cf.get("evidence_text", ""),
                            client=client,
                            model=args.model,
                            max_tokens=args.max_tokens,
                            verbose=args.verbose,
                        )
                        initial_yp = run.get("yes_prob")
                        updated_yp = None
                        if parsed:
                            updated_yp = parsed.get("yes_prob")
                            if updated_yp is None:
                                h1 = next((h for h in parsed.get("hypotheses", []) if h.get("id") == "H1"), {})
                                updated_yp = h1.get("posterior_probability")
                        delta = (
                            round(float(updated_yp) - float(initial_yp), 4)
                            if updated_yp is not None and initial_yp is not None else None
                        )
                        out_rec = {
                            # field names match frontier updated_forecast.py schema so viewer works
                            "update_id":                  f"{task_id}_{cf_id}_run{run_id}",
                            "forecast_model":             args.model,
                            "task_id":                    task_id,
                            "market_id":                  fc_rec.get("market_id"),
                            "question":                   question,
                            "cf_id":                      cf_id,
                            "direction":                  cf.get("direction"),
                            "cf_index":                   cf.get("cf_index"),
                            "initial_run_id":             run_id,
                            "initial_yes_prob":           initial_yp,
                            "updated_yes_prob":           updated_yp,
                            "delta_yes_prob":             delta,
                            "updated_structured_forecast": parsed,
                            "parse_error":                err,
                            # local-specific extras
                            "backend":                    "local-ollama",
                            "generation": {
                                "response":    raw_text,
                                "used_tools":  False,
                            },
                            "generated_at":               datetime.now(timezone.utc).isoformat(),
                        }
                        out_f.write(json.dumps(out_rec) + "\n")
                        out_f.flush()
                        n_written += 1

                    except Exception as exc:
                        n_err += 1
                        if args.verbose:
                            print(f"    ERROR: {exc}")

                    time.sleep(args.delay)

    print(f"\nDone — {n_written} updates written, {n_err} errors")


if __name__ == "__main__":
    main()
