#!/usr/bin/env python3
"""Step 4: Re-run agent with counterfactual evidence injected — no web search.

For each market × each k-run × each counterfactual packet:
  - Takes the structured forecast from that specific k-run as context
  - Appends the CF evidence text as "NEW EVIDENCE"
  - Makes a single-turn call WITHOUT agent_reference (no tools, no web search)
  - Asks the model to produce an updated structured forecast JSON

This gives one updated forecast per (k-run, CF packet) pair, enabling a clean
paired comparison:
    delta_yes_prob_i = updated_yes_prob_i - initial_yes_prob_i   (for run i)

Total calls: N_markets × k × 9_cf_packets
For 5 markets with k=5 and 9 packets each: 5 × 5 × 9 = 225 calls.

No-internet guarantee: the call is made WITHOUT extra_body agent_reference,
so the model has no tool access.  The prompt also explicitly forbids searching.

Usage (from exp1_prospective/):
    python agent/updated_forecast.py
    python agent/updated_forecast.py --n 1 --k 2    # 1 market, 2 runs only
    python agent/updated_forecast.py --cf-direction pro_H1  # one direction only
    python agent/updated_forecast.py --dry-run
    python agent/updated_forecast.py --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── path / env setup ───────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_ENV_FILE = _HERE / ".env"
if _ENV_FILE.exists() and not os.environ.get("AZURE_AI_API_KEY"):
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from forecast_agent import (
    make_openai_client,
    _call_with_retry,
    _extract_text,
    _extract_usage,
    DEFAULT_MODEL,
)

_ROOT   = _HERE.parent
_FC_DIR = _ROOT / "data" / "initial_forecasts"
_CF_DIR = _ROOT / "data" / "counterfactuals"
_UF_DIR = _ROOT / "data" / "updated_forecasts"


# ── injection prompt ────────────────────────────────────────────────────────────
# No {agent_reference} → no tools → no web search.
# The prompt also explicitly forbids searching.

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


# ── helpers ─────────────────────────────────────────────────────────────────────

def _load_forecasts(path: Path) -> list[dict]:
    latest: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                tid = rec.get("task_id")
                if tid:
                    latest[tid] = rec
            except json.JSONDecodeError:
                pass
    return list(latest.values())


def _load_cf_packets(cf_dir: Path, task_id: str | None = None) -> list[dict]:
    """Load all indexed CF packets, optionally filtered to one task_id."""
    packets: list[dict] = []
    seen: set[str] = set()
    for path in sorted(cf_dir.glob("counterfactuals_*.jsonl"), reverse=True):
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("cf_index") is None:
                        continue  # skip old single-packet format
                    if task_id and rec.get("task_id") != task_id:
                        continue
                    cf_id = rec.get("cf_id", "")
                    if cf_id and cf_id not in seen:
                        seen.add(cf_id)
                        packets.append(rec)
                except json.JSONDecodeError:
                    pass
    return packets


def _load_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    with out_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                if uid := r.get("update_id"):
                    done.add(uid)
            except json.JSONDecodeError:
                pass
    return done


def _parse_forecast_json(text: str) -> tuple[dict | None, str | None]:
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)), None
        except json.JSONDecodeError as e:
            return None, f"JSON parse failed: {e}"
    return None, "No JSON object found"


# ── single update call ──────────────────────────────────────────────────────────

def run_update(
    rec: dict,
    run_idx: int,
    cf_packet: dict,
    *,
    client,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> dict:
    """Single-turn call — no agent_reference, no tools, no web search."""
    k_runs = rec.get("k_runs", [])
    run = k_runs[run_idx] if run_idx < len(k_runs) else {}
    initial_sf = run.get("structured_forecast") or rec.get("structured_forecast") or {}
    initial_yp = run.get("yes_prob")

    prompt = _UPDATE_PROMPT.format(
        question        = rec.get("question", ""),
        initial_sf_json = json.dumps(initial_sf, indent=2),
        evidence_text   = cf_packet.get("evidence_text", ""),
    )

    update_id = f"{rec['task_id']}_{cf_packet['cf_id']}_run{run_idx}"

    if verbose:
        print(f"    [{cf_packet['direction']}/{cf_packet['cf_index']} run{run_idx}] …", end="", flush=True)

    # NO extra_body agent_reference → plain model call, no tools available
    resp = _call_with_retry(
        client.responses.create,
        model  = model,
        input  = [{"role": "user", "content": prompt}],
        store  = True,
    )

    text = _extract_text(resp)
    in_tok, out_tok = _extract_usage(resp)

    if verbose:
        print(f" done ({out_tok} tok)")

    updated_sf, parse_error = _parse_forecast_json(text)
    updated_yp = None
    if updated_sf:
        updated_yp = updated_sf.get("yes_prob")
        if updated_yp is None:
            hyps = updated_sf.get("hypotheses", [])
            h1 = next((h for h in hyps if h.get("id") == "H1"), {})
            updated_yp = h1.get("posterior_probability")

    delta = (updated_yp - initial_yp) if (updated_yp is not None and initial_yp is not None) else None

    return {
        "update_id":                  update_id,
        "forecast_model":             model,
        "task_id":                    rec["task_id"],
        "market_id":                  rec["market_id"],
        "question":                   rec.get("question", ""),
        "cf_id":                      cf_packet["cf_id"],
        "direction":                  cf_packet["direction"],
        "cf_index":                   cf_packet.get("cf_index"),
        "initial_run_id":             run_idx,
        "initial_yes_prob":           initial_yp,
        "updated_yes_prob":           updated_yp,
        "delta_yes_prob":             round(delta, 4) if delta is not None else None,
        "expected_shift":             cf_packet.get("expected_hypothesis_shift"),
        "shift_correct":              (
            (delta > 0 if cf_packet.get("expected_hypothesis_shift") == "increase" else
             delta < 0 if cf_packet.get("expected_hypothesis_shift") == "decrease" else
             None)
            if delta is not None else None
        ),
        "updated_structured_forecast": updated_sf,
        "parse_error":                parse_error,
        "generation": {
            "prompt":        prompt,
            "response":      text,
            "input_tokens":  in_tok,
            "output_tokens": out_tok,
            "response_id":   getattr(resp, "id", None),
            "used_tools":    False,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ── main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Step 4: inject CF evidence into each k-run, no web search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input",        type=Path, default=None,
                    help="Initial forecasts JSONL. Defaults to latest forecasts_*.jsonl.")
    ap.add_argument("--cf-input",     type=Path, default=None,
                    help="Counterfactuals JSONL dir. Defaults to data/counterfactuals/.")
    ap.add_argument("--out",          type=Path, default=_UF_DIR)
    ap.add_argument("--n",            type=int,  default=0,
                    help="Max markets (0 = all).")
    ap.add_argument("--k",            type=int,  default=0,
                    help="Max k-runs per market (0 = all runs present).")
    ap.add_argument("--cf-direction", type=str,  default=None,
                    help="Restrict to one direction: pro_H1 | anti_H1 | orthogonal.")
    ap.add_argument("--delay",        type=float, default=1.5,
                    help="Seconds between API calls.")
    ap.add_argument("--model",        type=str,  default=DEFAULT_MODEL)
    ap.add_argument("--api-key",      type=str,  default=None)
    ap.add_argument("--endpoint",     type=str,  default=None)
    ap.add_argument("--verbose",      action="store_true")
    ap.add_argument("--dry-run",      action="store_true",
                    help="Print first prompt; no API calls.")
    args = ap.parse_args()

    # ── resolve inputs ──────────────────────────────────────────────────────────
    fc_path = args.input
    if fc_path is None:
        candidates = sorted(_FC_DIR.glob("forecasts_*.jsonl"), reverse=True)
        fc_path = candidates[0] if candidates else None
    if not fc_path or not fc_path.exists():
        print("No forecasts JSONL found.")
        sys.exit(1)

    cf_dir = args.cf_input or _CF_DIR

    records = _load_forecasts(fc_path)
    if args.n > 0:
        records = records[:args.n]

    # ── build work list ─────────────────────────────────────────────────────────
    # work items: (rec, run_idx, cf_packet)
    work: list[tuple[dict, int, dict]] = []
    for rec in records:
        k_runs = rec.get("k_runs", [])
        n_runs = len(k_runs) if k_runs else 1
        if args.k > 0:
            n_runs = min(n_runs, args.k)

        cf_packets = _load_cf_packets(cf_dir, task_id=rec["task_id"])
        if args.cf_direction:
            cf_packets = [p for p in cf_packets if p["direction"] == args.cf_direction]

        for run_idx in range(n_runs):
            for cf_packet in cf_packets:
                work.append((rec, run_idx, cf_packet))

    total = len(work)
    print(f"Input:   {fc_path}")
    print(f"Markets: {len(records)}   Total update calls: {total}")

    if args.dry_run:
        rec, run_idx, cf_packet = work[0]
        k_runs = rec.get("k_runs", [])
        run    = k_runs[run_idx] if run_idx < len(k_runs) else {}
        sf     = run.get("structured_forecast") or rec.get("structured_forecast") or {}
        prompt = _UPDATE_PROMPT.format(
            question        = rec.get("question", ""),
            initial_sf_json = json.dumps(sf, indent=2)[:1500] + "\n  [... truncated for dry-run ...]",
            evidence_text   = cf_packet.get("evidence_text", ""),
        )
        print(f"\n{'='*70}\n[DRY RUN — {cf_packet['cf_id']} run{run_idx}]\n{prompt}\n")
        print("\n(NOTE: this call would be made WITHOUT agent_reference — no tools, no web search)")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = args.out / f"updated_{date_str}.jsonl"

    done_ids = _load_done(out_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} already done")
    print(f"Output:  {out_path}\n")

    client = make_openai_client(api_key=args.api_key, endpoint=args.endpoint)

    n_done = n_ok = n_err = 0

    with out_path.open("a") as out_f:
        for rec, run_idx, cf_packet in work:
            update_id = f"{rec['task_id']}_{cf_packet['cf_id']}_run{run_idx}"

            if update_id in done_ids:
                n_done += 1
                continue

            q40 = rec.get("question", "")[:40]
            if not args.verbose:
                label = f"{cf_packet['direction']}/{cf_packet['cf_index']} run{run_idx}"
                print(f"  {q40:<40} {label:<22}", end="", flush=True)

            try:
                result = run_update(
                    rec, run_idx, cf_packet,
                    client=client, model=args.model, verbose=args.verbose,
                )
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                n_ok += 1
                if not args.verbose:
                    delta = result.get("delta_yes_prob")
                    correct = result.get("shift_correct")
                    d_str = f"{delta:+.2f}" if delta is not None else "n/a"
                    ok_str = "✓" if correct else ("✗" if correct is False else "—")
                    print(f" {d_str}  {ok_str}")
            except Exception as exc:
                err = {
                    "update_id":  update_id,
                    "task_id":    rec["task_id"],
                    "cf_id":      cf_packet["cf_id"],
                    "run_idx":    run_idx,
                    "error":      str(exc),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
                out_f.write(json.dumps(err) + "\n")
                out_f.flush()
                n_err += 1
                if not args.verbose:
                    print(f" ERROR: {exc}")
                else:
                    print(f"    ERROR {update_id}: {exc}")

            n_done += 1
            if n_done < total:
                time.sleep(args.delay)

    print(f"\nDone — {n_ok} ok  {n_err} errors  →  {out_path}")


if __name__ == "__main__":
    main()
