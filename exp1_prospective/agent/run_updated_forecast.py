#!/usr/bin/env python3
"""Stage 3: Inject counterfactual evidence into initial forecasts and re-elicit.

Supports GPT, Claude, and DeepSeek via --model.  Each model has a thin wrapper
script (gpt_updated_forecast.py, claude_updated_forecast.py, deepseek_updated_forecast.py)
that injects model-specific defaults and delegates here via subprocess.

Usage (direct, from exp1_prospective/):
    python agent/run_updated_forecast.py --model gpt-5.4 --api-key <key>
    python agent/run_updated_forecast.py --model claude-opus-4-8 --api-key <key>
    python agent/run_updated_forecast.py --model DeepSeek-V4-Pro --api-key <key>

Via thin wrappers (recommended):
    AZURE_AI_API_KEY=<key>          python agent/gpt_updated_forecast.py
    CLAUDE_AZURE_API_KEY=<key>      python agent/claude_updated_forecast.py
    DEEPSEEK_AZURE_API_KEY=<key>    python agent/deepseek_updated_forecast.py

Claude uses the Anthropic SDK (messages.create); GPT and DeepSeek use the
OpenAI-compatible Azure AI Foundry Responses API.  The script infers the SDK
from the model name: any model containing "claude" uses Anthropic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from forecast_agent import (
    make_openai_client,
    _call_with_retry,
    _extract_text,
    _extract_usage,
    _parse_json_forecast as _parse_forecast_json,
)

_ROOT   = _HERE.parent
_FC_DIR = _ROOT / "data" / "initial_forecasts"
_CF_DIR = _ROOT / "data" / "counterfactuals"
_UF_DIR = _ROOT / "data" / "updated_forecasts"

# ── per-model defaults ──────────────────────────────────────────────────────────
# Thin wrappers override these via CLI flags; direct callers can rely on them.

MODEL_DEFAULTS: dict[str, dict] = {
    "gpt-5.4": {
        "endpoint":     "https://liv-forecast.services.ai.azure.com",
        "agent_name":   "forecasting-agent",
        "api_key_envs": ["AZURE_AI_API_KEY", "CLAUDE_AZURE_API_KEY"],
        "novelty":      True,   # adds novelty_assessment rule to prompt
    },
    "claude-opus-4-8": {
        "endpoint":     "https://liv-forecast.services.ai.azure.com/anthropic",
        "agent_name":   "",
        "api_key_envs": ["CLAUDE_AZURE_API_KEY"],
        "novelty":      False,
    },
    "DeepSeek-V4-Pro": {
        "endpoint":     "https://cos-tiktok-annotation-a-resource.services.ai.azure.com",
        "agent_name":   "",
        "api_key_envs": ["DEEPSEEK_AZURE_API_KEY", "CLAUDE_AZURE_API_KEY"],
        "novelty":      False,
    },
}

# ── prompts ─────────────────────────────────────────────────────────────────────

_UPDATE_PROMPT_BASE = """\
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

_NOVELTY_RULE = """\
- Add one new field "novelty_assessment": a single sentence stating whether this \
evidence was likely already reflected in your initial forecast as background knowledge \
(suggesting a small update) or is genuinely new and surprising (warranting a larger revision)."""

_NOVELTY_OUTPUT_NOTE = ", plus the \"novelty_assessment\" field"


def _build_prompt(novelty: bool) -> str:
    if not novelty:
        return _UPDATE_PROMPT_BASE
    lines = _UPDATE_PROMPT_BASE.rsplit("- Output ONLY", 1)
    rule_block = _NOVELTY_RULE + "\n"
    output_line = "- Output ONLY" + lines[1].replace(
        "forecast above.", f"forecast above{_NOVELTY_OUTPUT_NOTE}.", 1
    )
    return lines[0] + rule_block + output_line


# ── helpers ─────────────────────────────────────────────────────────────────────

def _is_anthropic(model: str) -> bool:
    return "claude" in model.lower()


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


def _find_forecasts(model: str) -> Path | None:
    slug = model.replace("/", "-").replace(":", "-").replace(".", "-")
    candidates = sorted(_FC_DIR.glob(f"forecasts_{slug}_*.jsonl"), reverse=True)
    if candidates:
        return candidates[0]
    # Fallback: scan manifests for a model match
    for path in sorted(_FC_DIR.glob("forecasts_*.jsonl"), reverse=True):
        mani = path.with_suffix("").with_suffix(".manifest.json")
        if mani.exists():
            try:
                m = json.loads(mani.read_text())
                if model in m.get("model", ""):
                    return path
            except (json.JSONDecodeError, OSError):
                pass
    return None


def _load_cf_packets(cf_dir: Path, task_id: str | None = None) -> list[dict]:
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
                        continue
                    if task_id and rec.get("task_id") != task_id:
                        continue
                    cf_id = rec.get("cf_id", "")
                    if cf_id and cf_id not in seen:
                        seen.add(cf_id)
                        packets.append(rec)
                except json.JSONDecodeError:
                    pass
    return packets


def _load_done(out_dir: Path, model: str) -> set[str]:
    """Return update_ids already written for this model across all updated_*.jsonl files."""
    done: set[str] = set()
    for path in sorted(out_dir.glob("updated_*.jsonl"), reverse=True):
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("forecast_model") == model and not r.get("error"):
                        uid = r.get("update_id")
                        if uid:
                            done.add(uid)
                except json.JSONDecodeError:
                    pass
    return done


# ── single update call ──────────────────────────────────────────────────────────

def run_update(
    rec: dict,
    run_idx: int,
    cf_packet: dict,
    *,
    client,
    model: str,
    novelty: bool = False,
    verbose: bool = False,
) -> dict:
    k_runs = rec.get("k_runs", [])
    run = k_runs[run_idx] if run_idx < len(k_runs) else {}
    initial_sf = run.get("structured_forecast") or rec.get("structured_forecast") or {}
    initial_yp = run.get("yes_prob")

    prompt_template = _build_prompt(novelty)
    prompt = prompt_template.format(
        question        = rec.get("question", ""),
        initial_sf_json = json.dumps(initial_sf, indent=2),
        evidence_text   = cf_packet.get("evidence_text", ""),
    )
    update_id = f"{rec['task_id']}_{cf_packet['cf_id']}_run{run_idx}"

    if verbose:
        print(f"    [{cf_packet['direction']}/{cf_packet['cf_index']} run{run_idx}] …", end="", flush=True)

    if _is_anthropic(model):
        resp = _call_with_retry(
            client.messages.create,
            model      = model,
            messages   = [{"role": "user", "content": prompt}],
            max_tokens = 2000,
        )
        text    = resp.content[0].text
        in_tok  = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
    else:
        resp = _call_with_retry(
            client.responses.create,
            model = model,
            input = [{"role": "user", "content": prompt}],
            store = True,
        )
        text            = _extract_text(resp)
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
    exp_shift = cf_packet.get("expected_hypothesis_shift")
    if delta is not None and exp_shift:
        shift_correct = (delta > 0 if exp_shift == "increase" else
                         delta < 0 if exp_shift == "decrease" else None)
    else:
        shift_correct = None

    result = {
        "update_id":                   update_id,
        "forecast_model":              model,
        "task_id":                     rec["task_id"],
        "market_id":                   rec["market_id"],
        "question":                    rec.get("question", ""),
        "cf_id":                       cf_packet["cf_id"],
        "direction":                   cf_packet["direction"],
        "cf_index":                    cf_packet.get("cf_index"),
        "initial_run_id":              run_idx,
        "initial_yes_prob":            initial_yp,
        "updated_yes_prob":            updated_yp,
        "delta_yes_prob":              round(delta, 4) if delta is not None else None,
        "expected_shift":              exp_shift,
        "shift_correct":               shift_correct,
        "updated_structured_forecast": updated_sf,
        "parse_error":                 parse_error,
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
    if novelty and updated_sf:
        result["novelty_assessment"] = updated_sf.get("novelty_assessment")
    return result


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Stage 3: inject counterfactual evidence into initial forecasts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model",            type=str,  required=True,
                    help="Forecast model name (e.g. gpt-5.4, claude-opus-4-8, DeepSeek-V4-Pro).")
    ap.add_argument("--input",            type=Path, default=None,
                    help="Initial-forecasts JSONL. Auto-detected from data/initial_forecasts/ if not set.")
    ap.add_argument("--cf-input",         type=Path, default=None,
                    help="Counterfactuals dir. Defaults to data/counterfactuals/.")
    ap.add_argument("--out",              type=Path, default=_UF_DIR)
    ap.add_argument("--n",                type=int,  default=0,
                    help="Process only the first N markets (0 = all).")
    ap.add_argument("--k",                type=int,  default=0,
                    help="Use only the first K runs per market (0 = all).")
    ap.add_argument("--cf-direction",     type=str,  default=None,
                    help="Filter to one CF direction (e.g. pro_H1).")
    ap.add_argument("--delay",            type=float, default=1.5,
                    help="Seconds between API calls.")
    ap.add_argument("--api-key",          type=str,  default=None)
    ap.add_argument("--endpoint",         type=str,  default=None,
                    help="API endpoint. Defaults to model's known endpoint.")
    ap.add_argument("--agent-name",       type=str,  default=None,
                    help="Azure agent name (OpenAI path only). Defaults to model's known agent.")
    ap.add_argument("--novelty-assessment", action="store_true",
                    help="Add novelty_assessment rule to the update prompt.")
    ap.add_argument("--verbose",          action="store_true")
    ap.add_argument("--dry-run",          action="store_true")
    args = ap.parse_args()

    cfg = MODEL_DEFAULTS.get(args.model, {})

    # Resolve API key
    key = args.api_key
    if not key:
        for env in cfg.get("api_key_envs", []):
            key = os.environ.get(env, "")
            if key:
                break
    if not key:
        envs = ", ".join(cfg.get("api_key_envs", ["AZURE_AI_API_KEY"]))
        print(f"Set {envs} or pass --api-key")
        sys.exit(1)

    endpoint   = args.endpoint   or cfg.get("endpoint", "")
    agent_name = args.agent_name if args.agent_name is not None else cfg.get("agent_name", "")
    novelty    = args.novelty_assessment or cfg.get("novelty", False)

    fc_path = args.input or _find_forecasts(args.model)
    if not fc_path or not fc_path.exists():
        print(f"No forecasts JSONL found for {args.model}. Run the Stage 1 script first.")
        sys.exit(1)

    cf_dir  = args.cf_input or _CF_DIR
    records = _load_forecasts(fc_path)
    if args.n > 0:
        records = records[:args.n]

    work: list[tuple] = []
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
    print(f"Model:   {args.model}")
    print(f"Input:   {fc_path}")
    print(f"Markets: {len(records)}   Total update calls: {total}")

    if args.dry_run:
        rec, run_idx, cf_packet = work[0]
        k_runs = rec.get("k_runs", [])
        run    = k_runs[run_idx] if run_idx < len(k_runs) else {}
        sf     = run.get("structured_forecast") or rec.get("structured_forecast") or {}
        prompt = _build_prompt(novelty).format(
            question        = rec.get("question", ""),
            initial_sf_json = json.dumps(sf, indent=2)[:1500] + "\n  [...truncated...]",
            evidence_text   = cf_packet.get("evidence_text", ""),
        )
        print(f"\n{'='*70}\n[DRY RUN — {cf_packet['cf_id']} run{run_idx}]\n{prompt}\n")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug     = args.model.replace("/", "-").replace(":", "-").replace(".", "-")
    out_path = args.out / f"updated_{slug}_{date_str}.jsonl"

    done_ids = _load_done(args.out, args.model)
    if done_ids:
        print(f"Resuming: {len(done_ids)} already done")
    print(f"Output:  {out_path}\n")

    if _is_anthropic(args.model):
        from anthropic import AnthropicFoundry
        client = AnthropicFoundry(
            azure_ad_token_provider=lambda: key,
            base_url=endpoint,
        )
    else:
        client = make_openai_client(api_key=key, endpoint=endpoint, agent_name=agent_name)

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
                    client=client, model=args.model,
                    novelty=novelty, verbose=args.verbose,
                )
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                n_ok += 1
                if not args.verbose:
                    delta   = result.get("delta_yes_prob")
                    correct = result.get("shift_correct")
                    d_str   = f"{delta:+.2f}" if delta is not None else "n/a"
                    ok_str  = "✓" if correct else ("✗" if correct is False else "—")
                    print(f" {d_str}  {ok_str}")
            except Exception as exc:
                err = {
                    "update_id":      update_id,
                    "forecast_model": args.model,
                    "task_id":        rec["task_id"],
                    "cf_id":          cf_packet["cf_id"],
                    "run_idx":        run_idx,
                    "error":          str(exc),
                    "generated_at":   datetime.now(timezone.utc).isoformat(),
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
