#!/usr/bin/env python3
"""Step 3: Build counterfactual evidence packets — one LLM call per snippet.

Generates 9 packets per market (3 directions × 3 snippets each).  Each call
is one single-turn Azure API call given:
  - the full world model from the canonical run-0 structured forecast
  - the generation date (so every snippet is dateline-stamped to today)
  - an angle instruction that diversifies the 3 snippets within a direction

Directions:
  pro_H1    — 3 snippets that each INCREASE P(H1=YES), via different mechanisms
  anti_H1   — 3 snippets that each DECREASE P(H1=YES), via different mechanisms
  orthogonal — 3 snippets topically related but causally disconnected from H1/H2

Usage (from exp1_prospective/):
    python agent/build_counterfactuals.py
    python agent/build_counterfactuals.py --n 2        # first 2 markets
    python agent/build_counterfactuals.py --dry-run    # print prompts, no API calls
    python agent/build_counterfactuals.py --verbose
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
    _extract_tool_calls,
    _extract_usage,
    DEFAULT_MODEL,
    AGENT_NAME,
    AGENT_VERSION,
)

_ROOT   = _HERE.parent
_FC_DIR = _ROOT / "data" / "initial_forecasts"
_CF_DIR = _ROOT / "data" / "counterfactuals"


# ── snippet plan: (direction, index, angle instruction) ────────────────────────
# 9 snippets per market, 3 per direction, each targeting a different angle so
# the three snippets within a direction are meaningfully distinct.

_SNIPPETS: list[tuple[str, int, str]] = [
    # direction   idx  angle ─────────────────────────────────────────────────────
    ("pro_H1",    0,  "Target the PRIMARY causal mechanism: the single most important "
                      "link that would drive H1 to happen."),
    ("pro_H1",    1,  "Target a DIFFERENT ACTOR or institution from the world model — "
                      "pick one not featured in angle 0, acting through a distinct pathway."),
    ("pro_H1",    2,  "Target a LATENT VARIABLE — something uncertain or not yet measured "
                      "in the evidence that, if revealed favorably, tips the balance toward H1."),

    ("anti_H1",   0,  "Target the PRIMARY obstacle to H1: the single most important "
                      "blocking factor or adverse development."),
    ("anti_H1",   1,  "Target a DIFFERENT ACTOR or institution — pick one not featured in "
                      "angle 0, acting through a distinct pathway that prevents H1."),
    ("anti_H1",   2,  "Target a LATENT VARIABLE — something uncertain that, if revealed "
                      "unfavorably, makes H1 significantly less likely."),

    ("orthogonal", 0, "An institutional, regulatory, or procedural development in the same "
                      "domain that has no causal bearing on whether H1 or H2 occurs."),
    ("orthogonal", 1, "A development involving one of the key actors on an UNRELATED matter "
                      "(e.g., a personnel change, an adjacent policy issue, a domestic concern)."),
    ("orthogonal", 2, "Background context: economic, logistical, or social news that sets the "
                      "scene but does not change the odds of H1 vs H2."),
]


# ── prompt templates ────────────────────────────────────────────────────────────

_CONTEXT_BLOCK = """\
MARKET QUESTION: {question}

RESOLUTION CRITERIA (abridged):
{description}

CURRENT WORLD MODEL:
  H1 (YES): {h1_description}
            Posterior probability: {h1_posterior:.0%}
            Supporting evidence:   {h1_supporting}
            Contradicting evidence:{h1_contradicting}

  H2 (NO):  {h2_description}
            Posterior probability: {h2_posterior:.0%}

Key actors:       {key_actors}
Key mechanisms:   {key_mechanisms}
Latent variables: {latent_variables}

Today's date: {today}"""


CF_PRO_H1_PROMPT = """\
You are a researcher designing a prediction-market consistency experiment.

""" + _CONTEXT_BLOCK + """

ANGLE FOR THIS SNIPPET: {angle}

TASK: Write ONE short SYNTHETIC news snippet dated {today} that, if real, would
INCREASE P(H1=YES).  Address the angle above specifically — do not default to the
most obvious pro-H1 framing if the angle says to target something different.

Rules:
- 2-4 sentences in AP/Reuters wire style; open with a dateline ("CITY, {today_short} —")
- Plausible given the current situation; invent names, figures, quotes as needed
- DO NOT search the web — generate entirely from the context above

Output ONLY a valid JSON object with NO markdown fences or commentary:
{{
  "news_date": "{today}",
  "evidence_headline": "Short headline 10-15 words",
  "evidence_text": "Dateline + 2-4 sentence news snippet.",
  "mechanism_targeted": "Which mechanism or latent variable this addresses",
  "rationale": "1-2 sentences: why this specifically INCREASES P(H1=YES)."
}}"""


CF_ANTI_H1_PROMPT = """\
You are a researcher designing a prediction-market consistency experiment.

""" + _CONTEXT_BLOCK + """

ANGLE FOR THIS SNIPPET: {angle}

TASK: Write ONE short SYNTHETIC news snippet dated {today} that, if real, would
DECREASE P(H1=YES).  Address the angle above specifically.

Rules:
- 2-4 sentences in AP/Reuters wire style; open with a dateline ("CITY, {today_short} —")
- Plausible given the current situation; invent names, figures, quotes as needed
- DO NOT search the web — generate entirely from the context above

Output ONLY a valid JSON object with NO markdown fences or commentary:
{{
  "news_date": "{today}",
  "evidence_headline": "Short headline 10-15 words",
  "evidence_text": "Dateline + 2-4 sentence news snippet.",
  "mechanism_targeted": "Which mechanism or latent variable this addresses",
  "rationale": "1-2 sentences: why this specifically DECREASES P(H1=YES)."
}}"""


CF_ORTHOGONAL_PROMPT = """\
You are a researcher designing a prediction-market consistency experiment.

""" + _CONTEXT_BLOCK + """

ANGLE FOR THIS SNIPPET: {angle}

TASK: Write ONE short SYNTHETIC news snippet dated {today} that is TOPICALLY
RELATED to this market but does NOT shift P(H1=YES) in either direction.
Address the angle above specifically.

Rules:
- 2-4 sentences in AP/Reuters wire style; open with a dateline ("CITY, {today_short} —")
- Same general domain, but a clearly different causal facet from H1/H2
- Plausible; invent names, figures, quotes as needed
- DO NOT search the web — generate entirely from the context above

Output ONLY a valid JSON object with NO markdown fences or commentary:
{{
  "news_date": "{today}",
  "evidence_headline": "Short headline 10-15 words",
  "evidence_text": "Dateline + 2-4 sentence news snippet.",
  "mechanism_targeted": "Which aspect of the situation this addresses (not H1/H2 causal path)",
  "rationale": "1-2 sentences: why this does NOT shift P(H1=YES) in either direction."
}}"""


_DIRECTION_PROMPTS = {
    "pro_H1":    CF_PRO_H1_PROMPT,
    "anti_H1":   CF_ANTI_H1_PROMPT,
    "orthogonal": CF_ORTHOGONAL_PROMPT,
}


# ── helpers ─────────────────────────────────────────────────────────────────────

def _today_strings() -> tuple[str, str]:
    """Return (long, short) date strings: ('June 6, 2026', 'June 6')."""
    now = datetime.now()
    long  = now.strftime("%B %-d, %Y")
    short = now.strftime("%B %-d")
    return long, short


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


def _canonical_sf(rec: dict) -> dict | None:
    for run in rec.get("k_runs", []):
        sf = run.get("structured_forecast")
        if sf and sf.get("hypotheses"):
            return sf
    sf = rec.get("structured_forecast")
    return sf if (sf and sf.get("hypotheses")) else None


def _get_hyp(sf: dict, hid: str) -> dict:
    return next((h for h in sf.get("hypotheses", []) if h["id"] == hid), {})


def _fmt_list(items: list, n: int = 3) -> str:
    if not items:
        return "(none)"
    return "; ".join(str(x) for x in items[:n])


def _build_prompt(direction: str, index: int, angle: str,
                  rec: dict, sf: dict, today: str, today_short: str) -> str:
    em = sf.get("event_model", {})
    h1 = _get_hyp(sf, "H1")
    h2 = _get_hyp(sf, "H2")
    return _DIRECTION_PROMPTS[direction].format(
        question         = rec.get("question", ""),
        description      = (rec.get("description") or "")[:700],
        h1_description   = h1.get("description", "(H1=YES outcome)"),
        h1_posterior     = h1.get("posterior_probability", 0.5),
        h2_description   = h2.get("description", "(H2=NO outcome)"),
        h2_posterior     = h2.get("posterior_probability", 0.5),
        key_actors       = _fmt_list(em.get("key_actors", []), 4),
        key_mechanisms   = _fmt_list(em.get("key_mechanisms", []), 4),
        latent_variables = _fmt_list(em.get("latent_variables", []), 4),
        h1_supporting    = _fmt_list(h1.get("supporting_evidence", []), 2),
        h1_contradicting = _fmt_list(h1.get("contradicting_evidence", []), 2),
        today            = today,
        today_short      = today_short,
        angle            = angle,
    )


def _parse_snippet(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError as e:
            return {"_parse_error": f"JSON parse failed: {e}", "_raw": text}
    return {"_parse_error": "No JSON object found", "_raw": text}


# ── single-turn generation ──────────────────────────────────────────────────────

def generate_snippet(
    direction: str,
    index: int,
    angle: str,
    rec: dict,
    sf: dict,
    *,
    client,
    today: str,
    today_short: str,
    model: str = DEFAULT_MODEL,
    agent_name: str = AGENT_NAME,
    agent_version: str = AGENT_VERSION,
    verbose: bool = False,
) -> dict:
    """One Azure API call → one counterfactual packet."""
    prompt = _build_prompt(direction, index, angle, rec, sf, today, today_short)
    h1 = _get_hyp(sf, "H1")

    if verbose:
        print(f"    [{direction}/{index}] sending …", end="", flush=True)

    extra = {
        "agent_reference": {
            "name":    agent_name,
            "version": agent_version,
            "type":    "agent_reference",
        }
    }
    resp = _call_with_retry(
        client.responses.create,
        model      = model,
        input      = [{"role": "user", "content": prompt}],
        store      = True,
        extra_body = extra,
    )

    text = _extract_text(resp)
    tool_calls, search_queries = _extract_tool_calls(resp)
    in_tok, out_tok = _extract_usage(resp)

    if verbose:
        print(f" done ({out_tok} tok, {len(search_queries)} searches)")

    parsed = _parse_snippet(text)

    return {
        "task_id":                   rec["task_id"],
        "market_id":                 rec["market_id"],
        "question":                  rec.get("question", ""),
        "cf_id":                     f"{rec['task_id']}_{direction}_{index}",
        "direction":                 direction,
        "cf_index":                  index,
        "angle":                     angle,
        "target_hypothesis":         "H1" if direction in ("pro_H1", "anti_H1") else None,
        "expected_hypothesis_shift": (
            "increase" if direction == "pro_H1" else
            "decrease" if direction == "anti_H1" else
            None
        ),
        "h1_posterior_at_generation": h1.get("posterior_probability"),
        "yes_prob_at_generation":     rec.get("yes_prob"),
        "source":                     "SYNTHETIC",
        "news_date":                  parsed.get("news_date", today),
        "generation": {
            "prompt":         prompt,
            "response":       text,
            "search_queries": search_queries,
            "tool_calls":     tool_calls,
            "input_tokens":   in_tok,
            "output_tokens":  out_tok,
            "response_id":    getattr(resp, "id", None),
        },
        "evidence_headline":  parsed.get("evidence_headline", ""),
        "evidence_text":      parsed.get("evidence_text", ""),
        "mechanism_targeted": parsed.get("mechanism_targeted", ""),
        "rationale":          parsed.get("rationale", ""),
        "parse_error":        parsed.get("_parse_error"),
        "generated_at":       datetime.now(timezone.utc).isoformat(),
    }


# ── resume ──────────────────────────────────────────────────────────────────────

def _load_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    with out_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                if cf_id := r.get("cf_id"):
                    done.add(cf_id)
            except json.JSONDecodeError:
                pass
    return done


# ── main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build counterfactual evidence packets (one LLM call per snippet, 9 per market).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input",    type=Path, default=None)
    ap.add_argument("--out",      type=Path, default=_CF_DIR)
    ap.add_argument("--n",        type=int,  default=0,
                    help="Max markets (0 = all).")
    ap.add_argument("--delay",    type=float, default=2.0,
                    help="Seconds between API calls.")
    ap.add_argument("--model",    type=str,  default=DEFAULT_MODEL)
    ap.add_argument("--api-key",  type=str,  default=None)
    ap.add_argument("--endpoint", type=str,  default=None)
    ap.add_argument("--verbose",  action="store_true")
    ap.add_argument("--dry-run",  action="store_true")
    args = ap.parse_args()

    input_path = args.input
    if input_path is None:
        candidates = sorted(_FC_DIR.glob("forecasts_*.jsonl"), reverse=True)
        input_path = candidates[0] if candidates else None
    if not input_path or not input_path.exists():
        print("No forecasts JSONL found.")
        sys.exit(1)

    records = _load_forecasts(input_path)
    valid   = [r for r in records if _canonical_sf(r)]
    if args.n > 0:
        valid = valid[:args.n]

    today, today_short = _today_strings()

    print(f"Input:   {input_path}")
    print(f"Markets: {len(valid)}   Date: {today}")
    print(f"Packets: {len(valid) * len(_SNIPPETS)} total  ({len(valid)} markets × {len(_SNIPPETS)} snippets)")

    if args.dry_run:
        rec = valid[0]
        sf  = _canonical_sf(rec)
        for direction, index, angle in _SNIPPETS[:3]:
            prompt = _build_prompt(direction, index, angle, rec, sf, today, today_short)
            print(f"\n{'='*70}\n[DRY RUN — {direction}/{index}]\n{prompt}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = args.out / f"counterfactuals_{date_str}.jsonl"

    done_ids = _load_done(out_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} packets already written")
    print(f"Output:  {out_path}\n")

    client = make_openai_client(api_key=args.api_key, endpoint=args.endpoint)

    total  = len(valid) * len(_SNIPPETS)
    n_done = n_ok = n_err = 0

    with out_path.open("a") as out_f:
        for rec in valid:
            sf  = _canonical_sf(rec)
            q60 = rec.get("question", "")[:60]
            if args.verbose:
                print(f"\n[{rec['task_id']}] {q60}")
            else:
                print(f"\n  {q60}")

            for direction, index, angle in _SNIPPETS:
                cf_id = f"{rec['task_id']}_{direction}_{index}"

                if cf_id in done_ids:
                    if args.verbose:
                        print(f"    SKIP  {direction}/{index}")
                    n_done += 1
                    continue

                label = f"{direction}/{index}"
                if not args.verbose:
                    print(f"    {label:<14}", end="", flush=True)

                try:
                    packet = generate_snippet(
                        direction, index, angle, rec, sf,
                        client=client, today=today, today_short=today_short,
                        model=args.model, verbose=args.verbose,
                    )
                    out_f.write(json.dumps(packet) + "\n")
                    out_f.flush()
                    n_ok += 1
                    if not args.verbose:
                        hl = packet.get("evidence_headline", "")[:50]
                        print(f" ✓  {hl}")
                except Exception as exc:
                    err_packet = {
                        "task_id":      rec["task_id"],
                        "market_id":    rec["market_id"],
                        "question":     rec.get("question", ""),
                        "cf_id":        cf_id,
                        "direction":    direction,
                        "cf_index":     index,
                        "error":        str(exc),
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    out_f.write(json.dumps(err_packet) + "\n")
                    out_f.flush()
                    n_err += 1
                    print(f" ERROR: {exc}")

                n_done += 1
                if n_done < total:
                    time.sleep(args.delay)

    print(f"\nDone — {n_ok} ok  {n_err} errors  →  {out_path}")


if __name__ == "__main__":
    main()
