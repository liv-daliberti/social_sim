#!/usr/bin/env python3
"""Diagnostic: is qwen2.5:72b truncating at the token cap on high-k curve prompts?

Runs the EXACT curve prompts at k=8,9,10 for a handful of cities and reports, per
call: finish_reason (length=truncated), whether a clean JSON object with
predicted_poll is present, and what parse() returns (to see fallback-to-stray-number).
"""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent; _ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_HERE))
from engine.news_response import generate_episode
from run_batch import make_client, _strip_reasoning, _AZURE_ENDPOINT
from run_recovery_probe import chat_call, parse
from local_agent import DEFAULT_OLLAMA_ENDPOINT
import run_recovery_curve as rc


def has_clean_json(raw):
    m = re.search(r"\{[^{}]*\"predicted_poll\"[^{}]*\}", raw, re.DOTALL)
    if not m: return False
    try:
        return json.loads(m.group()).get("predicted_poll") is not None
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:72b")
    ap.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--ks", default="8,9,10")
    ap.add_argument("--backend", default="ollama")       # accepted/ignored (slurm passes these)
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--azure-auth", action="store_true")
    args = ap.parse_args()
    client = make_client("ollama", endpoint=args.endpoint, api_key="", azure_auth=False)
    ks = [int(x) for x in args.ks.split(",")]

    n_trunc = n_total = n_nojson = n_fallback = 0
    examples = []
    for i in range(args.n):
        ep = generate_episode(seed=i, T=10)
        for k in ks:
            for s in (-10, 10):
                prompt = rc.build_prompt(ep["news"][:k], ep["polls"][:k], s)
                resp = chat_call(client, args.model, [{"role": "user", "content": prompt}], args.max_tokens, 0.1)
                fr = resp.choices[0].finish_reason
                raw = _strip_reasoning((resp.choices[0].message.content or "").strip())
                clean = has_clean_json(raw)
                pred, _ = parse(raw)
                n_total += 1
                if fr == "length": n_trunc += 1
                if not clean: n_nojson += 1
                if not clean and pred is not None: n_fallback += 1   # parse invented a number
                if fr == "length" and len(examples) < 3:
                    examples.append((k, s, fr, clean, pred, raw[-220:]))
    print(f"\nmodel={args.model}  calls={n_total}  ks={ks}  max_tokens={args.max_tokens}")
    print(f"  truncated (finish_reason=length): {n_trunc}/{n_total} ({100*n_trunc/n_total:.0f}%)")
    print(f"  no clean JSON in output:          {n_nojson}/{n_total} ({100*n_nojson/n_total:.0f}%)")
    print(f"  parse() invented a stray number:  {n_fallback}/{n_total} ({100*n_fallback/n_total:.0f}%)")
    for k, s, fr, clean, pred, tail in examples:
        print(f"\n  --- TRUNCATED k={k} shock={s:+d} clean_json={clean} parse->{pred} ---\n  ...{tail}")


if __name__ == "__main__":
    main()
