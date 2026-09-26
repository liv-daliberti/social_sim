"""run_dag_eval.py — evaluate an open-weight (Ollama) or API model on the LG DAG family.

Runs the four-shock probe (eval.py) per (structure, city, prefix k), saves the raw predictions to a
resumable JSONL, and prints the model's forecast-MAE / recovery against the analytic reference
bracket (oracle / prior-blind / naive-freq / persistence). Reuses Experiment 2's OpenAI-compatible
client (`make_client`/`chat_call`), so `--backend ollama --endpoint http://localhost:PORT/v1` calls a
local Ollama server exactly as the exp2 open-weight runs did.

Typical (under slurm_dag_eval.sh, which serves Ollama):
  python3 run_dag_eval.py --backend ollama --model qwen2.5:7b --endpoint $ENDPOINT \
          --structures test --n 25 --ks 1,3,5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Foundry OpenAI-compatible resource (gpt-5.4 / DeepSeek-V4-Pro live here; Claude on /anthropic)
_AZURE_ENDPOINT = "https://liv-forecast.services.ai.azure.com/openai/v1"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lg_dag import generate_episode, true_next_output, true_recovery_gain
from catalog import CATALOG, TRAIN, TEST, BY_NAME
from prompt import build_prompt, parse_forecast
from eval import _shock_inputs, eval_forecaster, fc_oracle, fc_prior_blind, fc_naive_freq, fc_persist, SHOCKS

# reuse the exp2 OpenAI-compatible clients + call wrapper
sys.path.insert(0, "/n/fs/similarity/social_sim/exp2_simulated_worlds/biased_news/eval")
from run_batch import make_client, _strip_reasoning          # noqa: E402
from run_recovery_probe import chat_call                      # noqa: E402

SEED0 = 20000                                                 # same cities as the reference baseline
MAX_TRIES = 3


def pick_structures(spec):
    if spec == "test":  return TEST
    if spec == "train": return TRAIN
    if spec == "all":   return CATALOG
    return [BY_NAME[n] for n in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama", choices=["ollama", "azure"])
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--endpoint", default=None)          # auto-resolved per frontier model when azure
    ap.add_argument("--api-key", default="")
    ap.add_argument("--azure-auth", action="store_true") # Foundry header (api-key) auth
    ap.add_argument("--structures", default="test")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--ks", default="1,3,5")
    ap.add_argument("--reason-cap", type=int, default=350)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--delay", type=float, default=0.02)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "dag_eval"))
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    structs = pick_structures(args.structures)

    # ---- endpoint / key / client resolution (mirrors run_recovery_curve_parity.py) ----
    if args.backend == "azure":
        ml = args.model.lower()
        if not args.api_key:
            args.api_key = os.environ.get("AZURE_AI_API_KEY", "")
        if args.endpoint is None:
            args.endpoint = ("https://liv-forecast.services.ai.azure.com/anthropic"
                             if "claude" in ml else _AZURE_ENDPOINT)
        if "claude" in ml:                               # Anthropic messages API (chat_call handles it)
            from anthropic import AnthropicFoundry
            client = AnthropicFoundry(azure_ad_token_provider=lambda: args.api_key,
                                      base_url=args.endpoint)
        else:                                            # gpt-5.4 / DeepSeek: OpenAI-shaped Foundry client
            client = make_client(args.backend, endpoint=args.endpoint, api_key=args.api_key,
                                 azure_auth=args.azure_auth, timeout=args.timeout)
    else:
        if args.endpoint is None:
            args.endpoint = "http://localhost:11434/v1"
        client = make_client(args.backend, endpoint=args.endpoint, api_key=args.api_key,
                             timeout=args.timeout)

    def call_model(prompt):
        resp = chat_call(client, args.model, [{"role": "user", "content": prompt}],
                         args.max_tokens, args.temperature)
        return _strip_reasoning((resp.choices[0].message.content or "").strip())

    slug = args.model.replace(":", "-").replace("/", "-")
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"dageval_{slug}_{date}.jsonl"
    done = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            if ln.strip():
                try:
                    d = json.loads(ln); done.add((d["structure"], d["seed"], d["k"]))
                except Exception:
                    pass
    total = len(structs) * args.n * len(ks)
    print(f"Model {args.model} [{args.backend}]  structures={[s.name for s in structs]}  "
          f"n={args.n} ks={ks}  ({len(SHOCKS)} calls/row, {total} rows, {total-len(done)} pending)  {out_path}",
          flush=True)

    t0 = time.time(); n_done = 0
    with out_path.open("a") as fh:
        for st in structs:
            for i in range(args.n):
                seed = SEED0 + i
                ep = generate_episode(st, seed=seed, T=10)
                U, Y, S, g = ep["U"], ep["Y"], ep["S"], ep["g"]
                for k in ks:
                    if (st.name, seed, k) in done:
                        continue
                    next_us = _shock_inputs(st)
                    mu = [float(true_next_output(st, S, k, g, u)[st.probe_output]) for u in next_us]
                    preds = []
                    for u in next_us:
                        p = None
                        for _ in range(MAX_TRIES):
                            try:
                                p = parse_forecast(call_model(build_prompt(st, U[:k], Y[:k], u, args.reason_cap)))
                            except Exception:
                                p = None
                            if p is not None:
                                break
                            if args.delay:
                                time.sleep(args.delay)
                        preds.append(p)
                    rec = {"structure": st.name, "seed": seed, "g": g, "k": k, "shocks": list(SHOCKS),
                           "preds": preds, "mu": mu,
                           "true_gain": (true_recovery_gain(st, g) if st.recover else None),
                           "recover": st.recover}
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    n_done += 1
                    if n_done % 20 == 0:
                        rate = n_done / max(time.time() - t0, 1e-9)
                        print(f"  {n_done} rows  ({rate:.1f}/s)", flush=True)

    print("\n=== done; scoring model vs references ===", flush=True)
    score_and_report(out_path, structs, ks)


def score_and_report(out_path, structs, ks):
    rows = [json.loads(l) for l in Path(out_path).read_text().splitlines() if l.strip()]
    # model aggregates from saved raw
    def model_agg(st, k):
        rs = [r for r in rows if r["structure"] == st.name and r["k"] == k]
        fes, res = [], []
        for r in rs:
            pr = r["preds"]; mu = r["mu"]
            errs = [abs(p - m) for p, m in zip(pr, mu) if p is not None]
            if errs:
                fes.append(float(np.mean(errs)))
            if r["recover"] and r["true_gain"] is not None and all(p is not None for p in pr):
                xs = np.asarray(r["shocks"], float); ys = np.asarray(pr, float)
                gh = float(np.cov(xs, ys, bias=True)[0, 1] / np.var(xs))
                res.append(abs(gh - r["true_gain"]))
        miss = 1 - len([1 for r in rs if all(p is not None for p in r["preds"])]) / max(len(rs), 1)
        return (round(np.mean(fes), 2) if fes else None,
                round(np.mean(res), 2) if res else None, round(miss, 2))

    # analytic references on the same structures/ks (fast)
    refs = {ref: {st.name: eval_forecaster(st, fn, N=len(set(r["seed"] for r in rows)) or 25, ks=ks)
                  for st in structs}
            for ref, fn in (("oracle", fc_oracle), ("naive_freq", fc_naive_freq), ("persist", fc_persist))}
    k = ks[-1] if len(ks) > 1 else ks[0]
    print(f"\nforecast MAE / recovery MAE  (k={k}); model vs references")
    print(f"{'structure':16s} {'MODEL':>14s} {'oracle':>12s} {'naive':>12s} {'persist':>7s}  miss")
    for st in structs:
        mf, mr, miss = model_agg(st, k)
        o = refs["oracle"][st.name]; nf = refs["naive_freq"][st.name]; pe = refs["persist"][st.name]
        def cell(fmae, rmae):
            return (f"{fmae:.2f}" if fmae is not None else " -- ") + ("/" + (f"{rmae:.2f}" if rmae is not None else "--"))
        print(f"{st.name:16s} {cell(mf,mr):>14s} {cell(o['fmae'][k],o['rmae'][k]):>12s} "
              f"{cell(nf['fmae'][k],nf['rmae'][k]):>12s} {pe['fmae'][k]:>7.2f}  {miss}")


if __name__ == "__main__":
    main()
