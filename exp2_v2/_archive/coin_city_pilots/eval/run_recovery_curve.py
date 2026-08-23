#!/usr/bin/env python3
"""Exp 2 — recovery LEARNING CURVE.

For each city we fix a K-week history and probe the forecaster after each prefix
length k = 0,1,...,K, reading its recovered gain ĝ_k from two test shocks (±10):

    ĝ_k = ( pred(+10) - pred(-10) ) / 20

(a 2-point slope, so the level anchor cancels). Plotting ρ(ĝ_k, g) vs k shows how
fast each forecaster infers the latent g as evidence accumulates; the Bayes oracle's
posterior-mean ĝ_k is the ceiling. At k=0 there is no evidence, so recovery ≈ 0.

Output: data/recovery_curve/results_{slug}_{date}.jsonl
"""
from __future__ import annotations

import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_HERE))
from engine.news_response import generate_episode, compute_bayes
from run_batch import make_client, _strip_reasoning, _slug, _Progress, _write_line, _AZURE_ENDPOINT
from run_recovery_probe import chat_call, parse, _slope
from local_agent import DEFAULT_OLLAMA_ENDPOINT

_OUTDIR = _ROOT / "data" / "recovery_curve"
SHOCKS = [-10, -5, 5, 10]      # 4-point regression slope (cleaner than 2-point)
KMAX = 10
MAX_TRIES = 3                  # retry a call that yields no parseable JSON before giving up


def build_prompt(news_k, polls_k, shock, *, no_think=False):
    head = [
        "You are tracking a local election campaign with a weekly tracking poll.",
        "Each week reports the net news (a signed number: positive = good news for the "
        "candidate, negative = bad) and the resulting poll (support, 0-100).",
        "Different cities' polls respond to news by different amounts: some local media "
        "environments let news move opinion strongly, others mute it so the polls barely budge.",
        "",
    ]
    k = len(news_k)
    if k == 0:
        body = ["No weeks have been observed yet for this city."]
    else:
        body = ["| Week | Net news | Poll |", "|------|----------|------|"]
        body += [f"| {i:>4} | {n:>+5d} | {p:>4} |" for i, (n, p) in enumerate(zip(news_k, polls_k), 1)]
        body += ["", "From the history above, first estimate how many poll points THIS city's "
                 "poll tends to move for each +1 point of net news."]
    tail = [
        f"Given that next week's net news will be {shock:+d}, predict next week's poll.",
        "You may reason step by step, but be brief: your reply is capped at about 500 tokens, "
        "so keep any reasoning to a few short sentences and be sure to finish within the limit.",
        'END your reply with the answer as a JSON object on its own line: '
        '{"rationale": "one short sentence", "predicted_poll": <integer 0-100>}',
    ]
    return "\n".join(head + body + [""] + tail)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "azure"])
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--azure-auth", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, default=_OUTDIR)
    ap.add_argument("--delay", type=float, default=0.02)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.backend == "azure":
        # per-model Azure endpoint + key (deployments live on different resources / APIs)
        ml = args.model.lower()
        if "claude" in ml:        # Anthropic messages API on the Foundry resource
            if args.endpoint is None:
                args.endpoint = "https://liv-forecast.services.ai.azure.com/anthropic"
            if not args.api_key:
                args.api_key = os.environ.get("AZURE_AI_API_KEY", "")
        elif "deepseek" in ml:
            if args.endpoint is None:
                args.endpoint = "https://cos-tiktok-annotation-a-resource.services.ai.azure.com/openai/v1"
            if not args.api_key:
                args.api_key = os.environ.get("DEEPSEEK_AZURE_API_KEY", "")
        else:
            if args.endpoint is None:
                args.endpoint = _AZURE_ENDPOINT
            if not args.api_key:
                args.api_key = os.environ.get("AZURE_AI_API_KEY", "")
    elif args.endpoint is None:
        args.endpoint = DEFAULT_OLLAMA_ENDPOINT
    no_think = "qwen" in args.model.lower()

    episodes = []
    for i in range(args.n):
        ep = generate_episode(seed=i, T=KMAX); ep["_id"] = f"city_{i:05d}"; episodes.append(ep)

    if args.dry_run:
        ep = episodes[0]
        print(f"city {ep['_id']} g={ep['g']}  K={KMAX}  shocks={SHOCKS}\n--- k=0 ---")
        print(build_prompt([], [], SHOCKS[0], no_think=no_think))
        print("\n--- k=3 ---")
        print(build_prompt(ep["news"][:3], ep["polls"][:3], SHOCKS[1], no_think=no_think))
        return

    slug = _slug(args.model); date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"results_{slug}_{date}.jsonl"
    done = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            if ln.strip():
                try: done.add(json.loads(ln)["episode_id"])
                except Exception: pass
    pending = [ep for ep in episodes if ep["_id"] not in done]
    print(f"Model {args.model} [{args.backend}]  cities pending {len(pending)}  "
          f"({(KMAX+1)*len(SHOCKS)} calls each)  out {out_path}")

    if args.backend == "azure" and "claude" in args.model.lower():
        from anthropic import AnthropicFoundry
        client = AnthropicFoundry(azure_ad_token_provider=lambda: args.api_key, base_url=args.endpoint)
    else:
        client = make_client(args.backend, endpoint=args.endpoint, api_key=args.api_key, azure_auth=args.azure_auth)
    prog = _Progress(len(pending)); n_ok = n_err = 0

    with out_path.open("a") as fh:
        for ep in pending:
            curve = []
            for k in range(0, KMAX + 1):
                nk, pk = ep["news"][:k], ep["polls"][:k]
                preds = {}
                for s in SHOCKS:
                    p = None
                    for _try in range(MAX_TRIES):
                        try:
                            resp = chat_call(client, args.model,
                                             [{"role": "user", "content": build_prompt(nk, pk, s, no_think=no_think)}],
                                             args.max_tokens, args.temperature)
                            p, _ = parse(_strip_reasoning((resp.choices[0].message.content or "").strip()))
                        except Exception:
                            p = None
                        if p is not None:
                            break
                        if args.delay: time.sleep(args.delay)
                    preds[s] = p
                pairs = [(s, preds[s]) for s in SHOCKS if preds[s] is not None]
                lm_g = _slope([s for s, _ in pairs], [p for _, p in pairs]) if len(pairs) >= 2 else None
                curve.append({"k": k, "lm_g": round(lm_g, 4) if lm_g is not None else None,
                              "bayes_g": compute_bayes(nk, pk, 0.0)["g_hat"],
                              "preds": [preds[s] for s in SHOCKS]})
            rec = {"episode_id": ep["_id"], "model": slug, "g": ep["g"],
                   "news": ep["news"], "polls": ep["polls"], "curve": curve,
                   "created_at": datetime.now(timezone.utc).isoformat()}
            _write_line(fh, rec)
            got = sum(1 for c in curve if c["lm_g"] is not None)
            if got: n_ok += 1; prog.tick("ok")
            else: n_err += 1; prog.tick("err")
            if args.verbose:
                print(f"  {ep['_id']} g={ep['g']:.2f}  ĝ_k=" + ",".join(
                    f"{c['lm_g']:.2f}" if c['lm_g'] is not None else "NA" for c in curve))

    prog.finish()
    (args.out / f"results_{slug}_{date}.manifest.json").write_text(json.dumps({
        "model": args.model, "slug": slug, "n": args.n, "K": KMAX, "shocks": SHOCKS,
        "n_ok": n_ok, "n_err": n_err, "world": "news_response_curve",
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "complete"}, indent=2))
    print(f"Done — {n_ok} ok, {n_err} err → {out_path}")


if __name__ == "__main__":
    main()
