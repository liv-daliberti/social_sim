#!/usr/bin/env python3
"""Exp 2 — recovery LEARNING CURVE, information-PARITY variant.

Same learning-curve design as run_recovery_curve.py (probe ĝ_k after each prefix
k=0..K), but with two fairness changes that level the field with the Bayes oracle:

  1. INFORMATION PARITY. The oracle is handed the exact generative model; here we
     tell the LLM the same structure it (implicitly) knows — g is a fixed number in
     [0.1, 1.0], opinion accumulates g*news each week with ~1 pt drift, and the poll
     carries ~2 pts of survey noise. This turns the task from "guess the spec" into
     pure latent inference.

  2. TWO READOUTS. We record both the behavioural SLOPE ĝ_k (slope of predicted poll
     on the ±shock, as before) AND the model's directly STATED points-per-news. The
     stated readout removes the forecast-application + integer-rounding noise that
     only the LLM (not the oracle) pays.

Output: data/recovery_curve_parity/results_{slug}_{date}.jsonl
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_HERE))
from engine.news_response import generate_episode, compute_bayes, G_LO, G_HI
from run_batch import make_client, _strip_reasoning, _slug, _Progress, _write_line, _AZURE_ENDPOINT
from run_recovery_probe import chat_call, parse, _slope
from local_agent import DEFAULT_OLLAMA_ENDPOINT

_OUTDIR = _ROOT / "data" / "recovery_curve_parity"
SHOCKS = [-10, -5, 5, 10]      # 4-point regression slope
KMAX = 10
MAX_TRIES = 3                  # retry a call that yields no parseable JSON before giving up


def build_prompt(news_k, polls_k, shock, *, no_think=False, reason_cap=500):
    head = [
        "You are tracking a local election campaign with a weekly tracking poll.",
        "Each week reports the net news (a signed number: positive = good news for the "
        "candidate, negative = bad) and the resulting poll (support, 0-100).",
        "",
        "How this city works (the same for every week):",
        f"  - This city has a fixed news-responsiveness g, a number between {G_LO} and {G_HI} "
        "(you do not know which).",
        "  - Each week the underlying opinion moves by about g times that week's net news, "
        "plus a small random drift of about 1 point.",
        "  - The reported poll is that opinion plus survey noise of about 2 points.",
        "  - Low g = sticky coverage (news barely moves the poll); high g = news moves it "
        "nearly one-for-one.",
        "",
    ]
    k = len(news_k)
    if k == 0:
        body = ["No weeks have been observed yet for this city, so g could be anything in its range."]
    else:
        body = ["| Week | Net news | Poll |", "|------|----------|------|"]
        body += [f"| {i:>4} | {n:>+5d} | {p:>4} |" for i, (n, p) in enumerate(zip(news_k, polls_k), 1)]
        # NEUTRAL elicitation: state the structure (head) and the data, then plainly ask for g — with
        # NO steer on how to handle sparsity/noise (neither "see through the noise" nor "hedge toward
        # the middle"). Either steer pre-loads the answer; saying nothing measures the model's OWN
        # disposition to over-commit or regularize when a short history can't identify g.
        body += ["", "From the history above, estimate this city's g (poll points moved per +1 point "
                 "of net news)."]
    tail = [
        f"Given that next week's net news will be {shock:+d}, predict next week's poll.",
        f"You may reason step by step. Your reply is capped at about {reason_cap} tokens, which is "
        "ample room: use as much or as little reasoning as the inference needs, and be sure to finish "
        "the JSON within the limit.",
        'END your reply with the answer as a JSON object on its own line: '
        '{"rationale": "one short sentence", "points_per_news": <number>, '
        '"predicted_poll": <integer 0-100>}',
    ]
    if no_think:                       # Qwen control token: suppress step-by-step reasoning
        tail.append("/no_think")
    return "\n".join(head + body + [""] + tail)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "azure"])
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--azure-auth", action="store_true")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--out", type=Path, default=_OUTDIR)
    ap.add_argument("--delay", type=float, default=0.02)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--timeout", type=float, default=90.0,
                    help="per-call client timeout (s); raise high (e.g. 1200) for slow "
                         "cold-starting deployments like DeepSeek so the model is never cut off")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    # probe depth: read prefixes k=0..kmax. Episodes are still generated at T=KMAX (=10)
    # so cities stay byte-identical to the open-weight shared seeds; only fewer prefixes
    # are queried. Use --kmax 5 to ~halve cost for the frontier probes (24 vs 44 calls/city).
    ap.add_argument("--kmax", type=int, default=KMAX)
    # Reasoning is ON by default so local models think step-by-step like the frontier
    # agents (fair comparison). --no-think restores the old Qwen /no_think behaviour.
    ap.add_argument("--no-think", action="store_true",
                    help="append Qwen /no_think token, disabling step-by-step reasoning")
    # token cap stated in the prompt ("reply is capped at about N tokens"); pair it with
    # --max-tokens so the stated and enforced limits agree (e.g. both 1024 for locals).
    ap.add_argument("--reason-cap", type=int, default=2048)
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
            # DeepSeek-V4-Pro is also deployed on the liv-forecast Foundry resource (same as gpt-5.4),
            # auth'd with AZURE_AI_API_KEY. The old cos-tiktok deployment went dead/degraded (every
            # call timed out from every node — same Front-Door IP, just an unhealthy backend), so we
            # use the healthy liv-forecast deployment, which is reachable even from the login node.
            if args.endpoint is None:
                args.endpoint = _AZURE_ENDPOINT
            if not args.api_key:
                args.api_key = os.environ.get("AZURE_AI_API_KEY", "")
        else:                     # gpt-5.4 etc. on the default Foundry resource
            if args.endpoint is None:
                args.endpoint = _AZURE_ENDPOINT
            if not args.api_key:
                args.api_key = os.environ.get("AZURE_AI_API_KEY", "")
    elif args.endpoint is None:
        args.endpoint = DEFAULT_OLLAMA_ENDPOINT
    no_think = args.no_think   # default False: models reason like the frontier agents

    episodes = []
    for i in range(args.n):
        ep = generate_episode(seed=i, T=KMAX); ep["_id"] = f"city_{i:05d}"; episodes.append(ep)

    if args.dry_run:
        ep = episodes[0]
        print(f"city {ep['_id']} g={ep['g']}  K={args.kmax}  shocks={SHOCKS}\n--- k=0 ---")
        print(build_prompt([], [], SHOCKS[0], no_think=no_think, reason_cap=args.reason_cap))
        print("\n--- k=3 ---")
        print(build_prompt(ep["news"][:3], ep["polls"][:3], SHOCKS[1], no_think=no_think, reason_cap=args.reason_cap))
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
    print(f"Model {args.model} [{args.backend}] PARITY  cities pending {len(pending)}  "
          f"({(args.kmax+1)*len(SHOCKS)} calls each, k=0..{args.kmax})  out {out_path}")

    if args.backend == "azure" and "claude" in args.model.lower():
        from anthropic import AnthropicFoundry
        client = AnthropicFoundry(azure_ad_token_provider=lambda: args.api_key, base_url=args.endpoint)
    else:
        client = make_client(args.backend, endpoint=args.endpoint, api_key=args.api_key,
                             azure_auth=args.azure_auth, timeout=args.timeout)
    prog = _Progress(len(pending)); n_ok = n_err = 0

    with out_path.open("a") as fh:
        for ep in pending:
            curve = []
            for k in range(0, args.kmax + 1):
                nk, pk = ep["news"][:k], ep["polls"][:k]
                preds, rates = {}, []
                for s in SHOCKS:
                    p = r = None
                    for _try in range(MAX_TRIES):
                        try:
                            resp = chat_call(client, args.model,
                                             [{"role": "user", "content": build_prompt(nk, pk, s, no_think=no_think, reason_cap=args.reason_cap)}],
                                             args.max_tokens, args.temperature)
                            p, r = parse(_strip_reasoning((resp.choices[0].message.content or "").strip()))
                        except Exception:
                            p = r = None
                        if p is not None:
                            break
                        if args.delay: time.sleep(args.delay)
                    if r is not None: rates.append(r)
                    preds[s] = p
                pairs = [(s, preds[s]) for s in SHOCKS if preds[s] is not None]
                lm_g = _slope([s for s, _ in pairs], [p for _, p in pairs]) if len(pairs) >= 2 else None
                lm_g_stated = sum(rates) / len(rates) if rates else None
                curve.append({"k": k, "lm_g": round(lm_g, 4) if lm_g is not None else None,
                              "lm_g_stated": round(lm_g_stated, 4) if lm_g_stated is not None else None,
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
    # config fingerprint so a run's prompt/budget is always recoverable (the data files
    # don't store it). prompt_sha hashes a fixed-input prompt, so it changes iff the
    # template wording, reason_cap, or no_think changes.
    prompt_sha = hashlib.sha256(
        build_prompt([5, -3], [55, 52], -10, no_think=no_think,
                     reason_cap=args.reason_cap).encode()).hexdigest()[:12]
    (args.out / f"results_{slug}_{date}.manifest.json").write_text(json.dumps({
        "model": args.model, "slug": slug, "n": args.n, "K": args.kmax, "shocks": SHOCKS,
        "n_ok": n_ok, "n_err": n_err, "world": "news_response_curve_parity",
        "max_tokens": args.max_tokens, "temperature": args.temperature,
        "reason_cap": args.reason_cap, "no_think": args.no_think, "prompt_sha": prompt_sha,
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "complete"}, indent=2))
    print(f"Done — {n_ok} ok, {n_err} err → {out_path}")


if __name__ == "__main__":
    main()
