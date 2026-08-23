#!/usr/bin/env python3
"""Exp 2 — fair latent-recovery probe (multi-shock + reasoning prompt).

For each city we (a) ask the model to first estimate this city's news-response rate,
then predict, and (b) query it at several test shocks {-10,-5,+5,+10}. The per-city
recovered gain is the SLOPE of predicted Δpoll on the shock magnitude — so any error
in where the model anchors the level lands in the intercept, not the gain. We also
keep the model's STATED rate (points-per-news) as a direct probe.

This is the apples-to-apples version of run_recovery.py: it reads each model's gain
the way the Bayes oracle's posterior reads it, instead of from one noisy shock.

Output: data/recovery_probe/results_{slug}_{date}.jsonl (+ .manifest.json)
"""
from __future__ import annotations

import argparse, json, os, re, sys, time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_HERE))
from engine.news_response import generate_episode, compute_bayes
from run_batch import (make_client, _call_with_retry, _strip_reasoning, _slug,
                       _Progress, _write_line, _AZURE_ENDPOINT)
from local_agent import DEFAULT_OLLAMA_ENDPOINT

_OUTDIR = _ROOT / "data" / "recovery_probe"
SHOCKS = [-10, -5, 5, 10]


def build_prompt(news, polls, shock, *, no_think=False):
    lines = [
        "You are tracking a local election campaign with a weekly tracking poll.",
        "Each week reports the net news for that week (a signed number: positive = good "
        "news for the candidate, negative = bad news) and the resulting poll "
        "(candidate support, 0-100).",
        "",
        "| Week | Net news | Poll |",
        "|------|----------|------|",
    ]
    for i, (n, p) in enumerate(zip(news, polls), 1):
        lines.append(f"| {i:>4} | {n:>+5d} | {p:>4} |")
    wk = len(news) + 1
    lines += [
        "",
        "Different cities' polls respond to news by different amounts: some local media "
        "environments let news move opinion strongly, others mute it so the polls barely budge.",
        f"First, from the history above, estimate how many poll points THIS city's poll tends "
        f"to move for each +1 point of net news. Then, given that in week {wk} the net news "
        f"will be {shock:+d}, predict week {wk}'s poll.",
        'Respond with a JSON object only: '
        '{"rationale": "one short sentence", "points_per_news": <number>, '
        '"predicted_poll": <integer 0-100>}',
    ]
    if no_think:
        lines.append("/no_think")
    return "\n".join(lines)


def _is_reasoning_azure(model: str) -> bool:
    m = model.lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def chat_call(client, model, messages, max_tokens, temperature):
    """GPT-5/o-series Azure deployments require max_completion_tokens (and reject a
    custom temperature); everything else uses the classic max_tokens + temperature.
    Claude (AnthropicFoundry) speaks the Anthropic messages API — we shim its reply into
    the OpenAI .choices[0].message.content shape so callers are unchanged."""
    if "claude" in model.lower():
        from types import SimpleNamespace
        resp = _call_with_retry(client.messages.create, model=model, messages=messages,
                                max_tokens=max(max_tokens, 1024))
        text = resp.content[0].text if getattr(resp, "content", None) else ""
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=text), finish_reason=getattr(resp, "stop_reason", "stop"))])
    kw = dict(model=model, messages=messages)
    if _is_reasoning_azure(model):
        kw["max_completion_tokens"] = max(max_tokens, 4096)   # leave room for hidden reasoning
    else:
        kw["max_tokens"] = max_tokens
        kw["temperature"] = temperature
    return _call_with_retry(client.chat.completions.create, **kw)


def parse(raw):
    """Extract (predicted_poll, points_per_news) from a model reply.

    Only accepts a genuine JSON object containing predicted_poll. If the reply has
    no parseable JSON (e.g. the model ran out of tokens mid-reasoning), returns
    (None, None) — we never scrape a stray number from the reasoning text, since
    that invents a fake answer and corrupts the recovered gain."""
    pred = rate = None
    for m in re.finditer(r"\{[^{}]*\}", raw, re.DOTALL):   # flat JSON objects, last valid wins
        try:
            o = json.loads(m.group())
        except Exception:
            continue
        if not isinstance(o, dict) or o.get("predicted_poll") is None:
            continue
        try:
            pred = max(0, min(100, int(round(float(o["predicted_poll"])))))
        except Exception:
            continue
        rate = None
        if o.get("points_per_news") is not None:
            try:
                rate = float(o["points_per_news"])
            except Exception:
                rate = None
    return pred, rate


def _slope(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n; my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "azure"])
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--azure-auth", action="store_true")
    ap.add_argument("--n", type=int, default=250, help="cities")
    ap.add_argument("--out", type=Path, default=_OUTDIR)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.02)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.endpoint is None:
        args.endpoint = _AZURE_ENDPOINT if args.backend == "azure" else DEFAULT_OLLAMA_ENDPOINT
    if args.backend == "azure" and not args.api_key:
        args.api_key = (os.environ.get("AZURE_AI_API_KEY", "")
                        or os.environ.get("CLAUDE_AZURE_API_KEY", "")
                        or os.environ.get("DEEPSEEK_AZURE_API_KEY", ""))
    no_think = "qwen" in args.model.lower()

    episodes = []
    for i in range(args.n):
        ep = generate_episode(seed=args.seed_offset + i)
        ep["_id"] = f"city_{i:05d}"
        episodes.append(ep)

    if args.dry_run:
        ep = episodes[0]
        print(f"Episode {ep['_id']}  g={ep['g']}  shocks={SHOCKS}\n")
        print(build_prompt(ep["news"], ep["polls"], SHOCKS[0], no_think=no_think))
        return

    slug = _slug(args.model)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"results_{slug}_{date_str}.jsonl"
    mani_path = args.out / f"results_{slug}_{date_str}.manifest.json"

    done = set()
    if out_path.exists():
        for ln in out_path.read_text().splitlines():
            if ln.strip():
                try: done.add(json.loads(ln)["episode_id"])
                except Exception: pass
    pending = [ep for ep in episodes if ep["_id"] not in done]
    print(f"Model {args.model} [{args.backend}]  cities pending {len(pending)}  "
          f"({len(SHOCKS)} shocks each)  out {out_path}")

    client = make_client(args.backend, endpoint=args.endpoint,
                         api_key=args.api_key, azure_auth=args.azure_auth)
    prog = _Progress(len(pending)); n_ok = n_err = 0

    with out_path.open("a") as fh:
        for ep in pending:
            news, polls, last = ep["news"], ep["polls"], ep["last_poll"]
            preds, rates, raws = [], [], []
            for shock in SHOCKS:
                try:
                    resp = chat_call(
                        client, args.model,
                        [{"role": "user", "content": build_prompt(news, polls, shock, no_think=no_think)}],
                        args.max_tokens, args.temperature)
                    raw = _strip_reasoning((resp.choices[0].message.content or "").strip())
                    p, r = parse(raw)
                    if args.delay: time.sleep(args.delay)
                except Exception as exc:
                    p, r, raw = None, None, f"ERROR: {exc}"
                preds.append(p); rates.append(r); raws.append(raw[:160])

            # behavioural recovered gain = slope of Δpoll on shock (level → intercept)
            pairs = [(s, p - last) for s, p in zip(SHOCKS, preds) if p is not None]
            g_slope = _slope([s for s, _ in pairs], [d for _, d in pairs]) if len(pairs) >= 2 else None
            good_rates = [r for r in rates if r is not None]
            g_stated = sum(good_rates) / len(good_rates) if good_rates else None
            bayes = compute_bayes(news, polls, 0.0)   # posterior-mean ĝ reference

            rec = {
                "episode_id": ep["_id"], "model": slug, "g": ep["g"], "T": ep["T"],
                "news": news, "polls": polls, "last_poll": last, "shocks": SHOCKS,
                "preds": preds, "rates": rates,
                "lm_g_slope":  round(g_slope, 4) if g_slope is not None else None,
                "lm_g_stated": round(g_stated, 4) if g_stated is not None else None,
                "bayes_g_hat": bayes["g_hat"],
                "raws": raws,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_line(fh, rec)
            if g_slope is None: n_err += 1; prog.tick("err")
            else: n_ok += 1; prog.tick("ok")
            if args.verbose and g_slope is not None:
                print(f"  {ep['_id']} g={ep['g']:.2f} ĝ_slope={rec['lm_g_slope']} ĝ_stated={rec['lm_g_stated']}")

    prog.finish()
    mani_path.write_text(json.dumps({
        "model": args.model, "slug": slug, "n": args.n, "shocks": SHOCKS,
        "n_ok": n_ok, "n_errors": n_err, "world": "news_response_probe",
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "complete"}, indent=2))
    print(f"Done — {n_ok} ok, {n_err} err → {out_path}")


if __name__ == "__main__":
    main()
