#!/usr/bin/env python3
"""Exp 2 (news-response world) — latent-recovery eval.

Each episode = one city with a hidden news-responsiveness gain g. The model sees
a weekly (net-news, poll) history, then a large test shock, and predicts next
week's poll. We record its prediction → implied ĝ = (pred − last_poll)/test_news,
alongside the Bayes oracle and the two fixed shortcuts (face-value ĝ=1, ignore ĝ=0).

Output: data/recovery/results_{slug}_{date}.jsonl  (+ .manifest.json)
Reuses run_batch.py's LLM client / retry / progress plumbing.

Examples:
    python eval/run_recovery.py --dry-run --n 1
    python eval/run_recovery.py --model qwen2.5:7b  --backend ollama --n 400
    python eval/run_recovery.py --model claude-opus-4-8 --backend azure --azure-auth --n 400
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

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))    # engine.news_response
sys.path.insert(0, str(_HERE))    # run_batch (sibling)

from engine.news_response import (
    generate_episode, compute_bayes, predict_fixed_gain, implied_g,
)
# reuse the client / retry / progress plumbing from the batch eval
from run_batch import (
    make_client, _call_with_retry, _strip_reasoning, _slug, _Progress,
    _write_line, _AZURE_ENDPOINT,
)
from local_agent import DEFAULT_OLLAMA_ENDPOINT

_OUTDIR = _ROOT / "data" / "recovery"


# ── prompt / parse ─────────────────────────────────────────────────────────────
def build_prompt(news: list[int], polls: list[int], test_news: int, *, no_think: bool = False) -> str:
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
        f"| {wk:>4} | {test_news:>+5d} | {'?':>4} |",
        "",
        f"In week {wk} a major story breaks with net news {test_news:+d}. "
        "Judging from how strongly THIS city's polls have responded to news so far, "
        "predict next week's poll.",
        'Respond with a JSON object only: '
        '{"rationale": "one short sentence", "predicted_poll": <integer 0-100>}',
    ]
    if no_think:
        lines.append("/no_think")
    return "\n".join(lines)


def parse_poll(raw: str) -> int | None:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            v = obj.get("predicted_poll")
            if v is not None:
                return max(0, min(100, int(round(float(v)))))
        except Exception:
            pass
    m = re.search(r"-?\d+(\.\d+)?", raw)            # fallback: first number
    return max(0, min(100, int(round(float(m.group()))))) if m else None


def predict_lm(news, polls, test_news, *, client, model, max_tokens, temperature, no_think):
    prompt = build_prompt(news, polls, test_news, no_think=no_think)
    resp = _call_with_retry(
        client.chat.completions.create,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    raw = _strip_reasoning((resp.choices[0].message.content or "").strip())
    return parse_poll(raw), raw[:240]


def _load_done(path: Path) -> set[str]:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["episode_id"])
                except Exception:
                    pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "azure"])
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--azure-auth", action="store_true")
    ap.add_argument("--n", type=int, default=400, help="number of cities (episodes)")
    ap.add_argument("--out", type=Path, default=_OUTDIR)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.05)
    ap.add_argument("--max-tokens", type=int, default=160)
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

    # deterministic episode set (shared across models via fixed seeds)
    episodes = []
    for i in range(args.n):
        ep = generate_episode(seed=args.seed_offset + i)
        ep["_id"] = f"city_{i:05d}"
        episodes.append(ep)

    if args.dry_run:
        ep = episodes[0]
        print(f"Episode {ep['_id']}  g={ep['g']}  T={ep['T']}  test_news={ep['test_news']}\n")
        print(build_prompt(ep["news"], ep["polls"], ep["test_news"], no_think=no_think))
        return

    slug = _slug(args.model)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"results_{slug}_{date_str}.jsonl"
    mani_path = args.out / f"results_{slug}_{date_str}.manifest.json"

    done = _load_done(out_path)
    pending = [ep for ep in episodes if ep["_id"] not in done]
    print(f"Model: {args.model} [{args.backend}] @ {args.endpoint}")
    print(f"Cities: {args.n}   pending: {len(pending)}   out: {out_path}")

    client = make_client(args.backend, endpoint=args.endpoint,
                         api_key=args.api_key, azure_auth=args.azure_auth)
    prog = _Progress(len(pending))
    n_ok = n_err = 0

    with out_path.open("a") as fh:
        for ep in pending:
            news, polls, tn, last = ep["news"], ep["polls"], ep["test_news"], ep["last_poll"]
            bayes = compute_bayes(news, polls, tn)
            try:
                lm_pred, lm_raw = predict_lm(news, polls, tn, client=client, model=args.model,
                                             max_tokens=args.max_tokens, temperature=args.temperature,
                                             no_think=no_think)
                if args.delay:
                    time.sleep(args.delay)
            except Exception as exc:
                lm_pred, lm_raw = None, f"ERROR: {exc}"

            rec = {
                "episode_id": ep["_id"],
                "model":      slug,
                "g":          ep["g"],
                "T":          ep["T"],
                "news":       news,
                "polls":      polls,
                "test_news":  tn,
                "last_poll":  last,
                "gold_poll":  ep["gold_poll"],
                "lm_pred":    lm_pred,
                "lm_raw":     lm_raw,
                "lm_g_hat":   round(implied_g(lm_pred, last, tn), 4) if lm_pred is not None else None,
                "bayes_pred":  bayes["predicted_poll"],
                "bayes_g_hat": bayes["g_hat"],
                "face_pred":   round(predict_fixed_gain(last, tn, 1.0), 2),  # ĝ=1
                "ignore_pred": round(predict_fixed_gain(last, tn, 0.0), 2),  # ĝ=0
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_line(fh, rec)
            if lm_pred is None:
                n_err += 1; prog.tick("err")
            else:
                n_ok += 1; prog.tick("ok")
            if args.verbose and lm_pred is not None:
                print(f"  {ep['_id']} g={ep['g']:.2f} ĝ_lm={rec['lm_g_hat']} ĝ_bayes={bayes['g_hat']}")

    prog.finish()
    mani_path.write_text(json.dumps({
        "model": args.model, "slug": slug, "backend": args.backend, "n": args.n,
        "n_ok": n_ok, "n_errors": n_err, "world": "news_response",
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "complete",
    }, indent=2))
    print(f"Done — {n_ok} ok, {n_err} err → {out_path}")


if __name__ == "__main__":
    main()
