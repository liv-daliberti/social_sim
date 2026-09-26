#!/usr/bin/env python3
"""Exp 2 Biased News — batch LM evaluation.

For N neutral + N biased episodes, at every step-prefix T_obs = 1 .. T_max,
records:
  • Local LM prediction   (Ollama — same client as Exp 1)
  • Bayesian Kalman-filter ceiling
  • Naïve EMA baseline

Output: data/batch/results_{slug}_{date}.jsonl  +  .manifest.json

Usage (from biased_news/):
    # local (Ollama)
    python eval/run_batch.py --model qwen2.5:7b
    python eval/run_batch.py --model llama3.1:8b
    python eval/run_batch.py --n 10 --dry-run
    # frontier (Azure AI Foundry); key from AZURE_AI_API_KEY/CLAUDE_AZURE_API_KEY env
    python eval/run_batch.py --backend azure --azure-auth --model claude-opus-4-8 --n 20
    python eval/run_batch.py --backend azure --azure-auth --model gpt-5.4         --n 20
    python eval/run_batch.py --backend azure --azure-auth --model DeepSeek-V4-Pro --n 20
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent                          # biased_news/

# ── engine ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(_ROOT))
from engine.temporal_dag import (
    generate_episode,
    compute_bayes_forecast,
    compute_blind_forecast,
    VARIANTS,
    BASE_EFFECT,
    BIASED_GAIN,
    SIGMA_OPINION,
    SIGMA_SURVEY,
)

# ── reuse Exp 1's Ollama client ────────────────────────────────────────────────
_EXP1_AGENT = (_ROOT.parent.parent / "exp1_prospective" / "agent").resolve()
sys.path.insert(0, str(_EXP1_AGENT))
from local_agent import make_local_client, DEFAULT_OLLAMA_ENDPOINT, _call_with_retry

DEFAULT_MODEL = "qwen2.5:7b"
_OUTDIR       = _ROOT / "data" / "batch"

# Frontier (Azure AI Foundry) defaults — mirror scripts/run_*_eval.py
_AZURE_ENDPOINT = "https://liv-forecast.services.ai.azure.com/openai/v1"


def make_client(backend: str, *, endpoint: str, api_key: str = "", azure_auth: bool = False,
                timeout: float = 90.0):
    """Return an OpenAI-compatible client for the chosen backend.

    backend="ollama" → local Ollama (Exp 1's client; no auth needed).
    backend="azure"  → Azure AI Foundry, header-based auth (matches run_grpo.py).
    Both expose the same chat.completions.create interface, so the rest of the
    batch loop is backend-agnostic and every model emits one identical schema.

    `timeout` is the per-call client timeout in seconds. Default 90 is fine for fast,
    always-warm deployments (GPT/Claude), but a slow reasoning model on a serverless
    backend (DeepSeek-V4-Pro) cold-starts well past 90s, so raise it (e.g. 1200) to give
    the model as much time as it needs rather than cutting it off with an APITimeoutError.
    """
    if backend == "azure":
        from openai import OpenAI
        if azure_auth:
            return OpenAI(
                base_url        = endpoint,
                api_key         = "placeholder",
                default_headers = {"api-key": api_key},
                timeout         = timeout,
                max_retries     = 2,
            )
        return OpenAI(base_url=endpoint, api_key=api_key, timeout=timeout, max_retries=2)
    return make_local_client(endpoint=endpoint)


def _strip_reasoning(text: str) -> str:
    """Drop <think>…</think> blocks emitted by reasoning models before parsing."""
    return re.sub(r"(?is)<think>.*?</think>", "", text).strip()


# ── prompt (mirrors server.py exactly) ────────────────────────────────────────

def _build_prompt(variant: str, events: list, surveys: list, next_event: dict,
                  *, no_think: bool = True) -> str:
    T_obs = len(events)
    lines = [
        "You are a political analyst tracking a local election campaign.",
        "You have been monitoring public opinion through a sequence of events "
        "and weekly survey results (scale 0–100, where higher means more support "
        "for the leading candidate).",
        "",
    ]
    prefix = VARIANTS[variant]
    if prefix:
        lines += [prefix, ""]
    lines += [
        "Campaign timeline:",
        "",
        "| Week | Event | Survey |",
        "|------|-------|--------|",
    ]
    for i, (evt, surv) in enumerate(zip(events, surveys), 1):
        lines.append(f"| {i:>4} | {evt['text']:<70} | {surv:>6} |")
    lines.append(f"| {T_obs + 1:>4} | {next_event['text']:<70} | {'?':>6} |")
    lines += [
        "",
        f"Given the campaign history above and this week's event (Week {T_obs + 1}), "
        f"what survey result (an integer from 0 to 100) do you expect?",
        "",
        "Respond with a JSON object only — no prose outside it:",
        '{"rationale": "one or two sentence explanation", "predicted_survey": <integer 0–100>}',
    ]
    if no_think:                       # Qwen control token; omit for frontier models
        lines.append("/no_think")
    return "\n".join(lines)


def _parse_pred(raw: str) -> int | None:
    """Extract predicted_survey from JSON response (with regex fallback)."""
    try:
        obj = json.loads(raw)
        v   = obj.get("predicted_survey")
        if v is not None:
            return max(0, min(100, int(round(float(v)))))
    except Exception:
        pass
    # fallback: first integer in response
    m = re.search(r'\b(\d{1,3})\b', raw)
    if m:
        return max(0, min(100, int(m.group(1))))
    return None


def predict_lm(
    variant: str,
    events: list,
    surveys: list,
    next_event: dict,
    *,
    client,
    model: str,
    max_tokens: int = 128,
    temperature: float = 0.1,
    no_think: bool = True,
) -> tuple[int | None, str]:
    prompt = _build_prompt(variant, events, surveys, next_event, no_think=no_think)
    msgs   = [{"role": "user", "content": prompt}]
    resp   = _call_with_retry(
        client.chat.completions.create,
        model       = model,
        messages    = msgs,
        max_tokens  = max_tokens,
        temperature = temperature,
    )
    raw  = _strip_reasoning((resp.choices[0].message.content or "").strip())
    pred = _parse_pred(raw)
    return pred, raw[:200]


# ── progress bar ───────────────────────────────────────────────────────────────

class _Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.done = self.ok = self.errors = self.skipped = 0
        self._t0  = time.time()

    def tick(self, status: str) -> None:
        self.done += 1
        if   status == "ok":   self.ok      += 1
        elif status == "skip": self.skipped  += 1
        else:                  self.errors   += 1
        elapsed = time.time() - self._t0
        rate    = self.done / max(elapsed, 0.1)
        eta_s   = (self.total - self.done) / max(rate, 1e-9)
        eta_str = f"{eta_s/60:.1f}m" if eta_s > 90 else f"{eta_s:.0f}s"
        filled  = int(30 * self.done / max(self.total, 1))
        bar     = "█" * filled + "░" * (30 - filled)
        sys.stdout.write(
            f"\r  [{bar}] {self.done}/{self.total}"
            f"  ok={self.ok} err={self.errors}"
            f"  {100*self.done/max(self.total,1):.0f}%  ETA {eta_str}   "
        )
        sys.stdout.flush()

    def finish(self) -> None:
        elapsed = time.time() - self._t0
        sys.stdout.write("\n")
        print(f"  Done in {elapsed/60:.1f}m — {self.ok} ok  {self.errors} errors  {self.skipped} skipped")


# ── I/O helpers ────────────────────────────────────────────────────────────────

def _slug(model: str) -> str:
    return model.replace(":", "-").replace("/", "-")


def _load_done(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["episode_id"], r["variant"]))
            except Exception:
                pass
    return done


def _write_line(f, obj: dict) -> None:
    line = json.dumps(obj) + "\n"
    fcntl.flock(f, fcntl.LOCK_EX)
    try:
        f.write(line)
        f.flush()
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model",       default=DEFAULT_MODEL,
                    help="Model id  (ollama tag e.g. qwen2.5:7b, or azure model e.g. claude-opus-4-8)")
    ap.add_argument("--backend",     default="ollama", choices=["ollama", "azure"],
                    help="ollama = local; azure = Azure AI Foundry frontier models")
    ap.add_argument("--endpoint",    default=None,
                    help="API base URL (default: Ollama endpoint for ollama, Azure Foundry for azure)")
    ap.add_argument("--api-key",     default="",
                    help="API key (azure backend; falls back to AZURE_AI_API_KEY / CLAUDE_AZURE_API_KEY env)")
    ap.add_argument("--azure-auth",  action="store_true",
                    help="Use Azure header-based auth ({'api-key': key}); set for Foundry frontier models")
    ap.add_argument("--n",           type=int, default=100,
                    help="Episodes per bias condition (100 neutral + 100 biased)")
    ap.add_argument("--T",           type=int, default=None,
                    help="Fixed episode length (default: random 6-12 per episode)")
    # World parameters (default = canonical Regime B in engine constants).
    ap.add_argument("--base-effect",   type=float, default=BASE_EFFECT,
                    help="Neutral-world magnitude of a polar event (gain=1.0)")
    ap.add_argument("--biased-gain",   type=float, default=BIASED_GAIN,
                    help="Biased-world gain ×base_effect (e.g. 0.10 = 10%% responsive)")
    ap.add_argument("--sigma-opinion", type=float, default=SIGMA_OPINION,
                    help="Opinion random-walk noise")
    ap.add_argument("--sigma-survey",  type=float, default=SIGMA_SURVEY,
                    help="Survey measurement noise")
    ap.add_argument("--variants",    nargs="+", default=["V0", "V1", "V2"],
                    choices=["V0", "V1", "V2"])
    ap.add_argument("--out",         type=Path, default=_OUTDIR)
    ap.add_argument("--seed-offset", type=int,  default=0,
                    help="Add to all episode seeds (keep 0 so Qwen/Llama use same episodes)")
    ap.add_argument("--delay",       type=float, default=0.1,
                    help="Seconds between LM calls")
    ap.add_argument("--max-tokens",  type=int,   default=128)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--verbose",     action="store_true")
    ap.add_argument("--dry-run",     action="store_true",
                    help="Print first prompt and exit without calling API")
    args = ap.parse_args()

    # ── resolve backend-dependent defaults ────────────────────────────────────
    if args.endpoint is None:
        args.endpoint = _AZURE_ENDPOINT if args.backend == "azure" else DEFAULT_OLLAMA_ENDPOINT
    if args.backend == "azure":
        args.api_key = (
            args.api_key
            or os.environ.get("AZURE_AI_API_KEY", "")
            or os.environ.get("CLAUDE_AZURE_API_KEY", "")
            or os.environ.get("DEEPSEEK_AZURE_API_KEY", "")
        )
        if not args.api_key and not args.dry_run:
            print("ERROR: azure backend requires --api-key or AZURE_AI_API_KEY/"
                  "CLAUDE_AZURE_API_KEY/DEEPSEEK_AZURE_API_KEY in env.", file=sys.stderr)
            sys.exit(1)
    # /no_think is a Qwen-only control token; only emit it for Qwen models.
    no_think = "qwen" in args.model.lower()

    # World parameters: passed to the DGP and to the oracle/blind forecasters so
    # the Kalman filters use the same generative model the episodes were drawn from.
    world = dict(base_effect=args.base_effect, biased_gain=args.biased_gain,
                 sigma_opinion=args.sigma_opinion, sigma_survey=args.sigma_survey)

    slug     = _slug(args.model)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    args.out.mkdir(parents=True, exist_ok=True)
    out_path  = args.out / f"results_{slug}_{date_str}.jsonl"
    mani_path = args.out / f"results_{slug}_{date_str}.manifest.json"

    # ── generate deterministic episodes (interleaved neutral/biased) ──────────
    neutral_eps: list[tuple[str, dict]] = []
    biased_eps:  list[tuple[str, dict]] = []
    for i in range(args.n):
        seed = args.seed_offset + i
        ep   = generate_episode(bias="neutral", seed=seed, T=args.T, **world)
        ep["_seed"] = seed
        neutral_eps.append((f"neutral_{i:04d}", ep))
    for i in range(args.n):
        seed = args.seed_offset + 100_000 + i
        ep   = generate_episode(bias="biased", seed=seed, T=args.T, **world)
        ep["_seed"] = seed
        biased_eps.append((f"biased_{i:04d}", ep))

    # Interleave so results arrive mixed: neutral_0, biased_0, neutral_1, biased_1 …
    all_eps = [ep for pair in zip(neutral_eps, biased_eps) for ep in pair]
    if len(neutral_eps) > len(biased_eps):
        all_eps += neutral_eps[len(biased_eps):]
    else:
        all_eps += biased_eps[len(neutral_eps):]

    # ── work list: (episode_id, episode, variant) ─────────────────────────────
    work = [(eid, ep, v) for eid, ep in all_eps for v in args.variants]

    # ── dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        eid, ep, variant = work[0]
        prompt = _build_prompt(variant, ep["events"][:1], ep["surveys"][:1], ep["events"][1],
                               no_think=no_think)
        print(f"Episode: {eid}  bias={ep['bias']}  T={ep['T']}  variant={variant}\n")
        print("─" * 72)
        print(prompt)
        return

    # ── resume ────────────────────────────────────────────────────────────────
    done_set = _load_done(out_path)
    pending  = [(eid, ep, v) for eid, ep, v in work if (eid, v) not in done_set]

    print(f"Model:    {args.model}  [{args.backend}]  @ {args.endpoint}")
    print(f"Episodes: {args.n}N + {args.n}B  ×  {len(args.variants)} variants = {len(work)} records")
    print(f"Output:   {out_path}")
    if done_set:
        print(f"Resuming: {len(done_set)} done, {len(pending)} pending")

    client   = make_client(args.backend, endpoint=args.endpoint,
                           api_key=args.api_key, azure_auth=args.azure_auth)
    progress = _Progress(len(pending))
    n_ok = n_err = 0

    with out_path.open("a") as out_f:
        for eid, ep, variant in pending:
            T     = ep["T"]
            steps = []

            for T_obs in range(1, T + 1):
                obs_evts  = ep["events"][:T_obs]
                obs_survs = ep["surveys"][:T_obs]
                nxt_evt   = ep["events"][T_obs]
                gold_surv = ep["surveys"][T_obs]
                gold_opin = ep["opinion_traj"][T_obs]

                # Bayesian ceiling (marginalises the latent gain)
                bayes = compute_bayes_forecast(
                    events              = obs_evts,
                    surveys             = obs_survs,
                    next_event_polarity = nxt_evt["polarity"],
                    sigma_opinion       = args.sigma_opinion,
                    sigma_survey        = args.sigma_survey,
                    base_effect         = args.base_effect,
                    biased_gain         = args.biased_gain,
                )

                # Latent-blind reference: uses events but a fixed prior-mean gain
                # (never infers the latent). Bayes−blind = value of latent inference.
                blind = compute_blind_forecast(
                    events              = obs_evts,
                    surveys             = obs_survs,
                    next_event_polarity = nxt_evt["polarity"],
                    sigma_opinion       = args.sigma_opinion,
                    sigma_survey        = args.sigma_survey,
                    base_effect         = args.base_effect,
                    biased_gain         = args.biased_gain,
                )

                # Naïve EMA (α=0.35, mirrors server.py _ema_predict)
                alpha, ema = 0.35, float(obs_survs[0])
                for s in obs_survs[1:]:
                    ema = alpha * s + (1 - alpha) * ema
                naive_pred = round(ema)

                # LM
                try:
                    lm_pred, lm_raw = predict_lm(
                        variant, obs_evts, obs_survs, nxt_evt,
                        client=client, model=args.model,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        no_think=no_think,
                    )
                    if args.delay > 0:
                        time.sleep(args.delay)
                except Exception as exc:
                    lm_pred, lm_raw = None, f"ERROR: {exc}"

                steps.append({
                    "T_obs":               T_obs,
                    "next_event_polarity": nxt_evt["polarity"],
                    "next_event_text":     nxt_evt["text"],
                    "lm_pred":             lm_pred,
                    "lm_raw":              lm_raw,
                    "bayes_pred":          round(bayes["predicted_survey"], 2),
                    "bayes_p_biased":      bayes["p_biased"],
                    "blind_pred":          round(blind["predicted_survey"], 2),
                    "naive_pred":          float(naive_pred),
                    "gold_survey":         gold_surv,
                    "gold_opinion":        round(gold_opin, 2),
                    "lm_abs_err":   round(abs(lm_pred - gold_surv), 2) if lm_pred is not None else None,
                    "bayes_abs_err": round(abs(bayes["predicted_survey"] - gold_surv), 2),
                    "blind_abs_err": round(abs(blind["predicted_survey"] - gold_surv), 2),
                    "naive_abs_err": round(abs(naive_pred - gold_surv), 2),
                })

            rec = {
                "episode_id": eid,
                "bias":       ep["bias"],
                "gain":       ep.get("gain"),
                "variant":    variant,
                "model":      slug,
                "ep_seed":    ep["_seed"],
                "T_max":      T,
                "steps":      steps,
                "episode": {
                    "events":       ep["events"],
                    "surveys":      ep["surveys"],
                    "opinion_traj": ep["opinion_traj"],
                    "opinion_init": ep["opinion_init"],
                },
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_line(out_f, rec)

            if args.verbose:
                lm_errs = [s["lm_abs_err"] for s in steps if s["lm_abs_err"] is not None]
                ba_errs = [s["bayes_abs_err"] for s in steps]
                print(f"\n  {eid} {variant}  LM={sum(lm_errs)/max(len(lm_errs),1):.1f}"
                      f"  Bayes={sum(ba_errs)/len(ba_errs):.1f}")

            progress.tick("ok")
            n_ok += 1

    progress.finish()
    mani_path.write_text(json.dumps({
        "model":      args.model,
        "backend":    args.backend,
        "slug":       slug,
        "n":          args.n,
        "variants":   args.variants,
        "world":      world,
        "n_records":  len(work),
        "n_ok":       n_ok,
        "n_errors":   n_err,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status":     "complete",
    }, indent=2))
    print(f"Manifest → {mani_path}")


if __name__ == "__main__":
    main()
