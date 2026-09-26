#!/usr/bin/env python3
"""Reproduce the appendix's frontier-model coin probe.

The API helpers live here deliberately: this paper artifact no longer imports
the superseded biased-news evaluation pipeline. The Azure and Anthropic request
semantics are unchanged from the original run.

Sampling at a non-zero temperature N times per model shows the model's central
tendency (and spread), not a single cherry-picked completion.

Usage (from biased_news/):
    export AZURE_AI_API_KEY=...        # gpt-5.4 + DeepSeek-V4-Pro
    export CLAUDE_AZURE_API_KEY=...    # claude-opus-4-8 (falls back to AZURE_AI_API_KEY)
    python eval/run_coin_probe.py                 # all 3 models, 8 samples each
    python eval/run_coin_probe.py --arm fair      # control: coin stipulated fair
    python eval/run_coin_probe.py --arm neutral   # control: unfamiliar coin, no carnival
    python eval/run_coin_probe.py --arm carnival_calc  # carnival, calculation allowed
    python eval/run_coin_probe.py --n 1 --temperature 0   # one deterministic answer each
    python eval/run_coin_probe.py --dry-run       # just print the prompt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from types import SimpleNamespace
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

_AZURE_ENDPOINT = "https://liv-forecast.services.ai.azure.com/openai/v1"
_CLAUDE_ENDPOINT = "https://liv-forecast.services.ai.azure.com/anthropic"
_MAX_RETRIES = 3
_RETRY_BASE_SECONDS = 4.0

# (model, endpoint, env-keys-in-priority-order, is_claude)
MODELS = [
    ("claude-opus-4-8", _CLAUDE_ENDPOINT, ["CLAUDE_AZURE_API_KEY", "AZURE_AI_API_KEY"], True),
    ("gpt-5.4",         _AZURE_ENDPOINT,  ["AZURE_AI_API_KEY"],                          False),
    ("DeepSeek-V4-Pro", _AZURE_ENDPOINT,  ["DEEPSEEK_AZURE_API_KEY", "AZURE_AI_API_KEY"], False),
]

# NEUTRAL prompt: establish only that this is an UNFAMILIAR coin and that the two
# observed flips are the ONLY information available — without ever raising the
# fair/rigged framing ourselves (saying "a coin" alone cues the textbook fair-coin
# reflex; saying "we don't know if it's fair" would steer the other way). We just
# report what was seen and ask for the next-flip probability, so the number reflects
# the model's own prior, not ours.
# The original probe collected one arm only. On its own that arm cannot say what
# the carnival framing contributes: two heads from an unfamiliar coin already
# imply .75 under Laplace's rule, with no world knowledge involved, so a reading
# near .68 is inside ordinary sequential updating. These control arms separate
# the accounts. FAIR stipulates a fair coin, so a model that still departs from
# .5 is not reasoning about bias at all; NEUTRAL removes the carnival while
# keeping the coin unfamiliar, isolating the framing; CALC repeats the carnival
# arm without the no-calculation instruction, which in the original forecloses
# both the .5 and the .75 answers it is contrasted against.
ARMS = {}

PROMPT = (
    "You are at a carnival game. A coin you have never seen before is flipped twice in "
    "front of you, and it comes up heads both times. These two flips are everything you "
    "know about this coin.\n\n"
    "What is the probability that the next flip comes up heads?\n\n"
    "Think about it intuitively, the way you actually would in the moment — do NOT use "
    "any formula, rule, or calculation, and do not work out any numbers in your "
    "reasoning. Just reason in plain words about what you'd expect and why. Then END "
    "your reply with your gut answer as a JSON object on its own line:\n"
    '{"rationale": "one short sentence, no math", "p_heads": <number between 0 and 1>}'
)

_TAIL = PROMPT[PROMPT.index("What is the probability"):]

ARMS["carnival"] = PROMPT
ARMS["fair"] = (
    "You are at a carnival game. A coin that you have been reliably told is a fair "
    "coin is flipped twice in front of you, and it comes up heads both times. These "
    "two flips are everything else you know about this coin.\n\n" + _TAIL
)
ARMS["neutral"] = (
    "A coin you have never seen before is flipped twice in front of you, and it comes "
    "up heads both times. These two flips are everything you know about this "
    "coin.\n\n" + _TAIL
)
ARMS["carnival_calc"] = PROMPT.replace(
    "Think about it intuitively, the way you actually would in the moment \u2014 do NOT use "
    "any formula, rule, or calculation, and do not work out any numbers in your "
    "reasoning. Just reason in plain words about what you'd expect and why. Then END ",
    "Reason it through however you find natural. Then END ",
)


def parse_p(raw: str):
    """Extract p_heads from the last valid flat JSON object containing it."""
    val = None
    for m in re.finditer(r"\{[^{}]*\}", raw, re.DOTALL):
        try:
            o = json.loads(m.group())
        except Exception:
            continue
        if isinstance(o, dict) and o.get("p_heads") is not None:
            try:
                val = float(o["p_heads"])
            except Exception:
                pass
    return val


def _strip_reasoning(text: str) -> str:
    """Remove hidden-reasoning blocks before parsing the public answer."""
    return re.sub(r"(?is)<think>.*?</think>", "", text).strip()


def _call_with_retry(fn, *args, **kwargs):
    """Retry transient OpenAI-compatible API failures with bounded backoff."""
    from openai import APIError, APITimeoutError, RateLimitError

    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except RateLimitError:
            wait = _RETRY_BASE_SECONDS * (2**attempt)
            print(f"\n    [rate-limit] sleeping {wait:.0f}s …", end="", flush=True)
            time.sleep(wait)
        except APITimeoutError:
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_RETRY_BASE_SECONDS * (2**attempt))
        except APIError as exc:
            if getattr(exc, "status_code", None) == 408 or attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_RETRY_BASE_SECONDS * (2**attempt))
    raise RuntimeError("maximum retries exceeded")


def _is_reasoning_azure(model: str) -> bool:
    return model.lower().startswith(("gpt-5", "o1", "o3", "o4"))


def chat_call(client, model, messages, max_tokens, temperature):
    """Call Anthropic Foundry or an OpenAI-compatible Azure deployment."""
    if "claude" in model.lower():
        response = _call_with_retry(
            client.messages.create,
            model=model,
            messages=messages,
            max_tokens=max(max_tokens, 1024),
        )
        text = response.content[0].text if getattr(response, "content", None) else ""
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )

    request = {"model": model, "messages": messages}
    if _is_reasoning_azure(model):
        request["max_completion_tokens"] = max(max_tokens, 4096)
    else:
        request["max_tokens"] = max_tokens
        request["temperature"] = temperature
    return _call_with_retry(client.chat.completions.create, **request)


def build_client(model, endpoint, api_key, is_claude):
    if is_claude:
        from anthropic import AnthropicFoundry
        return AnthropicFoundry(azure_ad_token_provider=lambda: api_key, base_url=endpoint)
    from openai import OpenAI
    return OpenAI(
        base_url=endpoint,
        api_key="placeholder",
        default_headers={"api-key": api_key},
        timeout=300.0,
        max_retries=2,
    )


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--n", type=int, default=8, help="samples per model")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of model names to run (default: all 3)")
    ap.add_argument("--out", type=Path, default=_ROOT / "data" / "coin_probe",
                    help="dir for the saved JSONL log (one record per sample)")
    ap.add_argument("--no-save", action="store_true", help="print only, do not write JSONL")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--arm", default="carnival", choices=sorted(ARMS),
                    help="prompt arm; 'carnival' reproduces the frozen appendix run")
    args = ap.parse_args()

    prompt = ARMS[args.arm]
    if args.dry_run:
        print(prompt)
        return

    todo = [m for m in MODELS if args.models is None or m[0] in args.models]
    messages = [{"role": "user", "content": prompt}]

    fh = None
    if not args.no_save:
        from datetime import datetime, timezone
        args.out.mkdir(parents=True, exist_ok=True)
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        suffix = "" if args.arm == "carnival" else f"_{args.arm}"
        out_path = args.out / f"coin_probe{suffix}_{date}.jsonl"
        fh = out_path.open("w")
        # prompt provenance lands as the first line so the file is self-describing
        fh.write(json.dumps({"_meta": True, "arm": args.arm, "prompt": prompt, "n": args.n,
                             "temperature": args.temperature, "max_tokens": args.max_tokens,
                             "created_at": datetime.now(timezone.utc).isoformat()}) + "\n")
        print(f"Saving samples → {out_path}")

    for model, endpoint, env_keys, is_claude in todo:
        api_key = next((os.environ[k] for k in env_keys if os.environ.get(k)), "")
        print(f"\n=== {model} ===")
        if not api_key:
            print(f"  SKIP — no key in {' / '.join(env_keys)}")
            continue
        try:
            client = build_client(model, endpoint, api_key, is_claude)
        except Exception as e:
            print(f"  client error: {e}")
            continue

        vals = []
        for i in range(args.n):
            raw = ""
            try:
                resp = chat_call(client, model, messages, args.max_tokens, args.temperature)
                raw = _strip_reasoning((resp.choices[0].message.content or "").strip())
                p = parse_p(raw)
            except Exception as e:
                print(f"  [{i+1}] error: {e}")
                continue
            # pull the rationale for the first sample so we can see the reasoning
            rat = ""
            m = re.search(r'"rationale"\s*:\s*"([^"]*)"', raw)
            if m:
                rat = m.group(1)
            tag = f"p={p:.3f}" if p is not None else "p=NA (unparsed)"
            print(f"  [{i+1}] {tag}   {rat}")
            if p is not None:
                vals.append(p)
            if fh is not None:
                fh.write(json.dumps({"model": model, "sample": i + 1, "p_heads": p,
                                     "rationale": rat, "raw": raw}) + "\n")

        if vals:
            print(f"  --> n={len(vals)}  mean={statistics.mean(vals):.3f}  "
                  f"median={statistics.median(vals):.3f}  "
                  f"min={min(vals):.3f}  max={max(vals):.3f}")
        else:
            print("  --> no parseable answers")

    if fh is not None:
        fh.close()


if __name__ == "__main__":
    main()
