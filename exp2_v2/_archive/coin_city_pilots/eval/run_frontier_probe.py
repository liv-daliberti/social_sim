#!/usr/bin/env python3
"""Diagnostic: can a frontier Azure model be reached on the recovery prompt, with the
CORRECT per-model endpoint + client?  GPT/DeepSeek use the OpenAI client; Claude uses the
Anthropic SDK (AnthropicFoundry.messages.create).  Prints per-call timing + finish_reason +
parse, UNBUFFERED."""
import argparse, os, sys, time
from pathlib import Path
_HERE = Path(__file__).resolve().parent; _ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_HERE))
from engine.news_response import generate_episode
from run_batch import make_client, _strip_reasoning
from run_recovery_probe import chat_call, parse
import run_recovery_curve_parity as rcp

MODEL_CFG = {
    "claude-opus-4-8": dict(endpoint="https://liv-forecast.services.ai.azure.com/anthropic",
                            keys=["AZURE_AI_API_KEY", "CLAUDE_AZURE_API_KEY"], client="anthropic"),
    "DeepSeek-V4-Pro": dict(endpoint="https://cos-tiktok-annotation-a-resource.services.ai.azure.com/openai/v1",
                            keys=["DEEPSEEK_AZURE_API_KEY", "CLAUDE_AZURE_API_KEY"], client="openai"),
    "gpt-5.4": dict(endpoint="https://liv-forecast.services.ai.azure.com/openai/v1",
                    keys=["AZURE_AI_API_KEY"], client="openai"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--backend", default="azure"); ap.add_argument("--azure-auth", action="store_true")
    ap.add_argument("--endpoint", default=None); ap.add_argument("--api-key", default="")
    ap.add_argument("--n", type=int, default=1); ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.1); ap.add_argument("--delay", type=float, default=0.0)
    a = ap.parse_args()
    cfg = MODEL_CFG.get(a.model, MODEL_CFG["gpt-5.4"])
    endpoint = a.endpoint or cfg["endpoint"]
    key = a.api_key or next((os.environ[e] for e in cfg["keys"] if os.environ.get(e)), "")
    print(f"model={a.model} client={cfg['client']} endpoint={endpoint} key_set={bool(key)} "
          f"max_tokens={a.max_tokens}", flush=True)

    if cfg["client"] == "anthropic":
        from anthropic import AnthropicFoundry
        client = AnthropicFoundry(azure_ad_token_provider=lambda: key, base_url=endpoint)
    else:
        client = make_client("azure", endpoint=endpoint, api_key=key, azure_auth=True)

    ep = generate_episode(seed=0, T=10)
    for k in (3, 6, 10):
        prompt = rcp.build_prompt(ep["news"][:k], ep["polls"][:k], 10)
        t = time.time()
        try:
            if cfg["client"] == "anthropic":
                resp = client.messages.create(model=a.model, max_tokens=a.max_tokens,
                                              messages=[{"role": "user", "content": prompt}])
                out = resp.content[0].text; fr = resp.stop_reason
            else:
                resp = chat_call(client, a.model, [{"role": "user", "content": prompt}], a.max_tokens, a.temperature)
                ch = resp.choices[0]; out = ch.message.content or ""; fr = ch.finish_reason
            dt = time.time() - t
            pred, rate = parse(_strip_reasoning((out or "").strip()))
            print(f"k={k}: {dt:.0f}s  finish={fr}  pred={pred}  rate={rate}  chars={len(out or '')}  "
                  f"preview={(out or '')[:90]!r}", flush=True)
        except Exception as e:
            print(f"k={k}: ERROR after {time.time()-t:.0f}s: {type(e).__name__}: {str(e)[:160]}", flush=True)


if __name__ == "__main__":
    main()
