#!/usr/bin/env python3
"""Run one model on the additive misleading-cue arm of the frozen Coin City design.

Each deployment is called with the *same* client, protocol, and sampling
configuration as its truthful-arm run, so the wrong-context arm differs from the
``abc_context`` arm only in the substituted City C sentence. The per-model
settings below are copied from the individual frozen runners; changing one here
would break comparability with that model's other arms.

    python eval/run_coin_city_wrong_context.py --model claude-opus-5 --prefixes 0
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import run_frozen_task_shard as base  # noqa: E402
from engine.coin_city_stable_relationship_claude_n250 import (  # noqa: E402
    C_CASE_LEVELS,
    EPISODES,
    EXPERIMENT,
    HARD_CALL_CAP,
)
from engine.coin_city_wrong_context_arm import (  # noqa: E402
    ARM,
    CONTEXT_ALIGNMENT,
    PROMPT_ARMS,
    validate_arm_prompt,
)


RUN = ROOT / "data" / EXPERIMENT
DESIGN = RUN / "design"
OUTDIR = RUN / "responses"
MAX_TASKS_PER_SHARD = EPISODES * len(C_CASE_LEVELS)
PLANNED_CALLS_PER_MODEL = EPISODES * len(C_CASE_LEVELS)

ANTHROPIC_ENDPOINT = "https://liv.services.ai.azure.com/anthropic"
OPENAI_ENDPOINT = "https://liv.services.ai.azure.com/openai/v1"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Per-model settings, copied from each model's frozen truthful-arm runner.
MODEL_PROTOCOLS = {
    "claude-opus-4-8": {
        "protocol": "anthropic_messages",
        "endpoint": ANTHROPIC_ENDPOINT,
        "key_env": "AZURE_AI_API_KEY",
        "max_output_tokens": 1_024,
    },
    "claude-opus-5": {
        "protocol": "anthropic_messages",
        "endpoint": ANTHROPIC_ENDPOINT,
        "key_env": "AZURE_AI_API_KEY",
        "max_output_tokens": 4_096,
    },
    "gpt-5.6-sol": {
        "protocol": "openai_responses",
        "endpoint": OPENAI_ENDPOINT,
        "key_env": "AZURE_AI_API_KEY",
        "max_output_tokens": 4_096,
        "reasoning_effort": "low",
    },
    "DeepSeek-V4-Pro": {
        "protocol": "openai_chat",
        "endpoint": OPENAI_ENDPOINT,
        "key_env": "AZURE_AI_API_KEY",
        "max_output_tokens": 4_096,
        "temperature": 0.0,
    },
    "FW-Kimi-K3": {
        "protocol": "openai_chat",
        "endpoint": OPENAI_ENDPOINT,
        "key_env": "AZURE_AI_API_KEY",
        "max_output_tokens": 16_384,
        "temperature": 0.0,
        "reasoning_effort": "low",
    },
    "gemini-3.6-flash": {
        "protocol": "openai_chat_completion_tokens",
        "endpoint": GEMINI_ENDPOINT,
        "key_env": "GEMINI_API_KEY",
        "max_output_tokens": 16_384,
        "temperature": 0.0,
        "reasoning_effort": "low",
    },
}

MAX_PRE_MODEL_RETRIES = 12
_RETRY_STATUS = (408, 429, 500, 502, 503, 504)
_RETRY_AFTER = re.compile(r"retry[- ]after[\"']?[:=]\s*[\"']?([0-9.]+)", re.I)
_RETRY_HINT = re.compile(r"[Pp]lease retry in ([0-9.]+)s")
_TRANSPORT_RETRIES = 0

base.PROMPT_ARMS = PROMPT_ARMS
base.PREFIX_LADDER = C_CASE_LEVELS
base.EXPERIMENT = EXPERIMENT
base._RUN = RUN
base._DESIGN = DESIGN
base._OUTDIR = OUTDIR
base._TASK_ID = re.compile(r"^coincityclaude250v4_(\d{4})_c([0-4])$")
base.validate_arm_prompt = validate_arm_prompt
for _name, _config in MODEL_PROTOCOLS.items():
    base.MODEL_CONFIG[_name] = {
        "endpoint": _config["endpoint"],
        "keys": (_config["key_env"],),
        "claude": _config["protocol"] == "anthropic_messages",
    }


def _values_after(flag: str) -> list[str]:
    if flag not in sys.argv:
        return []
    values = []
    for value in sys.argv[sys.argv.index(flag) + 1 :]:
        if value.startswith("--"):
            break
        values.append(value)
    return values


def _make_client(model: str, *, endpoint_override: Optional[str], timeout: float):
    config = MODEL_PROTOCOLS[model]
    key = os.environ.get(config["key_env"], "")
    if not key:
        raise RuntimeError(f"no API key found in {config['key_env']}")
    endpoint = endpoint_override or config["endpoint"]
    if config["protocol"] == "anthropic_messages":
        from anthropic import AnthropicFoundry

        client = AnthropicFoundry(
            azure_ad_token_provider=lambda: key,
            base_url=endpoint,
            timeout=timeout,
            max_retries=0,
        )
    else:
        from openai import OpenAI

        client = OpenAI(
            base_url=endpoint, api_key=key, timeout=timeout, max_retries=0
        )
    return client, endpoint


def _answer_text(message) -> str:
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) == "text":
            return block.text or ""
    return ""


def _shim(text: str, finish: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text or ""), finish_reason=finish
            )
        ]
    )


def _issue(client, model: str, messages, max_tokens: int):
    config = MODEL_PROTOCOLS[model]
    ceiling = max(max_tokens, config["max_output_tokens"])
    protocol = config["protocol"]
    if protocol == "anthropic_messages":
        message = client.messages.create(
            model=model, messages=messages, max_tokens=ceiling
        )
        return _shim(_answer_text(message), getattr(message, "stop_reason", "stop"))
    if protocol == "openai_responses":
        response = client.responses.create(
            model=model,
            input=messages,
            max_output_tokens=ceiling,
            reasoning={"effort": config["reasoning_effort"]},
        )
        return _shim(response.output_text or "", getattr(response, "status", "ok"))
    kwargs = {"model": model, "messages": messages, "temperature": config["temperature"]}
    if protocol == "openai_chat_completion_tokens":
        kwargs["max_completion_tokens"] = ceiling
    else:
        kwargs["max_tokens"] = ceiling
    if "reasoning_effort" in config:
        kwargs["reasoning_effort"] = config["reasoning_effort"]
    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    return _shim(choice.message.content or "", getattr(choice, "finish_reason", "stop"))


def _chat_call(client, model, messages, max_tokens, temperature):
    """One model attempt; requests rejected before the model runs are re-issued."""
    global _TRANSPORT_RETRIES
    del temperature  # per-model temperature comes from MODEL_PROTOCOLS
    for attempt in range(1, MAX_PRE_MODEL_RETRIES + 2):
        try:
            return _issue(client, model, messages, max_tokens)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status not in _RETRY_STATUS or attempt > MAX_PRE_MODEL_RETRIES:
                raise
            _TRANSPORT_RETRIES += 1
            hint = _RETRY_AFTER.search(str(exc)) or _RETRY_HINT.search(str(exc))
            delay = (
                min(120.0, float(hint.group(1)) + 1.0)
                if hint
                else min(120.0, 4.0 * 2 ** (attempt - 1))
            )
            print(
                f"pre-model retry {attempt} after HTTP {status}; sleeping "
                f"{delay:.1f}s (shard retries={_TRANSPORT_RETRIES})",
                flush=True,
            )
            time.sleep(delay + random.uniform(0.0, 2.0))
    raise RuntimeError("unreachable")


def _all_attempts_terminal(path: Path) -> tuple[set[str], set[str]]:
    terminal: set[str] = set()
    parsed: set[str] = set()
    if not path.exists():
        return terminal, parsed
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        task_id = record.get("task_id")
        if not task_id:
            continue
        terminal.add(task_id)
        if record.get("predicted_poll") is not None:
            parsed.add(task_id)
    return terminal, parsed


base._make_client = _make_client
base.chat_call = _chat_call
base._response_state = _all_attempts_terminal


def main() -> None:
    if PLANNED_CALLS_PER_MODEL > HARD_CALL_CAP:
        raise SystemExit("wrong-context call-cap invariant failed")
    if "--model" not in sys.argv:
        raise SystemExit("--model is required")
    model = sys.argv[sys.argv.index("--model") + 1]
    if model not in MODEL_PROTOCOLS:
        raise SystemExit(f"unsupported model: {model}")
    if "--arm" not in sys.argv:
        sys.argv.extend(["--arm", ARM])
    if sys.argv[sys.argv.index("--arm") + 1] != ARM:
        raise SystemExit(f"this runner serves only {ARM}")
    if "--episodes" not in sys.argv:
        sys.argv.extend(["--episodes", *map(str, range(EPISODES))])
    if "--prefixes" not in sys.argv:
        sys.argv.extend(["--prefixes", *map(str, C_CASE_LEVELS)])
    if "--max-attempts" not in sys.argv:
        sys.argv.extend(["--max-attempts", "1"])
    if _values_after("--max-attempts") != ["1"]:
        raise SystemExit("exactly one application-level API attempt is permitted")
    episodes = {int(value) for value in _values_after("--episodes")}
    c_levels = {int(value) for value in _values_after("--prefixes")}
    if not episodes <= set(range(EPISODES)):
        raise SystemExit("episode selection lies outside the frozen design")
    if not c_levels <= set(C_CASE_LEVELS):
        raise SystemExit("City C selection lies outside the frozen design")
    if len(episodes) * len(c_levels) > MAX_TASKS_PER_SHARD:
        raise SystemExit("selection exceeds the frozen shard size")
    if "--tasks" not in sys.argv:
        sys.argv.extend(["--tasks", str(DESIGN / f"tasks_{ARM}.jsonl")])

    base.main()
    if "--dry-run" not in sys.argv:
        outdir = OUTDIR
        if "--outdir" in sys.argv:
            outdir = Path(sys.argv[sys.argv.index("--outdir") + 1])
        manifest_path = (
            outdir / f"responses_{base._slug(model)}_{ARM}.manifest.json"
        )
        record = json.loads(manifest_path.read_text())
        config = MODEL_PROTOCOLS[model]
        record.update(
            {
                "experiment": EXPERIMENT,
                "arm": ARM,
                "context_alignment": CONTEXT_ALIGNMENT,
                "api_protocol": config["protocol"],
                "sampling_matches_truthful_arm": True,
                "effective_max_output_tokens": config["max_output_tokens"],
                "reasoning_effort": config.get("reasoning_effort"),
                "temperature": config.get("temperature"),
                "http_client_retries": 0,
                "pre_model_transport_retry_policy": {
                    "max_retries": MAX_PRE_MODEL_RETRIES,
                    "model_responses_never_repeated": True,
                    "retried_http_status": list(_RETRY_STATUS),
                    "shard_transport_retries": _TRANSPORT_RETRIES,
                },
                "planned_calls_this_model": PLANNED_CALLS_PER_MODEL,
                "execution_mode": "local_login_node",
            }
        )
        manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
