#!/usr/bin/env python3
"""Run one DeepSeek-V4-Pro arm on the frozen Coin City n=250 design.

This additive replication layer reuses the exact task files and prompt hashes
from the completed Claude Opus 4.8 run. DeepSeek is called through Foundry's
OpenAI-compatible Chat Completions API. Authentication uses Entra ID when
available, with an ephemeral project-key environment variable as fallback.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import run_frozen_task_shard as base
from engine.coin_city_stable_relationship_claude_n250 import (
    C_CASE_LEVELS,
    EPISODES,
    EXPERIMENT,
    HARD_CALL_CAP,
    PROMPT_ARMS,
    validate_arm_prompt,
)


MODEL = "DeepSeek-V4-Pro"
MODEL_LABEL = "DeepSeek-V4-Pro"
ENDPOINT = "https://liv.services.ai.azure.com/openai/v1"
PROJECT_ENDPOINT = "https://liv.services.ai.azure.com/api/projects/proj-default"
KEY_ENV = "DEEPSEEK_V4_PRO_AZURE_API_KEY"
MAX_OUTPUT_TOKENS = 4096
PLANNED_CALLS = EPISODES * len(C_CASE_LEVELS) * len(PROMPT_ARMS)

RUN = ROOT / "data" / EXPERIMENT
DESIGN = RUN / "design"
OUTDIR = RUN / "responses"
MAX_TASKS_PER_SHARD = EPISODES * len(C_CASE_LEVELS)

base.PROMPT_ARMS = PROMPT_ARMS
base.PREFIX_LADDER = C_CASE_LEVELS
base.EXPERIMENT = EXPERIMENT
base._RUN = RUN
base._DESIGN = DESIGN
base._OUTDIR = OUTDIR
base._TASK_ID = re.compile(r"^coincityclaude250v4_(\d{4})_c([0-4])$")
base.validate_arm_prompt = validate_arm_prompt
base.MODEL_CONFIG[MODEL] = {
    "endpoint": ENDPOINT,
    "keys": (KEY_ENV,),
    "claude": False,
}

_AUTH_MODE: Optional[str] = None


def _values_after(flag: str) -> list[str]:
    if flag not in sys.argv:
        return []
    values = []
    for value in sys.argv[sys.argv.index(flag) + 1 :]:
        if value.startswith("--"):
            break
        values.append(value)
    return values


def _make_client(
    model: str,
    *,
    endpoint_override: Optional[str],
    timeout: float,
):
    global _AUTH_MODE
    if model != MODEL:
        raise RuntimeError(f"unsupported model: {model}")
    from openai import OpenAI

    endpoint = endpoint_override or ENDPOINT
    project_key = os.environ.get(KEY_ENV, "")
    if project_key:
        _AUTH_MODE = "project_api_key"
        client = OpenAI(
            base_url=endpoint,
            api_key=project_key,
            timeout=timeout,
            max_retries=0,
        )
    else:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        _AUTH_MODE = "entra_id"
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://ai.azure.com/.default",
        )
        client = OpenAI(
            base_url=endpoint,
            api_key=token_provider,
            timeout=timeout,
            max_retries=0,
        )
    return client, endpoint


def _chat_completions_call(client, model, messages, max_tokens, temperature):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max(max_tokens, MAX_OUTPUT_TOKENS),
        temperature=temperature,
    )


def _all_attempts_terminal(path: Path) -> tuple[set[str], set[str]]:
    """Never repeat a task after its single recorded application-level attempt."""
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
base.chat_call = _chat_completions_call
base._response_state = _all_attempts_terminal


def main() -> None:
    if PLANNED_CALLS != 3_750 or PLANNED_CALLS > HARD_CALL_CAP:
        raise SystemExit("DeepSeek replication call-cap invariant failed")
    if "--arm" not in sys.argv or "--model" not in sys.argv:
        raise SystemExit("--model and --arm are required")
    arm = sys.argv[sys.argv.index("--arm") + 1]
    model = sys.argv[sys.argv.index("--model") + 1]
    if arm not in PROMPT_ARMS:
        raise SystemExit(f"invalid arm: {arm}")
    if model != MODEL:
        raise SystemExit(f"invalid DeepSeek deployment: {model}")
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
        sys.argv.extend(["--tasks", str(DESIGN / f"tasks_{arm}.jsonl")])

    base.main()
    if "--dry-run" not in sys.argv:
        manifest_outdir = OUTDIR
        if "--outdir" in sys.argv:
            manifest_outdir = Path(sys.argv[sys.argv.index("--outdir") + 1])
        manifest_path = (
            manifest_outdir
            / f"responses_{base._slug(model)}_{arm}.manifest.json"
        )
        record = json.loads(manifest_path.read_text())
        record.update(
            {
                "experiment": EXPERIMENT,
                "replication_model": MODEL_LABEL,
                "deployment": MODEL,
                "api_protocol": "OpenAI Chat Completions API",
                "auth_mode": _AUTH_MODE,
                "project_endpoint": PROJECT_ENDPOINT,
                "planned_calls_all_shards": PLANNED_CALLS,
                "hard_call_cap": HARD_CALL_CAP,
                "max_calls_this_shard": MAX_TASKS_PER_SHARD,
                "city_c_case_levels": list(C_CASE_LEVELS),
                "context_alignment": "100_percent_correct_news_environment",
                "execution_mode": "local_login_node",
                "effective_max_output_tokens": MAX_OUTPUT_TOKENS,
                "http_client_retries": 0,
            }
        )
        manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
