#!/usr/bin/env python3
"""Independent GPT-5.6 repeat of the three frozen Coin City k=0 arms."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import run_coin_city_stable_relationship_gpt56 as shared  # noqa: E402
import run_frozen_task_shard as base  # noqa: E402
from engine.coin_city_symbol_context_arm import (  # noqa: E402
    validate_arm_prompt as validate_symbol_arm_prompt,
)

ARMS = ("abc_no_context", "abc_context", "abc_symbol_context")
PREFIXES = (0,)
PLANNED_CALLS = shared.EPISODES * len(ARMS)
OUTDIR = shared.RUN / "responses" / "gpt56_k0_repeat_20260826"

# Preserve the exact original deployment, API, parser, one-attempt policy, task
# ordering, and inference settings; isolate only the response destination and
# restrict selection to the registered k=0 repeat.
base.PROMPT_ARMS = ARMS
base.PREFIX_LADDER = PREFIXES
base._OUTDIR = OUTDIR
base.validate_arm_prompt = validate_symbol_arm_prompt


def values_after(flag: str) -> list[str]:
    if flag not in sys.argv:
        return []
    values: list[str] = []
    for value in sys.argv[sys.argv.index(flag) + 1 :]:
        if value.startswith("--"):
            break
        values.append(value)
    return values


def main() -> None:
    if PLANNED_CALLS != 750:
        raise SystemExit("GPT-5.6 k=0 repeat call-cap invariant failed")
    if "--arm" not in sys.argv or "--model" not in sys.argv:
        raise SystemExit("--model and --arm are required")
    arm = sys.argv[sys.argv.index("--arm") + 1]
    model = sys.argv[sys.argv.index("--model") + 1]
    if arm not in ARMS:
        raise SystemExit(f"invalid repeat arm: {arm}")
    if model != shared.MODEL:
        raise SystemExit(f"invalid repeat deployment: {model}")
    if "--episodes" not in sys.argv:
        sys.argv.extend(["--episodes", *map(str, range(shared.EPISODES))])
    if "--prefixes" not in sys.argv:
        sys.argv.extend(["--prefixes", "0"])
    if values_after("--prefixes") != ["0"]:
        raise SystemExit("the registered repeat permits only City-C k=0")
    if "--max-attempts" not in sys.argv:
        sys.argv.extend(["--max-attempts", "1"])
    if values_after("--max-attempts") != ["1"]:
        raise SystemExit("exactly one application-level attempt is permitted")
    episodes = {int(value) for value in values_after("--episodes")}
    if not episodes <= set(range(shared.EPISODES)):
        raise SystemExit("episode selection lies outside the frozen design")
    if "--tasks" not in sys.argv:
        sys.argv.extend(["--tasks", str(shared.DESIGN / f"tasks_{arm}.jsonl")])

    base.main()
    if "--dry-run" in sys.argv:
        return
    manifest_path = OUTDIR / f"responses_{base._slug(model)}_{arm}.manifest.json"
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    record.update(
        {
            "protocol_version": "coin_city_robustness_v1",
            "classification": "post_hoc_run_to_run_repeat",
            "repeat_of": "original GPT-5.6 low-reasoning deployment",
            "deployment": shared.MODEL,
            "api_protocol": "OpenAI Responses API",
            "auth_mode": shared._AUTH_MODE,
            "project_endpoint": shared.PROJECT_ENDPOINT,
            "planned_calls_all_arms": PLANNED_CALLS,
            "city_c_case_levels": [0],
            "reasoning_effort": shared.REASONING_EFFORT,
            "effective_max_output_tokens": shared.MAX_OUTPUT_TOKENS,
            "http_client_retries": 0,
            "provider_native_sampling": True,
        }
    )
    manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
