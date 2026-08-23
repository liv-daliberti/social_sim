#!/usr/bin/env python3
"""Run one model-by-arm shard of the capped coin-to-city LLM pilot."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import run_three_city_c2_v9_confirmatory as base
from engine.coin_city_llm_pilot import (
    EPISODES,
    EXPERIMENT,
    HARD_CALL_CAP,
    PLANNED_CALLS,
    PROMPT_ARMS,
    ROUNDS,
    validate_arm_prompt,
)


RUN = ROOT / "data" / EXPERIMENT
DESIGN = RUN / "design"
OUTDIR = RUN / "responses"
MAX_TASKS_PER_SHARD = EPISODES * len(ROUNDS)

base.PROMPT_ARMS = PROMPT_ARMS
base._RUN = RUN
base._DESIGN = DESIGN
base._OUTDIR = OUTDIR
base._TASK_ID = re.compile(r"^coincityp1_(\d{4})_k([1-5])$")
base.validate_arm_prompt = validate_arm_prompt


def _values_after(flag: str) -> list[str]:
    if flag not in sys.argv:
        return []
    start = sys.argv.index(flag) + 1
    values = []
    for value in sys.argv[start:]:
        if value.startswith("--"):
            break
        values.append(value)
    return values


def main() -> None:
    if PLANNED_CALLS != 1800 or PLANNED_CALLS > HARD_CALL_CAP:
        raise SystemExit("hard call-budget invariant failed")
    if "--arm" not in sys.argv:
        raise SystemExit("--arm is required")
    arm = sys.argv[sys.argv.index("--arm") + 1]
    if arm not in PROMPT_ARMS:
        raise SystemExit(f"invalid arm: {arm}")
    if "--episodes" not in sys.argv:
        sys.argv.extend(["--episodes", *map(str, range(EPISODES))])
    if "--prefixes" not in sys.argv:
        sys.argv.extend(["--prefixes", *map(str, ROUNDS)])
    if "--max-attempts" not in sys.argv:
        sys.argv.extend(["--max-attempts", "1"])
    if int(_values_after("--max-attempts")[0]) != 1:
        raise SystemExit("pilot permits exactly one API attempt per task")
    episodes = {int(value) for value in _values_after("--episodes")}
    prefixes = {int(value) for value in _values_after("--prefixes")}
    if not episodes <= set(range(EPISODES)) or not prefixes <= set(ROUNDS):
        raise SystemExit("selection lies outside the frozen pilot")
    requested = len(episodes) * len(prefixes)
    if requested > MAX_TASKS_PER_SHARD:
        raise SystemExit("worker selection exceeds the 200-call shard cap")
    if "--tasks" not in sys.argv:
        sys.argv.extend(["--tasks", str(DESIGN / f"tasks_{arm}.jsonl")])
    base.main()
    if "--dry-run" not in sys.argv and OUTDIR.exists():
        for path in OUTDIR.glob("responses_*.manifest.json"):
            record = json.loads(path.read_text())
            record.update(
                {
                    "experiment": EXPERIMENT,
                    "hard_call_cap": HARD_CALL_CAP,
                    "planned_calls_all_shards": PLANNED_CALLS,
                    "max_calls_this_shard": MAX_TASKS_PER_SHARD,
                }
            )
            path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
