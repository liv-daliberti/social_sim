#!/usr/bin/env python3
"""Run one model shard of the 100%-aligned context rerun."""

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
from build_coin_city_context100_rerun import (
    ARM,
    CUMULATIVE_CALLS,
    DESIGN,
    EXPERIMENT,
    NEW_CALLS,
    RUN,
    validate_context100_prompt,
)
from engine.coin_city_llm_pilot import EPISODES, HARD_CALL_CAP, ROUNDS


OUTDIR = RUN / "responses"
MAX_TASKS = EPISODES * len(ROUNDS)
base.PROMPT_ARMS = (ARM,)
base._RUN = RUN
base._DESIGN = DESIGN
base._OUTDIR = OUTDIR
base._TASK_ID = re.compile(r"^coincityp1_(\d{4})_k([1-5])$")
base.validate_arm_prompt = validate_context100_prompt


def _values_after(flag: str) -> list[str]:
    if flag not in sys.argv:
        return []
    values = []
    for value in sys.argv[sys.argv.index(flag) + 1 :]:
        if value.startswith("--"):
            break
        values.append(value)
    return values


def main() -> None:
    if NEW_CALLS != 600 or CUMULATIVE_CALLS != 2400 or CUMULATIVE_CALLS > HARD_CALL_CAP:
        raise SystemExit("aligned-context call cap invariant failed")
    if "--arm" not in sys.argv:
        sys.argv.extend(["--arm", ARM])
    elif sys.argv[sys.argv.index("--arm") + 1] != ARM:
        raise SystemExit("only the aligned-context arm is permitted")
    if "--episodes" not in sys.argv:
        sys.argv.extend(["--episodes", *map(str, range(EPISODES))])
    if "--prefixes" not in sys.argv:
        sys.argv.extend(["--prefixes", *map(str, ROUNDS)])
    if "--max-attempts" not in sys.argv:
        sys.argv.extend(["--max-attempts", "1"])
    if _values_after("--max-attempts") != ["1"]:
        raise SystemExit("exactly one API attempt per task is permitted")
    episodes = {int(value) for value in _values_after("--episodes")}
    prefixes = {int(value) for value in _values_after("--prefixes")}
    if not episodes <= set(range(EPISODES)) or not prefixes <= set(ROUNDS):
        raise SystemExit("selection lies outside the frozen extension")
    if len(episodes) * len(prefixes) > MAX_TASKS:
        raise SystemExit("selection exceeds the 200-call model-shard cap")
    if "--tasks" not in sys.argv:
        sys.argv.extend(["--tasks", str(DESIGN / f"tasks_{ARM}.jsonl")])
    base.main()
    if "--dry-run" not in sys.argv:
        for path in OUTDIR.glob("responses_*.manifest.json"):
            record = json.loads(path.read_text())
            record.update(
                {
                    "experiment": EXPERIMENT,
                    "new_calls_all_models": NEW_CALLS,
                    "cumulative_test_calls": CUMULATIVE_CALLS,
                    "hard_call_cap": HARD_CALL_CAP,
                }
            )
            path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
