#!/usr/bin/env python3
"""Run one model-by-arm shard of the frozen C=0–4 stable-response study."""

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
from engine.coin_city_stable_relationship_n100_fair import (
    C_CASE_LEVELS,
    EPISODES,
    EXPERIMENT,
    FUTURE_CALLS,
    HARD_CALL_CAP,
    PROMPT_ARMS,
    validate_arm_prompt,
)


RUN = ROOT / "data" / EXPERIMENT
DESIGN = RUN / "design"
OUTDIR = RUN / "responses"
MAX_TASKS_PER_SHARD = EPISODES * len(C_CASE_LEVELS)
base.PROMPT_ARMS = PROMPT_ARMS
base._RUN = RUN
base._DESIGN = DESIGN
base._OUTDIR = OUTDIR
base._TASK_ID = re.compile(r"^coincityfair100v3_(\d{4})_c([0-4])$")
base.validate_arm_prompt = validate_arm_prompt


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
    if FUTURE_CALLS != 4_500 or FUTURE_CALLS > HARD_CALL_CAP:
        raise SystemExit("stable-response call-cap invariant failed")
    if "--arm" not in sys.argv or "--model" not in sys.argv:
        raise SystemExit("--model and --arm are required")
    arm = sys.argv[sys.argv.index("--arm") + 1]
    model = sys.argv[sys.argv.index("--model") + 1]
    if arm not in PROMPT_ARMS:
        raise SystemExit(f"invalid arm: {arm}")
    if "--episodes" not in sys.argv:
        sys.argv.extend(["--episodes", *map(str, range(EPISODES))])
    if "--prefixes" not in sys.argv:
        sys.argv.extend(["--prefixes", *map(str, C_CASE_LEVELS)])
    if "--max-attempts" not in sys.argv:
        sys.argv.extend(["--max-attempts", "1"])
    if _values_after("--max-attempts") != ["1"]:
        raise SystemExit("exactly one API attempt per task is permitted")
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
        manifest_path = OUTDIR / f"responses_{base._slug(model)}_{arm}.manifest.json"
        record = json.loads(manifest_path.read_text())
        record.update(
            {
                "experiment": EXPERIMENT,
                "planned_calls_all_shards": FUTURE_CALLS,
                "hard_call_cap": HARD_CALL_CAP,
                "max_calls_this_shard": MAX_TASKS_PER_SHARD,
                "city_c_case_levels": list(C_CASE_LEVELS),
                "context_alignment": "100_percent_correct_news_environment",
                "execution_mode": "local_login_node",
            }
        )
        manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
