#!/usr/bin/env python3
"""Build the misleading-cue task file for the frozen Coin City n=250 design.

Writes ``design/tasks_abc_wrong_context.jsonl`` plus a manifest. The frozen
files are read, never written. Every generated prompt is checked against its
``abc_context`` counterpart: the two must differ by exactly the one substituted
City C context sentence, so the arms remain numerically identical.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from engine.coin_city_stable_relationship_claude_n250 import (  # noqa: E402
    C_CASE_LEVELS,
    EPISODES,
    EXPERIMENT,
    TARGET_CONTEXT_STRONG,
    TARGET_CONTEXT_WEAK,
    task_id,
)
from engine.coin_city_wrong_context_arm import (  # noqa: E402
    ARM,
    CONTEXT_ALIGNMENT,
    make_prompt,
    target_background,
    truthful_background,
    validate_arm_prompt,
)


RUN = ROOT / "data" / EXPERIMENT
DESIGN = RUN / "design"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    episodes = {row["episode"]: row for row in _read_jsonl(DESIGN / "episodes.jsonl")}
    truthful = {
        row["task_id"]: row["prompt"]
        for row in _read_jsonl(DESIGN / "tasks_abc_context.jsonl")
    }
    if len(episodes) != EPISODES:
        raise SystemExit(f"expected {EPISODES} episodes, found {len(episodes)}")

    records = []
    flipped_strong_to_weak = 0
    for index in range(EPISODES):
        episode = episodes[index]
        for c_cases in C_CASE_LEVELS:
            prompt = make_prompt(episode, c_cases)
            validate_arm_prompt(prompt, ARM)
            identifier = task_id(index, c_cases)

            # The only permitted difference from the truthful arm is the swap of
            # the one context sentence; the numbers must be untouched.
            expected = truthful[identifier].replace(
                truthful_background(episode), target_background(episode)
            )
            if prompt != expected:
                raise SystemExit(f"prompt differs beyond the cue swap: {identifier}")
            if target_background(episode) == truthful_background(episode):
                raise SystemExit(f"cue was not flipped: {identifier}")
            if episode["cue_strong"]:
                flipped_strong_to_weak += 1

            records.append(
                {
                    "task_id": identifier,
                    "prompt": prompt,
                    "prompt_sha256": _sha256(prompt),
                }
            )

    out_path = DESIGN / f"tasks_{ARM}.jsonl"
    out_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )

    manifest = {
        "arm": ARM,
        "experiment": EXPERIMENT,
        "context_alignment": CONTEXT_ALIGNMENT,
        "derived_from": "tasks_abc_context.jsonl",
        "difference_from_truthful_arm": (
            "the City C context sentence is replaced by the description of the "
            "opposite response regime; all numeric rows, the query, and the "
            "shared instructions are byte-identical"
        ),
        "context_sentences": {
            "strong": TARGET_CONTEXT_STRONG,
            "weak": TARGET_CONTEXT_WEAK,
        },
        "episodes": EPISODES,
        "city_c_case_levels": list(C_CASE_LEVELS),
        "tasks": len(records),
        "flipped_strong_to_weak": flipped_strong_to_weak,
        "flipped_weak_to_strong": len(records) - flipped_strong_to_weak,
        "task_file_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "frozen_truthful_task_file_sha256": hashlib.sha256(
            (DESIGN / "tasks_abc_context.jsonl").read_bytes()
        ).hexdigest(),
    }
    manifest_path = DESIGN / f"tasks_{ARM}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
