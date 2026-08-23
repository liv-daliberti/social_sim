#!/usr/bin/env python3
"""Build the fresh 4,500-call, 100%-aligned 100-episode expansion."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from engine.coin_city_expanded_aligned import (
    CASES_BY_ROUND,
    EPISODES,
    EXPERIMENT,
    HARD_CALL_CAP,
    MODELS,
    PLANNED_CALLS,
    PROMPT_ARMS,
    ROUNDS,
    SEED_BASE,
    TARGET_CONTEXT_HIGH,
    TARGET_CONTEXT_LOW,
    baseline_estimates,
    make_episode,
    make_prompt,
    prompt_sections,
    prompt_sha256,
    task_id,
    validate_arm_prompt,
)


RUN = ROOT / "data" / EXPERIMENT
DESIGN = RUN / "design"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if PLANNED_CALLS != 4_500 or PLANNED_CALLS > HARD_CALL_CAP:
        raise SystemExit("expanded-run call cap invariant failed")
    if (RUN / "responses").exists() and any((RUN / "responses").glob("*.jsonl")):
        raise SystemExit("refusing to rebuild after expanded-run responses exist")
    DESIGN.mkdir(parents=True, exist_ok=True)
    episodes = [make_episode(index) for index in range(EPISODES)]
    _write_jsonl(DESIGN / "episodes.jsonl", episodes)
    tasks = {arm: [] for arm in PROMPT_ARMS}
    keys = []
    for episode in episodes:
        for k in ROUNDS:
            identifier = task_id(episode["episode"], k)
            baselines = baseline_estimates(episode, k)
            keys.append(
                {
                    "task_id": identifier,
                    "episode": episode["episode"],
                    "k": k,
                    "city_c_cases": CASES_BY_ROUND[k],
                    "gold_expected_poll": episode["target_truth"],
                    "target_high": episode["target_high"],
                    "cue_high": episode["cue_high"],
                    "cue_correct": True,
                    "high_reference_city": episode["high_reference_city"],
                    "baselines": baselines,
                }
            )
            for arm in PROMPT_ARMS:
                prompt_text = make_prompt(episode, k, arm)
                validate_arm_prompt(prompt_text, arm)
                tasks[arm].append(
                    {
                        "task_id": identifier,
                        "prompt": prompt_text,
                        "prompt_sha256": prompt_sha256(prompt_text),
                    }
                )
    _write_jsonl(DESIGN / "answer_key.jsonl", keys)
    for arm, rows in tasks.items():
        _write_jsonl(DESIGN / f"tasks_{arm}.jsonl", rows)
    example = episodes[0]
    (DESIGN / "example_prompt_sections.json").write_text(
        json.dumps(
            {arm: prompt_sections(example, 2, arm) for arm in PROMPT_ARMS},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for arm in PROMPT_ARMS:
        (DESIGN / f"example_prompt_{arm}.txt").write_text(
            make_prompt(example, 2, arm) + "\n"
        )
    source_paths = {
        "engine": ROOT / "engine" / "coin_city_expanded_aligned.py",
        "builder": HERE / "build_coin_city_expanded_aligned.py",
        "validator": HERE / "validate_coin_city_expanded_aligned.py",
        "runner": HERE / "run_coin_city_expanded_aligned.py",
        "launcher": HERE / "run_coin_city_expanded_aligned_local.sh",
        "live": ROOT / "analysis" / "live_coin_city_expanded_aligned.py",
        "renderer": ROOT / "analysis" / "render_coin_city_llm_pilot_by_condition.py",
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise SystemExit("expanded-run source missing: " + ", ".join(missing))
    manifest = {
        "experiment": EXPERIMENT,
        "status": "frozen_no_model_calls",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episodes": EPISODES,
        "rounds": list(ROUNDS),
        "cases_by_round": CASES_BY_ROUND,
        "arms": list(PROMPT_ARMS),
        "models": list(MODELS),
        "tasks_per_arm": EPISODES * len(ROUNDS),
        "planned_calls": PLANNED_CALLS,
        "hard_call_cap": HARD_CALL_CAP,
        "max_attempts_per_task": 1,
        "seed_base": SEED_BASE,
        "context_alignment": "100_percent_correct_direction",
        "task_sha256": {
            arm: _sha256(DESIGN / f"tasks_{arm}.jsonl") for arm in PROMPT_ARMS
        },
        "answer_key_sha256": _sha256(DESIGN / "answer_key.jsonl"),
        "episodes_sha256": _sha256(DESIGN / "episodes.jsonl"),
        "source_sha256": {
            name: _sha256(path) for name, path in source_paths.items()
        },
        "context_sentences": {
            "high_response_type": TARGET_CONTEXT_HIGH,
            "low_response_type": TARGET_CONTEXT_LOW,
        },
    }
    (DESIGN / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Built {PLANNED_CALLS}-call expansion at {RUN}")


if __name__ == "__main__":
    main()
