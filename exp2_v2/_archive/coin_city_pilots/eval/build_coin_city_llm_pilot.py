#!/usr/bin/env python3
"""Build the fresh, balanced 1,800-call coin-to-city development pilot."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from engine.coin_city_llm_pilot import (
    CASES_BY_ROUND,
    CUE_RELIABILITY,
    EPISODES,
    EXPERIMENT,
    HARD_CALL_CAP,
    MODELS,
    PLANNED_CALLS,
    PROMPT_ARMS,
    ROUNDS,
    baseline_estimates,
    make_episode,
    make_prompt,
    prompt_sha256,
    prompt_sections,
    task_id,
    validate_arm_prompt,
)


RUN = ROOT / "data" / EXPERIMENT
DESIGN = RUN / "design"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    if PLANNED_CALLS > HARD_CALL_CAP:
        raise SystemExit(f"planned calls {PLANNED_CALLS} exceed cap {HARD_CALL_CAP}")
    if (RUN / "responses").exists() and any((RUN / "responses").glob("*.jsonl")):
        raise SystemExit("refusing to rebuild a pilot that already has responses")
    DESIGN.mkdir(parents=True, exist_ok=True)
    episodes = [make_episode(index) for index in range(EPISODES)]
    _write_jsonl(DESIGN / "episodes.jsonl", episodes)

    arm_rows: dict[str, list[dict]] = {arm: [] for arm in PROMPT_ARMS}
    answer_key: list[dict] = []
    for episode in episodes:
        for k in ROUNDS:
            identifier = task_id(episode["episode"], k)
            baselines = baseline_estimates(episode, k)
            answer_key.append(
                {
                    "task_id": identifier,
                    "episode": episode["episode"],
                    "k": k,
                    "city_c_cases": CASES_BY_ROUND[k],
                    "gold_expected_poll": episode["target_truth"],
                    "target_high": episode["target_high"],
                    "cue_high": episode["cue_high"],
                    "cue_correct": episode["cue_correct"],
                    "high_reference_city": episode["high_reference_city"],
                    "baselines": baselines,
                }
            )
            for arm in PROMPT_ARMS:
                prompt_text = make_prompt(episode, k, arm)
                validate_arm_prompt(prompt_text, arm)
                arm_rows[arm].append(
                    {
                        "task_id": identifier,
                        "prompt": prompt_text,
                        "prompt_sha256": prompt_sha256(prompt_text),
                    }
                )
    _write_jsonl(DESIGN / "answer_key.jsonl", answer_key)
    for arm, rows in arm_rows.items():
        _write_jsonl(DESIGN / f"tasks_{arm}.jsonl", rows)

    example = episodes[0]
    example_k = 2
    exact_parts = [
        "# Frozen pilot prompt example",
        "",
        "This is episode 0 at k=2 (two City C cases).",
        "B and C contain identical numbers; only City C's background sentence changes.",
        "",
    ]
    for arm in PROMPT_ARMS:
        exact_parts.extend(
            [f"## {arm}", "", "```text", make_prompt(example, example_k, arm), "```", ""]
        )
        (DESIGN / f"example_prompt_{arm}.txt").write_text(
            make_prompt(example, example_k, arm) + "\n"
        )
    (DESIGN / "exact_prompt_example.md").write_text("\n".join(exact_parts))
    (DESIGN / "example_prompt_sections.json").write_text(
        json.dumps(
            {
                arm: prompt_sections(example, example_k, arm)
                for arm in PROMPT_ARMS
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    source_paths = {
        "engine": ROOT / "engine" / "coin_city_llm_pilot.py",
        "builder": HERE / "build_coin_city_llm_pilot.py",
        "validator": HERE / "validate_coin_city_llm_pilot.py",
        "runner": HERE / "run_coin_city_llm_pilot.py",
        "launcher": HERE / "run_coin_city_llm_pilot_local.sh",
        "analyzer": ROOT / "analysis" / "analyze_coin_city_llm_pilot.py",
        "live_monitor": ROOT / "analysis" / "live_coin_city_llm_pilot.py",
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise SystemExit("pilot source files missing: " + ", ".join(missing))
    manifest = {
        "experiment": EXPERIMENT,
        "status": "frozen_development_pilot_no_model_calls",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episodes": EPISODES,
        "rounds": list(ROUNDS),
        "cases_by_round": CASES_BY_ROUND,
        "arms": list(PROMPT_ARMS),
        "models": list(MODELS),
        "tasks_per_arm": EPISODES * len(ROUNDS),
        "planned_model_calls": PLANNED_CALLS,
        "hard_call_cap": HARD_CALL_CAP,
        "max_attempts_per_task": 1,
        "cue_reliability": CUE_RELIABILITY,
        "task_file_sha256": {
            arm: _file_sha256(DESIGN / f"tasks_{arm}.jsonl")
            for arm in PROMPT_ARMS
        },
        "answer_key_sha256": _file_sha256(DESIGN / "answer_key.jsonl"),
        "episodes_sha256": _file_sha256(DESIGN / "episodes.jsonl"),
        "source_sha256": {
            name: _file_sha256(path) for name, path in source_paths.items()
        },
    }
    (DESIGN / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Built {PLANNED_CALLS}-call pilot at {RUN}")


if __name__ == "__main__":
    main()
