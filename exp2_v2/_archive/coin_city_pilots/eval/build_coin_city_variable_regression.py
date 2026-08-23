#!/usr/bin/env python3
"""Build the frozen variable-predictor coin-city experiment."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from engine.coin_city_variable_regression import (
    CASE_NOISE_SD,
    C_CASE_LEVELS,
    CITY_SLOPE_SD,
    EPISODES,
    EXPERIMENT,
    HARD_CALL_CAP,
    LIVE_EPISODES_BY_CELL,
    LIVE_REPLICATES_BY_CELL,
    MODELS,
    NEWS_VALUES,
    PLANNED_CALLS,
    PROMPT_ARMS,
    REFERENCE_CASES,
    SEED_BASE,
    SHARED_PROMPT_INTRO,
    STARTING_POLL_RANGE,
    STRONG_SLOPE_MEAN,
    TARGET_CONTEXT_STRONG,
    TARGET_CONTEXT_WEAK,
    WEAK_SLOPE_MEAN,
    empirical_estimates,
    make_episode,
    make_prompt,
    prompt_sections,
    prompt_sha256,
    shared_prompt_closing,
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
    if PLANNED_CALLS != 3_150 or PLANNED_CALLS > HARD_CALL_CAP:
        raise SystemExit("variable-regression call cap invariant failed")
    if (RUN / "responses").exists() and any((RUN / "responses").glob("*.jsonl")):
        raise SystemExit("refusing to rebuild after model responses exist")
    DESIGN.mkdir(parents=True, exist_ok=True)
    episodes = [make_episode(index) for index in range(EPISODES)]
    _write_jsonl(DESIGN / "episodes.jsonl", episodes)
    tasks = {arm: [] for arm in PROMPT_ARMS}
    keys = []
    for episode in episodes:
        for c_cases in C_CASE_LEVELS:
            identifier = task_id(episode["episode"], c_cases)
            estimates = empirical_estimates(episode, c_cases)
            keys.append(
                {
                    "task_id": identifier,
                    "episode": episode["episode"],
                    "c_cases": c_cases,
                    "gold_expected_poll": episode["gold_expected_poll"],
                    "baselines": estimates,
                    "target_strong": episode["target_strong"],
                    "cue_strong": episode["cue_strong"],
                    "cue_correct": episode["cue_correct"],
                    "strong_reference_city": episode["strong_reference_city"],
                }
            )
            for arm in PROMPT_ARMS:
                prompt = make_prompt(episode, c_cases, arm)
                validate_arm_prompt(prompt, arm)
                tasks[arm].append(
                    {
                        "task_id": identifier,
                        "prompt": prompt,
                        "prompt_sha256": prompt_sha256(prompt),
                    }
                )
    _write_jsonl(DESIGN / "answer_key.jsonl", keys)
    scoring_keys = [
        {
            name: row[name]
            for name in (
                "task_id",
                "episode",
                "c_cases",
                "gold_expected_poll",
                "baselines",
            )
        }
        for row in keys
    ]
    _write_jsonl(DESIGN / "scoring_key.jsonl", scoring_keys)
    for arm, rows in tasks.items():
        _write_jsonl(DESIGN / f"tasks_{arm}.jsonl", rows)

    example = episodes[0]
    example_c_cases = 2
    (DESIGN / "example_prompt_components.json").write_text(
        json.dumps(
            {
                "shared_intro": SHARED_PROMPT_INTRO,
                "sections": {
                    arm: prompt_sections(example, example_c_cases, arm)
                    for arm in PROMPT_ARMS
                },
                "shared_closing": shared_prompt_closing(example),
                "episode": example["episode"],
                "c_cases": example_c_cases,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for arm in PROMPT_ARMS:
        (DESIGN / f"example_prompt_{arm}.txt").write_text(
            make_prompt(example, example_c_cases, arm) + "\n"
        )

    source_paths = {
        "engine": ROOT / "engine" / "coin_city_variable_regression.py",
        "builder": HERE / "build_coin_city_variable_regression.py",
        "validator": HERE / "validate_coin_city_variable_regression.py",
        "renderer": ROOT / "analysis" / "render_coin_city_variable_regression.py",
        "runner": HERE / "run_coin_city_variable_regression.py",
        "local_launcher": HERE / "run_coin_city_variable_regression_local.sh",
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise SystemExit("variable-regression source missing: " + ", ".join(missing))
    manifest = {
        "experiment": EXPERIMENT,
        "status": "frozen_no_model_calls",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episodes": EPISODES,
        "city_c_case_levels": list(C_CASE_LEVELS),
        "reference_cases_per_city": REFERENCE_CASES,
        "arms": list(PROMPT_ARMS),
        "models": list(MODELS),
        "tasks_per_arm": EPISODES * len(C_CASE_LEVELS),
        "planned_calls": PLANNED_CALLS,
        "hard_call_cap": HARD_CALL_CAP,
        "max_attempts_per_task": 1,
        "seed_base": SEED_BASE,
        "factorial_cell_episode_counts": list(LIVE_EPISODES_BY_CELL),
        "factorial_cell_source_replicates": [
            list(replicates) for replicates in LIVE_REPLICATES_BY_CELL
        ],
        "balanced_live_subset": (
            "50 alternating source replicates; 13/12/12/13 across four cells"
        ),
        "context_alignment": "100_percent_correct_news_environment",
        "data_generating_process": {
            "starting_poll_range": list(STARTING_POLL_RANGE),
            "net_news_values": list(NEWS_VALUES),
            "strong_slope_mean": STRONG_SLOPE_MEAN,
            "weak_slope_mean": WEAK_SLOPE_MEAN,
            "city_slope_sd": CITY_SLOPE_SD,
            "case_noise_sd": CASE_NOISE_SD,
        },
        "displayed_estimators": {
            "city_c": (
                "OLS through origin of (ending - starting) on net news, City C rows only"
            ),
            "abc_no_context": (
                "OLS through origin of (ending - starting) on net news, all displayed A/B/C rows"
            ),
            "estimation_inputs": "displayed numeric rows and displayed query only",
            "forbidden_inputs": [
                "latent city type",
                "local/national news clue",
                "simulation parameters",
                "future City C rows",
                "target truth",
            ],
        },
        "context_sentences": {
            "national_news": TARGET_CONTEXT_STRONG,
            "local_news": TARGET_CONTEXT_WEAK,
        },
        "task_sha256": {
            arm: _sha256(DESIGN / f"tasks_{arm}.jsonl") for arm in PROMPT_ARMS
        },
        "answer_key_sha256": _sha256(DESIGN / "answer_key.jsonl"),
        "scoring_key_sha256": _sha256(DESIGN / "scoring_key.jsonl"),
        "episodes_sha256": _sha256(DESIGN / "episodes.jsonl"),
        "source_sha256": {
            name: _sha256(path) for name, path in source_paths.items()
        },
    }
    (DESIGN / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Built frozen {PLANNED_CALLS}-call design at {RUN}")


if __name__ == "__main__":
    main()
