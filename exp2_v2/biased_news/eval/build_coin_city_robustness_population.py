#!/usr/bin/env python3
"""Build and freeze the harder Coin City generator population."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from engine.coin_city_robustness_population import (  # noqa: E402
    ARMS,
    CITY_SLOPE_SD,
    EXPERIMENT,
    PROTOCOL_VERSION,
    SEED_BASE,
    STRONG_SLOPE_MEAN,
    WEAK_SLOPE_MEAN,
    make_episode,
    make_prompt,
    prompt_sha256,
    task_id,
)
from engine.coin_city_stable_relationship_claude_n250 import (  # noqa: E402
    CASE_NOISE_SD,
    C_CASE_LEVELS,
    EPISODES,
    empirical_estimates,
    fit_change_slope,
)

RUN = ROOT / "data" / EXPERIMENT
DESIGN = RUN / "design"
RESPONSES = RUN / "responses"
PROTOCOL = ROOT / "ROBUSTNESS_EXTENSION_PROTOCOL.md"
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> None:
    if RESPONSES.exists() and any(RESPONSES.rglob("*.jsonl")):
        raise SystemExit("refusing to rebuild after robustness responses exist")
    if not PROTOCOL.is_file():
        raise SystemExit("robustness protocol is missing")
    DESIGN.mkdir(parents=True, exist_ok=True)

    episodes = [make_episode(index) for index in range(EPISODES)]
    keys: list[dict] = []
    tasks: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for episode in episodes:
        for c_cases in C_CASE_LEVELS:
            identifier = task_id(int(episode["episode"]), c_cases)
            keys.append(
                {
                    "task_id": identifier,
                    "episode": int(episode["episode"]),
                    "c_cases": c_cases,
                    "gold_expected_poll": float(episode["gold_expected_poll"]),
                    "baselines": empirical_estimates(episode, c_cases),
                    "target_strong": bool(episode["target_strong"]),
                    "cue_strong": bool(episode["cue_strong"]),
                    "strong_reference_city": episode["strong_reference_city"],
                }
            )
            arm_prompts: dict[str, str] = {}
            for arm in ARMS:
                prompt = make_prompt(episode, c_cases, arm)
                arm_prompts[arm] = prompt
                tasks[arm].append(
                    {
                        "task_id": identifier,
                        "prompt": prompt,
                        "prompt_sha256": prompt_sha256(prompt),
                    }
                )
            if NUMBER.findall(arm_prompts["abc_context"]) != NUMBER.findall(
                arm_prompts["abc_symbol_context"]
            ):
                raise SystemExit(f"symbol numeric content changed: {identifier}")

    write_jsonl(DESIGN / "episodes.jsonl", episodes)
    write_jsonl(DESIGN / "answer_key.jsonl", keys)
    write_jsonl(
        DESIGN / "scoring_key.jsonl",
        [
            {
                field: row[field]
                for field in (
                    "task_id",
                    "episode",
                    "c_cases",
                    "gold_expected_poll",
                    "baselines",
                )
            }
            for row in keys
        ],
    )
    for arm, rows in tasks.items():
        write_jsonl(DESIGN / f"tasks_{arm}.jsonl", rows)
        (DESIGN / f"example_prompt_{arm}.txt").write_text(rows[0]["prompt"] + "\n")

    source_paths = {
        "protocol": PROTOCOL,
        "robustness_engine": ROOT / "engine" / "coin_city_robustness_population.py",
        "frozen_prompt_engine": ROOT
        / "engine"
        / "coin_city_stable_relationship_claude_n250.py",
        "symbol_engine": ROOT / "engine" / "coin_city_symbol_context_arm.py",
        "builder": Path(__file__).resolve(),
    }
    strong = np.asarray([e["reference_strong_slope"] for e in episodes], dtype=float)
    weak = np.asarray([e["reference_weak_slope"] for e in episodes], dtype=float)
    target_strong = np.asarray(
        [e["target_slope"] for e in episodes if e["target_strong"]], dtype=float
    )
    target_weak = np.asarray(
        [e["target_slope"] for e in episodes if not e["target_strong"]], dtype=float
    )
    fitted_separation = np.asarray(
        [
            abs(fit_change_slope(e["reference_a"]) - fit_change_slope(e["reference_b"]))
            for e in episodes
        ],
        dtype=float,
    )
    balance = Counter(
        (e["strong_reference_city"], bool(e["target_strong"])) for e in episodes
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": EXPERIMENT,
        "classification": "post_hoc_generator_population_robustness",
        "status": "frozen_no_model_calls",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episodes": EPISODES,
        "seed_base": SEED_BASE,
        "seed_last": SEED_BASE + EPISODES - 1,
        "arms": list(ARMS),
        "tasks_per_arm": EPISODES * len(C_CASE_LEVELS),
        "planned_open_models": 10,
        "planned_responses": 10 * len(ARMS) * EPISODES * len(C_CASE_LEVELS),
        "data_generating_process": {
            "strong_slope_mean": STRONG_SLOPE_MEAN,
            "weak_slope_mean": WEAK_SLOPE_MEAN,
            "city_slope_sd": CITY_SLOPE_SD,
            "case_noise_sd": CASE_NOISE_SD,
            "draw_policy": "untruncated_unreordered",
        },
        "realized": {
            "reference_strong_mean": float(np.mean(strong)),
            "reference_strong_sd": float(np.std(strong, ddof=1)),
            "reference_weak_mean": float(np.mean(weak)),
            "reference_weak_sd": float(np.std(weak, ddof=1)),
            "target_strong_mean": float(np.mean(target_strong)),
            "target_strong_sd": float(np.std(target_strong, ddof=1)),
            "target_weak_mean": float(np.mean(target_weak)),
            "target_weak_sd": float(np.std(target_weak, ddof=1)),
            "reference_latent_reversals": int(np.sum(strong <= weak)),
            "median_fitted_reference_separation": float(np.median(fitted_separation)),
        },
        "factorial_balance": {f"{a}:{b}": n for (a, b), n in sorted(balance.items())},
        "task_sha256": {arm: sha256(DESIGN / f"tasks_{arm}.jsonl") for arm in ARMS},
        "episodes_sha256": sha256(DESIGN / "episodes.jsonl"),
        "answer_key_sha256": sha256(DESIGN / "answer_key.jsonl"),
        "scoring_key_sha256": sha256(DESIGN / "scoring_key.jsonl"),
        "source_sha256": {name: sha256(path) for name, path in source_paths.items()},
    }
    (DESIGN / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    checks = {
        "exact_episode_count": len(episodes) == 250,
        "fresh_seed_population": {e["seed"] for e in episodes}
        == set(range(940000, 940250)),
        "closest_factorial_balance": sorted(balance.values()) == [62, 62, 63, 63],
        "exact_task_dimensions": all(len(rows) == 1250 for rows in tasks.values()),
        "all_prompt_hashes_valid": all(
            prompt_sha256(row["prompt"]) == row["prompt_sha256"]
            for rows in tasks.values()
            for row in rows
        ),
        "symbol_numeric_content_matched": True,
        "no_response_files": not (
            RESPONSES.exists() and any(RESPONSES.rglob("*.jsonl"))
        ),
        "untruncated_population_retained": manifest["realized"][
            "reference_latent_reversals"
        ]
        >= 0,
    }
    validation = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "realized": manifest["realized"],
    }
    (DESIGN / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    if not all(checks.values()):
        raise SystemExit("robustness design validation failed")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
