#!/usr/bin/env python3
"""Freeze the v11 identifiable structural-choice tasks and private key."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

import build_three_city_c2_v9_tasks as v9_builder
from engine.three_city_c2_v11 import (
    CASES_BY_PREFIX,
    CASE_SIGMA,
    CITY_DEVIATION_SD,
    EPISODE_INTERCEPT_RANGE,
    EPISODE_PROFILE_SLOPE_RANGE,
    HIGH_PROFILE_RANGE,
    LOW_PROFILE_RANGE,
    PREFIX_LADDER,
    PROMPT_ARMS,
    REFERENCE_CASES,
    SEED_OFFSET,
    STARTING_POLL,
    TARGET_NEAR_ENDPOINT_RANGE,
    balanced_triplets,
    cases_for_prefix,
    compute_abc_shrinkage,
    compute_reference_interpolation,
    compute_target_only,
    gold_expected_poll,
)

_RUN = (
    _ROOT
    / "data"
    / "three_city_c2_v11"
    / "full_k1_5_identifiable_structural_choice"
)
_OUTDIR = _RUN / "design"
_PREREG = _ROOT / "PREREGISTRATION_THREE_CITY_C2_V11.md"
_CONFIG = _ROOT / "configs" / "three_city_c2_v11.yml"

STRUCTURAL_CHOICE_CLUE = (
    "Structural clue: One of Cities A and B is a more relevant analogue for "
    "City C than the other. The displayed regional-profile indices are "
    "informative about which reference is more relevant, although the "
    "resemblance is imperfect and City C may still respond differently."
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def strip_structural_clue(prompt: str) -> str:
    return prompt.replace(STRUCTURAL_CHOICE_CLUE + "\n\n", "")


def target_suffix(prompt: str) -> str:
    marker = "CITY C\n"
    if marker not in prompt:
        raise ValueError("prompt is missing City C")
    return strip_structural_clue(marker + prompt.split(marker, 1)[1])


def validate_arm_prompt(prompt: str, arm: str) -> None:
    if arm in ("c_only", "abc"):
        v9_builder.validate_arm_prompt(prompt, arm)
        return
    if arm != "abc_structural_clue":
        raise ValueError(f"unknown prompt arm: {arm!r}")
    if prompt.count(STRUCTURAL_CHOICE_CLUE) != 1:
        raise ValueError("structural-choice arm lacks exactly one clue block")
    v9_builder.validate_arm_prompt(strip_structural_clue(prompt), "abc")


def build_prompt(
    triplet: Mapping[str, Any],
    *,
    k: int,
    arm: str,
) -> str:
    if arm in ("c_only", "abc"):
        prompt = v9_builder.build_prompt(triplet, k=k, arm=arm)
        validate_arm_prompt(prompt, arm)
        return prompt
    if arm != "abc_structural_clue":
        raise ValueError(f"unknown prompt arm: {arm!r}")
    neutral = v9_builder.build_prompt(triplet, k=k, arm="abc")
    marker = v9_builder._REASONING_MARKER
    if neutral.count(marker) != 1:
        raise ValueError("reasoning insertion marker is not unique")
    prompt = neutral.replace(
        marker,
        STRUCTURAL_CHOICE_CLUE + "\n\n" + marker,
        1,
    )
    validate_arm_prompt(prompt, arm)
    return prompt


def _nearer(value: float, a: float, b: float) -> str:
    return "A" if abs(value - a) < abs(value - b) else "B"


def build_records(
    triplet: Mapping[str, Any],
    *,
    episode_index: int,
    k: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    task_id = f"c2v11_{episode_index:04d}_k{k}"
    prompts = {
        arm: build_prompt(triplet, k=k, arm=arm)
        for arm in PROMPT_ARMS
    }
    model_records = {
        arm: {
            "task_id": task_id,
            "prompt": prompt,
            "prompt_sha256": _text_sha256(prompt),
        }
        for arm, prompt in prompts.items()
    }
    cases_seen = cases_for_prefix(k)
    city_a = triplet["reference_a"]
    city_b = triplet["reference_b"]
    target = triplet["target"]
    profile_nearest = _nearer(
        float(target["profile_index"]),
        float(city_a["profile_index"]),
        float(city_b["profile_index"]),
    )
    response_nearest = _nearer(
        float(target["response"]),
        float(city_a["response"]),
        float(city_b["response"]),
    )
    evaluator = {
        "task_id": task_id,
        "episode_id": f"c2v11_{episode_index:04d}",
        "episode_seed": triplet["seed"],
        "k": k,
        "target_cases_seen": cases_seen,
        "public": {
            "reference_a": {
                "profile_index": city_a["profile_index"],
                "starting_poll": city_a["starting_poll"],
                "news": city_a["news"],
                "ending_polls": city_a["ending_polls"],
            },
            "reference_b": {
                "profile_index": city_b["profile_index"],
                "starting_poll": city_b["starting_poll"],
                "news": city_b["news"],
                "ending_polls": city_b["ending_polls"],
            },
            "target": {
                "profile_index": target["profile_index"],
                "starting_poll": target["starting_poll"],
                "news": target["news"][:cases_seen],
                "ending_polls": target["ending_polls"][:cases_seen],
            },
            "test_starting_poll": STARTING_POLL,
            "test_news": triplet["test_news"],
        },
        "design_truth": {
            "high_reference": triplet["high_reference"],
            "target_near_high": triplet["target_near_high"],
            "profile_nearest_reference": profile_nearest,
            "response_nearest_reference": response_nearest,
            "target_profile_position": round(
                float(triplet["target_profile_position"]), 8
            ),
            "true_response_a": round(float(city_a["response"]), 8),
            "true_response_b": round(float(city_b["response"]), 8),
        },
        "gold": {
            "expected_poll": gold_expected_poll(triplet),
            "target_response": round(float(target["response"]), 8),
        },
        "baselines": {
            "target_only": compute_target_only(triplet, k),
            "reference_interpolation": compute_reference_interpolation(triplet),
            "abc_shrinkage": compute_abc_shrinkage(triplet, k),
        },
    }
    return model_records, evaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed-offset", type=int, default=SEED_OFFSET)
    parser.add_argument("--outdir", type=Path, default=_OUTDIR)
    parser.add_argument("--preregistration", type=Path, default=_PREREG)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.n <= 0 or args.n % 4:
        parser.error("--n must be a positive multiple of four")
    triplets = list(balanced_triplets(args.n, seed_offset=args.seed_offset))
    if args.dry_run:
        for arm in PROMPT_ARMS:
            print(f"\n===== {arm} =====\n")
            print(build_prompt(triplets[0], k=2, arm=arm))
        return

    task_paths = {
        arm: args.outdir / f"tasks_c2_v11_{arm}.jsonl"
        for arm in PROMPT_ARMS
    }
    key_path = args.outdir / "answer_key_c2_v11.jsonl"
    manifest_path = args.outdir / "tasks_c2_v11.manifest.json"
    existing = [
        path
        for path in (*task_paths.values(), key_path, manifest_path)
        if path.exists()
    ]
    if existing and not args.overwrite:
        parser.error("outputs exist; pass --overwrite intentionally")
    args.outdir.mkdir(parents=True, exist_ok=True)
    handles = {arm: path.open("w") for arm, path in task_paths.items()}
    aggregates = {arm: hashlib.sha256() for arm in PROMPT_ARMS}
    task_count = 0
    try:
        with key_path.open("w") as key_handle:
            for episode_index, triplet in enumerate(triplets):
                for k in PREFIX_LADDER:
                    records, evaluator = build_records(
                        triplet,
                        episode_index=episode_index,
                        k=k,
                    )
                    for arm, record in records.items():
                        aggregates[arm].update(record["task_id"].encode())
                        aggregates[arm].update(record["prompt"].encode())
                        handles[arm].write(
                            json.dumps(record, sort_keys=True) + "\n"
                        )
                    key_handle.write(
                        json.dumps(evaluator, sort_keys=True) + "\n"
                    )
                    task_count += 1
    finally:
        for handle in handles.values():
            handle.close()

    source_paths = {
        "engine_sha256": _ROOT / "engine" / "three_city_c2_v11.py",
        "source_engine_v9_sha256": _ROOT / "engine" / "three_city_c2_v9.py",
        "builder_sha256": Path(__file__),
        "source_builder_v9_sha256": _HERE / "build_three_city_c2_v9_tasks.py",
        "config_sha256": _CONFIG,
        "validator_sha256": _HERE / "validate_three_city_c2_v11_tasks.py",
        "runner_sha256": _HERE / "run_three_city_c2_v11_confirmatory.py",
        "source_runner_v9_sha256": _HERE / "run_three_city_c2_v9_confirmatory.py",
        "launcher_sha256": _HERE / "run_three_city_c2_v11_local.sh",
        "analyzer_sha256": _ROOT / "analysis" / "analyze_three_city_c2_v11_confirmatory.py",
        "source_analyzer_v9_sha256": _ROOT / "analysis" / "analyze_three_city_c2_v9_confirmatory.py",
        "live_monitor_sha256": _ROOT / "analysis" / "live_monitor_three_city_c2_v11.py",
        "prompt_figure_sha256": _ROOT / "analysis" / "preview_three_city_c2_v11_prompts.py",
        "preregistration_sha256": args.preregistration,
    }
    manifest = {
        "experiment": "three_city_c2_v11_identifiable_structural_choice",
        "status": "frozen_no_model_runs",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episodes": args.n,
        "seed_offset": args.seed_offset,
        "arms": list(PROMPT_ARMS),
        "prefixes": list(PREFIX_LADDER),
        "cases_by_prefix": {
            str(k): CASES_BY_PREFIX[k] for k in PREFIX_LADDER
        },
        "tasks_per_arm": task_count,
        "calls_per_model": task_count * len(PROMPT_ARMS),
        "total_calls_three_models": task_count * len(PROMPT_ARMS) * 3,
        "primary_contrast": "abc_minus_c_only_information_value",
        "secondary_contrast": "abc_structural_clue_minus_abc_reasoning_value",
        "reference_cases_per_city": REFERENCE_CASES,
        "case_sigma": CASE_SIGMA,
        "city_deviation_sd": CITY_DEVIATION_SD,
        "low_profile_range": list(LOW_PROFILE_RANGE),
        "high_profile_range": list(HIGH_PROFILE_RANGE),
        "target_near_endpoint_range": list(TARGET_NEAR_ENDPOINT_RANGE),
        "intercept_range": list(EPISODE_INTERCEPT_RANGE),
        "profile_slope_range": list(EPISODE_PROFILE_SLOPE_RANGE),
        "numerical_dgp_identical_to_v9": False,
        "a_and_b_prompt_templates_identical_to_v9": True,
        "balanced_target_nearer_reference": True,
        "clue_reveals_closer_city": False,
        "structural_choice_clue": STRUCTURAL_CHOICE_CLUE,
        "model_task_files": {
            arm: {
                "file": path.name,
                "sha256": _file_sha256(path),
                "aggregate_prompt_sha256": aggregates[arm].hexdigest(),
            }
            for arm, path in task_paths.items()
        },
        "answer_key_file": key_path.name,
        "answer_key_sha256": _file_sha256(key_path),
        **{
            field: _file_sha256(path)
            for field, path in source_paths.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Wrote {task_count} tasks per arm "
        f"({task_count * len(PROMPT_ARMS)} per model; "
        f"{task_count * len(PROMPT_ARMS) * 3} total calls)"
    )
    for path in (*task_paths.values(), key_path, manifest_path):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
