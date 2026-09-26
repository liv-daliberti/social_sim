#!/usr/bin/env python3
"""Freeze the v12 noisier structural-choice tasks and private key."""

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

import build_three_city_c2_v11_tasks as prompt_base
from engine.three_city_c2_v12 import (
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
    _ROOT / "data" / "three_city_c2_v12"
    / "full_k1_5_noisier_structural_choice"
)
_OUTDIR = _RUN / "design"
_PREREG = _ROOT / "PREREGISTRATION_THREE_CITY_C2_V12.md"
_CONFIG = _ROOT / "configs" / "three_city_c2_v12.yml"

STRUCTURAL_CHOICE_CLUE = prompt_base.STRUCTURAL_CHOICE_CLUE
strip_structural_clue = prompt_base.strip_structural_clue
target_suffix = prompt_base.target_suffix
validate_arm_prompt = prompt_base.validate_arm_prompt
build_prompt = prompt_base.build_prompt


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _nearer(value: float, a: float, b: float) -> str:
    return "A" if abs(value - a) < abs(value - b) else "B"


def build_records(
    triplet: Mapping[str, Any],
    *,
    episode_index: int,
    k: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    task_id = f"c2v12_{episode_index:04d}_k{k}"
    prompts = {
        arm: build_prompt(triplet, k=k, arm=arm) for arm in PROMPT_ARMS
    }
    model_records = {
        arm: {
            "task_id": task_id,
            "prompt": prompt,
            "prompt_sha256": _text_sha256(prompt),
        }
        for arm, prompt in prompts.items()
    }
    city_a = triplet["reference_a"]
    city_b = triplet["reference_b"]
    target = triplet["target"]
    cases_seen = cases_for_prefix(k)
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
        "episode_id": f"c2v12_{episode_index:04d}",
        "episode_seed": triplet["seed"],
        "k": k,
        "target_cases_seen": cases_seen,
        "public": {
            "reference_a": {
                key: city_a[key]
                for key in (
                    "profile_index", "starting_poll", "news", "ending_polls"
                )
            },
            "reference_b": {
                key: city_b[key]
                for key in (
                    "profile_index", "starting_poll", "news", "ending_polls"
                )
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
        arm: args.outdir / f"tasks_c2_v12_{arm}.jsonl"
        for arm in PROMPT_ARMS
    }
    key_path = args.outdir / "answer_key_c2_v12.jsonl"
    manifest_path = args.outdir / "tasks_c2_v12.manifest.json"
    existing = [
        path for path in (*task_paths.values(), key_path, manifest_path)
        if path.exists()
    ]
    if existing and not args.overwrite:
        parser.error("outputs exist; pass --overwrite intentionally")
    args.outdir.mkdir(parents=True, exist_ok=True)
    handles = {arm: path.open("w") for arm, path in task_paths.items()}
    aggregates = {arm: hashlib.sha256() for arm in PROMPT_ARMS}
    count = 0
    try:
        with key_path.open("w") as key_handle:
            for index, triplet in enumerate(triplets):
                for k in PREFIX_LADDER:
                    records, evaluator = build_records(
                        triplet, episode_index=index, k=k
                    )
                    for arm, record in records.items():
                        aggregates[arm].update(record["task_id"].encode())
                        aggregates[arm].update(record["prompt"].encode())
                        handles[arm].write(json.dumps(record, sort_keys=True) + "\n")
                    key_handle.write(json.dumps(evaluator, sort_keys=True) + "\n")
                    count += 1
    finally:
        for handle in handles.values():
            handle.close()
    source_paths = {
        "engine_sha256": _ROOT / "engine" / "three_city_c2_v12.py",
        "builder_sha256": Path(__file__),
        "source_prompt_builder_v11_sha256": _HERE / "build_three_city_c2_v11_tasks.py",
        "config_sha256": _CONFIG,
        "validator_sha256": _HERE / "validate_three_city_c2_v12_tasks.py",
        "runner_sha256": _HERE / "run_three_city_c2_v12_confirmatory.py",
        "launcher_sha256": _HERE / "run_three_city_c2_v12_local.sh",
        "analyzer_sha256": _ROOT / "analysis" / "analyze_three_city_c2_v12_confirmatory.py",
        "preregistration_sha256": args.preregistration,
    }
    manifest = {
        "experiment": "three_city_c2_v12_noisier_structural_choice",
        "status": "frozen_no_model_runs",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episodes": args.n,
        "seed_offset": args.seed_offset,
        "arms": list(PROMPT_ARMS),
        "prefixes": list(PREFIX_LADDER),
        "cases_by_prefix": {
            str(k): CASES_BY_PREFIX[k] for k in PREFIX_LADDER
        },
        "tasks_per_arm": count,
        "calls_per_model": count * len(PROMPT_ARMS),
        "total_calls_three_models": count * len(PROMPT_ARMS) * 3,
        "case_sigma": CASE_SIGMA,
        "city_deviation_sd": CITY_DEVIATION_SD,
        "profile_slope_range": list(EPISODE_PROFILE_SLOPE_RANGE),
        "intercept_range": list(EPISODE_INTERCEPT_RANGE),
        "low_profile_range": list(LOW_PROFILE_RANGE),
        "high_profile_range": list(HIGH_PROFILE_RANGE),
        "target_near_endpoint_range": list(TARGET_NEAR_ENDPOINT_RANGE),
        "fresh_seeds_relative_to_v11": True,
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
        **{field: _file_sha256(path) for field, path in source_paths.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {count} tasks per arm ({count * 9} total calls)")
    for path in (*task_paths.values(), key_path, manifest_path):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
