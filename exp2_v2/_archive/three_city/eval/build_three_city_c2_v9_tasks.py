#!/usr/bin/env python3
"""Freeze the natural continuous-profile C2 v9 tasks and private key."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from engine.three_city_c2_v9 import (
    CASES_BY_PREFIX,
    CASE_SIGMA,
    CITY_DEVIATION_SD,
    PREFIX_LADDER,
    PROMPT_ARMS,
    REFERENCE_CASES,
    SEED_OFFSET,
    STARTING_POLL,
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
    / "three_city_c2_v9"
    / "full_k1_5_clean_information"
)
_OUTDIR = _RUN / "design"
_PREREG = _ROOT / "PREREGISTRATION_THREE_CITY_C2_V9.md"
_CONFIG = _ROOT / "configs" / "three_city_c2_v9.yml"

RELEVANCE_HINT_BLOCK = (
    "Regional-profile similarity is informative: cities with closer profile "
    "scores generally have more similar responses, although substantial "
    "city-specific variation remains."
)
_REASONING_MARKER = (
    "Think carefully about the expected poll, but do not provide step-by-step "
    "working."
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _table(
    city: Mapping[str, Any],
    k: Optional[int] = None,
) -> list[str]:
    if k is None:
        k = len(city["news"])
    rows = [
        "| Case | Starting poll | Net news | Poll at end of case |",
        "|------|--------------:|---------:|--------------------:|",
    ]
    rows.extend(
        f"| {case:>4} | {city['starting_poll']:>13.1f} | "
        f"{dose:>+8d} | {poll:>19.1f} |"
        for case, (dose, poll) in enumerate(
            zip(city["news"][:k], city["ending_polls"][:k]),
            1,
        )
    )
    return rows


def _city_lines(
    name: str,
    city: Mapping[str, Any],
    *,
    k: Optional[int] = None,
) -> list[str]:
    return [
        f"CITY {name}",
        f"Regional-profile index: {float(city['profile_index']):.1f} / 100",
        *_table(city, k=k),
    ]


def _common_opening() -> list[str]:
    return [
        "You are forecasting a post-news local-election poll.",
        (
            "Each table row is a separate polling case, not a time series. "
            "Every case began with a poll of 50.0. The signed net-news value "
            "then occurred, and the poll at the end of that case was measured."
        ),
        (
            "End-of-case polls vary from case to case because polling and local "
            "opinion are not perfectly stable."
        ),
        (
            "Positive net news is favorable to the candidate and negative net "
            "news is unfavorable."
        ),
        (
            "The regional-profile index is a continuous descriptive score "
            "from 0 to 100, not a category or an outcome label."
        ),
        "",
    ]


def _target_and_question(
    triplet: Mapping[str, Any],
    *,
    k: int,
) -> list[str]:
    cases_seen = cases_for_prefix(k)
    return [
        *_city_lines("C", triplet["target"], k=cases_seen),
        "",
        (
            f"A new City C case begins with a poll of {STARTING_POLL:.1f} "
            f"and has net news {int(triplet['test_news']):+d}."
        ),
        "What poll do you predict at the end of this new case?",
        "",
        _REASONING_MARKER,
        "Respond with one JSON object only:",
        (
            '{"rationale": "one short sentence", '
            '"predicted_poll": <number from 0 to 100>}'
        ),
    ]


def build_prompt(
    triplet: Mapping[str, Any],
    *,
    k: int,
    arm: str,
) -> str:
    if arm not in PROMPT_ARMS:
        raise ValueError(f"unknown prompt arm: {arm!r}")
    lines = _common_opening()
    if arm != "c_only":
        lines.extend(
            [
                "Below are records from two earlier cities in the same region.",
                "",
                *_city_lines("A", triplet["reference_a"]),
                "",
                *_city_lines("B", triplet["reference_b"]),
                "",
            ]
        )
    lines.extend(_target_and_question(triplet, k=k))
    prompt = "\n".join(lines)
    if arm == "abc_relevance":
        if prompt.count(_REASONING_MARKER) != 1:
            raise ValueError("reasoning insertion marker is not unique")
        prompt = prompt.replace(
            _REASONING_MARKER,
            RELEVANCE_HINT_BLOCK + "\n\n" + _REASONING_MARKER,
            1,
        )
    validate_arm_prompt(prompt, arm)
    return prompt


def strip_relevance_hint(prompt: str) -> str:
    return prompt.replace(RELEVANCE_HINT_BLOCK + "\n\n", "")


def target_suffix(prompt: str) -> str:
    marker = "CITY C\n"
    if marker not in prompt:
        raise ValueError("prompt is missing City C")
    return strip_relevance_hint(marker + prompt.split(marker, 1)[1])


def validate_arm_prompt(prompt: str, arm: str) -> None:
    if arm not in PROMPT_ARMS:
        raise ValueError(f"unknown prompt arm: {arm!r}")
    required = (
        "separate polling case",
        "not a time series",
        "continuous descriptive score",
        "CITY C",
        "do not provide step-by-step working",
        '"predicted_poll": <number from 0 to 100>',
    )
    missing = [phrase for phrase in required if phrase not in prompt]
    if missing:
        raise ValueError("missing required prompt text: " + ", ".join(missing))
    has_references = "CITY A\n" in prompt and "CITY B\n" in prompt
    if arm == "c_only":
        if has_references or RELEVANCE_HINT_BLOCK in prompt:
            raise ValueError("City-C-only arm contains reference information")
    elif not has_references:
        raise ValueError("A/B/C arm is missing reference information")
    if arm == "abc_relevance":
        if prompt.count(RELEVANCE_HINT_BLOCK) != 1:
            raise ValueError("relevance arm lacks exactly one hint block")
    elif RELEVANCE_HINT_BLOCK in prompt:
        raise ValueError("neutral arm contains the relevance hint")


def build_records(
    triplet: Mapping[str, Any],
    *,
    episode_index: int,
    k: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    task_id = f"c2v9_{episode_index:04d}_k{k}"
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
    evaluator = {
        "task_id": task_id,
        "episode_id": f"c2v9_{episode_index:04d}",
        "episode_seed": triplet["seed"],
        "k": k,
        "target_cases_seen": cases_seen,
        "public": {
            "reference_a": {
                "profile_index": triplet["reference_a"]["profile_index"],
                "starting_poll": triplet["reference_a"]["starting_poll"],
                "news": triplet["reference_a"]["news"],
                "ending_polls": triplet["reference_a"]["ending_polls"],
            },
            "reference_b": {
                "profile_index": triplet["reference_b"]["profile_index"],
                "starting_poll": triplet["reference_b"]["starting_poll"],
                "news": triplet["reference_b"]["news"],
                "ending_polls": triplet["reference_b"]["ending_polls"],
            },
            "target": {
                "profile_index": triplet["target"]["profile_index"],
                "starting_poll": triplet["target"]["starting_poll"],
                "news": triplet["target"]["news"][:cases_seen],
                "ending_polls": triplet["target"]["ending_polls"][:cases_seen],
            },
            "test_starting_poll": STARTING_POLL,
            "test_news": triplet["test_news"],
        },
        "gold": {
            "expected_poll": gold_expected_poll(triplet),
            "target_response": round(float(triplet["target"]["response"]), 8),
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
    if args.n <= 0 or args.n % 2:
        parser.error("--n must be a positive even number")
    triplets = list(balanced_triplets(args.n, seed_offset=args.seed_offset))
    if args.dry_run:
        for arm in PROMPT_ARMS:
            print(f"\n===== {arm} =====\n")
            print(build_prompt(triplets[0], k=2, arm=arm))
        return

    task_paths = {
        arm: args.outdir / f"tasks_c2_v9_{arm}.jsonl"
        for arm in PROMPT_ARMS
    }
    key_path = args.outdir / "answer_key_c2_v9.jsonl"
    manifest_path = args.outdir / "tasks_c2_v9.manifest.json"
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
        "engine_sha256": _ROOT / "engine" / "three_city_c2_v9.py",
        "builder_sha256": Path(__file__),
        "config_sha256": _CONFIG,
        "validator_sha256": _HERE / "validate_three_city_c2_v9_tasks.py",
        "runner_sha256": _HERE / "run_three_city_c2_v9_confirmatory.py",
        "launcher_sha256": _HERE / "run_three_city_c2_v9_local.sh",
        "analyzer_sha256": _ROOT / "analysis" / "analyze_three_city_c2_v9_confirmatory.py",
        "live_monitor_sha256": _ROOT / "analysis" / "live_monitor_three_city_c2_v9.py",
        "preregistration_sha256": args.preregistration,
    }
    manifest = {
        "experiment": "three_city_c2_v9_clean_information_relevance",
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
        "secondary_contrast": "abc_relevance_minus_abc_guidance_value",
        "reference_cases_per_city": REFERENCE_CASES,
        "case_sigma": CASE_SIGMA,
        "city_deviation_sd": CITY_DEVIATION_SD,
        "continuous_profiles_no_discrete_types": True,
        "relevance_hint_block": RELEVANCE_HINT_BLOCK,
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
    for path in task_paths.values():
        print(f"Wrote {path}")
    print(f"Wrote {key_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
