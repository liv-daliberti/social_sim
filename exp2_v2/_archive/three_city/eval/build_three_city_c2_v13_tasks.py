#!/usr/bin/env python3
"""Freeze the v13 two-regime structural-choice tasks and private key."""

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

from engine.three_city_c2_v13 import (
    CASES_BY_PREFIX,
    CASE_SIGMA,
    CITY_DEVIATION_SD,
    EPISODE_REGIME_GAP_RANGE,
    EPISODE_RESPONSE_CENTER_RANGE,
    HIGH_PROFILE_RANGE,
    LOW_PROFILE_RANGE,
    PREFIX_LADDER,
    PRIOR_EFFECTIVE_CASES,
    PROMPT_ARMS,
    REFERENCE_CASES,
    SEED_OFFSET,
    STARTING_POLL,
    TARGET_NEAR_ENDPOINT_RANGE,
    balanced_triplets,
    cases_for_prefix,
    compute_abc_no_structure,
    compute_abc_structural,
    compute_reference_estimates,
    compute_target_only,
    gold_expected_poll,
)

_RUN = (
    _ROOT / "data" / "three_city_c2_v13"
    / "full_k1_5_two_regime_structural_choice"
)
_OUTDIR = _RUN / "design"
_PREREG = _ROOT / "PREREGISTRATION_THREE_CITY_C2_V13.md"
_CONFIG = _ROOT / "configs" / "three_city_c2_v13.yml"

_REASONING_MARKER = (
    "Think carefully about the expected poll, but do not provide step-by-step "
    "working."
)
STRUCTURAL_CHOICE_CLUE = (
    "Structural clue: The cities come from two distinct response regimes, "
    "represented by Cities A and B. City C is drawn from the same response "
    "regime as whichever reference city has the closer regional-profile index. "
    "Use the closer-profile reference as the relevant analogue rather than "
    "averaging Cities A and B. This clue does not say whether A or B is closer; "
    "infer that from the displayed indices. City-specific differences and "
    "case noise remain possible."
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _table(city: Mapping[str, Any], k: Optional[int] = None) -> list[str]:
    if k is None:
        k = len(city["news"])
    rows = [
        "| Case | Starting poll | Net news | Poll at end of case |",
        "|------|--------------:|---------:|--------------------:|",
    ]
    rows.extend(
        f"| {case:>4} | {float(city['starting_poll']):>13.1f} | "
        f"{int(dose):>+8d} | {float(poll):>19.1f} |"
        for case, (dose, poll) in enumerate(
            zip(city["news"][:k], city["ending_polls"][:k]), 1
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
            "End-of-case polls vary because polling and local opinion are not "
            "perfectly stable."
        ),
        (
            "Positive net news is favorable to the candidate and negative net "
            "news is unfavorable."
        ),
        (
            "The regional-profile index is a descriptive score from 0 to 100. "
            "Its relationship to polling responses is not otherwise specified."
        ),
        "",
    ]


def _target_and_question(triplet: Mapping[str, Any], *, k: int) -> list[str]:
    return [
        *_city_lines("C", triplet["target"], k=cases_for_prefix(k)),
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
    if arm == "abc_structural_clue":
        if prompt.count(_REASONING_MARKER) != 1:
            raise ValueError("reasoning insertion marker is not unique")
        prompt = prompt.replace(
            _REASONING_MARKER,
            STRUCTURAL_CHOICE_CLUE + "\n\n" + _REASONING_MARKER,
            1,
        )
    validate_arm_prompt(prompt, arm)
    return prompt


def strip_structural_clue(prompt: str) -> str:
    return prompt.replace(STRUCTURAL_CHOICE_CLUE + "\n\n", "")


def target_suffix(prompt: str) -> str:
    marker = "CITY C\n"
    if marker not in prompt:
        raise ValueError("prompt is missing City C")
    return strip_structural_clue(marker + prompt.split(marker, 1)[1])


def validate_arm_prompt(prompt: str, arm: str) -> None:
    if arm not in PROMPT_ARMS:
        raise ValueError(f"unknown prompt arm: {arm!r}")
    required = (
        "separate polling case",
        "not a time series",
        "descriptive score from 0 to 100",
        "CITY C",
        "do not provide step-by-step working",
        '"predicted_poll": <number from 0 to 100>',
    )
    missing = [phrase for phrase in required if phrase not in prompt]
    if missing:
        raise ValueError("missing required prompt text: " + ", ".join(missing))
    has_references = "CITY A\n" in prompt and "CITY B\n" in prompt
    if arm == "c_only":
        if has_references or STRUCTURAL_CHOICE_CLUE in prompt:
            raise ValueError("City-C-only arm contains added information")
    elif not has_references:
        raise ValueError("A/B/C arm is missing reference information")
    if arm == "abc_structural_clue":
        if prompt.count(STRUCTURAL_CHOICE_CLUE) != 1:
            raise ValueError("structural arm lacks exactly one clue")
    elif STRUCTURAL_CHOICE_CLUE in prompt:
        raise ValueError("non-structural arm contains the clue")


def _nearer(value: float, a: float, b: float) -> str:
    return "A" if abs(value - a) < abs(value - b) else "B"


def build_records(
    triplet: Mapping[str, Any],
    *,
    episode_index: int,
    k: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    task_id = f"c2v13_{episode_index:04d}_k{k}"
    prompts = {arm: build_prompt(triplet, k=k, arm=arm) for arm in PROMPT_ARMS}
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
    reference_estimates = compute_reference_estimates(triplet)
    evaluator = {
        "task_id": task_id,
        "episode_id": f"c2v13_{episode_index:04d}",
        "episode_seed": triplet["seed"],
        "k": k,
        "target_cases_seen": cases_seen,
        "public": {
            "reference_a": {
                key: city_a[key]
                for key in ("profile_index", "starting_poll", "news", "ending_polls")
            },
            "reference_b": {
                key: city_b[key]
                for key in ("profile_index", "starting_poll", "news", "ending_polls")
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
            "relevant_reference": triplet["relevant_reference"],
            "profile_selected_reference": reference_estimates["profile_selected_reference"],
            "response_nearest_reference": _nearer(
                float(target["response"]),
                float(city_a["response"]),
                float(city_b["response"]),
            ),
            "target_profile_position": round(float(triplet["target_profile_position"]), 8),
            "true_response_a": round(float(city_a["response"]), 8),
            "true_response_b": round(float(city_b["response"]), 8),
            "latent_regime_gap": round(float(triplet["latent_regime_gap"]), 8),
        },
        "gold": {
            "expected_poll": gold_expected_poll(triplet),
            "target_response": round(float(target["response"]), 8),
        },
        "baselines": {
            "target_only": compute_target_only(triplet, k),
            "reference_estimates": reference_estimates,
            "abc_no_structure": compute_abc_no_structure(triplet, k),
            "abc_structural": compute_abc_structural(triplet, k),
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
        arm: args.outdir / f"tasks_c2_v13_{arm}.jsonl" for arm in PROMPT_ARMS
    }
    key_path = args.outdir / "answer_key_c2_v13.jsonl"
    manifest_path = args.outdir / "tasks_c2_v13.manifest.json"
    existing = [
        path for path in (*task_paths.values(), key_path, manifest_path) if path.exists()
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
                    records, evaluator = build_records(triplet, episode_index=index, k=k)
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
        "engine_sha256": _ROOT / "engine" / "three_city_c2_v13.py",
        "builder_sha256": Path(__file__),
        "config_sha256": _CONFIG,
        "validator_sha256": _HERE / "validate_three_city_c2_v13_tasks.py",
        "runner_sha256": _HERE / "run_three_city_c2_v13_confirmatory.py",
        "launcher_sha256": _HERE / "run_three_city_c2_v13_local.sh",
        "analyzer_sha256": _ROOT / "analysis" / "analyze_three_city_c2_v13_confirmatory.py",
        "renderer_sha256": _ROOT / "analysis" / "render_three_city_c2_v13_exact_structure.py",
        "live_renderer_sha256": _ROOT / "analysis" / "live_exact_structure_three_city_c2_v13.py",
        "preregistration_sha256": args.preregistration,
    }
    manifest = {
        "experiment": "three_city_c2_v13_two_regime_structural_choice",
        "status": "frozen_no_model_runs",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episodes": args.n,
        "seed_offset": args.seed_offset,
        "arms": list(PROMPT_ARMS),
        "prefixes": list(PREFIX_LADDER),
        "cases_by_prefix": {str(k): CASES_BY_PREFIX[k] for k in PREFIX_LADDER},
        "tasks_per_arm": count,
        "calls_per_model": count * len(PROMPT_ARMS),
        "total_calls_three_models": count * len(PROMPT_ARMS) * 3,
        "reference_cases_per_city": REFERENCE_CASES,
        "case_sigma": CASE_SIGMA,
        "city_deviation_sd": CITY_DEVIATION_SD,
        "prior_effective_cases": PRIOR_EFFECTIVE_CASES,
        "response_center_range": list(EPISODE_RESPONSE_CENTER_RANGE),
        "regime_gap_range": list(EPISODE_REGIME_GAP_RANGE),
        "low_profile_range": list(LOW_PROFILE_RANGE),
        "high_profile_range": list(HIGH_PROFILE_RANGE),
        "target_near_endpoint_range": list(TARGET_NEAR_ENDPOINT_RANGE),
        "fresh_seeds_relative_to_v12": True,
        "balanced_relevant_reference": True,
        "clue_reveals_reference_identity": False,
        "neutral_and_structural_prior_strength_matched": True,
        "pre_model_revision": (
            "reference tables reduced from 8 to 4 cases and matched prior "
            "strength reduced from 2.5 to 1.75 before any model calls"
        ),
        "previous_frozen_reference_cases_per_city": 8,
        "previous_frozen_prior_effective_cases": 2.5,
        "parameter_development_seed_range": [200000, 200799],
        "pre_revision_holdout_seed_range": [103000, 103119],
        "final_confirmatory_seed_range": [105000, 105119],
        "holdout_evaluated_once_after_parameter_lock": True,
        "final_pre_model_parameter_revision": (
            "case noise 7.5 to 9.5, regime gap [1.3, 1.7] to [1.7, 2.1], "
            "matched prior strength 1.75 to 2.0, and final seeds 103000-103119"
        ),
        "final_reference_count_revision": (
            "reference tables reduced from 4 to 2 cases per city before model "
            "collection; no requirement that neutral pooling beat C-only after k=2"
        ),
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
