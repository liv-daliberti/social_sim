#!/usr/bin/env python3
"""Validate the frozen C2 v13 matched two-regime design."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from build_three_city_c2_v13_tasks import (
    STRUCTURAL_CHOICE_CLUE,
    strip_structural_clue,
    target_suffix,
    validate_arm_prompt,
)
from engine.three_city_c2_v13 import (
    CASES_BY_PREFIX,
    CASE_SIGMA,
    CITY_DEVIATION_SD,
    PREFIX_LADDER,
    PRIOR_EFFECTIVE_CASES,
    PROMPT_ARMS,
    REFERENCE_CASES,
    SEED_OFFSET,
    cases_for_prefix,
)

_DATA = (
    _ROOT / "data" / "three_city_c2_v13"
    / "full_k1_5_two_regime_structural_choice" / "design"
)
_EPISODES = 120
_TASKS = _EPISODES * len(PREFIX_LADDER)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _index(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(row["task_id"].split("_")[1]), int(row["task_id"].rsplit("k", 1)[1])): row
        for row in rows
    }


def _mae(keys: list[dict[str, Any]], method: str, k: int) -> float:
    return statistics.mean(
        abs(
            float(row["baselines"][method]["predicted_poll"])
            - float(row["gold"]["expected_poll"])
        )
        for row in keys if int(row["k"]) == k
    )


def validate(
    task_paths: Mapping[str, Path],
    key_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    tasks = {arm: _read(path) for arm, path in task_paths.items()}
    keys = _read(key_path)
    manifest = json.loads(manifest_path.read_text())
    checks: dict[str, bool] = {}
    by_arm = {arm: _index(rows) for arm, rows in tasks.items()}
    ids = {arm: {row["task_id"] for row in rows} for arm, rows in tasks.items()}
    key_ids = {row["task_id"] for row in keys}
    checks["counts_and_ids"] = (
        all(len(rows) == _TASKS for rows in tasks.values())
        and len(keys) == _TASKS
        and all(values == key_ids for values in ids.values())
    )
    checks["prompt_only_records_and_hashes"] = all(
        set(row) == {"task_id", "prompt", "prompt_sha256"}
        and row["prompt_sha256"] == hashlib.sha256(row["prompt"].encode()).hexdigest()
        for rows in tasks.values() for row in rows
    )
    prompt_failures = []
    for arm, rows in tasks.items():
        for row in rows:
            try:
                validate_arm_prompt(row["prompt"], arm)
            except ValueError as exc:
                prompt_failures.append(f"{arm}:{row['task_id']}:{exc}")
    checks["all_prompts_validate"] = not prompt_failures
    checks["c_differs_from_b_only_by_exact_clue"] = all(
        strip_structural_clue(by_arm["abc_structural_clue"][key]["prompt"])
        == by_arm["abc"][key]["prompt"]
        and by_arm["abc_structural_clue"][key]["prompt"].count(STRUCTURAL_CHOICE_CLUE) == 1
        for key in by_arm["abc"]
    )
    checks["city_c_information_identical"] = all(
        len({target_suffix(by_arm[arm][key]["prompt"]) for arm in PROMPT_ARMS}) == 1
        for key in by_arm["c_only"]
    )
    clue_lower = STRUCTURAL_CHOICE_CLUE.lower()
    checks["clue_gives_rule_not_reference_identity"] = (
        "whichever reference city has the closer" in clue_lower
        and "does not say whether a or b is closer" in clue_lower
        and "city a is closer" not in clue_lower
        and "city b is closer" not in clue_lower
    )
    episode_seeds = {int(row["episode_seed"]) for row in keys}
    checks["round_and_fresh_seed_design"] = (
        {int(row["k"]) for row in keys} == set(PREFIX_LADDER)
        and all(int(row["target_cases_seen"]) == cases_for_prefix(int(row["k"])) for row in keys)
        and episode_seeds == set(range(SEED_OFFSET, SEED_OFFSET + _EPISODES))
        and not (episode_seeds & set(range(93_000, 93_120)))
        and not (episode_seeds & set(range(95_000, 95_120)))
        and not (episode_seeds & set(range(103_000, 103_120)))
        and not (episode_seeds & set(range(200_000, 200_800)))
    )

    episodes = [row for row in keys if int(row["k"]) == 5]
    balance = Counter(
        (row["design_truth"]["high_reference"], row["design_truth"]["relevant_reference"])
        for row in episodes
    )
    checks["factorial_balance"] = balance == Counter(
        {("A", "A"): 30, ("A", "B"): 30, ("B", "A"): 30, ("B", "B"): 30}
    )
    profile_gaps = []
    choice_strengths = []
    displayed_separations = []
    profile_correct = []
    response_nearest = []
    for row in episodes:
        public = row["public"]
        truth = row["design_truth"]
        pa = float(public["reference_a"]["profile_index"])
        pb = float(public["reference_b"]["profile_index"])
        pc = float(public["target"]["profile_index"])
        span = abs(pa - pb)
        profile_gaps.append(span)
        choice_strengths.append(2.0 * abs(pc - (pa + pb) / 2.0) / span)
        profile_correct.append(truth["profile_selected_reference"] == truth["relevant_reference"])
        response_nearest.append(truth["response_nearest_reference"] == truth["relevant_reference"])
        displayed_separations.append(
            float(row["baselines"]["reference_estimates"]["displayed_poll_separation"])
        )
    checks["profile_choice_is_decisive"] = (
        min(profile_gaps) >= 43.0
        and min(choice_strengths) >= 0.63
        and all(profile_correct)
    )
    checks["response_regime_matches_choice"] = statistics.mean(response_nearest) >= 0.98
    checks["displayed_ab_separation"] = statistics.median(displayed_separations) >= 14.0

    target = {k: _mae(keys, "target_only", k) for k in PREFIX_LADDER}
    neutral = {k: _mae(keys, "abc_no_structure", k) for k in PREFIX_LADDER}
    structural = {k: _mae(keys, "abc_structural", k) for k in PREFIX_LADDER}
    checks["round1_ordered_contrast"] = (
        target[1] > neutral[1] > structural[1]
        and target[1] - structural[1] >= 2.5
    )
    checks["neutral_beats_c_only_first_two_rounds"] = all(
        neutral[k] < target[k] for k in (1, 2)
    )
    checks["structural_beats_neutral_every_round"] = all(
        structural[k] < neutral[k] for k in PREFIX_LADDER
    )
    checks["round5_shared_convergence"] = (
        max(target[5], neutral[5], structural[5])
        - min(target[5], neutral[5], structural[5]) < 0.35
    )
    checks["pooling_strength_exactly_matched"] = all(
        float(row["baselines"]["abc_no_structure"]["prior_weight"])
        == float(row["baselines"]["abc_structural"]["prior_weight"])
        and float(row["baselines"]["abc_no_structure"]["target_weight"])
        == float(row["baselines"]["abc_structural"]["target_weight"])
        for row in keys
    )

    checks["manifest_task_hashes"] = (
        all(manifest["model_task_files"][arm]["sha256"] == _hash(task_paths[arm]) for arm in PROMPT_ARMS)
        and manifest["answer_key_sha256"] == _hash(key_path)
    )
    source_paths = {
        "engine_sha256": _ROOT / "engine" / "three_city_c2_v13.py",
        "builder_sha256": _HERE / "build_three_city_c2_v13_tasks.py",
        "config_sha256": _ROOT / "configs" / "three_city_c2_v13.yml",
        "validator_sha256": Path(__file__),
        "runner_sha256": _HERE / "run_three_city_c2_v13_confirmatory.py",
        "launcher_sha256": _HERE / "run_three_city_c2_v13_local.sh",
        "analyzer_sha256": _ROOT / "analysis" / "analyze_three_city_c2_v13_confirmatory.py",
        "renderer_sha256": _ROOT / "analysis" / "render_three_city_c2_v13_exact_structure.py",
        "live_renderer_sha256": _ROOT / "analysis" / "live_exact_structure_three_city_c2_v13.py",
        "preregistration_sha256": _ROOT / "PREREGISTRATION_THREE_CITY_C2_V13.md",
    }
    checks["manifest_source_hashes"] = all(
        manifest[field] == _hash(path) for field, path in source_paths.items()
    )
    checks["manifest_parameters"] = (
        manifest["experiment"] == "three_city_c2_v13_two_regime_structural_choice"
        and float(manifest["case_sigma"]) == CASE_SIGMA
        and float(manifest["city_deviation_sd"]) == CITY_DEVIATION_SD
        and float(manifest["prior_effective_cases"]) == PRIOR_EFFECTIVE_CASES
        and REFERENCE_CASES == 2
        and int(manifest["reference_cases_per_city"]) == REFERENCE_CASES
        and manifest["fresh_seeds_relative_to_v12"] is True
        and manifest["clue_reveals_reference_identity"] is False
        and manifest["neutral_and_structural_prior_strength_matched"] is True
        and manifest["previous_frozen_reference_cases_per_city"] == 8
        and float(manifest["previous_frozen_prior_effective_cases"]) == 2.5
        and manifest["parameter_development_seed_range"] == [200000, 200799]
        and manifest["pre_revision_holdout_seed_range"] == [103000, 103119]
        and manifest["final_confirmatory_seed_range"] == [105000, 105119]
        and manifest["holdout_evaluated_once_after_parameter_lock"] is True
    )
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "experiment": "three_city_c2_v13_two_regime_structural_choice",
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "diagnostics": {
            "target_only_mae_by_round": target,
            "abc_no_structure_mae_by_round": neutral,
            "abc_structural_mae_by_round": structural,
            "neutral_gain_over_target_by_round": {k: target[k] - neutral[k] for k in PREFIX_LADDER},
            "structural_gain_over_neutral_by_round": {k: neutral[k] - structural[k] for k in PREFIX_LADDER},
            "round5_three_estimator_spread": max(target[5], neutral[5], structural[5]) - min(target[5], neutral[5], structural[5]),
            "minimum_profile_gap": min(profile_gaps),
            "minimum_choice_strength": min(choice_strengths),
            "median_displayed_ab_separation": statistics.median(displayed_separations),
            "profile_choice_accuracy": statistics.mean(profile_correct),
            "response_nearest_accuracy": statistics.mean(response_nearest),
            "prompt_failures": prompt_failures,
        },
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=_DATA)
    args = parser.parse_args()
    result = validate(
        {arm: args.data / f"tasks_c2_v13_{arm}.jsonl" for arm in PROMPT_ARMS},
        args.data / "answer_key_c2_v13.jsonl",
        args.data / "tasks_c2_v13.manifest.json",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    output = args.data / "validation_c2_v13.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
