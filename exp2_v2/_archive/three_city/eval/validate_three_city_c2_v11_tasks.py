#!/usr/bin/env python3
"""Validate the frozen C2 v11 identifiable structural-choice design."""

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

import build_three_city_c2_v9_tasks as v9_builder
from build_three_city_c2_v11_tasks import (
    STRUCTURAL_CHOICE_CLUE,
    strip_structural_clue,
    target_suffix,
    validate_arm_prompt,
)
from engine.three_city_c2_v11 import (
    CASES_BY_PREFIX,
    CASE_SIGMA,
    CITY_DEVIATION_SD,
    EPISODE_INTERCEPT_RANGE,
    EPISODE_PROFILE_SLOPE_RANGE,
    HIGH_PROFILE_RANGE,
    LOW_PROFILE_RANGE,
    NEWS_VALUE,
    PREFIX_LADDER,
    PROMPT_ARMS,
    TARGET_NEAR_ENDPOINT_RANGE,
    cases_for_prefix,
)

_DATA = (
    _ROOT
    / "data"
    / "three_city_c2_v11"
    / "full_k1_5_identifiable_structural_choice"
    / "design"
)
_EPISODES = 120
_TASKS_PER_ARM = _EPISODES * len(PREFIX_LADDER)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _mean_mae(keys: list[dict[str, Any]], method: str, k: int) -> float:
    rows = [row for row in keys if int(row["k"]) == k]
    return statistics.mean(
        abs(
            float(row["baselines"][method]["predicted_poll"])
            - float(row["gold"]["expected_poll"])
        )
        for row in rows
    )


def _index(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    indexed = {}
    for row in rows:
        episode = int(row["task_id"].split("_")[1])
        k = int(row["task_id"].rsplit("k", 1)[1])
        indexed[(episode, k)] = row
    return indexed


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _nearer(value: float, a: float, b: float) -> str:
    return "A" if abs(value - a) < abs(value - b) else "B"


def _prompt_triplet(row: Mapping[str, Any]) -> dict[str, Any]:
    public = row["public"]
    return {
        "reference_a": public["reference_a"],
        "reference_b": public["reference_b"],
        "target": public["target"],
        "test_news": public["test_news"],
    }


def validate(
    *,
    task_paths: Mapping[str, Path],
    answer_key_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    tasks = {arm: _read_jsonl(path) for arm, path in task_paths.items()}
    keys = _read_jsonl(answer_key_path)
    manifest = json.loads(manifest_path.read_text())
    checks: dict[str, bool] = {}
    diagnostics: dict[str, Any] = {}

    checks["expected_tasks_per_arm"] = all(
        len(rows) == _TASKS_PER_ARM for rows in tasks.values()
    )
    checks["expected_answer_keys"] = len(keys) == _TASKS_PER_ARM
    id_sets = {
        arm: {row["task_id"] for row in rows}
        for arm, rows in tasks.items()
    }
    key_ids = {row["task_id"] for row in keys}
    checks["unique_ids_within_every_file"] = all(
        len(id_sets[arm]) == len(tasks[arm]) for arm in PROMPT_ARMS
    ) and len(key_ids) == len(keys)
    checks["same_ids_across_arms_and_key"] = all(
        values == key_ids for values in id_sets.values()
    )
    checks["model_records_contain_only_prompt_material"] = all(
        set(row) == {"task_id", "prompt", "prompt_sha256"}
        for rows in tasks.values()
        for row in rows
    )
    checks["stored_prompt_hashes_match"] = all(
        row["prompt_sha256"]
        == hashlib.sha256(row["prompt"].encode()).hexdigest()
        for rows in tasks.values()
        for row in rows
    )

    prompt_failures = []
    by_arm = {arm: _index(rows) for arm, rows in tasks.items()}
    key_index = _index(keys)
    for arm, rows in tasks.items():
        for row in rows:
            try:
                validate_arm_prompt(row["prompt"], arm)
            except ValueError as exc:
                prompt_failures.append(f"{arm}:{row['task_id']}:{exc}")
    checks["all_arm_prompts_validate"] = not prompt_failures
    checks["structural_arm_differs_only_by_fixed_clue"] = all(
        strip_structural_clue(by_arm["abc_structural_clue"][key]["prompt"])
        == by_arm["abc"][key]["prompt"]
        and by_arm["abc_structural_clue"][key]["prompt"].count(
            STRUCTURAL_CHOICE_CLUE
        )
        == 1
        for key in by_arm["abc"]
    )
    checks["city_c_information_identical_across_arms"] = all(
        len(
            {
                target_suffix(by_arm[arm][key]["prompt"])
                for arm in PROMPT_ARMS
            }
        )
        == 1
        for key in by_arm["c_only"]
    )
    clue_lower = STRUCTURAL_CHOICE_CLUE.lower()
    checks["clue_does_not_reveal_city_choice"] = (
        "closer to city a" not in clue_lower
        and "closer to city b" not in clue_lower
        and "city a is" not in clue_lower
        and "city b is" not in clue_lower
        and "distance" not in clue_lower
        and "one of cities a and b" in clue_lower
    )
    checks["clue_is_fixed_across_all_tasks"] = all(
        row["prompt"].count(STRUCTURAL_CHOICE_CLUE) == 1
        for row in tasks["abc_structural_clue"]
    )
    checks["a_and_b_use_exact_v9_prompt_templates"] = all(
        by_arm[arm][index_key]["prompt"]
        == v9_builder.build_prompt(
            _prompt_triplet(key_index[index_key]),
            k=index_key[1],
            arm=arm,
        )
        for arm in ("c_only", "abc")
        for index_key in by_arm[arm]
    )

    checks["expected_120_episodes"] = len(
        {row["episode_id"] for row in keys}
    ) == _EPISODES
    checks["exact_round_ladder"] = {
        int(row["k"]) for row in keys
    } == set(PREFIX_LADDER)
    checks["exact_case_ladder"] = all(
        int(row["target_cases_seen"]) == cases_for_prefix(int(row["k"]))
        for row in keys
    )

    episode_rows = [row for row in keys if int(row["k"]) == 5]
    profile_gaps = []
    choice_strengths = []
    true_response_separations = []
    displayed_response_separations = []
    profile_response_agreement = []
    design_truth_consistent = []
    balance = Counter()
    for row in episode_rows:
        public = row["public"]
        truth = row["design_truth"]
        pa = float(public["reference_a"]["profile_index"])
        pb = float(public["reference_b"]["profile_index"])
        pc = float(public["target"]["profile_index"])
        span = abs(pa - pb)
        profile_gaps.append(span)
        choice_strengths.append(2.0 * abs(pc - (pa + pb) / 2.0) / span)
        ra = float(truth["true_response_a"])
        rb = float(truth["true_response_b"])
        rc = float(row["gold"]["target_response"])
        profile_label = _nearer(pc, pa, pb)
        response_label = _nearer(rc, ra, rb)
        design_truth_consistent.append(
            profile_label == truth["profile_nearest_reference"]
            and response_label == truth["response_nearest_reference"]
            and profile_label == row["design_truth"]["profile_nearest_reference"]
        )
        profile_response_agreement.append(profile_label == response_label)
        true_response_separations.append(NEWS_VALUE * abs(ra - rb))
        fitted = row["baselines"]["reference_interpolation"]
        displayed_response_separations.append(
            NEWS_VALUE
            * abs(float(fitted["response_a"]) - float(fitted["response_b"]))
        )
        balance[(truth["high_reference"], profile_label)] += 1

    observed_choice_accuracy: dict[int, float] = {}
    for k in PREFIX_LADDER:
        correct = []
        for row in keys:
            if int(row["k"]) != k:
                continue
            target_hat = float(row["baselines"]["target_only"]["response_hat"])
            fitted = row["baselines"]["reference_interpolation"]
            observed_label = _nearer(
                target_hat,
                float(fitted["response_a"]),
                float(fitted["response_b"]),
            )
            correct.append(
                observed_label
                == row["design_truth"]["profile_nearest_reference"]
            )
        observed_choice_accuracy[k] = statistics.mean(correct)

    checks["design_truth_is_derived_consistently"] = all(
        design_truth_consistent
    )
    checks["factorial_balance_is_exact"] = balance == Counter(
        {("A", "A"): 30, ("A", "B"): 30,
         ("B", "A"): 30, ("B", "B"): 30}
    )
    checks["reference_profiles_are_widely_separated"] = min(profile_gaps) >= 39.5
    checks["target_is_decisively_near_one_endpoint"] = min(choice_strengths) >= 0.54
    agreement_rate = statistics.mean(profile_response_agreement)
    checks["profile_and_response_analogue_agree"] = agreement_rate >= 0.90
    checks["displayed_reference_outcomes_are_separated"] = (
        statistics.median(displayed_response_separations) >= 5.0
        and _quantile(displayed_response_separations, 0.10) >= 2.0
    )
    choice_gates = {1: 0.65, 2: 0.70, 3: 0.75, 4: 0.80, 5: 0.85}
    checks["observed_choice_accuracy_passes_every_round"] = all(
        observed_choice_accuracy[k] >= choice_gates[k]
        for k in PREFIX_LADDER
    )

    target_mae = {
        k: _mean_mae(keys, "target_only", k) for k in PREFIX_LADDER
    }
    abc_mae = {
        k: _mean_mae(keys, "abc_shrinkage", k) for k in PREFIX_LADDER
    }
    gaps = {k: target_mae[k] - abc_mae[k] for k in PREFIX_LADDER}
    weights = {
        k: statistics.mean(
            float(row["baselines"]["abc_shrinkage"]["reference_weight"])
            for row in keys
            if int(row["k"]) == k
        )
        for k in PREFIX_LADDER
    }
    checks["large_round1_information_value"] = gaps[1] > 1.75
    checks["large_round2_information_value"] = gaps[2] > 0.90
    checks["abc_beats_c_only_rounds_1_to_4"] = all(
        abc_mae[k] < target_mae[k] for k in (1, 2, 3, 4)
    )
    checks["matched_convergence_by_round5"] = (
        abs(gaps[5]) < 0.12
        and target_mae[5] < 0.90
        and abc_mae[5] < 0.90
    )
    checks["reference_weight_strictly_decreases"] = all(
        weights[a] > weights[b]
        for a, b in zip(PREFIX_LADDER, PREFIX_LADDER[1:])
    )

    checks["manifest_declares_identifiable_design"] = (
        manifest.get("experiment")
        == "three_city_c2_v11_identifiable_structural_choice"
        and manifest.get("arms") == list(PROMPT_ARMS)
        and manifest.get("numerical_dgp_identical_to_v9") is False
        and manifest.get("a_and_b_prompt_templates_identical_to_v9") is True
        and manifest.get("balanced_target_nearer_reference") is True
        and manifest.get("clue_reveals_closer_city") is False
        and manifest.get("structural_choice_clue") == STRUCTURAL_CHOICE_CLUE
        and manifest.get("cases_by_prefix")
        == {str(k): CASES_BY_PREFIX[k] for k in PREFIX_LADDER}
        and float(manifest.get("case_sigma")) == CASE_SIGMA
        and float(manifest.get("city_deviation_sd")) == CITY_DEVIATION_SD
        and manifest.get("low_profile_range") == list(LOW_PROFILE_RANGE)
        and manifest.get("high_profile_range") == list(HIGH_PROFILE_RANGE)
        and manifest.get("target_near_endpoint_range")
        == list(TARGET_NEAR_ENDPOINT_RANGE)
        and manifest.get("intercept_range") == list(EPISODE_INTERCEPT_RANGE)
        and manifest.get("profile_slope_range")
        == list(EPISODE_PROFILE_SLOPE_RANGE)
    )
    checks["manifest_task_and_answer_hashes_match"] = (
        all(
            manifest["model_task_files"][arm]["sha256"]
            == _file_sha256(task_paths[arm])
            for arm in PROMPT_ARMS
        )
        and manifest["answer_key_sha256"] == _file_sha256(answer_key_path)
    )
    aggregate_ok = True
    for arm in PROMPT_ARMS:
        digest = hashlib.sha256()
        for row in tasks[arm]:
            digest.update(row["task_id"].encode())
            digest.update(row["prompt"].encode())
        aggregate_ok = aggregate_ok and (
            digest.hexdigest()
            == manifest["model_task_files"][arm]["aggregate_prompt_sha256"]
        )
    checks["manifest_aggregate_prompt_hashes_match"] = aggregate_ok

    source_paths = {
        "engine_sha256": _ROOT / "engine" / "three_city_c2_v11.py",
        "source_engine_v9_sha256": _ROOT / "engine" / "three_city_c2_v9.py",
        "builder_sha256": _HERE / "build_three_city_c2_v11_tasks.py",
        "source_builder_v9_sha256": _HERE / "build_three_city_c2_v9_tasks.py",
        "config_sha256": _ROOT / "configs" / "three_city_c2_v11.yml",
        "validator_sha256": Path(__file__),
        "runner_sha256": _HERE / "run_three_city_c2_v11_confirmatory.py",
        "source_runner_v9_sha256": _HERE / "run_three_city_c2_v9_confirmatory.py",
        "launcher_sha256": _HERE / "run_three_city_c2_v11_local.sh",
        "analyzer_sha256": _ROOT / "analysis" / "analyze_three_city_c2_v11_confirmatory.py",
        "source_analyzer_v9_sha256": _ROOT / "analysis" / "analyze_three_city_c2_v9_confirmatory.py",
        "live_monitor_sha256": _ROOT / "analysis" / "live_monitor_three_city_c2_v11.py",
        "prompt_figure_sha256": _ROOT / "analysis" / "preview_three_city_c2_v11_prompts.py",
        "preregistration_sha256": _ROOT / "PREREGISTRATION_THREE_CITY_C2_V11.md",
    }
    checks["manifest_source_hashes_match"] = all(
        manifest[field] == _file_sha256(path)
        for field, path in source_paths.items()
    )

    diagnostics.update(
        {
            "task_counts": {arm: len(rows) for arm, rows in tasks.items()},
            "answer_key_count": len(keys),
            "target_only_mae_by_round": target_mae,
            "abc_shrinkage_mae_by_round": abc_mae,
            "information_value_gap_by_round": gaps,
            "abc_reference_weight_by_round": weights,
            "reference_profile_gap": {
                "minimum": min(profile_gaps),
                "median": statistics.median(profile_gaps),
            },
            "endpoint_choice_strength": {
                "minimum": min(choice_strengths),
                "median": statistics.median(choice_strengths),
            },
            "true_reference_separation_poll_points": {
                "p10": _quantile(true_response_separations, 0.10),
                "median": statistics.median(true_response_separations),
            },
            "displayed_reference_separation_poll_points": {
                "p10": _quantile(displayed_response_separations, 0.10),
                "median": statistics.median(displayed_response_separations),
            },
            "profile_response_nearest_agreement": agreement_rate,
            "observed_choice_accuracy_by_round": observed_choice_accuracy,
            "factorial_balance_high_by_nearer": {
                f"high_{high}_near_{near}": count
                for (high, near), count in sorted(balance.items())
            },
            "structural_choice_clue": STRUCTURAL_CHOICE_CLUE,
            "prompt_failures": prompt_failures,
        }
    )
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "experiment": "three_city_c2_v11_identifiable_structural_choice",
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "diagnostics": diagnostics,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=_DATA)
    args = parser.parse_args()
    task_paths = {
        arm: args.data / f"tasks_c2_v11_{arm}.jsonl"
        for arm in PROMPT_ARMS
    }
    result = validate(
        task_paths=task_paths,
        answer_key_path=args.data / "answer_key_c2_v11.jsonl",
        manifest_path=args.data / "tasks_c2_v11.manifest.json",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    output = args.data / "validation_c2_v11.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
