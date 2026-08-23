#!/usr/bin/env python3
"""Validate the frozen C2 v12 noisier structural-choice design."""

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

from build_three_city_c2_v12_tasks import (
    STRUCTURAL_CHOICE_CLUE,
    strip_structural_clue,
    target_suffix,
    validate_arm_prompt,
)
from engine.three_city_c2_v12 import (
    CASES_BY_PREFIX,
    CASE_SIGMA,
    CITY_DEVIATION_SD,
    NEWS_VALUE,
    PREFIX_LADDER,
    PROMPT_ARMS,
    SEED_OFFSET,
    cases_for_prefix,
)

_DATA = (
    _ROOT / "data" / "three_city_c2_v12"
    / "full_k1_5_noisier_structural_choice" / "design"
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


def _nearer(value: float, a: float, b: float) -> str:
    return "A" if abs(value - a) < abs(value - b) else "B"


def _mae(keys: list[dict[str, Any]], method: str, k: int) -> float:
    return statistics.mean(
        abs(
            float(row["baselines"][method]["predicted_poll"])
            - float(row["gold"]["expected_poll"])
        )
        for row in keys if int(row["k"]) == k
    )


def validate(
    task_paths: Mapping[str, Path], key_path: Path, manifest_path: Path
) -> dict[str, Any]:
    tasks = {arm: _read(path) for arm, path in task_paths.items()}
    keys = _read(key_path)
    manifest = json.loads(manifest_path.read_text())
    checks: dict[str, bool] = {}
    by_arm = {arm: _index(rows) for arm, rows in tasks.items()}
    key_index = _index(keys)
    ids = {arm: set(row["task_id"] for row in rows) for arm, rows in tasks.items()}
    key_ids = set(row["task_id"] for row in keys)
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
    lower = STRUCTURAL_CHOICE_CLUE.lower()
    checks["clue_does_not_reveal_choice"] = (
        "one of cities a and b" in lower
        and "closer to city a" not in lower
        and "closer to city b" not in lower
        and "distance" not in lower
    )
    checks["round_and_fresh_seed_design"] = (
        {int(row["k"]) for row in keys} == set(PREFIX_LADDER)
        and all(int(row["target_cases_seen"]) == cases_for_prefix(int(row["k"])) for row in keys)
        and {int(row["episode_seed"]) for row in keys}
        == set(range(SEED_OFFSET, SEED_OFFSET + _EPISODES))
        and not ({int(row["episode_seed"]) for row in keys} & set(range(91_000, 91_120)))
    )

    episodes = [row for row in keys if int(row["k"]) == 5]
    gaps = []
    strengths = []
    displayed = []
    agreement = []
    balance = Counter()
    for row in episodes:
        public = row["public"]
        truth = row["design_truth"]
        pa = float(public["reference_a"]["profile_index"])
        pb = float(public["reference_b"]["profile_index"])
        pc = float(public["target"]["profile_index"])
        span = abs(pa - pb)
        gaps.append(span)
        strengths.append(2 * abs(pc - (pa + pb) / 2) / span)
        profile_label = _nearer(pc, pa, pb)
        response_label = _nearer(
            float(row["gold"]["target_response"]),
            float(truth["true_response_a"]),
            float(truth["true_response_b"]),
        )
        agreement.append(profile_label == response_label)
        balance[(truth["high_reference"], profile_label)] += 1
        ref = row["baselines"]["reference_interpolation"]
        displayed.append(
            NEWS_VALUE * abs(float(ref["response_a"]) - float(ref["response_b"]))
        )
    choice = {}
    for k in PREFIX_LADDER:
        correct = []
        for row in keys:
            if int(row["k"]) != k:
                continue
            ref = row["baselines"]["reference_interpolation"]
            observed = _nearer(
                float(row["baselines"]["target_only"]["response_hat"]),
                float(ref["response_a"]), float(ref["response_b"]),
            )
            correct.append(observed == row["design_truth"]["profile_nearest_reference"])
        choice[k] = statistics.mean(correct)
    checks["factorial_balance"] = balance == Counter(
        {("A", "A"): 30, ("A", "B"): 30, ("B", "A"): 30, ("B", "B"): 30}
    )
    checks["profile_structure_is_decisive"] = min(gaps) >= 39.5 and min(strengths) >= 0.54
    checks["profile_response_agreement"] = statistics.mean(agreement) >= 0.90
    checks["displayed_ab_separation"] = statistics.median(displayed) >= 7.0
    checks["observed_choice_identifiable"] = choice[1] >= 0.65 and choice[5] >= 0.85

    target = {k: _mae(keys, "target_only", k) for k in PREFIX_LADDER}
    matched = {k: _mae(keys, "abc_shrinkage", k) for k in PREFIX_LADDER}
    info = {k: target[k] - matched[k] for k in PREFIX_LADDER}
    checks["both_estimators_are_harder_than_v11"] = (
        target[1] > 3.320 and matched[1] > 1.218
        and target[5] > 0.727 and matched[5] > 0.658
    )
    checks["large_early_gap_and_late_convergence"] = info[1] > 2.5 and abs(info[5]) < 0.20
    checks["matched_beats_target_all_rounds"] = all(matched[k] < target[k] for k in PREFIX_LADDER)

    checks["manifest_task_hashes"] = (
        all(manifest["model_task_files"][arm]["sha256"] == _hash(task_paths[arm]) for arm in PROMPT_ARMS)
        and manifest["answer_key_sha256"] == _hash(key_path)
    )
    source_paths = {
        "engine_sha256": _ROOT / "engine" / "three_city_c2_v12.py",
        "builder_sha256": _HERE / "build_three_city_c2_v12_tasks.py",
        "source_prompt_builder_v11_sha256": _HERE / "build_three_city_c2_v11_tasks.py",
        "config_sha256": _ROOT / "configs" / "three_city_c2_v12.yml",
        "validator_sha256": Path(__file__),
        "runner_sha256": _HERE / "run_three_city_c2_v12_confirmatory.py",
        "launcher_sha256": _HERE / "run_three_city_c2_v12_local.sh",
        "analyzer_sha256": _ROOT / "analysis" / "analyze_three_city_c2_v12_confirmatory.py",
        "preregistration_sha256": _ROOT / "PREREGISTRATION_THREE_CITY_C2_V12.md",
    }
    checks["manifest_source_hashes"] = all(manifest[field] == _hash(path) for field, path in source_paths.items())
    checks["manifest_parameters"] = (
        manifest["experiment"] == "three_city_c2_v12_noisier_structural_choice"
        and float(manifest["case_sigma"]) == CASE_SIGMA
        and float(manifest["city_deviation_sd"]) == CITY_DEVIATION_SD
        and manifest["fresh_seeds_relative_to_v11"] is True
        and manifest["clue_reveals_closer_city"] is False
    )
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "experiment": "three_city_c2_v12_noisier_structural_choice",
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "diagnostics": {
            "target_only_mae_by_round": target,
            "abc_shrinkage_mae_by_round": matched,
            "information_gap_by_round": info,
            "minimum_profile_gap": min(gaps),
            "minimum_choice_strength": min(strengths),
            "median_displayed_ab_separation": statistics.median(displayed),
            "profile_response_agreement": statistics.mean(agreement),
            "observed_choice_accuracy_by_round": choice,
            "prompt_failures": prompt_failures,
        },
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=_DATA)
    args = parser.parse_args()
    result = validate(
        {arm: args.data / f"tasks_c2_v12_{arm}.jsonl" for arm in PROMPT_ARMS},
        args.data / "answer_key_c2_v12.jsonl",
        args.data / "tasks_c2_v12.manifest.json",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    output = args.data / "validation_c2_v12.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
