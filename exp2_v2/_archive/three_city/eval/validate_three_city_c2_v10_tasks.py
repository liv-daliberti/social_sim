#!/usr/bin/env python3
"""Validate the frozen C2 v10 structural-choice-clue paper design."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from build_three_city_c2_v10_tasks import (
    STRUCTURAL_CHOICE_CLUE,
    strip_structural_clue,
    target_suffix,
    validate_arm_prompt,
)
from engine.three_city_c2_v10 import (
    CASES_BY_PREFIX,
    CASE_SIGMA,
    CITY_DEVIATION_SD,
    PREFIX_LADDER,
    PROMPT_ARMS,
    cases_for_prefix,
)

_DATA = (
    _ROOT
    / "data"
    / "three_city_c2_v10"
    / "full_k1_5_structural_choice"
    / "design"
)
_V9_DATA = (
    _ROOT
    / "data"
    / "three_city_c2_v9"
    / "full_k1_5_clean_information"
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


def _without_ids(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in ("task_id", "episode_id")
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
    v9_tasks = {
        arm: _read_jsonl(_V9_DATA / f"tasks_c2_v9_{arm}.jsonl")
        for arm in ("c_only", "abc")
    }
    v9_keys = _read_jsonl(_V9_DATA / "answer_key_c2_v9.jsonl")
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

    v9_by_arm = {arm: _index(rows) for arm, rows in v9_tasks.items()}
    checks["a_and_b_prompts_byte_identical_to_v9"] = all(
        by_arm[arm][key]["prompt"] == v9_by_arm[arm][key]["prompt"]
        for arm in ("c_only", "abc")
        for key in by_arm[arm]
    )
    v10_key_index = _index(keys)
    v9_key_index = _index(v9_keys)
    checks["numerical_design_byte_identical_to_v9"] = all(
        _without_ids(v10_key_index[key]) == _without_ids(v9_key_index[key])
        for key in v10_key_index
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
    checks["large_round1_information_value"] = gaps[1] > 1.50
    checks["large_round2_information_value"] = gaps[2] > 0.75
    checks["abc_beats_c_only_rounds_1_to_4"] = all(
        abc_mae[k] < target_mae[k] for k in (1, 2, 3, 4)
    )
    checks["matched_convergence_by_round5"] = (
        abs(gaps[5]) < 0.10
        and target_mae[5] < 0.90
        and abc_mae[5] < 0.90
    )
    checks["reference_weight_strictly_decreases"] = all(
        weights[a] > weights[b]
        for a, b in zip(PREFIX_LADDER, PREFIX_LADDER[1:])
    )

    checks["manifest_declares_structural_choice_design"] = (
        manifest.get("experiment") == "three_city_c2_v10_structural_choice_clue"
        and manifest.get("arms") == list(PROMPT_ARMS)
        and manifest.get("numerical_dgp_identical_to_v9") is True
        and manifest.get("a_and_b_prompts_byte_identical_to_v9") is True
        and manifest.get("clue_reveals_closer_city") is False
        and manifest.get("structural_choice_clue") == STRUCTURAL_CHOICE_CLUE
        and manifest.get("cases_by_prefix")
        == {str(k): CASES_BY_PREFIX[k] for k in PREFIX_LADDER}
        and float(manifest.get("case_sigma")) == CASE_SIGMA
        and float(manifest.get("city_deviation_sd")) == CITY_DEVIATION_SD
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
        "engine_sha256": _ROOT / "engine" / "three_city_c2_v10.py",
        "source_engine_v9_sha256": _ROOT / "engine" / "three_city_c2_v9.py",
        "builder_sha256": _HERE / "build_three_city_c2_v10_tasks.py",
        "source_builder_v9_sha256": _HERE / "build_three_city_c2_v9_tasks.py",
        "config_sha256": _ROOT / "configs" / "three_city_c2_v10.yml",
        "validator_sha256": Path(__file__),
        "runner_sha256": _HERE / "run_three_city_c2_v10_confirmatory.py",
        "source_runner_v9_sha256": _HERE / "run_three_city_c2_v9_confirmatory.py",
        "launcher_sha256": _HERE / "run_three_city_c2_v10_local.sh",
        "analyzer_sha256": _ROOT / "analysis" / "analyze_three_city_c2_v10_confirmatory.py",
        "source_analyzer_v9_sha256": _ROOT / "analysis" / "analyze_three_city_c2_v9_confirmatory.py",
        "live_monitor_sha256": _ROOT / "analysis" / "live_monitor_three_city_c2_v10.py",
        "prompt_figure_sha256": _ROOT / "analysis" / "preview_three_city_c2_v10_prompts.py",
        "preregistration_sha256": _ROOT / "PREREGISTRATION_THREE_CITY_C2_V10.md",
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
            "structural_choice_clue": STRUCTURAL_CHOICE_CLUE,
            "prompt_failures": prompt_failures,
        }
    )
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "experiment": "three_city_c2_v10_structural_choice_clue",
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
        arm: args.data / f"tasks_c2_v10_{arm}.jsonl"
        for arm in PROMPT_ARMS
    }
    result = validate(
        task_paths=task_paths,
        answer_key_path=args.data / "answer_key_c2_v10.jsonl",
        manifest_path=args.data / "tasks_c2_v10.manifest.json",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    output = args.data / "validation_c2_v10.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
