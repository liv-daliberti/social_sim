#!/usr/bin/env python3
"""Fail-closed validation for the C=0..6 variable-regression design."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from engine.coin_city_variable_regression import (
    C_CASE_LEVELS,
    EPISODES,
    EXPERIMENT,
    HARD_CALL_CAP,
    LIVE_EPISODES_BY_CELL,
    LIVE_REPLICATES_BY_CELL,
    MODELS,
    PLANNED_CALLS,
    PROMPT_ARMS,
    REFERENCE_CASES,
    SEED_BASE,
    SOURCE_EPISODES_PER_CELL,
    TARGET_CONTEXT_NONE,
    TARGET_CONTEXT_STRONG,
    TARGET_CONTEXT_WEAK,
    empirical_estimates,
    fit_change_slope,
    make_episode,
    make_prompt,
    predict_from_slope,
    prompt_sha256,
    target_background,
    task_id,
    validate_arm_prompt,
)


RUN = ROOT / "data" / EXPERIMENT
DESIGN = RUN / "design"
TASK_RE = re.compile(r"^coincityvr1_(\d{4})_c([0-6])$")
CITY_BLOCK = re.compile(
    r"CITY ([ABC])\n.*?\n(\| Case .*?)(?=\n\nCITY |\n\nA new City C)",
    re.DOTALL,
)
ROW = re.compile(
    r"^\|\s*\d+\s*\|\s*([0-9.]+)\s*\|\s*([+-]\d+)\s*\|\s*([0-9.]+)\s*\|$",
    re.MULTILINE,
)
QUERY = re.compile(
    r"A new City C case begins with a poll of ([0-9.]+) and has net news ([+-]\d+)\."
)
FINAL_CONVERGENCE_THRESHOLD = 0.50


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _visible(prompt: str) -> tuple[dict[str, list[dict]], float, int]:
    cities = {}
    for block in CITY_BLOCK.finditer(prompt):
        cities[block.group(1)] = [
            {
                "starting_poll": float(start),
                "net_news": int(news),
                "ending_poll": float(end),
            }
            for start, news, end in ROW.findall(block.group(2))
        ]
    query = QUERY.search(prompt)
    if not cities or query is None:
        raise ValueError("failed to reconstruct displayed rows/query")
    return cities, float(query.group(1)), int(query.group(2))


def _prompt_prediction(
    prompt: str, included_cities: tuple[str, ...]
) -> float | None:
    cities, query_start, query_news = _visible(prompt)
    rows = [row for city in included_cities for row in cities[city]]
    if not rows:
        return None
    return float(query_start + query_news * fit_change_slope(rows))


def _same_optional(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= 1e-11


def _mae(predictions: list[float], truths: list[float]) -> float:
    return float(
        np.mean(np.abs(np.asarray(predictions, dtype=float) - np.asarray(truths)))
    )


def main() -> None:
    manifest = json.loads((DESIGN / "manifest.json").read_text())
    episodes = _read_jsonl(DESIGN / "episodes.jsonl")
    episode_by_id = {row["episode"]: row for row in episodes}
    audit_keys = _read_jsonl(DESIGN / "answer_key.jsonl")
    audit_key_by_id = {row["task_id"]: row for row in audit_keys}
    keys = _read_jsonl(DESIGN / "scoring_key.jsonl")
    key_by_id = {row["task_id"]: row for row in keys}
    tasks = {
        arm: _read_jsonl(DESIGN / f"tasks_{arm}.jsonl")
        for arm in PROMPT_ARMS
    }
    task_by_arm = {
        arm: {row["task_id"]: row for row in rows}
        for arm, rows in tasks.items()
    }
    expected_ids = {
        task_id(episode, c_cases)
        for episode in range(EPISODES)
        for c_cases in C_CASE_LEVELS
    }
    scoring_fields = {
        "task_id",
        "episode",
        "c_cases",
        "gold_expected_poll",
        "baselines",
    }
    checks = {
        "experiment": manifest.get("experiment") == EXPERIMENT,
        "direct_city_c_levels_0_through_6": (
            tuple(manifest.get("city_c_case_levels", ())) == C_CASE_LEVELS
        ),
        "planned_calls_exactly_3150": (
            PLANNED_CALLS == manifest.get("planned_calls") == 3_150
        ),
        "planned_calls_below_5000_cap": (
            PLANNED_CALLS <= HARD_CALL_CAP == manifest.get("hard_call_cap") == 5_000
        ),
        "one_attempt_per_task": manifest.get("max_attempts_per_task") == 1,
        "dimensions": (
            EPISODES * len(C_CASE_LEVELS) * len(PROMPT_ARMS) * len(MODELS)
            == PLANNED_CALLS
        ),
        "episode_count_and_ids": (
            len(episodes) == len(episode_by_id) == EPISODES
            and set(episode_by_id) == set(range(EPISODES))
        ),
        "audit_and_scoring_key_counts_and_ids": (
            len(keys) == len(key_by_id) == len(expected_ids)
            and set(key_by_id) == expected_ids
            and len(audit_keys) == len(audit_key_by_id) == len(expected_ids)
            and set(audit_key_by_id) == expected_ids
        ),
        "scoring_key_excludes_latent_design_fields": all(
            set(row) == scoring_fields for row in keys
        ),
        "no_responses_before_freeze": not any(
            (RUN / "responses").glob("*.jsonl")
        ),
    }

    task_structure = True
    prompt_hashes = True
    prompt_contracts = True
    exact_regeneration = True
    prompt_only_estimators = True
    condition_identity = True
    c_prefixes = True
    for arm, rows in tasks.items():
        if len(rows) != len(expected_ids) or set(task_by_arm[arm]) != expected_ids:
            task_structure = False
        for row in rows:
            if set(row) != {"task_id", "prompt", "prompt_sha256"}:
                task_structure = False
            match = TASK_RE.fullmatch(row.get("task_id", ""))
            if match is None:
                task_structure = False
                continue
            episode_index, c_cases = map(int, match.groups())
            if episode_index not in episode_by_id or c_cases not in C_CASE_LEVELS:
                task_structure = False
                continue
            if prompt_sha256(row["prompt"]) != row["prompt_sha256"]:
                prompt_hashes = False
            try:
                validate_arm_prompt(row["prompt"], arm)
            except ValueError:
                prompt_contracts = False
            if row["prompt"] != make_prompt(
                episode_by_id[episode_index], c_cases, arm
            ):
                exact_regeneration = False

    for identifier in expected_ids:
        key = key_by_id[identifier]
        episode = episode_by_id[key["episode"]]
        c_cases = key["c_cases"]
        expected = empirical_estimates(episode, c_cases)
        if any(
            not _same_optional(key["baselines"][name], value)
            for name, value in expected.items()
        ):
            exact_regeneration = False
        c_prompt = task_by_arm["baseline"][identifier]["prompt"]
        abc_prompt = task_by_arm["abc_no_context"][identifier]["prompt"]
        c_from_text = _prompt_prediction(c_prompt, ("C",))
        abc_from_text = _prompt_prediction(abc_prompt, ("A", "B", "C"))
        if (
            not _same_optional(c_from_text, key["baselines"]["baseline"])
            or not _same_optional(
                abc_from_text, key["baselines"]["abc_no_context"]
            )
        ):
            prompt_only_estimators = False
        context = target_background(episode, "abc_context")
        if (
            task_by_arm["abc_context"][identifier]["prompt"].replace(
                context, TARGET_CONTEXT_NONE
            )
            != abc_prompt
        ):
            condition_identity = False
        c_rows = _visible(c_prompt)[0]["C"]
        abc_rows = _visible(abc_prompt)[0]
        if c_rows != abc_rows["C"]:
            condition_identity = False
        if c_rows != episode["target"][:c_cases]:
            c_prefixes = False

    checks.update(
        {
            "task_counts_and_ids": task_structure,
            "prompt_hashes": prompt_hashes,
            "prompt_contracts": prompt_contracts,
            "exact_regeneration": exact_regeneration,
            "estimators_reconstruct_from_displayed_prompt_only": prompt_only_estimators,
            "b_and_c_differ_only_by_city_c_context": condition_identity,
            "city_c_rows_identical_across_arms_and_nested": c_prefixes,
        }
    )

    balance = Counter(
        (row["strong_reference_city"], bool(row["target_strong"]))
        for row in episodes
    )
    expected_balance = {
        ("A", True): 13,
        ("A", False): 12,
        ("B", True): 12,
        ("B", False): 13,
    }
    checks["closest_possible_50_episode_factorial_balance"] = (
        balance == Counter(expected_balance)
        and tuple(manifest.get("factorial_cell_episode_counts", ()))
        == LIVE_EPISODES_BY_CELL
        and tuple(
            tuple(replicates)
            for replicates in manifest.get("factorial_cell_source_replicates", ())
        )
        == LIVE_REPLICATES_BY_CELL
    )
    checks["balanced_aligned_national_local_clues"] = (
        sum(bool(row["target_strong"]) for row in episodes) == 25
        and all(
            row["cue_correct"] is True
            and bool(row["cue_strong"]) == bool(row["target_strong"])
            and target_background(row, "abc_context")
            == (
                TARGET_CONTEXT_STRONG
                if row["target_strong"]
                else TARGET_CONTEXT_WEAK
            )
            for row in episodes
        )
    )
    expected_source_indices = [
        block * SOURCE_EPISODES_PER_CELL + replicate
        for block, replicates in enumerate(LIVE_REPLICATES_BY_CELL)
        for replicate in replicates
    ]
    checks["predetermined_balanced_source_subset"] = all(
        row["source_episode"] == expected_source_indices[row["episode"]]
        and row["seed"] == SEED_BASE + row["source_episode"]
        for row in episodes
    ) and len(expected_source_indices) == EPISODES
    all_rows = [
        row
        for episode in episodes
        for name in ("reference_a", "reference_b", "target")
        for row in episode[name]
    ]
    starting_values = {row["starting_poll"] for row in all_rows}
    news_values = {row["net_news"] for row in all_rows}
    checks["starting_polls_vary_widely"] = (
        min(starting_values) <= 35.1
        and max(starting_values) >= 64.9
        and len(starting_values) > 200
    )
    checks["net_news_varies_in_both_directions"] = (
        min(news_values) == -10
        and max(news_values) == 10
        and any(value < 0 for value in news_values)
        and any(value > 0 for value in news_values)
    )
    checks["three_reference_rows_per_city_with_both_signs"] = all(
        len(episode[name]) == REFERENCE_CASES
        and {int(np.sign(row["net_news"])) for row in episode[name]} == {-1, 1}
        for episode in episodes
        for name in ("reference_a", "reference_b")
    )
    fitted_reference_separation = [
        abs(
            fit_change_slope(episode["reference_a"])
            - fit_change_slope(episode["reference_b"])
        )
        for episode in episodes
    ]
    checks["reference_response_patterns_observably_different"] = (
        float(np.median(fitted_reference_separation)) > 0.60
    )

    curves = {"city_c": {}, "abc_no_context": {}, "structural_diagnostic": {}}
    for c_cases in C_CASE_LEVELS:
        level_keys = [row for row in keys if row["c_cases"] == c_cases]
        truths = [row["gold_expected_poll"] for row in level_keys]
        city_predictions = [
            row["baselines"]["baseline"] for row in level_keys
        ]
        curves["city_c"][c_cases] = (
            None
            if c_cases == 0
            else _mae(city_predictions, truths)
        )
        curves["abc_no_context"][c_cases] = _mae(
            [row["baselines"]["abc_no_context"] for row in level_keys],
            truths,
        )
        structural_predictions = []
        for key in level_keys:
            episode = episode_by_id[key["episode"]]
            if episode["target_strong"]:
                matching_city = episode["strong_reference_city"]
            else:
                matching_city = (
                    "B" if episode["strong_reference_city"] == "A" else "A"
                )
            reference = (
                episode["reference_a"]
                if matching_city == "A"
                else episode["reference_b"]
            )
            slope = fit_change_slope(
                reference + episode["target"][:c_cases]
            )
            structural_predictions.append(predict_from_slope(episode, slope))
        curves["structural_diagnostic"][c_cases] = _mae(
            structural_predictions, truths
        )

    c_curve = [curves["city_c"][n] for n in C_CASE_LEVELS if n > 0]
    abc_curve = [curves["abc_no_context"][n] for n in C_CASE_LEVELS]
    structural_curve = [
        curves["structural_diagnostic"][n] for n in C_CASE_LEVELS
    ]
    checks["city_c_regression_undefined_only_at_zero"] = (
        curves["city_c"][0] is None
        and all(value is not None for value in c_curve)
    )
    checks["abc_regression_defined_at_zero"] = curves["abc_no_context"][0] is not None
    checks["city_c_regression_declines_each_case"] = all(
        later < earlier for earlier, later in zip(c_curve, c_curve[1:])
    )
    checks["abc_regression_declines_each_case"] = all(
        later < earlier for earlier, later in zip(abc_curve, abc_curve[1:])
    )
    checks["abc_empirically_beats_city_c_when_defined"] = all(
        curves["abc_no_context"][n] < curves["city_c"][n]
        for n in C_CASE_LEVELS
        if n > 0
    )
    checks["large_sparse_empirical_benefit"] = (
        curves["city_c"][1] - curves["abc_no_context"][1] > 1.25
        and curves["city_c"][2] - curves["abc_no_context"][2] > 0.75
    )
    checks["news_environment_signal_helps_with_observed_c"] = all(
        structural_curve[n] < abc_curve[n] for n in range(1, 7)
    )
    checks["zero_c_news_environment_diagnostic_helps"] = (
        abc_curve[0] - structural_curve[0] > 0.25
    )
    final_gap = curves["city_c"][6] - curves["abc_no_context"][6]
    checks["empirical_regressions_converge_by_six_c_cases"] = (
        0.0 <= final_gap < FINAL_CONVERGENCE_THRESHOLD
    )

    checks["task_file_hashes"] = all(
        _sha256(DESIGN / f"tasks_{arm}.jsonl") == manifest["task_sha256"][arm]
        for arm in PROMPT_ARMS
    )
    checks["answer_scoring_and_episode_hashes"] = (
        _sha256(DESIGN / "answer_key.jsonl") == manifest["answer_key_sha256"]
        and _sha256(DESIGN / "scoring_key.jsonl")
        == manifest["scoring_key_sha256"]
        and _sha256(DESIGN / "episodes.jsonl") == manifest["episodes_sha256"]
    )
    source_paths = {
        "engine": ROOT / "engine" / "coin_city_variable_regression.py",
        "builder": HERE / "build_coin_city_variable_regression.py",
        "validator": HERE / "validate_coin_city_variable_regression.py",
        "renderer": ROOT / "analysis" / "render_coin_city_variable_regression.py",
        "runner": HERE / "run_coin_city_variable_regression.py",
        "local_launcher": HERE / "run_coin_city_variable_regression_local.sh",
    }
    checks["frozen_source_hashes"] = all(
        path.exists() and _sha256(path) == manifest["source_sha256"].get(name)
        for name, path in source_paths.items()
    )

    result = {
        "experiment": EXPERIMENT,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "episodes": EPISODES,
        "planned_calls": PLANNED_CALLS,
        "hard_call_cap": HARD_CALL_CAP,
        "factorial_balance": {
            f"strong_reference_{city}_target_{'strong' if target else 'weak'}": balance[(city, target)]
            for city in ("A", "B")
            for target in (False, True)
        },
        "correctly_aligned_contexts": sum(
            bool(row["cue_correct"]) for row in episodes
        ),
        "benchmark_mae": {
            "city_c": {str(n): curves["city_c"][n] for n in C_CASE_LEVELS},
            "abc_no_context": {
                str(n): curves["abc_no_context"][n] for n in C_CASE_LEVELS
            },
        },
        "diagnostic_not_plotted": {
            "description": "uses the local/national news clue to select the matching displayed reference city",
            "mae": {
                str(n): curves["structural_diagnostic"][n]
                for n in C_CASE_LEVELS
            },
        },
        "final_convergence_threshold": FINAL_CONVERGENCE_THRESHOLD,
        "final_gap": final_gap,
        "median_fitted_reference_slope_separation": float(
            np.median(fitted_reference_separation)
        ),
    }
    (DESIGN / "validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("variable-regression preflight failed: " + ", ".join(failed))
    print(
        f"Variable-regression preflight passed: {len(checks)} checks; "
        f"{PLANNED_CALLS:,}/{HARD_CALL_CAP:,} planned calls"
    )


if __name__ == "__main__":
    main()
