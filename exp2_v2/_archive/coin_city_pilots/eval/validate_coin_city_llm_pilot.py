#!/usr/bin/env python3
"""Fail-closed preflight checks for the bounded coin-to-city LLM pilot."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from engine.coin_city_llm_pilot import (
    CASES_BY_ROUND,
    EPISODES,
    EXPERIMENT,
    HARD_CALL_CAP,
    MODELS,
    PLANNED_CALLS,
    PROMPT_ARMS,
    ROUNDS,
    TARGET_CONTEXT_NONE,
    baseline_estimates,
    make_prompt,
    prompt_sections,
    prompt_sha256,
    target_background,
    task_id,
    validate_arm_prompt,
)


RUN = ROOT / "data" / EXPERIMENT
DESIGN = RUN / "design"
TASK_RE = re.compile(r"^coincityp1_(\d{4})_k([1-5])$")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_absolute(values: list[float], truths: list[float]) -> float:
    return float(np.mean(np.abs(np.asarray(values) - np.asarray(truths))))


def main() -> None:
    manifest = json.loads((DESIGN / "manifest.json").read_text())
    episodes = _read_jsonl(DESIGN / "episodes.jsonl")
    episode_by_id = {row["episode"]: row for row in episodes}
    answer_rows = _read_jsonl(DESIGN / "answer_key.jsonl")
    answer_by_id = {row["task_id"]: row for row in answer_rows}
    tasks = {
        arm: _read_jsonl(DESIGN / f"tasks_{arm}.jsonl")
        for arm in PROMPT_ARMS
    }
    task_by_arm = {
        arm: {row["task_id"]: row for row in rows}
        for arm, rows in tasks.items()
    }
    checks: dict[str, bool] = {}

    checks["experiment_name"] = manifest.get("experiment") == EXPERIMENT
    checks["planned_calls_exactly_1800"] = PLANNED_CALLS == 1800 == manifest.get(
        "planned_model_calls"
    )
    checks["planned_calls_below_5000_cap"] = (
        PLANNED_CALLS <= HARD_CALL_CAP == 5000
        and manifest.get("hard_call_cap") == HARD_CALL_CAP
    )
    checks["one_attempt_per_task"] = manifest.get("max_attempts_per_task") == 1
    checks["dimensions_match_call_count"] = (
        EPISODES * len(ROUNDS) * len(PROMPT_ARMS) * len(MODELS) == PLANNED_CALLS
    )
    checks["episode_count_and_ids"] = (
        len(episodes) == EPISODES
        and len(episode_by_id) == EPISODES
        and set(episode_by_id) == set(range(EPISODES))
    )
    expected_task_ids = {
        task_id(episode, k) for episode in range(EPISODES) for k in ROUNDS
    }
    checks["answer_key_count_and_ids"] = (
        len(answer_rows) == len(expected_task_ids)
        and len(answer_by_id) == len(expected_task_ids)
        and set(answer_by_id) == expected_task_ids
    )

    task_structure_ok = True
    prompt_hashes_ok = True
    exact_regeneration_ok = True
    arm_identity_ok = True
    b_c_substitution_ok = True
    c_numeric_block_ok = True
    for arm, rows in tasks.items():
        if len(rows) != len(expected_task_ids) or set(task_by_arm[arm]) != expected_task_ids:
            task_structure_ok = False
        for row in rows:
            if set(row) != {"task_id", "prompt", "prompt_sha256"}:
                task_structure_ok = False
            match = TASK_RE.fullmatch(row.get("task_id", ""))
            if match is None:
                task_structure_ok = False
                continue
            episode_index, k = map(int, match.groups())
            if episode_index not in episode_by_id or k not in ROUNDS:
                task_structure_ok = False
                continue
            if prompt_sha256(row["prompt"]) != row["prompt_sha256"]:
                prompt_hashes_ok = False
            try:
                validate_arm_prompt(row["prompt"], arm)
            except ValueError:
                arm_identity_ok = False
            if row["prompt"] != make_prompt(episode_by_id[episode_index], k, arm):
                exact_regeneration_ok = False

    for identifier in expected_task_ids:
        key = answer_by_id[identifier]
        episode = episode_by_id[key["episode"]]
        k = key["k"]
        b_prompt = task_by_arm["abc_no_context"][identifier]["prompt"]
        c_prompt = task_by_arm["abc_context"][identifier]["prompt"]
        context_sentence = target_background(episode, "abc_context")
        if c_prompt.replace(context_sentence, TARGET_CONTEXT_NONE) != b_prompt:
            b_c_substitution_ok = False
        baseline_section = prompt_sections(episode, k, "baseline")
        b_section = prompt_sections(episode, k, "abc_no_context")
        if b_section[b_section.rfind("CITY C\n") :] != baseline_section:
            c_numeric_block_ok = False
        expected_baselines = baseline_estimates(episode, k)
        if key["city_c_cases"] != CASES_BY_ROUND[k] or any(
            abs(key["baselines"][name] - value) > 1e-12
            for name, value in expected_baselines.items()
        ):
            exact_regeneration_ok = False

    checks["task_file_counts_and_ids"] = task_structure_ok
    checks["prompt_sha256_values"] = prompt_hashes_ok
    checks["all_prompts_regenerate_exactly"] = exact_regeneration_ok
    checks["arm_prompt_contracts"] = arm_identity_ok
    checks["b_and_c_differ_only_by_one_c_background_sentence"] = b_c_substitution_ok
    checks["city_c_numeric_evidence_identical_across_arms"] = c_numeric_block_ok
    checks["task_file_manifest_hashes"] = all(
        _sha256(DESIGN / f"tasks_{arm}.jsonl")
        == manifest["task_file_sha256"][arm]
        for arm in PROMPT_ARMS
    )
    checks["answer_and_episode_manifest_hashes"] = (
        _sha256(DESIGN / "answer_key.jsonl") == manifest["answer_key_sha256"]
        and _sha256(DESIGN / "episodes.jsonl") == manifest["episodes_sha256"]
    )

    source_paths = {
        "engine": ROOT / "engine" / "coin_city_llm_pilot.py",
        "builder": HERE / "build_coin_city_llm_pilot.py",
        "validator": HERE / "validate_coin_city_llm_pilot.py",
        "runner": HERE / "run_coin_city_llm_pilot.py",
        "launcher": HERE / "run_coin_city_llm_pilot_local.sh",
        "analyzer": ROOT / "analysis" / "analyze_coin_city_llm_pilot.py",
        "live_monitor": ROOT / "analysis" / "live_coin_city_llm_pilot.py",
    }
    checks["frozen_source_hashes"] = all(
        path.exists() and _sha256(path) == manifest["source_sha256"].get(name)
        for name, path in source_paths.items()
    )

    balance = Counter(
        (
            row["high_reference_city"],
            bool(row["target_high"]),
            bool(row["cue_correct"]),
        )
        for row in episodes
    )
    checks["factorial_reference_and_target_balance"] = all(
        sum(
            count
            for (observed_city, observed_target, _), count in balance.items()
            if observed_city == city and observed_target == target_high
        )
        == 10
        for city in ("A", "B")
        for target_high in (False, True)
    )
    checks["cue_is_exactly_80_percent_reliable"] = (
        sum(bool(row["cue_correct"]) for row in episodes) == 32
        and sum(not bool(row["cue_correct"]) for row in episodes) == 8
        and all(
            balance[(city, target_high, True)] == 8
            and balance[(city, target_high, False)] == 2
            for city in ("A", "B")
            for target_high in (False, True)
        )
    )
    checks["fresh_seed_range"] = all(
        row["seed"] == 310_000 + row["episode"] for row in episodes
    )

    curve_values: dict[str, dict[int, float]] = defaultdict(dict)
    for k in ROUNDS:
        keys = [row for row in answer_rows if row["k"] == k]
        truths = [row["gold_expected_poll"] for row in keys]
        for arm in PROMPT_ARMS:
            curve_values[arm][k] = _mean_absolute(
                [row["baselines"][arm] for row in keys], truths
            )
    c_curve = [curve_values["baseline"][k] for k in ROUNDS]
    no_context_curve = [curve_values["abc_no_context"][k] for k in ROUNDS]
    context_curve = [curve_values["abc_context"][k] for k in ROUNDS]
    checks["city_c_benchmark_declines_each_round"] = all(
        later < earlier for earlier, later in zip(c_curve, c_curve[1:])
    )
    # In a 40-episode development pilot, forcing a strict ordering after the
    # estimators have nearly converged would amount to selecting lucky seeds.
    # Require the intended aggregate benefit while C evidence is sparse, then
    # require the small late-round crossover to remain substantively negligible.
    checks["abc_no_context_beats_c_only_in_sparse_rounds"] = all(
        no_context_curve[index] < c_curve[index] for index in range(3)
    )
    checks["abc_no_context_nearly_converges_with_c_only_late"] = all(
        abs(no_context_curve[index] - c_curve[index]) < 0.10
        for index in range(3, 5)
    )
    checks["context_beats_no_context_each_round"] = all(
        context < neutral
        for context, neutral in zip(context_curve, no_context_curve)
    )
    checks["three_benchmarks_nearly_converge_by_k5"] = (
        max(c_curve[-1], no_context_curve[-1], context_curve[-1])
        - min(c_curve[-1], no_context_curve[-1], context_curve[-1])
        < 0.20
    )
    reference_separations = []
    for episode in episodes:
        a_mean = float(np.mean(episode["reference_a"]))
        b_mean = float(np.mean(episode["reference_b"]))
        reference_separations.append(abs(a_mean - b_mean))
    checks["reference_cities_observably_different"] = (
        float(np.median(reference_separations)) > 4.0
    )

    validation = {
        "experiment": EXPERIMENT,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "planned_model_calls": PLANNED_CALLS,
        "hard_call_cap": HARD_CALL_CAP,
        "cue_balance": {
            "correct": sum(bool(row["cue_correct"]) for row in episodes),
            "misleading": sum(not bool(row["cue_correct"]) for row in episodes),
        },
        "benchmark_mae": {
            arm: {str(k): curve_values[arm][k] for k in ROUNDS}
            for arm in PROMPT_ARMS
        },
        "median_observed_reference_separation": float(
            np.median(reference_separations)
        ),
    }
    (DESIGN / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("preflight failed: " + ", ".join(failed))
    print(
        f"Preflight passed: {PLANNED_CALLS} planned calls <= {HARD_CALL_CAP}; "
        f"{len(checks)} checks passed"
    )


if __name__ == "__main__":
    main()
