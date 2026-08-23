#!/usr/bin/env python3
"""Fail-closed preflight for the 100-episode always-aligned expansion."""

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

from engine.coin_city_expanded_aligned import (
    CASES_BY_ROUND,
    EPISODES,
    EXPERIMENT,
    HARD_CALL_CAP,
    MODELS,
    PLANNED_CALLS,
    PROMPT_ARMS,
    ROUNDS,
    SEED_BASE,
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
TASK_RE = re.compile(r"^coincitya100_(\d{4})_k([1-5])$")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mae(values: list[float], truths: list[float]) -> float:
    return float(np.mean(np.abs(np.asarray(values) - np.asarray(truths))))


def main() -> None:
    manifest = json.loads((DESIGN / "manifest.json").read_text())
    episodes = _read_jsonl(DESIGN / "episodes.jsonl")
    episode_by_id = {row["episode"]: row for row in episodes}
    keys = _read_jsonl(DESIGN / "answer_key.jsonl")
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
        task_id(episode, k) for episode in range(EPISODES) for k in ROUNDS
    }
    checks = {
        "experiment": manifest.get("experiment") == EXPERIMENT,
        "planned_calls_exactly_4500": (
            PLANNED_CALLS == manifest.get("planned_calls") == 4_500
        ),
        "planned_calls_below_5000_cap": (
            PLANNED_CALLS <= HARD_CALL_CAP == manifest.get("hard_call_cap") == 5_000
        ),
        "one_attempt": manifest.get("max_attempts_per_task") == 1,
        "dimensions": (
            EPISODES * len(ROUNDS) * len(PROMPT_ARMS) * len(MODELS)
            == PLANNED_CALLS
        ),
        "episode_count_and_ids": (
            len(episodes) == EPISODES
            and len(episode_by_id) == EPISODES
            and set(episode_by_id) == set(range(EPISODES))
        ),
        "answer_key_count_and_ids": (
            len(keys) == len(expected_ids)
            and len(key_by_id) == len(expected_ids)
            and set(key_by_id) == expected_ids
        ),
    }
    structure = True
    hashes = True
    contracts = True
    regeneration = True
    bc_identity = True
    c_identity = True
    for arm, rows in tasks.items():
        if len(rows) != len(expected_ids) or set(task_by_arm[arm]) != expected_ids:
            structure = False
        for row in rows:
            if set(row) != {"task_id", "prompt", "prompt_sha256"}:
                structure = False
            match = TASK_RE.fullmatch(row.get("task_id", ""))
            if match is None:
                structure = False
                continue
            episode_index, k = map(int, match.groups())
            if episode_index not in episode_by_id or k not in ROUNDS:
                structure = False
                continue
            if prompt_sha256(row["prompt"]) != row["prompt_sha256"]:
                hashes = False
            try:
                validate_arm_prompt(row["prompt"], arm)
            except ValueError:
                contracts = False
            if row["prompt"] != make_prompt(episode_by_id[episode_index], k, arm):
                regeneration = False
    for identifier in expected_ids:
        key = key_by_id[identifier]
        episode = episode_by_id[key["episode"]]
        k = key["k"]
        context = target_background(episode, "abc_context")
        if (
            task_by_arm["abc_context"][identifier]["prompt"].replace(
                context, TARGET_CONTEXT_NONE
            )
            != task_by_arm["abc_no_context"][identifier]["prompt"]
        ):
            bc_identity = False
        baseline_section = prompt_sections(episode, k, "baseline")
        b_section = prompt_sections(episode, k, "abc_no_context")
        if b_section[b_section.rfind("CITY C\n") :] != baseline_section:
            c_identity = False
        expected_baselines = baseline_estimates(episode, k)
        if key["city_c_cases"] != CASES_BY_ROUND[k] or any(
            abs(key["baselines"][name] - value) > 1e-12
            for name, value in expected_baselines.items()
        ):
            regeneration = False
    checks.update(
        {
            "task_counts_and_ids": structure,
            "prompt_hashes": hashes,
            "prompt_contracts": contracts,
            "exact_regeneration": regeneration,
            "b_c_differ_only_by_context_sentence": bc_identity,
            "city_c_evidence_identical_across_arms": c_identity,
        }
    )
    balance = Counter(
        (row["high_reference_city"], bool(row["target_high"]))
        for row in episodes
    )
    checks["exact_25_per_factorial_cell"] = all(
        balance[(city, target_high)] == 25
        for city in ("A", "B")
        for target_high in (False, True)
    )
    checks["all_100_context_clues_point_correctly"] = all(
        row["cue_correct"] is True
        and bool(row["cue_high"]) == bool(row["target_high"])
        for row in episodes
    )
    checks["fresh_seed_range"] = all(
        row["seed"] == SEED_BASE + row["episode"] for row in episodes
    )
    checks["fresh_task_prefix"] = all(
        row["task_id"].startswith("coincitya100_") for row in keys
    )
    reference_separation = [
        abs(float(np.mean(row["reference_a"])) - float(np.mean(row["reference_b"])))
        for row in episodes
    ]
    checks["reference_cities_observably_different"] = (
        float(np.median(reference_separation)) > 4.0
    )

    curves: dict[str, dict[int, float]] = defaultdict(dict)
    for k in ROUNDS:
        round_keys = [row for row in keys if row["k"] == k]
        truths = [row["gold_expected_poll"] for row in round_keys]
        for arm in PROMPT_ARMS:
            curves[arm][k] = _mae(
                [row["baselines"][arm] for row in round_keys], truths
            )
    c_curve = [curves["baseline"][k] for k in ROUNDS]
    no_context = [curves["abc_no_context"][k] for k in ROUNDS]
    aligned = [curves["abc_context"][k] for k in ROUNDS]
    checks["city_c_regression_declines_each_round"] = all(
        later < earlier for earlier, later in zip(c_curve, c_curve[1:])
    )
    checks["abc_no_context_helps_when_sparse"] = all(
        no_context[index] < c_curve[index] for index in range(3)
    )
    late_advantage = [
        abs(no_context[index] - c_curve[index]) for index in range(3, 5)
    ]
    checks["abc_no_context_advantage_shrinks_and_is_small_by_k5"] = (
        late_advantage[1] < late_advantage[0]
        and late_advantage[1] < 0.20
    )
    checks["aligned_context_benchmark_beats_no_context"] = all(
        aligned_value < neutral
        for aligned_value, neutral in zip(aligned, no_context)
    )
    checks["all_benchmarks_nearly_converge_by_k5"] = (
        max(c_curve[-1], no_context[-1], aligned[-1])
        - min(c_curve[-1], no_context[-1], aligned[-1])
        < 0.20
    )
    checks["task_file_hashes"] = all(
        _sha256(DESIGN / f"tasks_{arm}.jsonl") == manifest["task_sha256"][arm]
        for arm in PROMPT_ARMS
    )
    checks["answer_and_episode_hashes"] = (
        _sha256(DESIGN / "answer_key.jsonl") == manifest["answer_key_sha256"]
        and _sha256(DESIGN / "episodes.jsonl") == manifest["episodes_sha256"]
    )
    source_paths = {
        "engine": ROOT / "engine" / "coin_city_expanded_aligned.py",
        "builder": HERE / "build_coin_city_expanded_aligned.py",
        "validator": HERE / "validate_coin_city_expanded_aligned.py",
        "runner": HERE / "run_coin_city_expanded_aligned.py",
        "launcher": HERE / "run_coin_city_expanded_aligned_local.sh",
        "live": ROOT / "analysis" / "live_coin_city_expanded_aligned.py",
        "renderer": ROOT / "analysis" / "render_coin_city_llm_pilot_by_condition.py",
    }
    checks["frozen_source_hashes"] = all(
        path.exists() and _sha256(path) == manifest["source_sha256"].get(name)
        for name, path in source_paths.items()
    )
    result = {
        "experiment": EXPERIMENT,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "planned_calls": PLANNED_CALLS,
        "hard_call_cap": HARD_CALL_CAP,
        "factorial_balance": {
            f"high_reference_{city}_target_{'high' if target else 'low'}": balance[(city, target)]
            for city in ("A", "B")
            for target in (False, True)
        },
        "correctly_aligned_contexts": sum(
            bool(row["cue_correct"]) for row in episodes
        ),
        "benchmark_mae": {
            arm: {str(k): curves[arm][k] for k in ROUNDS} for arm in PROMPT_ARMS
        },
        "median_reference_separation": float(np.median(reference_separation)),
    }
    (DESIGN / "validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("expanded preflight failed: " + ", ".join(failed))
    print(f"Expanded preflight passed: {len(checks)} checks; 4,500/5,000 calls")


if __name__ == "__main__":
    main()
