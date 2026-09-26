"""Registered hierarchical contrasts for the expanded C3 mechanism factorial.

Every stochastic decode is first averaged within its episode. Training seeds are
then resampled at the top level and paired episodes are resampled within seed.
Positive estimates always mean that the first-named condition has the larger
causal-training benefit or the lower response error, as stated by the contrast.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

MODEL_BENEFIT_PAIRS = (
    ("qwen3_8b", "qwen3_4b", "qwen3_8b_minus_qwen3_4b_causal_benefit"),
    ("qwen3_8b", "llama3_1_8b", "qwen3_8b_minus_llama3_1_8b_causal_benefit"),
)

_TASK_INDEX_CACHE: dict[tuple[int, str], dict[tuple, dict[str, float]]] = {}


def _task_index(rows: list[dict], metric: str) -> dict[tuple, dict[str, float]]:
    """Average draws once and index tasks by the full experimental cell."""
    cache_key = (id(rows), metric)
    cached = _TASK_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    grouped: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        cell = (
            row["model"], row["disclosure"], row["decode"], row["block"],
            int(row["seed"]), row["arm"],
        )
        grouped[cell][str(row["task_id"])].append(float(row[metric]))
    result = {
        cell: {task_id: float(np.mean(values)) for task_id, values in tasks.items()}
        for cell, tasks in grouped.items()
    }
    _TASK_INDEX_CACHE[cache_key] = result
    return result


def _task_means(rows: list[dict], selectors: dict[str, Any],
                metric: str = "response_mae") -> dict[str, float]:
    fields = ("model", "disclosure", "decode", "block", "seed", "arm")
    grouped: dict[str, list[float]] = defaultdict(list)
    for cell, tasks in _task_index(rows, metric).items():
        metadata = dict(zip(fields, cell))
        if all(metadata.get(key) == value for key, value in selectors.items()):
            for task_id, value in tasks.items():
                grouped[task_id].append(value)
    return {task_id: float(np.mean(values)) for task_id, values in grouped.items()}


def _paired_values(left: dict[str, float], right: dict[str, float]) -> np.ndarray:
    task_ids = sorted(set(left) & set(right))
    return np.asarray([left[item] - right[item] for item in task_ids], dtype=float)


def _bootstrap_draws(paired: dict[int, np.ndarray], repetitions: int,
                     rng: np.random.Generator, batch_size: int = 256) -> np.ndarray:
    """Batched seed-first/episode-within-seed hierarchical bootstrap."""
    seeds = sorted(paired)
    lengths = {len(paired[seed]) for seed in seeds}
    if len(lengths) != 1:
        raise AssertionError(f"paired episode counts differ across seeds: {lengths}")
    values = np.stack([paired[seed] for seed in seeds])
    seed_count, episode_count = values.shape
    draws = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        batch = stop - start
        selected_seeds = rng.integers(0, seed_count, size=(batch, seed_count))
        selected_episodes = rng.integers(
            0, episode_count, size=(batch, seed_count, episode_count)
        )
        sampled = values[selected_seeds[..., None], selected_episodes]
        draws[start:stop] = sampled.mean(axis=2).mean(axis=1)
    return draws


def _bootstrap_result(paired: dict[int, np.ndarray], metadata: dict[str, Any],
                      repetitions: int, bootstrap_seed: int) -> dict | None:
    paired = {seed: values for seed, values in paired.items() if len(values)}
    if len(paired) < 2:
        return None
    seeds = sorted(paired)
    estimate = float(np.mean([np.mean(paired[seed]) for seed in seeds]))
    rng = np.random.default_rng(bootstrap_seed)
    draws = _bootstrap_draws(paired, repetitions, rng)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        **metadata,
        "estimate": estimate,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "training_seeds": seeds,
        "episodes_per_seed": {str(seed): len(paired[seed]) for seed in seeds},
        "bootstrap_repetitions": repetitions,
    }


def hierarchical_base_contrasts(rows: list[dict], repetitions: int,
                                bootstrap_seed: int) -> list[dict]:
    """Untrained-base error minus causal-training error; positive favors training."""
    dimensions = sorted({
        (row["model"], row["disclosure"], row["decode"], row["block"])
        for row in rows if row["arm"] == "causal_family"
    })
    results = []
    for model, disclosure, decode, block in dimensions:
        common = {
            "model": model, "disclosure": disclosure,
            "decode": decode, "block": block,
        }
        base = _task_means(rows, {**common, "arm": "base"})
        seeds = sorted({
            int(row["seed"]) for row in rows
            if row["arm"] == "causal_family"
            and all(row.get(key) == value for key, value in common.items())
        })
        paired = {}
        for seed in seeds:
            causal = _task_means(
                rows, {**common, "arm": "causal_family", "seed": seed}
            )
            paired[seed] = _paired_values(base, causal)
        result = _bootstrap_result(
            paired,
            {**common, "contrast": "base_minus_causal_response_mae"},
            repetitions,
            bootstrap_seed,
        )
        if result is not None:
            results.append(result)
    return results


def hierarchical_structureless_contrasts(rows: list[dict], repetitions: int,
                                         bootstrap_seed: int) -> list[dict]:
    """Structureless error minus causal-training error; positive favors causal training."""
    dimensions = sorted({
        (row["model"], row["disclosure"], row["decode"], row["block"])
        for row in rows if row["arm"] == "structureless"
    })
    results = []
    for model, disclosure, decode, block in dimensions:
        common = {
            "model": model, "disclosure": disclosure,
            "decode": decode, "block": block,
        }
        seeds = sorted({
            int(row["seed"]) for row in rows
            if row["arm"] == "structureless"
            and all(row.get(key) == value for key, value in common.items())
        })
        paired = {}
        for seed in seeds:
            structureless = _task_means(
                rows, {**common, "arm": "structureless", "seed": seed}
            )
            causal = _task_means(
                rows, {**common, "arm": "causal_family", "seed": seed}
            )
            paired[seed] = _paired_values(structureless, causal)
        result = _bootstrap_result(
            paired,
            {**common, "contrast": "structureless_minus_causal_response_mae"},
            repetitions,
            bootstrap_seed,
        )
        if result is not None:
            results.append(result)
    return results


def _causal_benefit(rows: list[dict], model: str, disclosure: str,
                    decode: str, block: str, seed: int) -> dict[str, float]:
    common = {
        "model": model, "disclosure": disclosure, "decode": decode,
        "block": block, "seed": seed,
    }
    prior = _task_means(rows, {**common, "arm": "population_prior"})
    causal = _task_means(rows, {**common, "arm": "causal_family"})
    task_ids = sorted(set(prior) & set(causal))
    return {task_id: prior[task_id] - causal[task_id] for task_id in task_ids}


def hierarchical_model_benefit_contrasts(rows: list[dict], repetitions: int,
                                         bootstrap_seed: int) -> list[dict]:
    """Difference in causal-vs-prior benefit for the registered model contrasts."""
    dimensions = sorted({
        (row["disclosure"], row["decode"], row["block"])
        for row in rows if row["arm"] in {"causal_family", "population_prior"}
    })
    results = []
    for first, second, label in MODEL_BENEFIT_PAIRS:
        for disclosure, decode, block in dimensions:
            seeds = sorted({
                int(row["seed"]) for row in rows
                if row["model"] == first and row["disclosure"] == disclosure
                and row["decode"] == decode and row["block"] == block
                and row["arm"] == "causal_family"
            } & {
                int(row["seed"]) for row in rows
                if row["model"] == second and row["disclosure"] == disclosure
                and row["decode"] == decode and row["block"] == block
                and row["arm"] == "causal_family"
            })
            paired = {}
            for seed in seeds:
                first_benefit = _causal_benefit(
                    rows, first, disclosure, decode, block, seed
                )
                second_benefit = _causal_benefit(
                    rows, second, disclosure, decode, block, seed
                )
                paired[seed] = _paired_values(first_benefit, second_benefit)
            result = _bootstrap_result(
                paired,
                {
                    "model_first": first,
                    "model_second": second,
                    "disclosure": disclosure,
                    "decode": decode,
                    "block": block,
                    "contrast": label,
                },
                repetitions,
                bootstrap_seed,
            )
            if result is not None:
                results.append(result)
    return results


def hierarchical_disclosure_benefit_contrasts(rows: list[dict], repetitions: int,
                                              bootstrap_seed: int) -> list[dict]:
    """Disclosed minus undisclosed causal-vs-prior benefit on paired numeric tasks."""
    dimensions = sorted({
        (row["model"], row["decode"], row["block"])
        for row in rows if row["arm"] in {"causal_family", "population_prior"}
    })
    results = []
    for model, decode, block in dimensions:
        seeds = sorted({
            int(row["seed"]) for row in rows
            if row["model"] == model and row["decode"] == decode
            and row["block"] == block and row["disclosure"] == "disclosed"
            and row["arm"] == "causal_family"
        } & {
            int(row["seed"]) for row in rows
            if row["model"] == model and row["decode"] == decode
            and row["block"] == block and row["disclosure"] == "undisclosed"
            and row["arm"] == "causal_family"
        })
        paired = {}
        for seed in seeds:
            disclosed = _causal_benefit(
                rows, model, "disclosed", decode, block, seed
            )
            undisclosed = _causal_benefit(
                rows, model, "undisclosed", decode, block, seed
            )
            paired[seed] = _paired_values(disclosed, undisclosed)
        result = _bootstrap_result(
            paired,
            {
                "model": model,
                "decode": decode,
                "block": block,
                "contrast": "disclosed_minus_undisclosed_causal_benefit",
            },
            repetitions,
            bootstrap_seed,
        )
        if result is not None:
            results.append(result)
    return results
