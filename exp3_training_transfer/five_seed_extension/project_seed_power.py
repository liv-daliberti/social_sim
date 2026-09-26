#!/usr/bin/env python3
"""Seed-level contrasts and forward power projection for the Coin City rosters.

Recomputes the seed-level population-prior-minus-episode-matched contrast for
each transfer cell directly from the registered ``stochastic_n5.scores.jsonl``
endpoint files, reproduces the published Student-t interval on the seed-level
means, and projects that interval forward at larger seed counts holding the
between-seed standard deviation fixed at its observed estimate.

This is the projection cited by QWEN3_8B_EIGHT_SEED_AMENDMENT.md. It reads only
already-registered endpoints and writes nothing.

    python exp3_training_transfer/five_seed_extension/project_seed_power.py
    python ... --model qwen3_8b --project 5 6 8
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

REPORTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "coin_city_structural", "reports",
)

# Same four transfer cells, in the same order, as
# coin_city_structural/make_interim_outputs.py::CELLS.
CELLS = (
    ("City / A (ID)", "coin_city", "direct_a"),
    ("Harbor / A (domain)", "coin_harbor", "direct_a"),
    ("City / B (structure)", "coin_city", "mediated_b"),
    ("Harbor / B (joint)", "coin_harbor", "mediated_b"),
)

# Two-sided 95% Student-t critical values, indexed by degrees of freedom.
TCRIT = {
    1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582,
    6: 2.446912, 7: 2.364624, 8: 2.306004, 9: 2.262157, 10: 2.228139,
    11: 2.200985, 12: 2.178813, 13: 2.160369, 14: 2.144787, 19: 2.093024,
    24: 2.063899, 29: 2.045230, 39: 2.022691,
}


def tcrit(df: int) -> float:
    """Two-sided 95% t critical value, interpolating conservatively above the table."""
    if df in TCRIT:
        return TCRIT[df]
    known = sorted(TCRIT)
    below = max(k for k in known if k < df)
    return TCRIT[below]


def load_arm(model: str, arm: str) -> list[dict]:
    rows = []
    pattern = os.path.join(REPORTS, f"{arm}_{model}_s4*", "stochastic_n5.scores.jsonl")
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            rows.extend(json.loads(line) for line in handle)
    if not rows:
        raise SystemExit(f"no registered endpoints matched {pattern}")
    return rows


def episode_means(rows: list[dict], domain: str, structure: str) -> dict[int, dict[str, float]]:
    """{seed: {task_id: mean response MAE over the five draws}} for one cell."""
    acc: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if (row["domain"] == domain and row["target_structure"] == structure
                and row["cue"] == "correct"):
            acc[row["seed"]][row["task_id"]].append(row["response_mae"])
    return {seed: {task: sum(v) / len(v) for task, v in tasks.items()}
            for seed, tasks in acc.items()}


def seed_contrasts(model: str, domain: str, structure: str) -> tuple[list[int], list[float]]:
    """Per-seed mean over paired episodes of (population_prior - episode_matched)."""
    matched = episode_means(load_arm(model, "causal"), domain, structure)
    prior = episode_means(load_arm(model, "population_prior"), domain, structure)
    seeds = sorted(set(matched) & set(prior))
    contrasts = []
    for seed in seeds:
        tasks = sorted(set(matched[seed]) & set(prior[seed]))
        contrasts.append(sum(prior[seed][t] - matched[seed][t] for t in tasks) / len(tasks))
    return seeds, contrasts


def interval(mean: float, sd: float, n: int) -> tuple[float, float]:
    half = tcrit(n - 1) * sd / math.sqrt(n)
    return mean - half, mean + half


def smallest_excluding_zero(mean: float, sd: float, limit: int = 40) -> int | None:
    for n in range(3, limit + 1):
        low, high = interval(mean, sd, n)
        if low * high > 0:
            return n
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", dest="models",
                        help="repeatable; default qwen3_8b and qwen3_4b")
    parser.add_argument("--project", type=int, nargs="+", default=(5, 6, 8, 10),
                        help="seed counts to project the t interval to")
    args = parser.parse_args()

    for model in args.models or ["qwen3_8b", "qwen3_4b"]:
        print(f"\n=== {model} ===")
        for label, domain, structure in CELLS:
            seeds, contrasts = seed_contrasts(model, domain, structure)
            n = len(contrasts)
            mean, sd = st.mean(contrasts), st.stdev(contrasts)
            low, high = interval(mean, sd, n)
            print(f"  {label}")
            print(f"    per-seed: " + ", ".join(f"s{s}={c:+.4f}" for s, c in zip(seeds, contrasts)))
            print(f"    G={n} observed  mean={mean:+.4f} sd={sd:.4f} "
                  f"t({n - 1}) 95% CI=[{low:+.4f}, {high:+.4f}]")
            for target in args.project:
                low_g, high_g = interval(mean, sd, target)
                verdict = "excludes 0" if low_g * high_g > 0 else "spans 0"
                print(f"    G={target} projected            [{low_g:+.4f}, {high_g:+.4f}]  {verdict}")
            need = smallest_excluding_zero(mean, sd)
            print(f"    smallest G excluding zero at this sd: {need if need else '>40'}")
            unanimous = all(c > 0 for c in contrasts) or all(c < 0 for c in contrasts)
            if unanimous:
                print(f"    one-sided exact sign test at G={n}: p={2.0 ** -n:.4f}")
            else:
                print(f"    sign test: seed contrasts not unanimous at G={n}")


if __name__ == "__main__":
    main()
