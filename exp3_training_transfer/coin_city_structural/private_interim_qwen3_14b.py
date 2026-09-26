#!/usr/bin/env python3
"""Paper-external preview of the five completed Qwen3-14B adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import make_paper_outputs as core


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
REPORTS = ROOT / "reports"
MODEL = "qwen3_14b"
COMPLETED = (
    ("causal", 42, "30880155"),
    ("causal", 44, "30880157"),
    ("population_prior", 42, "30880158"),
    ("population_prior", 43, "30880159"),
    ("population_prior", 44, "30880160"),
)
BASE_JOB = "30879415"
PAIRED_SEEDS = (42, 44)
CELLS = core.TRANSFER_CELLS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_root(arm: str, seed: int, job_id: str) -> Path:
    matches = sorted(
        REPORTS.glob(f"scale_{arm}_{MODEL}_s{seed}_*_j{job_id}")
    )
    if len(matches) != 1:
        raise AssertionError(
            f"{arm} seed {seed}: expected one job-{job_id} report, found {matches}"
        )
    return matches[0]


def load_endpoint(
    root: Path, item: dict, decode: str
) -> tuple[list[dict], Path, frozenset[tuple]]:
    path = core.endpoint_path(root, decode)
    rows = core.read_jsonl(path)
    universe = core.validate_rows(rows, item, decode)
    return rows, path, universe


def task_values(rows: list[dict], selectors: dict) -> dict[int, dict[str, float]]:
    grouped: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if all(row.get(key) == value for key, value in selectors.items()):
            grouped[int(row["seed"])][str(row["task_id"])].append(
                float(row["response_mae"])
            )
    return {
        seed: {
            task_id: float(np.mean(draws)) for task_id, draws in tasks.items()
        }
        for seed, tasks in grouped.items()
    }


def seed_means(values: dict[int, dict[str, float]]) -> dict[str, float]:
    return {
        str(seed): float(np.mean(list(tasks.values())))
        for seed, tasks in sorted(values.items())
    }


def paired_preview(
    left: dict[int, dict[str, float]],
    right: dict[int, dict[str, float]],
    *,
    repetitions: int,
    seed: int,
) -> dict:
    differences: dict[int, np.ndarray] = {}
    for training_seed in PAIRED_SEEDS:
        shared = sorted(set(left[training_seed]) & set(right[training_seed]))
        if len(shared) != 120:
            raise AssertionError(
                f"seed {training_seed}: expected 120 paired primary-cell tasks, "
                f"found {len(shared)}"
            )
        differences[training_seed] = np.asarray(
            [left[training_seed][task] - right[training_seed][task] for task in shared]
        )
    per_seed = {
        str(training_seed): float(np.mean(values))
        for training_seed, values in differences.items()
    }
    estimate = float(np.mean(list(per_seed.values())))
    rng = np.random.default_rng(seed)
    draws = []
    seed_array = np.asarray(PAIRED_SEEDS)
    for _ in range(repetitions):
        sampled = rng.choice(seed_array, len(seed_array), replace=True)
        draws.append(
            float(
                np.mean(
                    [
                        np.mean(
                            rng.choice(
                                differences[int(selected)],
                                len(differences[int(selected)]),
                                replace=True,
                            )
                        )
                        for selected in sampled
                    ]
                )
            )
        )
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "estimate": estimate,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "per_seed": per_seed,
        "paired_seeds": list(PAIRED_SEEDS),
    }


def unbalanced_preview(
    left: dict[int, dict[str, float]],
    right: dict[int, dict[str, float]],
    *,
    repetitions: int,
    seed: int,
) -> dict:
    left_seeds = np.asarray(sorted(left))
    right_seeds = np.asarray(sorted(right))
    if tuple(left_seeds) != (42, 43, 44) or tuple(right_seeds) != PAIRED_SEEDS:
        raise AssertionError("unexpected interim arm roster")

    def arm_mean(values: dict[int, dict[str, float]]) -> float:
        return float(
            np.mean([np.mean(list(tasks.values())) for tasks in values.values()])
        )

    estimate = arm_mean(left) - arm_mean(right)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        left_sample = rng.choice(left_seeds, len(left_seeds), replace=True)
        right_sample = rng.choice(right_seeds, len(right_seeds), replace=True)
        left_mean = np.mean(
            [
                np.mean(
                    rng.choice(
                        list(left[int(selected)].values()),
                        len(left[int(selected)]),
                        replace=True,
                    )
                )
                for selected in left_sample
            ]
        )
        right_mean = np.mean(
            [
                np.mean(
                    rng.choice(
                        list(right[int(selected)].values()),
                        len(right[int(selected)]),
                        replace=True,
                    )
                )
                for selected in right_sample
            ]
        )
        draws.append(float(left_mean - right_mean))
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "estimate": estimate,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "left_seeds": [int(value) for value in left_seeds],
        "right_seeds": [int(value) for value in right_seeds],
    }


def analyze_cell(
    rows: list[dict],
    *,
    decode: str,
    domain: str,
    structure: str,
    repetitions: int,
    seed: int,
) -> dict:
    common = {
        "model": MODEL,
        "decode": decode,
        "domain": domain,
        "target_structure": structure,
        "cue": "correct",
    }
    prior = task_values(rows, {**common, "arm": "population_prior"})
    matched = task_values(rows, {**common, "arm": "causal"})
    base = task_values(rows, {**common, "arm": "base"})
    if set(prior) != {42, 43, 44} or set(matched) != {42, 44} or set(base) != {0}:
        raise AssertionError("cell does not contain the exact five-adapter plus base roster")
    return {
        "domain": domain,
        "target_structure": structure,
        "mean_response_mae": {
            "base": float(np.mean(list(base[0].values()))),
            "population_prior_three_seed": float(
                np.mean([np.mean(list(tasks.values())) for tasks in prior.values()])
            ),
            "episode_matched_two_seed": float(
                np.mean([np.mean(list(tasks.values())) for tasks in matched.values()])
            ),
        },
        "per_seed_response_mae": {
            "population_prior": seed_means(prior),
            "episode_matched": seed_means(matched),
        },
        "paired_seed_preview_population_prior_minus_episode_matched": paired_preview(
            prior, matched, repetitions=repetitions, seed=seed
        ),
        "unbalanced_three_vs_two_population_prior_minus_episode_matched": unbalanced_preview(
            prior, matched, repetitions=repetitions, seed=seed + 1000
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    args = parser.parse_args()

    rows: list[dict] = []
    inputs: list[Path] = []
    reference: frozenset[tuple] | None = None
    for arm, training_seed, job_id in COMPLETED:
        item = {
            "kind": "confirmatory",
            "model": MODEL,
            "arm": arm,
            "seed": training_seed,
            "job_id": job_id,
        }
        root = find_root(arm, training_seed, job_id)
        for decode in ("greedy", "stochastic"):
            endpoint_rows, path, universe = load_endpoint(root, item, decode)
            if reference is None:
                reference = universe
            elif universe != reference:
                raise AssertionError(f"task universe drifted: {path}")
            rows.extend(endpoint_rows)
            inputs.append(path)

    base_matches = sorted(REPORTS.glob(f"base_{MODEL}_j{BASE_JOB}"))
    if len(base_matches) != 1:
        raise AssertionError(f"expected one base report, found {base_matches}")
    base_item = {
        "kind": "base",
        "model": MODEL,
        "job_id": BASE_JOB,
    }
    for decode in ("greedy", "stochastic"):
        endpoint_rows, path, universe = load_endpoint(
            base_matches[0], base_item, decode
        )
        if reference != universe:
            raise AssertionError(f"base task universe drifted: {path}")
        rows.extend(endpoint_rows)
        inputs.append(path)

    expected_rows = 6 * (1440 + 7200)
    if len(inputs) != 12 or len(rows) != expected_rows:
        raise AssertionError(
            f"expected 12 endpoint files and {expected_rows} rows, "
            f"found {len(inputs)} and {len(rows)}"
        )

    estimates = {}
    for decode_index, decode in enumerate(("greedy", "stochastic")):
        estimates[decode] = [
            analyze_cell(
                rows,
                decode=decode,
                domain=domain,
                structure=structure,
                repetitions=args.bootstrap_repetitions,
                seed=args.bootstrap_seed + decode_index * 100 + cell_index,
            )
            for cell_index, (domain, structure) in enumerate(CELLS)
        ]

    artifact = {
        "analysis": "private_interim_qwen3_14b_five_completed_adapters",
        "status": "paper_external_incomplete_roster_preview",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warning": (
            "Not the registered Qwen3-14B result. Episode-matched seed 43 is "
            "missing; confirmatory rendering remains fail-closed."
        ),
        "completed_adapter_roster": [
            {"arm": arm, "seed": seed, "job_id": job_id}
            for arm, seed, job_id in COMPLETED
        ],
        "missing_adapter": {"arm": "causal", "seed": 43},
        "closed_task_universe": {
            "tasks_per_endpoint": 1440,
            "greedy_draws_per_task": 1,
            "stochastic_draws_per_task": 5,
            "temperature": 0.7,
            "validated_endpoint_files": len(inputs),
            "validated_row_draws": len(rows),
        },
        "bootstrap": {
            "repetitions": args.bootstrap_repetitions,
            "seed": args.bootstrap_seed,
            "paired_preview_variance_unit": (
                "two available paired training seeds, then paired task within seed"
            ),
            "unbalanced_preview_variance_unit": (
                "arm-specific training seed, then task within seed"
            ),
        },
        "input_score_files": [str(path.relative_to(REPO)) for path in inputs],
        "input_sha256": {
            str(path.relative_to(REPO)): sha256(path) for path in inputs
        },
        "estimates": estimates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"validated {len(inputs)} closed-eval files and {len(rows)} row-draws")
    print(f"paper-external interim result -> {args.output}")


if __name__ == "__main__":
    main()
