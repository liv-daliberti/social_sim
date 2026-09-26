#!/usr/bin/env python3
"""Paper-external analysis of the in-progress Llama-3.1-70B seed-42 runs.

This analysis is deliberately fail-closed on the two exact jobs and their
temperature-zero online evaluations at rounds 0, 50, and 100.  It is not a
replacement for the registered three-seed, five-draw round-300 endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import make_paper_outputs as core


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
REPORTS = ROOT / "reports"
MODEL = "llama3_1_70b"
SEED = 42
CHECKPOINTS = (0, 50, 100)
RUNS = {
    "population_prior": {
        "job_id": "30929139",
        "report": (
            "population_prior_llama3_1_70b_s42_20260828_063001_j30929139"
        ),
        "runtime": "debug_0828T06:30:33",
    },
    "causal": {
        "job_id": "30929140",
        "report": "causal_llama3_1_70b_s42_20260828_075600_j30929140",
        "runtime": "debug_0828T08:02:01",
    },
}
CELLS = core.TRANSFER_CELLS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_path(arm: str, checkpoint: int) -> Path:
    run = RUNS[arm]
    return (
        REPORTS
        / run["report"]
        / run["runtime"]
        / "eval_results"
        / f"{checkpoint}.scores.jsonl"
    )


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def binary_mean(rows: list[dict], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows]))


def select(rows: list[dict], **selectors: object) -> list[dict]:
    return [
        row
        for row in rows
        if all(row.get(key) == value for key, value in selectors.items())
    ]


def task_map(rows: list[dict], key: str) -> dict[str, float]:
    result = {}
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in result:
            raise AssertionError(f"duplicate task: {task_id}")
        result[task_id] = float(row[key])
    return result


def paired_task_interval(
    left: list[dict],
    right: list[dict],
    *,
    key: str,
    repetitions: int,
    seed: int,
) -> dict:
    left_values = task_map(left, key)
    right_values = task_map(right, key)
    shared = sorted(set(left_values) & set(right_values))
    if len(shared) != len(left_values) or len(shared) != len(right_values):
        raise AssertionError("paired task universes differ")
    differences = np.asarray(
        [left_values[task] - right_values[task] for task in shared],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    draws = [
        float(np.mean(rng.choice(differences, len(differences), replace=True)))
        for _ in range(repetitions)
    ]
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "estimate": float(np.mean(differences)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "paired_tasks": len(shared),
    }


def summarize_arm(rows: list[dict]) -> dict:
    return {
        "n_tasks": len(rows),
        "parse_rate": binary_mean(rows, "parsed"),
        "response_mae": mean(rows, "response_mae"),
        "structure_accuracy": binary_mean(rows, "structure_correct"),
    }


def analyze_cell(
    rows: list[dict],
    *,
    checkpoint: int,
    domain: str,
    structure: str,
    repetitions: int,
    seed: int,
) -> dict:
    by_arm = {
        arm: select(
            rows,
            checkpoint=checkpoint,
            arm=arm,
            cue="correct",
            domain=domain,
            target_structure=structure,
        )
        for arm in RUNS
    }
    if any(len(values) != 120 for values in by_arm.values()):
        raise AssertionError(
            f"checkpoint {checkpoint} {domain}/{structure}: expected 120 tasks per arm"
        )
    paired = paired_task_interval(
        by_arm["population_prior"],
        by_arm["causal"],
        key="response_mae",
        repetitions=repetitions,
        seed=seed,
    )
    prior_by_id = {str(row["task_id"]): row for row in by_arm["population_prior"]}
    causal_by_id = {str(row["task_id"]): row for row in by_arm["causal"]}
    both_parsed_ids = sorted(
        task
        for task in prior_by_id
        if prior_by_id[task]["parsed"] and causal_by_id[task]["parsed"]
    )
    parsed_only = paired_task_interval(
        [prior_by_id[task] for task in both_parsed_ids],
        [causal_by_id[task] for task in both_parsed_ids],
        key="response_mae",
        repetitions=repetitions,
        seed=seed + 10_000,
    )
    return {
        "domain": domain,
        "target_structure": structure,
        "arms": {arm: summarize_arm(values) for arm, values in by_arm.items()},
        "population_prior_minus_causal_response_mae": paired,
        "both_parsed_only_diagnostic": parsed_only,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    args = parser.parse_args()

    rows: list[dict] = []
    inputs: list[Path] = []
    task_universe: frozenset[tuple] | None = None
    for arm, run in RUNS.items():
        item = {
            "kind": "confirmatory",
            "model": MODEL,
            "arm": arm,
            "seed": SEED,
            "job_id": run["job_id"],
        }
        for checkpoint in CHECKPOINTS:
            path = score_path(arm, checkpoint)
            checkpoint_rows = core.read_jsonl(path)
            universe = core.validate_rows(checkpoint_rows, item, "greedy")
            if task_universe is None:
                task_universe = universe
            elif universe != task_universe:
                raise AssertionError(f"task universe drifted: {path}")
            rows.extend(dict(row, checkpoint=checkpoint) for row in checkpoint_rows)
            inputs.append(path)

    if len(inputs) != 6 or len(rows) != 6 * 1440:
        raise AssertionError("expected six complete 1,440-task evaluations")

    checkpoints = {}
    for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
        checkpoints[str(checkpoint)] = [
            analyze_cell(
                rows,
                checkpoint=checkpoint,
                domain=domain,
                structure=structure,
                repetitions=args.bootstrap_repetitions,
                seed=(
                    args.bootstrap_seed
                    + checkpoint_index * 100
                    + cell_index
                ),
            )
            for cell_index, (domain, structure) in enumerate(CELLS)
        ]

    artifact = {
        "analysis": "private_interim_llama3_1_70b_seed42_round100",
        "status": "paper_external_incomplete_roster_preview",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warning": (
            "Exploratory interim only: one training seed, one deterministic draw, "
            "and round 100 of 300. Task-bootstrap intervals condition on seed 42 "
            "and do not estimate between-training-seed uncertainty."
        ),
        "model": MODEL,
        "training_seed": SEED,
        "checkpoints": list(CHECKPOINTS),
        "closed_task_universe": {
            "tasks_per_arm_checkpoint": 1440,
            "correct_cue_tasks_per_transfer_cell": 120,
            "temperature": 0.0,
            "draws_per_task": 1,
            "validated_files": len(inputs),
            "validated_rows": len(rows),
        },
        "bootstrap": {
            "repetitions": args.bootstrap_repetitions,
            "seed": args.bootstrap_seed,
            "variance_unit": (
                "paired held-out task conditional on the single training seed"
            ),
        },
        "input_score_files": [str(path.relative_to(REPO)) for path in inputs],
        "input_sha256": {
            str(path.relative_to(REPO)): sha256(path) for path in inputs
        },
        "cell_estimates_by_checkpoint": checkpoints,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"validated {len(inputs)} online evaluations and {len(rows)} rows")
    print(f"paper-external interim result -> {args.output}")


if __name__ == "__main__":
    main()
