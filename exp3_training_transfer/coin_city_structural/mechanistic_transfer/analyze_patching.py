#!/usr/bin/env python3
"""Score the sealed patch intervention and estimate registered causal contrasts."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import (  # noqa: E402
    RUNS_DIR,
    STUDY,
    TRAINING_SEEDS,
    atomic_json,
    load_frozen_tasks,
    paired_seed_bootstrap,
)
from worlds import response_vector  # noqa: E402


def parse(text: str, clip: tuple[float, float]) -> np.ndarray | None:
    try:
        payload = json.loads(text.strip())
    except (AttributeError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"forecasts"}:
        return None
    values = payload["forecasts"]
    if not isinstance(values, list) or len(values) != 10:
        return None
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        return None
    return np.clip(np.asarray(values, dtype=float), *clip)


def main() -> None:
    tasks, _ = load_frozen_tasks()
    task_by_id = {row["sample_id"]: row for row in tasks}
    scored = []
    exact_self = []
    parse_flags = []
    for seed in TRAINING_SEEDS:
        paths = sorted((RUNS_DIR / f"qwen3_8b_s{seed}").glob("patch_outputs_shard_*.jsonl"))
        if not paths:
            fallback = RUNS_DIR / f"qwen3_8b_s{seed}" / "patch_outputs.jsonl"
            paths = [fallback] if fallback.is_file() else []
        records = [
            json.loads(line)
            for path in paths
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        if len(records) != 64 * 8:
            raise ValueError(f"seed {seed} has {len(records)} patch conditions, expected 512")
        by_sample = defaultdict(dict)
        for record in records:
            task = task_by_id[record["sample_id"]]
            truth = np.asarray(task["truth_response"], dtype=float)
            output_rows = []
            for draw, output in enumerate(record["outputs"]):
                predicted = parse(output, (40.0, 320.0))
                parsed = predicted is not None
                parse_flags.append(parsed)
                mae = 280.0 if predicted is None else float(
                    np.mean(np.abs(response_vector(predicted) - truth))
                )
                output_rows.append(mae)
                scored.append(
                    {
                        "seed": seed,
                        "sample_id": record["sample_id"],
                        "episode_id": task["episode_id"],
                        "k": task["k"],
                        "condition": record["condition"],
                        "draw": draw,
                        "parsed": parsed,
                        "response_mae": mae,
                    }
                )
            by_sample[record["sample_id"]][record["condition"]] = {
                "outputs": record["outputs"], "mean_mae": float(np.mean(output_rows))
            }
        for sample_id, conditions in by_sample.items():
            for base, self_condition in (
                ("prior_unpatched", "prior_self"),
                ("matched_unpatched", "matched_self"),
            ):
                exact_self.extend(
                    left.strip() == right.strip()
                    for left, right in zip(
                        conditions[base]["outputs"], conditions[self_condition]["outputs"]
                    )
                )

    means = defaultdict(list)
    for row in scored:
        means[(row["seed"], row["sample_id"], row["condition"])].append(row["response_mae"])
    means = {key: float(np.mean(value)) for key, value in means.items()}
    by_k = {}
    for k in (0, 8):
        forward: dict[int, np.ndarray] = {}
        reverse: dict[int, np.ndarray] = {}
        specificity: dict[int, np.ndarray] = {}
        symmetric: dict[int, np.ndarray] = {}
        for seed in TRAINING_SEEDS:
            sample_ids = sorted(
                row["sample_id"]
                for row in tasks
                if row["split"] == "test" and row["variant"] == "original" and row["k"] == k
            )
            f_values, r_values, s_values, sym_values = [], [], [], []
            for sample_id in sample_ids:
                get = lambda condition: means[(seed, sample_id, condition)]
                f = get("prior_unpatched") - get("matched_same_to_prior")
                r = get("prior_same_to_matched") - get("matched_unpatched")
                wrong_f = get("prior_unpatched") - get("matched_wrong_to_prior")
                wrong_r = get("prior_wrong_to_matched") - get("matched_unpatched")
                symmetric_value = 0.5 * (f + r)
                f_values.append(f)
                r_values.append(r)
                sym_values.append(symmetric_value)
                s_values.append(symmetric_value - 0.5 * (wrong_f + wrong_r))
            forward[seed] = np.asarray(f_values)
            reverse[seed] = np.asarray(r_values)
            symmetric[seed] = np.asarray(sym_values)
            specificity[seed] = np.asarray(s_values)
        by_k[str(k)] = {
            "matched_to_prior_improvement": paired_seed_bootstrap(
                forward, seed=202608253 + k
            ),
            "prior_to_matched_harm": paired_seed_bootstrap(reverse, seed=202608263 + k),
            "symmetric_aligned_effect": paired_seed_bootstrap(
                symmetric, seed=202608273 + k
            ),
            "same_minus_wrong_episode": paired_seed_bootstrap(
                specificity, seed=202608283 + k
            ),
        }
    summary = {
        "study": STUDY,
        "status": "sealed_patching_analysis_complete",
        "strict_parse_rate": float(np.mean(parse_flags)),
        "self_patch_exact_forecast_fidelity": float(np.mean(exact_self)),
        "effects_by_k": by_k,
    }
    atomic_json(RUNS_DIR / "patching_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
