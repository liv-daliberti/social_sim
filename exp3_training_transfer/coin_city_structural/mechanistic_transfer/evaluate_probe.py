#!/usr/bin/env python3
"""Apply the frozen common readout once to the sealed mechanistic test episodes."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    ANCHORS,
    PRIMARY_ANCHOR,
    RUNS_DIR,
    STUDY,
    TRAINING_SEEDS,
    atomic_json,
    atomic_jsonl,
    file_sha256,
    load_frozen_tasks,
    paired_seed_bootstrap,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, default=RUNS_DIR / "common_probe")
    args = parser.parse_args()
    selection_path = args.probe_dir / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "development_selection_complete_test_unopened":
        raise ValueError("probe was not frozen before test evaluation")
    decoder_path = Path(selection["decoder_path"])
    if file_sha256(decoder_path) != selection["decoder_sha256"]:
        raise ValueError("decoder changed after development selection")
    decoder = np.load(decoder_path, allow_pickle=False)
    tasks, manifest = load_frozen_tasks()
    if selection["tasks_sha256"] != manifest["tasks_sha256"]:
        raise ValueError("selection/task mismatch")
    test_indices = np.asarray(
        [index for index, row in enumerate(tasks) if row["split"] == "test" and row["k"] == 0]
    )
    if len(test_indices) != 64:
        raise ValueError("unexpected sealed test surface")
    layer = int(selection["selected_layer"])
    anchor = ANCHORS.index(PRIMARY_ANCHOR)
    predictions = []
    effects: dict[int, np.ndarray] = {}
    tracking: dict[int, np.ndarray] = {}
    for seed in TRAINING_SEEDS:
        by_arm = {}
        for arm in ("matched", "prior"):
            path = RUNS_DIR / f"qwen3_8b_s{seed}" / arm / f"states_{arm}.npy"
            states = np.load(path, mmap_mode="r", allow_pickle=False)
            x = np.asarray(states[test_indices, layer, anchor, :], dtype=np.float64)
            standardized = (x - decoder["x_mean"]) @ decoder["beta"] + decoder["y_mean"]
            natural = standardized * decoder["target_sd"] + decoder["target_mean"]
            by_arm[arm] = standardized
            for local_index, task_index in enumerate(test_indices):
                row = tasks[int(task_index)]
                truth_standardized = (
                    np.asarray(row["kernel_h1_h3"], dtype=float) - decoder["target_mean"]
                ) / decoder["target_sd"]
                predictions.append(
                    {
                        "study": STUDY,
                        "seed": seed,
                        "arm": arm,
                        "sample_id": row["sample_id"],
                        "episode_id": row["episode_id"],
                        "variant": row["variant"],
                        "k": row["k"],
                        "predicted_kernel_standardized": standardized[local_index].tolist(),
                        "predicted_kernel": natural[local_index].tolist(),
                        "truth_kernel_standardized": truth_standardized.tolist(),
                        "squared_error": float(
                            np.mean((standardized[local_index] - truth_standardized) ** 2)
                        ),
                    }
                )
        task_truth = np.asarray(
            [
                (np.asarray(tasks[index]["kernel_h1_h3"]) - decoder["target_mean"])
                / decoder["target_sd"]
                for index in test_indices
            ]
        )
        effects[seed] = np.mean((by_arm["prior"] - task_truth) ** 2, axis=1) - np.mean(
            (by_arm["matched"] - task_truth) ** 2, axis=1
        )
        episode_rows: dict[str, dict[str, int]] = defaultdict(dict)
        for local_index, task_index in enumerate(test_indices):
            episode_rows[tasks[int(task_index)]["episode_id"]][
                tasks[int(task_index)]["variant"]
            ] = local_index
        tracking_values = []
        for episode_id, variants in episode_rows.items():
            original_index = variants["original"]
            transplant_index = variants["transplant"]
            original_task = tasks[int(test_indices[original_index])]
            displacement = (
                np.asarray(original_task["donor_kernel_h1_h3"]) - np.asarray(original_task["original_kernel_h1_h3"])
            ) / decoder["target_sd"]
            decoded_shift = (
                by_arm["matched"][transplant_index] - by_arm["matched"][original_index]
            )
            tracking_values.append(
                float(np.dot(decoded_shift, displacement) / max(np.dot(displacement, displacement), 1e-12))
            )
        tracking[seed] = np.asarray(tracking_values)
    output_path = args.probe_dir / "sealed_test_predictions.jsonl"
    atomic_jsonl(output_path, predictions)
    summary = {
        "study": STUDY,
        "status": "sealed_probe_test_complete",
        "selection_sha256": file_sha256(selection_path),
        "predictions_path": str(output_path),
        "predictions_sha256": file_sha256(output_path),
        "common_readout_matched_advantage": paired_seed_bootstrap(effects, seed=202608251),
        "matched_reference_transplant_tracking": paired_seed_bootstrap(
            tracking, seed=202608252
        ),
    }
    atomic_json(args.probe_dir / "sealed_test_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
