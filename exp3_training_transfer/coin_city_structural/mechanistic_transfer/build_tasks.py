#!/usr/bin/env python3
"""Build and freeze the paired reference-transplant mechanistic task set."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import (  # noqa: E402
    DATA_DIR,
    DEV_EPISODES,
    EPISODE_COUNT,
    K_VALUES,
    REFERENCE_LENGTH,
    SEED_BASE,
    STUDY,
    TARGET_LENGTH,
    TEST_EPISODES,
    VARIANTS,
    atomic_json,
    atomic_jsonl,
    file_sha256,
    impulse_kernel,
    json_sha256,
    persistence_ratio,
)
from prompt import render_prompt  # noqa: E402
from worlds import (  # noqa: E402
    COIN_HARBOR,
    balanced_inputs,
    context_for,
    forecast_scenarios,
    response_vector,
    sample_parameters,
    simulate_series,
)


def series(structure: str, parameters: dict[str, float], seed: int, length: int) -> dict:
    rng = np.random.default_rng(seed)
    inputs = balanced_inputs(COIN_HARBOR, length, rng)
    return simulate_series(COIN_HARBOR, structure, inputs, parameters, rng)


def build_episode(index: int) -> dict[str, Any]:
    seed = SEED_BASE + index
    rng = np.random.default_rng(seed)
    parameters = {
        "direct_a": sample_parameters("direct_a", rng),
        "mediated_b": sample_parameters("mediated_b", rng),
    }
    specifications = list(parameters.items())
    rng.shuffle(specifications)
    references = []
    for ref_index, (structure, item_parameters) in enumerate(specifications):
        references.append(
            {
                **series(
                    structure,
                    item_parameters,
                    seed + 100_000 * (ref_index + 1),
                    REFERENCE_LENGTH,
                ),
                "structure": structure,
                "parameters": item_parameters,
                "context": context_for(COIN_HARBOR, structure),
            }
        )
    target_series = series(
        "mediated_b", parameters["mediated_b"], seed + 7_000_000, TARGET_LENGTH
    )
    return {
        "episode_id": f"harbor-mediated-{index:03d}",
        "episode_index": index,
        "generator_seed": seed,
        "parameters": parameters,
        "references": references,
        "selected_reference_index": next(
            position for position, row in enumerate(references) if row["structure"] == "mediated_b"
        ),
        "target_series": target_series,
        "kernel": impulse_kernel(parameters["mediated_b"]).tolist(),
    }


def assign_donors_and_splits(episodes: list[dict[str, Any]]) -> None:
    ordered = sorted(episodes, key=lambda row: (row["kernel"][0], row["episode_id"]))
    donor_pairs = list(zip(ordered[: len(ordered) // 2], ordered[len(ordered) // 2 :]))
    split_rng = np.random.default_rng(0xE3C0A51)
    test_pair_ranks = set(
        int(value)
        for value in split_rng.choice(len(donor_pairs), TEST_EPISODES // 2, replace=False)
    )
    dev_rank = 0
    for pair_rank, (low, high) in enumerate(donor_pairs):
        pair_id = f"donor-pair-{pair_rank:02d}"
        split = "test" if pair_rank in test_pair_ranks else "dev"
        fold = None if split == "test" else dev_rank % 4
        if split == "dev":
            dev_rank += 1
        for recipient, donor in ((low, high), (high, low)):
            recipient["donor_episode_id"] = donor["episode_id"]
            recipient["donor_pair_id"] = pair_id
            recipient["split"] = split
            recipient["dev_fold"] = fold
            recipient["donor_kernel"] = donor["kernel"]


def anchors_for(prompt: str) -> dict[str, int]:
    ref2 = prompt.index("\n\nREFERENCE SYSTEM 2")
    target = prompt.index("\n\nTARGET SYSTEM")
    background_start = prompt.index("Background:", target)
    background_end = prompt.index("\n\n", background_start)
    query_start = prompt.index("\n\nCOUNTERFACTUAL FORECASTS", background_end)
    return_start = prompt.index("\n\nReturn only strict JSON", query_start)
    anchors = {
        "reference_1_end": ref2,
        "reference_2_end": target,
        "target_background_end": background_end,
        "target_evidence_end": query_start,
        "query_end": return_start,
        "final_prompt": len(prompt),
    }
    if any(position <= 0 or position > len(prompt) for position in anchors.values()):
        raise ValueError("invalid prompt anchor")
    return anchors


def scenario_ends_for(prompt: str) -> list[int]:
    """Exclusive character ends for scenario rows A--J in the raw prompt."""
    positions = []
    search_from = prompt.index("\n\nCOUNTERFACTUAL FORECASTS")
    for label in "ABCDEFGHIJ":
        start = prompt.index(f"\n\n  {label}:", search_from) + 2
        end = prompt.index("\n\n", start)
        positions.append(end)
        search_from = end
    if positions[-1] != prompt.index("\n\nReturn only strict JSON", positions[-2]):
        raise ValueError("scenario J does not end at the frozen query anchor")
    return positions


def make_rows(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["episode_id"]: row for row in episodes}
    rows = []
    for episode in sorted(episodes, key=lambda row: row["episode_id"]):
        donor = by_id[episode["donor_episode_id"]]
        selected = int(episode["selected_reference_index"])
        for k in K_VALUES:
            original_parameters = episode["parameters"]["mediated_b"]
            donor_parameters = donor["parameters"]["mediated_b"]
            history = episode["target_series"]["inputs"][:k]
            original_targets = forecast_scenarios(
                COIN_HARBOR, "mediated_b", history, original_parameters
            )
            donor_targets = forecast_scenarios(
                COIN_HARBOR, "mediated_b", history, donor_parameters
            )
            target = {
                **episode["target_series"],
                "shown_context": context_for(COIN_HARBOR, "mediated_b", target=True),
            }
            for variant in VARIANTS:
                references = deepcopy(episode["references"])
                shown_parameters = original_parameters
                if variant == "transplant":
                    donor_reference = deepcopy(donor["references"][donor["selected_reference_index"]])
                    # The selected slot and its semantic background remain fixed; only its
                    # numeric trajectory is transplanted.
                    for field in ("inputs", "observed", "expected"):
                        references[selected][field] = donor_reference[field]
                    shown_parameters = donor_parameters
                prompt = render_prompt(
                    COIN_HARBOR, references, target, "correct", k, training=False
                )
                task_id = f"{episode['episode_id']}:k{k}:{variant}"
                selected_identity = selected + 1
                row = {
                    "study": STUDY,
                    "sample_id": task_id,
                    "task_id": task_id,
                    "episode_id": episode["episode_id"],
                    "episode_index": episode["episode_index"],
                    "generator_seed": episode["generator_seed"],
                    "donor_episode_id": donor["episode_id"],
                    "donor_pair_id": episode["donor_pair_id"],
                    "split": episode["split"],
                    "dev_fold": episode["dev_fold"],
                    "domain": "coin_harbor",
                    "structure": "mediated_b",
                    "cue": "correct",
                    "k": k,
                    "variant": variant,
                    "selected_reference_identity": selected_identity,
                    "kernel_h1_h3": [float(value) for value in impulse_kernel(shown_parameters)],
                    "original_kernel_h1_h3": [
                        float(value) for value in impulse_kernel(original_parameters)
                    ],
                    "donor_kernel_h1_h3": [
                        float(value) for value in impulse_kernel(donor_parameters)
                    ],
                    "persistence_ratio": persistence_ratio(shown_parameters),
                    "original_parameters": original_parameters,
                    "donor_parameters": donor_parameters,
                    "truth_targets": original_targets.round(6).tolist(),
                    "donor_targets": donor_targets.round(6).tolist(),
                    "truth_response": response_vector(original_targets).round(6).tolist(),
                    "donor_response": response_vector(donor_targets).round(6).tolist(),
                    "displayed_reference_numeric_hash": json_sha256(
                        [
                            {"inputs": ref["inputs"], "observed": ref["observed"]}
                            for ref in references
                        ]
                    ),
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "anchor_char_ends": anchors_for(prompt),
                    "patch_scenario_char_ends": scenario_ends_for(prompt),
                }
                rows.append(row)
    rows.sort(key=lambda row: row["sample_id"])
    return rows


def validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = EPISODE_COUNT * len(K_VALUES) * len(VARIANTS)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, observed {len(rows)}")
    if len({row["sample_id"] for row in rows}) != expected:
        raise ValueError("duplicate sample IDs")
    episode_splits: dict[str, str] = {}
    for row in rows:
        prior = episode_splits.setdefault(row["episode_id"], row["split"])
        if prior != row["split"]:
            raise ValueError("episode crosses development/test split")
        donor = next(item for item in rows if item["episode_id"] == row["donor_episode_id"])
        if donor["split"] != row["split"] or donor["donor_episode_id"] != row["episode_id"]:
            raise ValueError("donor pairing crosses split or is not reciprocal")
    counts = Counter((row["split"], row["k"], row["variant"]) for row in rows)
    for split, episode_count in (("dev", DEV_EPISODES), ("test", TEST_EPISODES)):
        for k in K_VALUES:
            for variant in VARIANTS:
                if counts[(split, k, variant)] != episode_count:
                    raise ValueError(f"unbalanced cell {(split, k, variant)}")
    for episode_id in episode_splits:
        items = [row for row in rows if row["episode_id"] == episode_id]
        originals = {(row["k"]): row for row in items if row["variant"] == "original"}
        transplants = {(row["k"]): row for row in items if row["variant"] == "transplant"}
        for k in K_VALUES:
            left, right = originals[k], transplants[k]
            if left["truth_targets"] != right["truth_targets"] or left["prompt"] == right["prompt"]:
                raise ValueError("invalid original/transplant pair")
    separations = [
        abs(row["donor_kernel_h1_h3"][0] - row["original_kernel_h1_h3"][0])
        for row in rows
    ]
    return {
        "record_count": len(rows),
        "episode_count": len(episode_splits),
        "development_episodes": sum(value == "dev" for value in episode_splits.values()),
        "test_episodes": sum(value == "test" for value in episode_splits.values()),
        "donor_pair_count": len({row["donor_pair_id"] for row in rows}),
        "minimum_h1_donor_separation": float(min(separations)),
        "median_h1_donor_separation": float(np.median(separations)),
        "cell_counts": {str(key): value for key, value in sorted(counts.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    episodes = [build_episode(index) for index in range(EPISODE_COUNT)]
    assign_donors_and_splits(episodes)
    rows = make_rows(episodes)
    summary = validate(rows)
    tasks_path = args.outdir / "tasks.jsonl"
    atomic_jsonl(tasks_path, rows)
    manifest = {
        "study": STUDY,
        "status": "tasks_frozen",
        "source_protocol": "coin_city_structural_transfer_v1",
        "task_seed_band": [SEED_BASE, SEED_BASE + EPISODE_COUNT - 1],
        "domain": "coin_harbor",
        "structure": "mediated_b",
        "cue": "correct",
        "k_values": list(K_VALUES),
        "variants": list(VARIANTS),
        "target_variables": [
            "immediate_per_unit_response_h1",
            "horizon_three_per_unit_response_h3",
            "persistence_ratio",
            "selected_reference_identity",
            "eight_dimensional_nonzero_shock_response",
        ],
        "split_rule": (
            "sort episodes by h1; pair lower and upper halves; choose 16 of 64 reciprocal "
            "donor pairs for sealed test with RNG seed 0xE3C0A51; assign four dev folds by "
            "remaining donor-pair rank modulo four"
        ),
        "transplant_rule": (
            "replace only the selected mediated reference numeric trajectory with its "
            "same-split paired donor; preserve slot, background, target, query, and formatting"
        ),
        "tasks_sha256": file_sha256(tasks_path),
        **summary,
    }
    atomic_json(args.outdir / "task_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
