#!/usr/bin/env python3
"""Build paired disclosed/undisclosed datasets for every C3 mechanism-family arm."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from prompt import gold_response, render_prompt
from worlds import (
    SCENARIO_LABELS,
    TEST,
    TRAIN,
    balanced_inputs,
    forecast_scenarios,
    population_prior_targets,
    response_pairs,
    response_vector,
    scenario_grid,
    simulate_series,
)

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
DISCLOSURES = ("disclosed", "undisclosed")
MODES = ("causal_family", "population_prior", "structureless")
TRAIN_SEED_BASE = 31_000_000
EVAL_SEED_BASE = 73_000_000
K_VALUES = (3, 6, 9)
TARGET_LENGTH = 9
CALIBRATION_LENGTH = 10
N_CALIBRATIONS = 3


def _to_list(array) -> list:
    return np.asarray(array, dtype=float).round(4).tolist()


def _series(world, seed: int, length: int) -> dict:
    rng = np.random.default_rng(seed)
    gain = float(rng.uniform(world.gain_lo, world.gain_hi))
    inputs = balanced_inputs(world, length, rng)
    observed, expected, _ = simulate_series(world, inputs, gain, rng)
    return {
        "inputs": _to_list(inputs),
        "observed": _to_list(observed),
        "expected": _to_list(expected),
        "gain": round(gain, 6),
    }


def make_bundle(world, seed: int, k: int, structureless: bool = False) -> tuple[dict, dict]:
    calibrations = [
        _series(world, seed + 100_000 * (index + 1), CALIBRATION_LENGTH)
        for index in range(N_CALIBRATIONS)
    ]
    target = _series(world, seed, TARGET_LENGTH)
    causal = {
        "calibrations": calibrations,
        "target_inputs": target["inputs"][:k],
        "target_observed": target["observed"][:k],
    }
    shown = json.loads(json.dumps(causal))
    if structureless:
        # Preserve every outcome and task identity, but independently resample every displayed
        # driver. The intervention target is flat, so the training reward contains no driver signal.
        for index, calibration in enumerate(shown["calibrations"]):
            rng = np.random.default_rng(seed ^ (0x51A7 + index))
            calibration["inputs"] = _to_list(balanced_inputs(world, len(calibration["inputs"]), rng))
        rng = np.random.default_rng(seed ^ 0xC0117)
        shown["target_inputs"] = _to_list(balanced_inputs(world, k, rng))
    truth_targets = forecast_scenarios(
        world,
        np.asarray(causal["target_inputs"], dtype=float),
        float(target["gain"]),
    )
    metadata = {
        "truth_targets": truth_targets,
        "target_gain": float(target["gain"]),
        "causal_bundle": causal,
    }
    return shown, metadata


def numeric_hash(bundle: dict) -> str:
    payload = {
        "calibrations": [
            {"inputs": item["inputs"], "observed": item["observed"]}
            for item in bundle["calibrations"]
        ],
        "target_inputs": bundle["target_inputs"],
        "target_observed": bundle["target_observed"],
        "scenarios": scenario_grid(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_row(world, seed: int, k: int, disclosure: str, mode: str, *, evaluation: bool = False):
    effective_mode = "causal_family" if evaluation else mode
    bundle, metadata = make_bundle(
        world,
        seed,
        k,
        structureless=(effective_mode == "structureless"),
    )
    truth_targets = np.asarray(metadata["truth_targets"], dtype=float)
    last_observed = float(bundle["target_observed"][-1])
    prior_targets = population_prior_targets(world, last_observed)
    if effective_mode == "causal_family":
        targets = truth_targets
    elif effective_mode == "population_prior":
        targets = prior_targets
    elif effective_mode == "structureless":
        targets = np.full(len(SCENARIO_LABELS), last_observed)
    else:
        raise ValueError(effective_mode)

    pairs = response_pairs()
    reference = {
        "protocol": "c3_mechanism_disclosure_v3",
        "task_id": f"{world.name}:{seed}:{k}",
        "world": world.name,
        "block": world.block,
        "split": world.split,
        "seed": int(seed),
        "k": int(k),
        "disclosure": disclosure,
        "mode": "evaluation" if evaluation else effective_mode,
        "g": float(metadata["target_gain"]),
        "targets": _to_list(targets),
        "truth_targets": _to_list(truth_targets),
        "prior_targets": _to_list(prior_targets),
        "scenario_labels": list(SCENARIO_LABELS),
        "scenario_shocks": [item["shock"] for item in scenario_grid()],
        "scenario_horizons": [item["horizon"] for item in scenario_grid()],
        # The legacy key is retained for the shared trainer; response_targets selects the new
        # response-vector reward branch instead of its slope branch.
        "shocks": [item["shock"] for item in scenario_grid()],
        "response_pairs": [list(pair) for pair in pairs],
        "response_targets": _to_list(response_vector(targets)),
        "truth_response": _to_list(response_vector(truth_targets)),
        "prior_response": _to_list(response_vector(prior_targets)),
        "reward_scale": 10.0,
        "response_scale": 4.0,
        "clip": list(world.clip),
        "numeric_hash": numeric_hash(bundle),
        "calibrations": metadata["causal_bundle"]["calibrations"],
        "target_inputs": metadata["causal_bundle"]["target_inputs"],
        "target_observed": metadata["causal_bundle"]["target_observed"],
        "displayed_calibration_inputs": [item["inputs"] for item in bundle["calibrations"]],
        "displayed_target_inputs": bundle["target_inputs"],
        "gold": gold_response(targets),
    }
    return {
        "input": render_prompt(world, bundle, disclosure),
        "reference": json.dumps(reference, sort_keys=True),
    }


def build_train(disclosure: str, mode: str, n_per_world: int) -> list[dict]:
    rows = []
    for world_index, world in enumerate(TRAIN):
        for item_index in range(n_per_world):
            seed = TRAIN_SEED_BASE + world_index * 1_000_000 + item_index
            k = K_VALUES[item_index % len(K_VALUES)]
            rows.append(make_row(world, seed, k, disclosure, mode))
    np.random.default_rng(0xC3D15C).shuffle(rows)
    return rows


def build_eval(disclosure: str, n_per_world: int) -> list[dict]:
    rows = []
    for world_index, world in enumerate(TEST):
        for item_index in range(n_per_world):
            seed = EVAL_SEED_BASE + world_index * 1_000_000 + item_index
            for k in K_VALUES:
                rows.append(make_row(
                    world,
                    seed,
                    k,
                    disclosure,
                    "causal_family",
                    evaluation=True,
                ))
    return rows


def write_one(disclosure: str, mode: str, n_train_per: int, n_eval_per: int) -> dict:
    from datasets import Dataset, DatasetDict

    output = DATA_ROOT / f"{disclosure}_{mode}"
    train = build_train(disclosure, mode, n_train_per)
    heldout = build_eval(disclosure, n_eval_per)
    output.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": Dataset.from_list(train)}).save_to_disk(str(output / "train"))
    DatasetDict({"train": Dataset.from_list(heldout)}).save_to_disk(str(output / "heldout"))
    return {
        "disclosure": disclosure,
        "mode": mode,
        "path": str(output.relative_to(ROOT)),
        "train_rows": len(train),
        "heldout_rows": len(heldout),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disclosure", choices=DISCLOSURES)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--n-train-per", type=int, default=600)
    parser.add_argument("--n-eval-per", type=int, default=60)
    args = parser.parse_args()
    if not args.all and (args.disclosure is None or args.mode is None):
        parser.error("pass --all or both --disclosure and --mode")

    selections = (
        [(disclosure, mode) for disclosure in DISCLOSURES for mode in MODES]
        if args.all else [(args.disclosure, args.mode)]
    )
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    built = [
        write_one(disclosure, mode, args.n_train_per, args.n_eval_per)
        for disclosure, mode in selections
    ]
    manifest = {
        "protocol": "c3_mechanism_disclosure_v3",
        "n_train_per_world": args.n_train_per,
        "n_eval_per_world": args.n_eval_per,
        "k_values": list(K_VALUES),
        "calibration_trajectories": N_CALIBRATIONS,
        "calibration_length": CALIBRATION_LENGTH,
        "target_length": TARGET_LENGTH,
        "datasets": built,
    }
    (DATA_ROOT / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for item in built:
        print(f"{item['disclosure']:11s} {item['mode']:17s} "
              f"train={item['train_rows']} heldout={item['heldout_rows']} -> {item['path']}")


if __name__ == "__main__":
    main()
