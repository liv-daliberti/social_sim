#!/usr/bin/env python3
"""Fail-closed audit for the paired C3 disclosure-factorial datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_from_disk

from prompt import MECHANISM_BEGIN, OUTPUT_PREFIX, OUTPUT_SUFFIX, parse_forecasts
from worlds import BY_NAME, TEST, TRAIN, oracle_forecast, response_vector

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
DISCLOSURES = ("disclosed", "undisclosed")
MODES = ("causal_family", "population_prior", "structureless")
MODEL_ROSTER = (
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
)
FORBIDDEN_BLIND_TEXT = (
    MECHANISM_BEGIN,
    "saturating",
    "threshold",
    "mediated",
    "feedback state",
    "outcome retains",
    "gaussian noise",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def references(dataset) -> list[dict]:
    return [json.loads(value) for value in dataset["reference"]]


def normalized_reference(value: str) -> str:
    payload = json.loads(value)
    payload.pop("disclosure", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): file_hash(path)
        for path in sorted(DATA_ROOT.rglob("*"))
        if path.is_file() and not path.name.startswith("cache-")
    }


def load_all(expected_train: int, expected_eval: int):
    loaded = {}
    for disclosure in DISCLOSURES:
        for mode in MODES:
            path = DATA_ROOT / f"{disclosure}_{mode}"
            require(path.is_dir(), f"missing dataset {path}")
            train = load_from_disk(str(path / "train"))["train"]
            heldout = load_from_disk(str(path / "heldout"))["train"]
            require(len(train) == expected_train,
                    f"{disclosure}/{mode}: {len(train)} train rows != {expected_train}")
            require(len(heldout) == expected_eval,
                    f"{disclosure}/{mode}: {len(heldout)} heldout rows != {expected_eval}")
            require(set(train.column_names) == {"input", "reference"}, "train schema drift")
            require(set(heldout.column_names) == {"input", "reference"}, "heldout schema drift")
            loaded[(disclosure, mode)] = {"train": train, "heldout": heldout}
    return loaded


def audit_pairing(loaded) -> dict:
    canonical = loaded[("disclosed", "causal_family")]
    canonical_train_refs = references(canonical["train"])
    canonical_ids = [item["task_id"] for item in canonical_train_refs]
    require(len(canonical_ids) == len(set(canonical_ids)), "training task IDs are not unique")

    for disclosure in DISCLOSURES:
        causal = loaded[(disclosure, "causal_family")]
        prior = loaded[(disclosure, "population_prior")]
        require(causal["train"]["input"] == prior["train"]["input"],
                f"{disclosure}: causal/prior prompts differ")
        for mode in MODES:
            rows = loaded[(disclosure, mode)]
            ids = [item["task_id"] for item in references(rows["train"])]
            require(ids == canonical_ids, f"{disclosure}/{mode}: training identities drifted")
            require(rows["heldout"]["input"] == causal["heldout"]["input"],
                    f"{disclosure}/{mode}: heldout prompts differ across arms")
            require(rows["heldout"]["reference"] == causal["heldout"]["reference"],
                    f"{disclosure}/{mode}: heldout references differ across arms")

    disclosed = loaded[("disclosed", "causal_family")]
    undisclosed = loaded[("undisclosed", "causal_family")]
    d_train = references(disclosed["train"])
    u_train = references(undisclosed["train"])
    require([item["numeric_hash"] for item in d_train]
            == [item["numeric_hash"] for item in u_train],
            "numeric training tasks differ across disclosure regimes")
    require([normalized_reference(value) for value in disclosed["heldout"]["reference"]]
            == [normalized_reference(value) for value in undisclosed["heldout"]["reference"]],
            "heldout scored records differ across disclosure regimes")
    require(disclosed["train"]["input"] != undisclosed["train"]["input"],
            "disclosure manipulation did not change prompts")
    return {"train_task_ids_unique": True, "paired_across_disclosure": True,
            "heldout_identical_across_arms": True}


def audit_catalog(loaded) -> dict:
    refs = references(loaded[("disclosed", "causal_family")]["train"])
    counts = Counter(item["world"] for item in refs)
    require(set(counts) == {world.name for world in TRAIN}, "training world roster drifted")
    require(len(set(counts.values())) == 1, "training worlds are not balanced")
    held = references(loaded[("disclosed", "causal_family")]["heldout"])
    require({item["world"] for item in held} == {world.name for world in TEST},
            "heldout world roster drifted")
    require(not ({item["seed"] for item in refs} & {item["seed"] for item in held}),
            "train/heldout episode seeds overlap")
    train_keys = {world.mechanism_key for world in TRAIN}
    test_keys = {world.mechanism_key for world in TEST}
    require(train_keys.isdisjoint(test_keys), "heldout mechanism composition occurs in training")
    train_gain = max(world.gain_hi for world in TRAIN)
    train_noise = max(world.noise for world in TRAIN)
    extrap = BY_NAME["feedback_saturating_extrap"]
    require(extrap.gain_lo > train_gain and extrap.noise > train_noise,
            "extrapolation world lies inside the training hull")
    return {"balanced_training_worlds": dict(sorted(counts.items())),
            "composition_disjoint": True, "extrapolation_outside_hull": True}


def audit_prompts_and_gold(loaded) -> dict:
    for mode in MODES:
        disclosed = loaded[("disclosed", mode)]["train"]
        undisclosed = loaded[("undisclosed", mode)]["train"]
        require(all(MECHANISM_BEGIN in prompt for prompt in disclosed["input"]),
                f"{mode}: disclosed prompt missing description marker")
        for prompt in undisclosed["input"]:
            lower = prompt.lower()
            require(all(token.lower() not in lower for token in FORBIDDEN_BLIND_TEXT),
                    f"{mode}: undisclosed prompt leaks mechanism text")
        for split in ("train", "heldout"):
            for disclosure in DISCLOSURES:
                for prompt in loaded[(disclosure, mode)][split]["input"]:
                    require("<A>" not in prompt and "<J>" not in prompt,
                            f"{disclosure}/{mode}: non-JSON output placeholder remains")
                    require(f"Start your response with {OUTPUT_PREFIX}" in prompt,
                            f"{disclosure}/{mode}: output prefix instruction missing")
                    require(
                        f"After the tenth number, write {OUTPUT_SUFFIX} and stop." in prompt,
                        f"{disclosure}/{mode}: output suffix instruction missing",
                    )
    checked = 0
    for rows in loaded.values():
        for split in ("train", "heldout"):
            for ref in references(rows[split]):
                parsed = parse_forecasts(ref["gold"], ref["scenario_labels"], ref["clip"])
                require(parsed is not None and np.allclose(parsed, ref["targets"], atol=1e-3),
                        f"gold parser round trip failed: {ref['task_id']}")
                checked += 1
    return {"gold_round_trips": checked, "undisclosed_leak_scan": True}


def audit_controls(loaded, expected_train: int) -> dict:
    prior_refs = references(loaded[("disclosed", "population_prior")]["train"])
    prior_responses = {tuple(np.round(item["response_targets"], 6)) for item in prior_refs}
    require(len(prior_responses) == 1,
            "population-prior response varies by episode or mechanism")

    structureless = references(loaded[("undisclosed", "structureless")]["train"])
    for item in structureless:
        require(np.max(np.abs(item["response_targets"])) < 1e-10,
                "structureless target responds to intervention")
    xs, ys = [], []
    for item in structureless:
        x = np.asarray(item["displayed_target_inputs"], dtype=float)[:, 0]
        y = np.asarray(item["target_observed"], dtype=float)
        xs.extend(x.tolist())
        ys.extend(y.tolist())
    correlation = float(np.corrcoef(xs, ys)[0, 1])
    tolerance = 0.03 if expected_train >= 1_000 else 0.30
    require(abs(correlation) < tolerance,
            f"structureless displayed driver/outcome corr {correlation:.4f} >= {tolerance}")
    return {"population_prior_unique_response_vectors": len(prior_responses),
            "structureless_driver_outcome_correlation": correlation}


def audit_blind_identifiability(loaded, per_world: int) -> dict:
    refs = references(loaded[("undisclosed", "causal_family")]["heldout"])
    selected = {}
    for item in refs:
        if item["k"] != 9:
            continue
        selected.setdefault(item["world"], [])
        if len(selected[item["world"]]) < per_world:
            selected[item["world"]].append(item)
    report = {}
    for world_name, items in selected.items():
        require(len(items) == per_world, f"{world_name}: insufficient oracle audit rows")
        ratios, margins = [], []
        world = BY_NAME[world_name]
        for item in items:
            oracle, _ = oracle_forecast(
                world,
                item["calibrations"],
                np.asarray(item["target_inputs"], dtype=float),
                np.asarray(item["target_observed"], dtype=float),
                disclosed=False,
            )
            truth = np.asarray(item["truth_targets"], dtype=float)
            prior = np.asarray(item["prior_targets"], dtype=float)
            oracle_error = float(np.mean(np.abs(response_vector(oracle) - response_vector(truth))))
            prior_error = float(np.mean(np.abs(response_vector(prior) - response_vector(truth))))
            require(prior_error > 0.20, f"{world_name}: blind floor too close to oracle")
            ratios.append(oracle_error / prior_error)
            margins.append(prior_error - oracle_error)
        mean_ratio = float(np.mean(ratios))
        require(mean_ratio < 0.75,
                f"{world_name}: graph-blind oracle/floor ratio {mean_ratio:.3f} is too high")
        report[world_name] = {
            "oracle_to_prior_error_ratio": mean_ratio,
            "oracle_advantage": float(np.mean(margins)),
            "n": len(items),
        }
    require(set(report) == {world.name for world in TEST}, "oracle audit missed heldout blocks")
    return report


def audit(oracle_per_world: int) -> dict:
    build_manifest = json.loads((DATA_ROOT / "build_manifest.json").read_text())
    expected_train = int(build_manifest["n_train_per_world"]) * len(TRAIN)
    expected_eval = int(build_manifest["n_eval_per_world"]) * len(TEST) * 3
    loaded = load_all(expected_train, expected_eval)
    return {
        "protocol": "c3_mechanism_disclosure_v3",
        "models": list(MODEL_ROSTER),
        "disclosures": list(DISCLOSURES),
        "confirmatory_arms": ["causal_family", "population_prior"],
        "seeds": [42, 43, 44],
        "confirmatory_training_runs": 36,
        "diagnostic_structureless_runs": 6,
        "expected_train_rows": expected_train,
        "expected_heldout_rows": expected_eval,
        "pairing": audit_pairing(loaded),
        "catalog": audit_catalog(loaded),
        "prompts": audit_prompts_and_gold(loaded),
        "controls": audit_controls(loaded, expected_train),
        "blind_identifiability": audit_blind_identifiability(loaded, oracle_per_world),
        "dataset_hashes": tree_hashes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-per-world", type=int, default=3)
    parser.add_argument("--write-manifest", type=Path)
    args = parser.parse_args()
    result = audit(args.oracle_per_world)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.write_manifest:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(encoded, encoding="utf-8")
        print(f"PASS -> {args.write_manifest}")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
