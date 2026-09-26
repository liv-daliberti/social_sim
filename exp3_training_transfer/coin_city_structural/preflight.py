#!/usr/bin/env python3
"""Fail-closed validation and manifest writer for Coin City structural transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk

from worlds import DOMAINS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PROTOCOL = ROOT / "protocol" / "coin_city_structural_manifest.json"
MODES = ("causal", "population_prior", "structureless")
K_VALUES = (0, 2, 4, 8)
EXPECTED_TRAIN = 4800
EXPECTED_EVAL = 1440


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(mode: str, split: str) -> list[dict]:
    return list(load_from_disk(str(DATA / mode / split))["train"])


def references(items: list[dict]) -> list[dict]:
    return [json.loads(item["reference"]) for item in items]


def target_key(values) -> tuple[float, ...]:
    return tuple(round(float(value), 6) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", type=Path, default=PROTOCOL)
    args = parser.parse_args()

    build = json.loads((DATA / "build_manifest.json").read_text(encoding="utf-8"))
    train = {mode: rows(mode, "train") for mode in MODES}
    heldout = {mode: rows(mode, "heldout") for mode in MODES}
    train_refs = {mode: references(items) for mode, items in train.items()}
    eval_refs = references(heldout["causal"])

    checks: dict[str, bool] = {}
    checks["locked_build_counts"] = (
        build.get("train_rows") == EXPECTED_TRAIN
        and build.get("heldout_rows") == EXPECTED_EVAL
        and build.get("eval_per_domain_structure_cue_k") == 30
    )
    checks["all_dataset_counts"] = all(
        len(train[mode]) == EXPECTED_TRAIN and len(heldout[mode]) == EXPECTED_EVAL
        for mode in MODES
    )
    checks["causal_prior_prompts_byte_identical"] = (
        [row["input"] for row in train["causal"]]
        == [row["input"] for row in train["population_prior"]]
    )
    checks["heldout_byte_identical_across_training_arms"] = all(
        heldout[mode] == heldout["causal"] for mode in MODES[1:]
    )
    checks["training_contains_only_structure_a"] = all(
        ref["target_structure"] == "direct_a"
        and ref["reference_structures"] == ["direct_a", "direct_a"]
        and not ref["training_structure_b_present"]
        for mode in MODES for ref in train_refs[mode]
    )
    checks["structure_b_language_absent_from_training"] = all(
        all(term not in row["input"].lower() for term in (
            "build gradually", "persist after", "mediator", "rho", "lambda",
            "structure a", "structure b", "mediated_b",
        )) for row in train["causal"]
    )

    # The population-prior arm is an exact within-k derangement of causal reward targets.
    marginal_ok = True
    for k in K_VALUES:
        causal_targets = Counter(
            target_key(ref["truth_targets"]) for ref in train_refs["causal"] if ref["k"] == k
        )
        prior_targets = Counter(
            target_key(ref["targets"])
            for ref in train_refs["population_prior"] if ref["k"] == k
        )
        marginal_ok &= causal_targets == prior_targets
    checks["population_prior_exact_target_marginal"] = bool(marginal_ok)
    checks["population_prior_no_fixed_reward_links"] = all(
        ref["reward_source_task_id"] != ref["task_id"]
        for ref in train_refs["population_prior"]
    )
    checks["causal_rewards_are_episode_truth"] = all(
        target_key(ref["targets"]) == target_key(ref["truth_targets"])
        for ref in train_refs["causal"]
    )

    factorial = Counter(
        (ref["domain"], ref["target_structure"], ref["cue"], int(ref["k"]))
        for ref in eval_refs
    )
    expected_cells = {
        (domain, structure, cue, k)
        for domain in DOMAINS for structure in ("direct_a", "mediated_b")
        for cue in ("correct", "none", "misleading") for k in K_VALUES
    }
    checks["complete_balanced_2x2x3x4_evaluation"] = (
        set(factorial) == expected_cells and set(factorial.values()) == {30}
    )
    paired = defaultdict(list)
    for ref in eval_refs:
        paired[ref["pair_id"]].append(ref)
    checks["cue_arms_share_every_numeric_episode"] = all(
        len(group) == 3
        and {ref["cue"] for ref in group} == {"correct", "none", "misleading"}
        and len({ref["numeric_hash"] for ref in group}) == 1
        and len({target_key(ref["truth_targets"]) for ref in group}) == 1
        for group in paired.values()
    )
    checks["evaluation_references_show_a_and_b"] = all(
        set(ref["reference_structures"]) == {"direct_a", "mediated_b"}
        for ref in eval_refs
    )
    checks["equations_and_internal_labels_hidden"] = all(
        all(term not in row["input"].lower() for term in (
            "m_{t+1}", "p_{t+1}", "rho=", "lambda=", "phi(", "direct_a",
            "mediated_b", "structure a", "structure b",
        )) for row in heldout["causal"]
    )
    checks["disjoint_train_eval_identifiers"] = (
        not ({ref["task_id"] for ref in train_refs["causal"]}
             & {ref["task_id"] for ref in eval_refs})
    )

    # The two candidate pathways must make behaviorally separated forecasts, and the
    # correct answer must improve materially on a blind 50/50 pathway mixture.
    structure_gaps = defaultdict(list)
    prior_errors = defaultdict(list)
    for ref in eval_refs:
        if ref["cue"] != "correct":
            continue
        direct = np.asarray(ref["direct_template_response"], dtype=float)
        mediated = np.asarray(ref["mediated_template_response"], dtype=float)
        truth = np.asarray(ref["truth_response"], dtype=float)
        prior = np.asarray(ref["prior_response"], dtype=float)
        structure_gaps[ref["domain"]].append(float(np.mean(np.abs(direct - mediated))))
        prior_errors[ref["domain"]].append(float(np.mean(np.abs(prior - truth))))
    mean_structure_gap = {key: float(np.mean(values)) for key, values in structure_gaps.items()}
    mean_prior_error = {key: float(np.mean(values)) for key, values in prior_errors.items()}
    checks["a_b_behaviorally_separated"] = all(value > 0.75 for value in mean_structure_gap.values())
    checks["oracle_beats_blind_pathway_mixture"] = all(value > 0.35 for value in mean_prior_error.values())

    # Canonical gold responses exercise the same strict ten-number output contract.
    gold_ok = True
    for ref in train_refs["causal"][:64] + eval_refs[:128]:
        try:
            payload = json.loads(ref["gold"])
            values = payload["forecasts"]
            gold_ok &= set(payload) == {"forecasts"} and len(values) == 10
            gold_ok &= all(np.isfinite(float(value)) for value in values)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            gold_ok = False
    checks["gold_round_trips_strict_contract"] = bool(gold_ok)

    source_paths = {
        "worlds.py": ROOT / "worlds.py",
        "prompt.py": ROOT / "prompt.py",
        "make_dataset.py": ROOT / "make_dataset.py",
        "preflight.py": ROOT / "preflight.py",
    }
    dataset_paths = sorted(DATA.glob("*/train/train/*.arrow")) + sorted(
        DATA.glob("*/heldout/train/*.arrow")
    ) + [DATA / "build_manifest.json"]
    manifest = {
        "protocol": build["protocol"],
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "registered_design": {
            "models": ["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-8B",
                       "meta-llama/Llama-3.1-8B-Instruct"],
            "confirmatory_arms": ["causal", "population_prior"],
            "training_seeds": [42, 43, 44],
            "confirmatory_training_runs": 18,
            "diagnostic_structureless_runs": 3,
            "base_evaluations": 3,
            "train_rows_per_run": EXPECTED_TRAIN,
            "heldout_rows_per_checkpoint": EXPECTED_EVAL,
            "stochastic_draws": 5,
            "primary_cell": {"domain": "coin_harbor", "target_structure": "mediated_b",
                             "cue": "correct", "k": "equal-weight over 0,2,4,8"},
        },
        "identifiability": {
            "mean_a_b_response_gap": mean_structure_gap,
            "mean_blind_pathway_response_mae": mean_prior_error,
        },
        "source_sha256": {name: sha256(path) for name, path in source_paths.items()},
        "dataset_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in dataset_paths},
    }
    args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.write_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("Coin City structural preflight failed: " + ", ".join(failed))
    print(f"Coin City structural preflight passed: {len(checks)} checks")
    print(f"manifest -> {args.write_manifest}")


if __name__ == "__main__":
    main()
