#!/usr/bin/env python3
"""Build the locked Coin City A-to-B training and paired evaluation datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from prompt import render_prompt
from worlds import (
    COIN_CITY,
    DOMAINS,
    STRUCTURES,
    balanced_inputs,
    context_for,
    forecast_scenarios,
    response_pairs,
    response_vector,
    sample_parameters,
    simulate_series,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PROTOCOL = "coin_city_structural_transfer_v1"
MODES = ("causal", "population_prior", "structureless")
K_VALUES = (0, 2, 4, 8)
TRAIN_ROWS = 4800
EVAL_PER_CELL = 30
REFERENCE_LENGTH = 14
TARGET_LENGTH = 8
TRAIN_SEED_BASE = 81_000_000
EVAL_SEED_BASE = 91_000_000

TRAIN_DIRECT_CONTEXTS = {
    "high": (
        "Residents receive campaign developments through immediate national alerts and "
        "unedited feeds, and historically react strongly within the same week."
    ),
    "low": (
        "Residents receive campaign developments through local civic briefings that "
        "filter outside news, and historically react weakly within the same week."
    ),
}


def _sha(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _series(domain, structure: str, parameters: dict, seed: int, length: int) -> dict:
    rng = np.random.default_rng(seed)
    inputs = balanced_inputs(domain, length, rng)
    return simulate_series(domain, structure, inputs, parameters, rng)


def _numeric_hash(domain, references: list[dict], target: dict, k: int) -> str:
    return _sha({
        "domain": domain.name,
        "references": [{"inputs": row["inputs"], "observed": row["observed"]}
                       for row in references],
        "target_inputs": target["inputs"][:k],
        "target_observed": target["observed"][:k],
        "scenarios": list(domain.scenarios),
    })


def _gold(targets) -> str:
    values = [round(float(value), 3) for value in targets]
    return json.dumps({"forecasts": values}, separators=(",", ":"))


def _reference(task_id: str, *, domain, structure: str, cue: str, k: int,
               references: list[dict], target: dict, targets, prior_targets,
               mode: str, reward_source_task_id: str | None = None) -> dict:
    truth_targets = np.asarray(target["truth_targets"], dtype=float)
    reward_targets = np.asarray(targets, dtype=float)
    return {
        "protocol": PROTOCOL,
        "task_id": task_id,
        "pair_id": target["pair_id"],
        "domain": domain.name,
        "target_structure": structure,
        "cue": cue,
        "k": int(k),
        "mode": mode,
        "targets": reward_targets.round(6).tolist(),
        "truth_targets": truth_targets.round(6).tolist(),
        "prior_targets": np.asarray(prior_targets, dtype=float).round(6).tolist(),
        "scenario_labels": [row["label"] for row in domain.scenarios],
        "scenario_shocks": [row["shock"] for row in domain.scenarios],
        "scenario_horizons": [row["horizon"] for row in domain.scenarios],
        "shocks": [row["shock"] for row in domain.scenarios],
        "response_pairs": [list(pair) for pair in response_pairs()],
        "response_targets": response_vector(reward_targets).round(6).tolist(),
        "truth_response": response_vector(truth_targets).round(6).tolist(),
        "prior_response": response_vector(prior_targets).round(6).tolist(),
        "direct_template_response": np.asarray(target["direct_template_response"]).round(6).tolist(),
        "mediated_template_response": np.asarray(target["mediated_template_response"]).round(6).tolist(),
        "clip": list(domain.clip),
        "reward_scale": float((domain.clip[1] - domain.clip[0]) * 0.10),
        "response_scale": float(domain.shock_unit * 0.50),
        "numeric_hash": _numeric_hash(domain, references, target, k),
        "reference_structures": [row["structure"] for row in references],
        "training_structure_b_present": any(row["structure"] == "mediated_b" for row in references),
        "target_inputs": target["inputs"][:k],
        "target_observed": target["observed"][:k],
        "reward_source_task_id": reward_source_task_id or task_id,
        "gold": _gold(reward_targets),
    }


def _training_base(index: int) -> dict:
    seed = TRAIN_SEED_BASE + index
    rng = np.random.default_rng(seed)
    k = K_VALUES[index % len(K_VALUES)]
    low = sample_parameters("direct_a", rng, gain_band=(0.18, 0.46))
    high = sample_parameters("direct_a", rng, gain_band=(0.68, 0.98))
    reference_specs = [("low", low), ("high", high)]
    rng.shuffle(reference_specs)
    references = []
    for ref_index, (band, parameters) in enumerate(reference_specs):
        series = _series(COIN_CITY, "direct_a", parameters,
                         seed + 100_000 * (ref_index + 1), REFERENCE_LENGTH)
        references.append({
            **series,
            "structure": "direct_a",
            "parameters": parameters,
            "context": TRAIN_DIRECT_CONTEXTS[band],
            "band": band,
        })
    selected_band = "high" if index % 2 else "low"
    selected = next(row for row in references if row["band"] == selected_band)
    target_series = _series(COIN_CITY, "direct_a", selected["parameters"], seed, TARGET_LENGTH)
    truth = forecast_scenarios(
        COIN_CITY, "direct_a", target_series["inputs"][:k], selected["parameters"]
    )
    mean_parameters = {"gain": 0.58, "rho": 0.0, "phi": 0.0, "lambda": 1.0}
    prior = forecast_scenarios(
        COIN_CITY, "direct_a", target_series["inputs"][:k], mean_parameters
    )
    target = {
        **target_series,
        "structure": "direct_a",
        "parameters": selected["parameters"],
        "shown_context": TRAIN_DIRECT_CONTEXTS[selected_band],
        "truth_targets": truth.tolist(),
        "pair_id": f"train:{index}:k{k}",
        "direct_template_response": response_vector(truth).tolist(),
        "mediated_template_response": response_vector(truth).tolist(),
    }
    task_id = target["pair_id"]
    prompt = render_prompt(COIN_CITY, references, target, "correct", k, training=True)
    reference = _reference(
        task_id, domain=COIN_CITY, structure="direct_a", cue="correct", k=k,
        references=references, target=target, targets=truth, prior_targets=prior,
        mode="causal",
    )
    return {"input": prompt, "reference": reference, "k": k}


def build_training() -> dict[str, list[dict]]:
    bases = [_training_base(index) for index in range(TRAIN_ROWS)]
    causal = []
    for row in bases:
        causal.append({"input": row["input"],
                       "reference": json.dumps(row["reference"], sort_keys=True)})

    # Exact within-k derangement: prompt inputs and the complete reward-target marginal are
    # unchanged, but no population-control row keeps its own episode target.
    prior_rows = [None] * len(bases)
    rng = np.random.default_rng(0xC01AC17)
    for k in K_VALUES:
        indices = np.asarray([i for i, row in enumerate(bases) if row["k"] == k])
        shift = int(rng.integers(1, len(indices)))
        sources = np.roll(indices, shift)
        for destination, source in zip(indices, sources):
            row = bases[int(destination)]
            source_ref = bases[int(source)]["reference"]
            ref = deepcopy(row["reference"])
            ref["mode"] = "population_prior"
            ref["targets"] = source_ref["truth_targets"]
            ref["response_targets"] = source_ref["truth_response"]
            ref["reward_source_task_id"] = source_ref["task_id"]
            ref["gold"] = _gold(ref["targets"])
            prior_rows[int(destination)] = {
                "input": row["input"], "reference": json.dumps(ref, sort_keys=True)
            }

    structureless = []
    for index, row in enumerate(bases):
        ref = deepcopy(row["reference"])
        ref["mode"] = "structureless"
        baseline = COIN_CITY.baseline
        ref["targets"] = [baseline] * 10
        ref["response_targets"] = [0.0] * 8
        ref["gold"] = _gold(ref["targets"])
        # Retain the textual shell but deterministically scramble every displayed driver token.
        # This is diagnostic only and is not part of the causal-vs-prior contrast.
        prompt = row["input"]
        for old, new in (("-12.0", "901.0"), ("12.0", "-12.0"), ("901.0", "12.0"),
                         ("-8.0", "902.0"), ("8.0", "-8.0"), ("902.0", "8.0")):
            prompt = prompt.replace(old, new)
        structureless.append({"input": prompt, "reference": json.dumps(ref, sort_keys=True)})

    return {"causal": causal, "population_prior": prior_rows,
            "structureless": structureless}


def _evaluation_core(domain, replicate: int) -> dict:
    seed = EVAL_SEED_BASE + (0 if domain.name == "coin_city" else 1_000_000) + replicate
    rng = np.random.default_rng(seed)
    direct_parameters = sample_parameters("direct_a", rng)
    mediated_parameters = sample_parameters("mediated_b", rng)
    specifications = [("direct_a", direct_parameters), ("mediated_b", mediated_parameters)]
    rng.shuffle(specifications)
    references = []
    for ref_index, (structure, parameters) in enumerate(specifications):
        series = _series(domain, structure, parameters,
                         seed + 100_000 * (ref_index + 1), REFERENCE_LENGTH)
        references.append({
            **series,
            "structure": structure,
            "parameters": parameters,
            "context": context_for(domain, structure),
        })
    return {
        "seed": seed,
        "references": references,
        "parameters": {
            "direct_a": direct_parameters,
            "mediated_b": mediated_parameters,
        },
    }


def _evaluation_rows(domain, structure: str, replicate: int) -> list[dict]:
    core = _evaluation_core(domain, replicate)
    parameters = core["parameters"][structure]
    target_series = _series(domain, structure, parameters, core["seed"] + 7_000_000,
                            TARGET_LENGTH)
    rows = []
    for k in K_VALUES:
        truth = forecast_scenarios(domain, structure, target_series["inputs"][:k], parameters)
        direct = forecast_scenarios(
            domain, "direct_a", target_series["inputs"][:k], core["parameters"]["direct_a"]
        )
        mediated = forecast_scenarios(
            domain, "mediated_b", target_series["inputs"][:k],
            core["parameters"]["mediated_b"]
        )
        prior = (direct + mediated) / 2.0
        pair_id = f"eval:{domain.name}:{structure}:r{replicate}:k{k}"
        for cue in ("correct", "none", "misleading"):
            shown_structure = structure if cue == "correct" else (
                "mediated_b" if structure == "direct_a" else "direct_a"
            )
            target = {
                **target_series,
                "structure": structure,
                "parameters": parameters,
                "shown_context": context_for(domain, shown_structure, target=True),
                "truth_targets": truth.tolist(),
                "pair_id": pair_id,
                "direct_template_response": response_vector(direct).tolist(),
                "mediated_template_response": response_vector(mediated).tolist(),
            }
            task_id = f"{pair_id}:cue-{cue}"
            prompt = render_prompt(domain, core["references"], target, cue, k, training=False)
            ref = _reference(
                task_id, domain=domain, structure=structure, cue=cue, k=k,
                references=core["references"], target=target, targets=truth,
                prior_targets=prior, mode="evaluation",
            )
            rows.append({"input": prompt, "reference": json.dumps(ref, sort_keys=True)})
    return rows


def build_evaluation(n_per_cell: int) -> list[dict]:
    rows = []
    for domain in DOMAINS.values():
        for structure in STRUCTURES:
            for replicate in range(n_per_cell):
                rows.extend(_evaluation_rows(domain, structure, replicate))
    return rows


def write_datasets(train_rows: int, n_eval_per_cell: int) -> dict:
    if train_rows != TRAIN_ROWS:
        raise ValueError(f"locked protocol requires exactly {TRAIN_ROWS} training rows")
    from datasets import Dataset, DatasetDict

    training = build_training()
    evaluation = build_evaluation(n_eval_per_cell)
    DATA.mkdir(parents=True, exist_ok=True)
    records = []
    for mode in MODES:
        destination = DATA / mode
        DatasetDict({"train": Dataset.from_list(training[mode])}).save_to_disk(
            str(destination / "train")
        )
        DatasetDict({"train": Dataset.from_list(evaluation)}).save_to_disk(
            str(destination / "heldout")
        )
        records.append({"mode": mode, "path": str(destination.relative_to(ROOT)),
                        "train_rows": len(training[mode]), "heldout_rows": len(evaluation)})
    manifest = {
        "protocol": PROTOCOL,
        "train_rows": train_rows,
        "eval_per_domain_structure_cue_k": n_eval_per_cell,
        "heldout_rows": len(evaluation),
        "k_values": list(K_VALUES),
        "train_structures": ["direct_a"],
        "eval_structures": list(STRUCTURES),
        "domains": list(DOMAINS),
        "reference_length": REFERENCE_LENGTH,
        "target_length": TARGET_LENGTH,
        "datasets": records,
    }
    (DATA / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rows", type=int, default=TRAIN_ROWS)
    parser.add_argument("--eval-per-cell", type=int, default=EVAL_PER_CELL)
    args = parser.parse_args()
    if args.eval_per_cell != EVAL_PER_CELL:
        raise SystemExit(f"locked protocol requires --eval-per-cell {EVAL_PER_CELL}")
    manifest = write_datasets(args.train_rows, args.eval_per_cell)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
