#!/usr/bin/env python3
"""Build the Coin City structure-selection training and evaluation datasets.

Protocol `coin_city_structure_selection_v1`.

Why this exists
---------------
The parent experiment (`coin_city_structural_transfer_v1`) trains on two
`direct_a` references differing in *gain* and evaluates, among other cells, on
targets whose *structure* is `mediated_b`. Structure never varies in training,
so "always emit zero at horizon 3" is the optimal training policy, and the
trained checkpoints adopt it unconditionally: measured horizon-3 output is
exactly 0.000 on every cell and every seed, and the cue moves structure choice
by less than 0.002. The structure-shift cells therefore do not measure whether
cue-driven selection transfers; they measure whether a model trained solely on
structure A spontaneously produces structure B.

This dataset makes structure a *selectable* dimension in training, so the
transfer question becomes answerable: train cue-driven structure selection on
Coin City, then ask whether it survives a surface-domain shift to Coin Harbor,
and whether it survives replacing the semantic cue with an arbitrary label
whose mapping changes every episode.

Design
------
Every episode shows two references that differ **only** in structure: one
`direct_a`, one `mediated_b`, with gains set so their horizon-1 responses
coincide exactly. Persistence at horizon 3 is therefore the sole discriminating
feature, and a model cannot select on response magnitude instead. The mediated parameters (rho, phi, lambda) are
resampled every episode, so a fixed memorized "mediated template" does not
solve the task -- the persistence must be read off the named reference.

Verified solvable before building: a one-feature classifier (coefficient on the
lagged outcome) applied to the 14 reference rows separates the structures with
d = 5.2 in Coin City and 3.6 in Coin Harbor, and choosing the right structure is
worth 2.18 response-MAE points averaged over the evaluation set.

Held out from training
----------------------
* **Coin Harbor entirely** -- the surface-domain transfer test.
* **Arbitrary labels entirely** -- the novel-induction test, matching the
  Experiment 2 design that carries the representational claim. Training uses
  only the semantic contexts.
* Evaluation episodes are drawn from a disjoint seed band from training.

Arms
----
`causal`            reward is (the parent's name for episode-matched) the cue-named structure's own forecast.
`population_prior`  reward is the structure-marginal blend (the equal-weight
                    average of the two structures' responses) -- the policy of
                    ignoring the cue and hedging.

Both arms see byte-identical prompts. Only the reward differs, exactly as in
the parent protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PARENT = Path(__file__).resolve().parent.parent / "coin_city_structural"
sys.path.insert(0, str(PARENT))

from prompt import render_prompt  # noqa: E402
from worlds import (  # noqa: E402
    COIN_CITY,
    DOMAINS,
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
PROTOCOL = "coin_city_structure_selection_v1"
# "causal" is the parent protocol's name for the episode-matched arm; kept
# so the parent train.sh (which validates ARM) runs unmodified.
ARMS = ("causal", "population_prior")
STRUCTURES = ("direct_a", "mediated_b")
CUES = ("correct", "none", "misleading")
LABEL_KINDS = ("semantic", "arbitrary")
K_VALUES = (0, 2, 4, 8)
TRAIN_ROWS = 4800
EVAL_PER_CELL = 30
REFERENCE_LENGTH = 14
TARGET_LENGTH = 8
GAIN_BAND = (0.22, 0.92)

# Disjoint from the parent protocol's 81/91-million bands so no episode of this
# experiment can coincide with one of the parent's.
TRAIN_SEED_BASE = 61_000_000
EVAL_SEED_BASE = 71_000_000

# Arbitrary labels carry no information about which structure they denote; the
# mapping is redrawn every episode, so it can only be recovered by reading the
# reference trajectories. Held out of training entirely.
ARBITRARY_SYMBOLS = ("ZETA", "KAPPA")


def _sha(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _gold(targets) -> str:
    return json.dumps({"forecasts": [round(float(v), 3) for v in targets]},
                      separators=(",", ":"))


def _matched_pair(rng: np.random.Generator) -> tuple[dict, dict]:
    """Two parameter sets whose horizon-1 responses coincide exactly.

    A mediated system's horizon-1 response is ``lambda * gain * shock`` while a
    direct one's is ``gain * shock``, so matching the gain *parameter* leaves
    the realised responses differing by lambda. Scaling the direct gain by
    lambda makes the two structures indistinguishable at horizon 1, leaving
    persistence at horizon 3 as the sole discriminating feature. A model cannot
    then select on response magnitude instead of on structure.
    """
    gain = float(rng.uniform(*GAIN_BAND))
    mediated = sample_parameters("mediated_b", rng)
    mediated["gain"] = gain
    direct = sample_parameters("direct_a", rng)
    direct["gain"] = gain * float(mediated["lambda"])
    return direct, mediated


def _labels(rng: np.random.Generator, domain, kind: str) -> dict:
    """Context strings for each structure, and for the target sentence."""
    if kind == "semantic":
        return {
            "reference": {s: context_for(domain, s) for s in STRUCTURES},
            "target": {s: context_for(domain, s, target=True) for s in STRUCTURES},
        }
    symbols = list(ARBITRARY_SYMBOLS)
    rng.shuffle(symbols)
    mapping = dict(zip(STRUCTURES, symbols))
    return {
        "reference": {s: f"This system is of type {mapping[s]}." for s in STRUCTURES},
        "target": {s: f"The target system is of type {mapping[s]}." for s in STRUCTURES},
        "mapping": mapping,
    }


def _episode(domain, seed: int, k: int, target_structure: str, cue: str,
             label_kind: str) -> dict:
    """One episode: two structure-contrasting references plus a cued target."""
    rng = np.random.default_rng(seed)
    parameters = dict(zip(STRUCTURES, _matched_pair(rng)))
    labels = _labels(rng, domain, label_kind)

    order = list(STRUCTURES)
    rng.shuffle(order)
    references = []
    for index, structure in enumerate(order):
        series = simulate_series(
            domain, structure,
            balanced_inputs(domain, REFERENCE_LENGTH, rng),
            parameters[structure], rng)
        references.append({
            **series,
            "structure": structure,
            "parameters": parameters[structure],
            "context": labels["reference"][structure],
        })

    target_series = simulate_series(
        domain, target_structure,
        balanced_inputs(domain, TARGET_LENGTH, rng),
        parameters[target_structure], rng)

    shown = target_structure if cue == "correct" else (
        "mediated_b" if target_structure == "direct_a" else "direct_a")

    history = target_series["inputs"][:k]
    truth = forecast_scenarios(domain, target_structure, history,
                               parameters[target_structure])
    other = "mediated_b" if target_structure == "direct_a" else "direct_a"
    counter = forecast_scenarios(domain, other, history, parameters[other])
    # The structure-marginal policy: ignore the cue, hedge across structures.
    blend = 0.5 * (truth + counter)

    templates = {
        s: response_vector(forecast_scenarios(domain, s, history, parameters[s]))
        for s in STRUCTURES
    }
    return {
        "domain": domain, "seed": seed, "k": k, "cue": cue,
        "label_kind": label_kind, "target_structure": target_structure,
        "shown_structure": shown, "parameters": parameters,
        "references": references, "target_series": target_series,
        "shown_context": labels["target"][shown],
        "truth": truth, "blend": blend, "templates": templates,
        "mapping": labels.get("mapping"),
    }


def _row(ep: dict, arm: str, pair_id: str, task_id: str) -> dict:
    domain = ep["domain"]
    target = {**ep["target_series"], "shown_context": ep["shown_context"]}
    prompt = render_prompt(domain, ep["references"], target, ep["cue"], ep["k"],
                           training=(arm is not None))
    reward = ep["truth"] if arm != "population_prior" else ep["blend"]
    truth = np.asarray(ep["truth"], dtype=float)
    reward = np.asarray(reward, dtype=float)
    persistence = {
        s: float(np.mean(np.abs(np.asarray(ep["templates"][s])[4:])) /
                 max(np.mean(np.abs(np.asarray(ep["templates"][s])[:4])), 1e-9))
        for s in STRUCTURES
    }
    reference = {
        "protocol": PROTOCOL,
        "task_id": task_id,
        "pair_id": pair_id,
        "domain": domain.name,
        "target_structure": ep["target_structure"],
        "shown_structure": ep["shown_structure"],
        "label_kind": ep["label_kind"],
        "cue": ep["cue"],
        "k": int(ep["k"]),
        "mode": arm or "evaluation",
        "targets": reward.round(6).tolist(),
        "truth_targets": truth.round(6).tolist(),
        "prior_targets": np.asarray(ep["blend"], dtype=float).round(6).tolist(),
        "scenario_labels": [r["label"] for r in domain.scenarios],
        "scenario_shocks": [r["shock"] for r in domain.scenarios],
        "scenario_horizons": [r["horizon"] for r in domain.scenarios],
        "shocks": [r["shock"] for r in domain.scenarios],
        "response_pairs": [list(p) for p in response_pairs()],
        "response_targets": response_vector(reward).round(6).tolist(),
        "truth_response": response_vector(truth).round(6).tolist(),
        "prior_response": response_vector(np.asarray(ep["blend"])).round(6).tolist(),
        "direct_template_response": np.asarray(ep["templates"]["direct_a"]).round(6).tolist(),
        "mediated_template_response": np.asarray(ep["templates"]["mediated_b"]).round(6).tolist(),
        "clip": list(domain.clip),
        "reward_scale": float((domain.clip[1] - domain.clip[0]) * 0.10),
        "response_scale": float(domain.shock_unit * 0.50),
        "reference_structures": [r["structure"] for r in ep["references"]],
        "cued_persistence_ratio": persistence[ep["shown_structure"]],
        "true_persistence_ratio": persistence[ep["target_structure"]],
        "gain": float(ep["parameters"]["direct_a"]["gain"]),
        "mediated_parameters": {k: float(v) for k, v in
                                ep["parameters"]["mediated_b"].items()},
        "arbitrary_mapping": ep["mapping"],
        "target_inputs": ep["target_series"]["inputs"][:ep["k"]],
        "target_observed": ep["target_series"]["observed"][:ep["k"]],
        "numeric_hash": _sha({
            "refs": [{"i": r["inputs"], "o": r["observed"]} for r in ep["references"]],
            "tgt": {"i": ep["target_series"]["inputs"][:ep["k"]],
                    "o": ep["target_series"]["observed"][:ep["k"]]},
        }),
        "gold": _gold(reward),
    }
    return {"input": prompt, "reference": json.dumps(reference, sort_keys=True)}


def build_training() -> dict[str, list[dict]]:
    """Coin City only, semantic labels only, cue always correct."""
    out = {arm: [] for arm in ARMS}
    for index in range(TRAIN_ROWS):
        seed = TRAIN_SEED_BASE + index
        k = K_VALUES[index % len(K_VALUES)]
        structure = STRUCTURES[(index // len(K_VALUES)) % len(STRUCTURES)]
        ep = _episode(COIN_CITY, seed, k, structure, "correct", "semantic")
        pair_id = f"train:{index}:k{k}"
        for arm in ARMS:
            out[arm].append(_row(ep, arm, pair_id, pair_id))
    return out


def build_evaluation() -> list[dict]:
    """Both domains, both structures, three cues, two label kinds, four depths."""
    rows = []
    for domain_name in ("coin_city", "coin_harbor"):
        domain = DOMAINS[domain_name]
        for label_kind in LABEL_KINDS:
            for structure in STRUCTURES:
                for replicate in range(EVAL_PER_CELL):
                    base = (EVAL_SEED_BASE
                            + 5_000_000 * ("coin_harbor" == domain_name)
                            + 2_000_000 * LABEL_KINDS.index(label_kind)
                            + 500_000 * STRUCTURES.index(structure)
                            + replicate)
                    for k in K_VALUES:
                        pair_id = (f"eval:{domain_name}:{label_kind}:{structure}"
                                   f":r{replicate}:k{k}")
                        for cue in CUES:
                            ep = _episode(domain, base, k, structure, cue, label_kind)
                            rows.append(_row(ep, None, pair_id, f"{pair_id}:cue-{cue}"))
    return rows


def write(destination: Path, rows: list[dict]) -> None:
    """Write a HuggingFace DatasetDict, the format the trainer loads.

    `run_mechanism_rl.py` calls `datasets.load_from_disk`, which requires an
    Arrow directory -- a raw .jsonl here made every job die with
    FileNotFoundError ~80s in and then hang until its wall limit. The parent
    protocol's builder uses exactly this call; matching it is not optional.
    """
    from datasets import Dataset, DatasetDict
    destination.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": Dataset.from_list(rows)}).save_to_disk(str(destination))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DATA)
    args = parser.parse_args()

    training = build_training()
    evaluation = build_evaluation()
    manifest = {"protocol": PROTOCOL, "arms": list(ARMS),
                "train_rows": {a: len(r) for a, r in training.items()},
                "eval_rows": len(evaluation),
                "held_out": ["coin_harbor", "arbitrary_labels"],
                "sha256": {}}
    for arm, rows in training.items():
        for split in ("train", "heldout"):
            payload = rows if split == "train" else evaluation
            destination = args.out / arm / split
            write(destination, payload)
            manifest["sha256"][f"{arm}/{split}"] = _sha(
                [r["reference"] for r in payload])
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()
