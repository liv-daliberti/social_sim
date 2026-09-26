#!/usr/bin/env python3
"""Fail-closed audit for the registered Experiment 3A datasets.

The three training arms must be episode-paired and all arms must be evaluated on
the same real held-out worlds. Older family/control datasets predate the
provenance-only ``mode`` field, so held-out references are compared after
normalizing that one field. Every scored field must remain identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_from_disk

from catalog import TEST, TRAIN


ROOT = Path(__file__).resolve().parent
ARMS = {
    "family": ROOT / "data" / "rl_multishock",
    "structureless": ROOT / "data" / "rl_multishock_control",
    "prior_mean": ROOT / "data" / "rl_multishock_priormean",
}
EXPECTED_TRAIN_ROWS = 4_800
EXPECTED_HELDOUT_ROWS = 1_440
PROTOCOL = {
    "version": "exp3a_registered_v1",
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "optimizer": "Dr. GRPO",
    "train_steps": 300,
    "train_batch_size": 16,
    "rollouts_per_prompt": 8,
    "temperature": 1.3,
    "learning_rate": 1e-6,
    "lora_rank": 32,
    "lora_alpha": 64,
    "reward_scale": 15.0,
    "slope_weight": 0.5,
    "slope_scale": 0.5,
    "eval_every_steps": 25,
    "eval_temperature": 0.0,
    "seeds": [42, 43, 44],
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _references(dataset: Any) -> list[dict[str, Any]]:
    return [json.loads(value) for value in dataset["reference"]]


def _identity(ref: dict[str, Any]) -> tuple[Any, ...]:
    return (
        ref["structure"],
        int(ref["seed"]),
        int(ref["k"]),
        int(ref["horizon"]),
        tuple(float(x) for x in ref["shocks"]),
        float(ref["g"]),
    )


def _normalized_eval_reference(value: str) -> str:
    ref = json.loads(value)
    # "mode" was added after the completed family/control datasets were built.
    # It is provenance only; the reward and reports do not read it.
    ref.pop("mode", None)
    return json.dumps(ref, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hashes(path: Path) -> dict[str, str]:
    return {
        str(file.relative_to(ROOT)): _sha256(file)
        for file in sorted(path.rglob("*"))
        if file.is_file() and not file.name.startswith("cache-")
    }


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def audit() -> dict[str, Any]:
    datasets: dict[str, dict[str, Any]] = {}
    refs: dict[str, list[dict[str, Any]]] = {}
    for arm, path in ARMS.items():
        _require(path.is_dir(), f"missing {arm} dataset: {path}")
        train = load_from_disk(str(path / "train"))["train"]
        heldout = load_from_disk(str(path / "heldout"))["train"]
        _require(
            len(train) == EXPECTED_TRAIN_ROWS,
            f"{arm}: expected 4800 train rows, got {len(train)}",
        )
        _require(
            len(heldout) == EXPECTED_HELDOUT_ROWS,
            f"{arm}: expected 1440 held-out rows, got {len(heldout)}",
        )
        _require(
            set(train.column_names) == {"input", "reference"},
            f"{arm}: unexpected train schema",
        )
        _require(
            set(heldout.column_names) == {"input", "reference"},
            f"{arm}: unexpected held-out schema",
        )
        datasets[arm] = {"train": train, "heldout": heldout}
        refs[arm] = _references(train)

    family_ids = [_identity(ref) for ref in refs["family"]]
    _require(
        len(set(family_ids)) == EXPECTED_TRAIN_ROWS,
        "family train identities are not unique",
    )
    for arm in ("structureless", "prior_mean"):
        _require(
            [_identity(ref) for ref in refs[arm]] == family_ids,
            f"{arm}: episodes are not paired with family",
        )

    _require(
        datasets["family"]["train"]["input"]
        == datasets["prior_mean"]["train"]["input"],
        "prior-mean prompts differ from family prompts",
    )

    for ref in refs["family"]:
        _require(not bool(ref["control"]), "family row marked as control")
        _require(
            ref.get("mode", "family") == "family",
            "family row has the wrong mode",
        )
    for ref in refs["structureless"]:
        _require(bool(ref["control"]), "structureless row not marked as control")
        _require(
            ref.get("mode", "structureless") == "structureless",
            "structureless row has the wrong mode",
        )
        _require(
            abs(float(ref["slope_target"])) < 1e-12,
            "structureless row has a nonzero slope target",
        )
        _require(
            len(set(float(x) for x in ref["targets"])) == 1,
            "structureless row reacts to a scenario shock",
        )
    for ref in refs["prior_mean"]:
        _require(bool(ref["control"]), "prior-mean row not marked as control")
        _require(
            ref.get("mode") == "prior_mean",
            "prior-mean row has the wrong mode",
        )

    train_structures = {ref["structure"] for ref in refs["family"]}
    _require(
        train_structures == {struct.name for struct in TRAIN},
        "training structure roster drifted",
    )
    _require(
        Counter(ref["structure"] for ref in refs["family"])
        == Counter({struct.name: 600 for struct in TRAIN}),
        "training structures are not balanced at 600 episodes each",
    )

    canonical_eval = datasets["family"]["heldout"]
    canonical_inputs = canonical_eval["input"]
    canonical_refs = [
        _normalized_eval_reference(value) for value in canonical_eval["reference"]
    ]
    eval_refs = [json.loads(value) for value in canonical_eval["reference"]]
    for arm, arm_data in datasets.items():
        _require(
            arm_data["heldout"]["input"] == canonical_inputs,
            f"{arm}: held-out prompts drifted",
        )
        normalized = [
            _normalized_eval_reference(value)
            for value in arm_data["heldout"]["reference"]
        ]
        _require(
            normalized == canonical_refs,
            f"{arm}: held-out scored references drifted",
        )
    _require(
        {ref["structure"] for ref in eval_refs}
        == {struct.name for struct in TEST},
        "held-out structure roster drifted",
    )
    _require(
        not (
            {ref["seed"] for ref in eval_refs}
            & {ref["seed"] for ref in refs["family"]}
        ),
        "train and held-out episode seeds overlap",
    )

    recover_family = [ref for ref in refs["family"] if ref["recover"]]
    recover_prior = [ref for ref in refs["prior_mean"] if ref["recover"]]
    family_corr = _correlation(
        [float(ref["g"]) for ref in recover_family],
        [float(ref["slope_target"]) for ref in recover_family],
    )
    prior_corr = _correlation(
        [float(ref["g"]) for ref in recover_prior],
        [float(ref["slope_target"]) for ref in recover_prior],
    )

    return {
        "status": "pass",
        "protocol": PROTOCOL,
        "arms": {
            arm: {
                "train_rows": len(data["train"]),
                "heldout_rows": len(data["heldout"]),
                "dataset_hashes": _tree_hashes(ARMS[arm]),
            }
            for arm, data in datasets.items()
        },
        "audits": {
            "paired_training_episodes": True,
            "family_prior_mean_prompts_identical": True,
            "normalized_heldout_data_identical": True,
            "train_heldout_seed_overlap": 0,
            "family_gain_slope_correlation": family_corr,
            "prior_mean_gain_slope_correlation": prior_corr,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", type=Path)
    args = parser.parse_args()
    result = audit()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.write_manifest:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
