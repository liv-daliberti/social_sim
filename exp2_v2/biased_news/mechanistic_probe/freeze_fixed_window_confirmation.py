#!/usr/bin/env python3
"""Freeze a fixed-window patch confirmation on fresh episodes.

The registered causal runs choose their window from development decoding. This
run chooses nothing: every window is fixed here, before any fresh-episode
prompt reaches a model, so the sealed test has no selection step to overfit.

Episodes are new draws from the unchanged generator under a seed base no other
design uses, so none of them was seen by any earlier development or test split.
Windows are fixed from the Llama-3.1-70B development sweep (flat from windows
11 to 22, half-sized at 28, gone at 35) and, for Qwen3-14B, from the two windows
its registered analyses already produced. The mid window for Llama is a central
plateau window rather than the sweep's maximum, so no noisy argmax enters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import engine.coin_city_stable_relationship_claude_n250 as design  # noqa: E402
from build_arbitrary_symbol_probe_tasks import (  # noqa: E402
    CELL_ORDER,
    factorial_cell,
    make_records,
)

STUDY = "fixed_window_confirmation_v1"
RUN_DIR = ROOT / "data" / design.EXPERIMENT / "mechanistic_probe" / STUDY
FRESH_SEED_BASE = 960_000
TEST_PER_CELL = 11
PRIMARY_DEPTH = 0
PERMUTATION_SEED = 20260825
SIGN_FLIP_DRAWS = 200_000

MODELS = {
    "qwen3_14b": {
        "model_id": "Qwen/Qwen3-14B",
        "model_commit": "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "transformer_layers": 40,
        "mid_window": [19, 20, 21],
        "late_window": [33, 34, 35],
        "window_source": (
            "mid: the window the 160-episode development set selected for the "
            "registered v2 test; late: the window the 64-episode development "
            "set selected for the registered v1 test"
        ),
    },
    "llama3_1_70b": {
        "model_id": "meta-llama/Llama-3.1-70B-Instruct",
        "model_commit": "1605565b47bb9346c5515c34102e054115b4f98b",
        "transformer_layers": 80,
        "mid_window": [19, 20, 21],
        "late_window": [35, 36, 37],
        "window_source": (
            "mid: a central window of the development sweep's plateau (start "
            "11, 15, 19, 22 give .78, .78, .77, .79), not its maximum; late: "
            "the registered v1 window, where the sweep gives .02"
        ),
    },
}

CONDITIONS = (
    "unpatched",
    "self_mid_window",
    "cross_mid_window",
    "cross_late_window",
    "cross_embedding_only",
)

HYPOTHESES = {
    "H1_mid_positive": (
        "At k=0, the mean symmetric shift d_e under cross_mid_window is "
        "greater than zero."
    ),
    "H2_mid_exceeds_late": (
        "At k=0, the mean of the per-episode difference d_e(cross_mid_window) "
        "- d_e(cross_late_window) is greater than zero."
    ),
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fresh_episodes() -> list[dict]:
    original = design.SEED_BASE
    design.SEED_BASE = FRESH_SEED_BASE
    try:
        pool = [design.make_episode(index) for index in range(design.EPISODES)]
    finally:
        design.SEED_BASE = original
    by_cell = {cell: [] for cell in CELL_ORDER}
    for episode in pool:
        by_cell[factorial_cell(episode)].append(episode)
    selected = []
    for cell in CELL_ORDER:
        chosen = by_cell[cell][:TEST_PER_CELL]
        if len(chosen) != TEST_PER_CELL:
            raise ValueError(f"cell {cell} has only {len(chosen)} fresh episodes")
        selected.extend(chosen)
    selected.sort(key=lambda row: int(row["episode"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    args = parser.parse_args()
    protocol_path = args.run_dir / "protocol.json"
    if protocol_path.exists():
        raise FileExistsError(f"{protocol_path} is frozen; refusing to overwrite")

    episodes = fresh_episodes()
    records = make_records(episodes, {int(e["episode"]): ("test", None) for e in episodes})
    for row in records:
        row["sample_id"] = f"{STUDY}__{row['sample_id']}"
        row["fresh_seed"] = FRESH_SEED_BASE + int(row["episode"])
    cells = Counter("{}:{}:{}".format(*factorial_cell(e)) for e in episodes)
    if set(cells.values()) != {TEST_PER_CELL} or len(cells) != len(CELL_ORDER):
        raise ValueError(f"factorial balance failed: {dict(cells)}")

    args.run_dir.mkdir(parents=True, exist_ok=False)
    episodes_path = args.run_dir / "episodes.jsonl"
    tasks_path = args.run_dir / "tasks.jsonl"
    episodes_path.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in episodes))
    tasks_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))

    code = {
        name: file_sha256(HERE / name)
        for name in (
            "freeze_fixed_window_confirmation.py",
            "patch_fixed_window_confirmation.py",
            "analyze_fixed_window_confirmation.py",
            "patch_symbol_label_activations.py",
            "symbol_relational_common.py",
            "build_arbitrary_symbol_probe_tasks.py",
        )
    }
    code["engine/coin_city_stable_relationship_claude_n250.py"] = file_sha256(
        ROOT / "engine" / "coin_city_stable_relationship_claude_n250.py"
    )
    protocol = {
        "study": STUDY,
        "status": "frozen_before_any_fresh_episode_generation",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "episodes": {
            "source": "engine.coin_city_stable_relationship_claude_n250.make_episode",
            "seed_base": FRESH_SEED_BASE,
            "registered_seed_base": original_seed_base(),
            "test_per_factorial_cell": TEST_PER_CELL,
            "sealed_test_episodes": len(episodes),
            "factorial_cells": dict(sorted(cells.items())),
            "development_episodes": 0,
            "selection": "first eleven generator indices in each factorial cell",
        },
        "models": MODELS,
        "hidden_state_layer_convention": (
            "0 is the token-embedding output; n is the output of transformer block n"
        ),
        "patch_site": "both exact City C symbol subtokens, prefill only",
        "conditions": list(CONDITIONS),
        "depths": [0, 4],
        "primary_depth": PRIMARY_DEPTH,
        "estimand": (
            "d_e = s_e/2 [(g_{W<-C} - g_W) + (g_C - g_{C<-W})], s_e = +1 for a "
            "strong-response City C; identical to analyze_symbol_relational_probe"
        ),
        "hypotheses": HYPOTHESES,
        "test": (
            f"one-sided episode sign flip, {SIGN_FLIP_DRAWS} draws, seed "
            f"{PERMUTATION_SEED}; complete-case episodes only"
        ),
        "multiplicity": (
            "Holm over the four primary tests (H1 and H2 in each model) at "
            "family-wise alpha .05"
        ),
        "gates": {
            "min_complete_episode_fraction": 14 / 16,
            "min_self_patch_exact_match_fraction": 58 / 64,
        },
        "reported_secondary": [
            "k=4 for every cross condition",
            "cross_late_window and cross_embedding_only at k=0 on their own",
            "episode-bootstrap 95% intervals",
        ],
        "decode": "greedy, max_new_tokens 160",
        "tasks_sha256": file_sha256(tasks_path),
        "episodes_sha256": file_sha256(episodes_path),
        "code_sha256": code,
    }
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: protocol[k] for k in ("study", "tasks_sha256")}, indent=2))
    print(f"{len(episodes)} fresh episodes, {len(records)} task records")


def original_seed_base() -> int:
    return int(design.SEED_BASE)


if __name__ == "__main__":
    main()
