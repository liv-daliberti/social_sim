#!/usr/bin/env python3
"""Register the immutable design, endpoints, estimands, gates, and task hashes."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    ADAPTER_JOBS,
    ANCHORS,
    DATA_DIR,
    K_VALUES,
    MODEL,
    MODEL_COMMIT,
    PRIMARY_ANCHOR,
    PROTOCOL_DIR,
    STUDY,
    TRAINING_SEEDS,
    adapter_freeze,
    atomic_json,
    file_sha256,
)


def protocol_payload() -> dict:
    task_manifest_path = DATA_DIR / "task_manifest.json"
    task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
    if task_manifest.get("study") != STUDY:
        raise ValueError("task study does not match protocol")
    endpoints = [
        adapter_freeze(seed, arm)
        for seed in TRAINING_SEEDS
        for arm in ("matched", "prior")
    ]
    return {
        "study": STUDY,
        "status": "protocol_frozen_before_activation_extraction",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "paper_status": "paper_external_do_not_render_or_import_into_manuscript",
        "model": MODEL,
        "model_commit": MODEL_COMMIT,
        "confirmatory_endpoints": endpoints,
        "secondary_reference": "untrained base, extracted only for seed 42 and never gated",
        "tasks": {
            "manifest_path": str(task_manifest_path.relative_to(HERE)),
            "manifest_sha256": file_sha256(task_manifest_path),
            "tasks_sha256": task_manifest["tasks_sha256"],
            "record_count": task_manifest["record_count"],
            "development_episodes": task_manifest["development_episodes"],
            "sealed_test_episodes": task_manifest["test_episodes"],
            "k_values": list(K_VALUES),
            "cell": "Coin Harbor / mediated_b / correct cue",
        },
        "activation_extraction": {
            "anchors": list(ANCHORS),
            "primary_anchor": PRIMARY_ANCHOR,
            "states": "embedding plus every transformer-block output, float16",
            "maximum_input_tokens": 2304,
            "thinking": False,
            "patch_positions": "last token of each of the ten scenario rows A--J",
        },
        "representation_estimand": {
            "targets": ["per-unit h1", "per-unit h3"],
            "standardization": "development-only target mean and SD",
            "readout": "one ridge decoder pooled symmetrically across matched and prior arm means",
            "cross_validation": "four development folds grouped by reciprocal donor pair",
            "selection": "development-only alpha and layer; transformer-state layers 1--35 eligible",
            "primary_effect": "standardized kernel MSE(prior) minus MSE(matched)",
            "transplant_effect": (
                "projection of decoded transplant-minus-original kernel onto "
                "donor-minus-original kernel, divided by donor displacement squared"
            ),
        },
        "behavior_estimands": {
            "decode": "five draws, temperature 0.7, top_p 0.95, strict forecast-array grammar",
            "accuracy": (
                "original-prompt response MAE(prior) minus response MAE(matched), equally "
                "averaged over k=0 and k=8"
            ),
            "transplant": (
                "projection of predicted response(transplant)-predicted response(original) "
                "onto donor-response minus original-response, normalized by squared donor displacement"
            ),
            "primary_transplant_k": 0,
            "evidence_resistance_k": 8,
        },
        "patching_estimand": {
            "selection": "development-selected three-state-layer window centered on probe layer",
            "positions": "scenario-row end tokens A--J",
            "test_inputs": "sealed original prompts at k=0 and k=8",
            "conditions": [
                "prior_unpatched",
                "prior_self",
                "matched_same_to_prior",
                "matched_wrong_to_prior",
                "matched_unpatched",
                "matched_self",
                "prior_same_to_matched",
                "prior_wrong_to_matched",
            ],
            "primary_k": 0,
            "forward": "MAE(prior_unpatched) minus MAE(matched_same_to_prior)",
            "reverse": "MAE(prior_same_to_matched) minus MAE(matched_unpatched)",
            "same_episode_specificity": (
                "mean aligned same-episode effect minus mean aligned reciprocal wrong-episode effect"
            ),
        },
        "uncertainty": {
            "method": "hierarchical paired bootstrap: training seed first, episode second",
            "repetitions": 10000,
            "interval": 0.95,
            "draws_averaged_within_episode": True,
        },
        "success_gate": [
            "frozen-subset matched accuracy advantage has CI lower bound > 0 and all seeds positive",
            "common-readout within-cell kernel advantage has CI lower bound > 0 and all seeds positive",
            "matched-state reference-transplant tracking has CI lower bound > 0",
            "matched-to-prior patch improvement has CI lower bound > 0",
            "prior-to-matched patch harm has CI lower bound > 0",
            "same-episode aligned patch effect exceeds wrong-episode effect with CI lower bound > 0",
            "all required directional effects are positive in seeds 42, 43, and 44",
            "strict parse rate is at least 0.99 and self-patch forecast fidelity is at least 0.99",
        ],
        "replication_rule": (
            "submit the complete Qwen3-4B three-seed protocol only if every Qwen3-8B gate passes; "
            "14B remains ineligible until its separate behavioral matched-training advantage is confirmed"
        ),
        "registered_jobs": ADAPTER_JOBS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROTOCOL_DIR / "frozen_protocol.json")
    args = parser.parse_args()
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        proposed = protocol_payload()
        for timestamp_free in (existing, proposed):
            timestamp_free.pop("frozen_at", None)
        if existing != proposed:
            raise RuntimeError("protocol already frozen with different contents")
        print(f"verified existing frozen protocol: {args.output}")
        return
    payload = protocol_payload()
    atomic_json(args.output, payload)
    atomic_json(
        args.output.with_name("freeze_receipt.json"),
        {
            "study": STUDY,
            "status": "frozen",
            "protocol_path": str(args.output),
            "protocol_sha256": file_sha256(args.output),
            "frozen_at": payload["frozen_at"],
        },
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
