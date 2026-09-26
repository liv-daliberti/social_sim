#!/usr/bin/env python3
"""Evaluate the all-or-nothing frozen Qwen3-8B mechanistic success gate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import RUNS_DIR, STUDY, atomic_json  # noqa: E402


def positive(result: dict) -> bool:
    return bool(result["ci95_low"] > 0 and result["all_seed_direction_positive"])


def main() -> None:
    behavior = json.loads((RUNS_DIR / "behavior_summary.json").read_text())
    probe = json.loads((RUNS_DIR / "common_probe" / "sealed_test_summary.json").read_text())
    patch = json.loads((RUNS_DIR / "patching_summary.json").read_text())
    primary_patch = patch["effects_by_k"]["0"]
    gates = {
        "frozen_subset_accuracy_advantage": positive(
            behavior["matched_accuracy_advantage"]
        ),
        "common_readout_kernel_advantage": positive(
            probe["common_readout_matched_advantage"]
        ),
        "matched_state_transplant_tracking": positive(
            probe["matched_reference_transplant_tracking"]
        ),
        "matched_to_prior_patch_improves": positive(
            primary_patch["matched_to_prior_improvement"]
        ),
        "prior_to_matched_patch_harms": positive(
            primary_patch["prior_to_matched_harm"]
        ),
        "same_episode_exceeds_wrong_episode": positive(
            primary_patch["same_minus_wrong_episode"]
        ),
        "strict_parse_and_self_patch_fidelity": bool(
            behavior["minimum_strict_parse_rate"] >= 0.99
            and patch["strict_parse_rate"] >= 0.99
            and patch["self_patch_exact_forecast_fidelity"] >= 0.99
        ),
    }
    passed = all(gates.values())
    result = {
        "study": STUDY,
        "status": "qwen3_8b_gate_passed" if passed else "qwen3_8b_gate_failed",
        "gates": gates,
        "all_gates_passed": passed,
        "qwen3_4b_replication_authorized": passed,
        "paper_status": "paper_external_do_not_render_or_import_into_manuscript",
        "behavioral_transplant_difference_in_differences_supportive_not_gated": behavior[
            "reference_transplant_by_k"
        ]["0"]["matched_minus_prior_difference_in_differences"],
    }
    atomic_json(RUNS_DIR / "final_gate.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
