import math
import sys
from pathlib import Path

import pytest

TRANSFER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRANSFER_ROOT))

from verify_paper_handoff import verify_coin_estimate_shapes  # noqa: E402

MODELS = ("qwen3_4b", "qwen3_8b", "llama3_1_8b")


def interval():
    return {"estimate": 1.0, "ci95_low": 0.5, "ci95_high": 1.5}


def complete_payload():
    return {"registered_estimates": {
        "primary": [
            {"model": model, "decode": decode,
             "mean_response_mae": {"base": 3.0, "population_prior": 2.0, "causal": 1.0},
             "population_prior_minus_causal": interval()}
            for model in MODELS for decode in ("greedy", "stochastic")
        ],
        "transfer_cells": [
            {"model": model, "domain": domain, "target_structure": structure,
             "population_prior_minus_causal": 1.0, "ci95_low": 0.5, "ci95_high": 1.5}
            for model in MODELS for domain in ("coin_city", "coin_harbor")
            for structure in ("direct_a", "mediated_b")
        ],
        "cue_by_k": [
            {"model": model, "k": k, "mean_response_mae": {"correct": 1.0,
             "none": 2.0, "misleading": 3.0}, "absent_minus_correct": interval(),
             "misleading_minus_correct": interval()}
            for model in MODELS for k in (0, 2, 4, 8)
        ],
        "cue_overall": [
            {"model": model, "absent_minus_correct": interval(),
             "misleading_minus_correct": interval()} for model in MODELS
        ],
        "evidence_override": [
            {"model": model, "contrast": "k8_minus_k0", **interval()}
            for model in MODELS
        ],
    }}


def test_exact_registered_estimate_factorial_passes():
    verify_coin_estimate_shapes(complete_payload())


def test_missing_or_nonfinite_registered_estimate_fails():
    payload = complete_payload()
    payload["registered_estimates"]["cue_by_k"].pop()
    with pytest.raises(AssertionError, match="cue-by-k"):
        verify_coin_estimate_shapes(payload)
    payload = complete_payload()
    payload["registered_estimates"]["evidence_override"][0]["estimate"] = math.nan
    with pytest.raises(AssertionError, match="nonfinite"):
        verify_coin_estimate_shapes(payload)
