#!/usr/bin/env python3
"""Fresh 100-episode coin-to-city design with always-aligned context."""

from __future__ import annotations

from typing import Any

import numpy as np

from engine import coin_city_llm_pilot as base


EXPERIMENT = "coin_city_expanded_aligned_v1"
TASK_PREFIX = "coincitya100"
EPISODES = 100
ROUNDS = base.ROUNDS
CASES_BY_ROUND = base.CASES_BY_ROUND
REFERENCE_CASES = base.REFERENCE_CASES
PROMPT_ARMS = base.PROMPT_ARMS
MODELS = base.MODELS
HARD_CALL_CAP = 5_000
PLANNED_CALLS = EPISODES * len(ROUNDS) * len(PROMPT_ARMS) * len(MODELS)
SEED_BASE = 420_000
CUE_RELIABILITY = 1.0

STARTING_POLL = base.STARTING_POLL
NEWS = base.NEWS
LOW_ENDPOINT_MEAN = base.LOW_ENDPOINT_MEAN
HIGH_ENDPOINT_MEAN = base.HIGH_ENDPOINT_MEAN
CITY_ENDPOINT_SD = base.CITY_ENDPOINT_SD
CASE_NOISE_SD = base.CASE_NOISE_SD

SHARED_PROMPT_INTRO = base.SHARED_PROMPT_INTRO
SHARED_PROMPT_CLOSING = base.SHARED_PROMPT_CLOSING
TARGET_CONTEXT_HIGH = base.TARGET_CONTEXT_HIGH
TARGET_CONTEXT_LOW = base.TARGET_CONTEXT_LOW
TARGET_CONTEXT_NONE = base.TARGET_CONTEXT_NONE

prompt_sha256 = base.prompt_sha256
prompt_sections = base.prompt_sections
make_prompt = base.make_prompt
target_background = base.target_background
validate_arm_prompt = base.validate_arm_prompt


def _rounded(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.round(values, 1)]


def make_episode(index: int) -> dict[str, Any]:
    """Use 25 episodes in each reference-label × target-type cell."""
    if not 0 <= index < EPISODES:
        raise ValueError(index)
    block = index // 25
    high_reference_city = "A" if block in (0, 1) else "B"
    target_high = block in (0, 2)
    rng = np.random.default_rng(SEED_BASE + index)
    low_truth = LOW_ENDPOINT_MEAN + rng.normal(0.0, CITY_ENDPOINT_SD)
    high_truth = HIGH_ENDPOINT_MEAN + rng.normal(0.0, CITY_ENDPOINT_SD)
    target_center = HIGH_ENDPOINT_MEAN if target_high else LOW_ENDPOINT_MEAN
    target_truth = target_center + rng.normal(0.0, CITY_ENDPOINT_SD)
    low_rows = _rounded(
        low_truth + rng.normal(0.0, CASE_NOISE_SD, size=REFERENCE_CASES)
    )
    high_rows = _rounded(
        high_truth + rng.normal(0.0, CASE_NOISE_SD, size=REFERENCE_CASES)
    )
    target_rows = _rounded(
        target_truth
        + rng.normal(0.0, CASE_NOISE_SD, size=max(CASES_BY_ROUND.values()))
    )
    references = (
        {"A": high_rows, "B": low_rows}
        if high_reference_city == "A"
        else {"A": low_rows, "B": high_rows}
    )
    return {
        "episode": index,
        "seed": SEED_BASE + index,
        "high_reference_city": high_reference_city,
        "target_high": target_high,
        "cue_high": target_high,
        "cue_correct": True,
        "reference_low_truth": float(low_truth),
        "reference_high_truth": float(high_truth),
        "target_truth": float(target_truth),
        "reference_a": references["A"],
        "reference_b": references["B"],
        "target": target_rows,
    }


def task_id(episode: int, k: int) -> str:
    return f"{TASK_PREFIX}_{episode:04d}_k{k}"


def baseline_estimates(episode: dict[str, Any], k: int) -> dict[str, float]:
    """Benchmarks use only prompt-visible values; context is an oracle bound."""
    cases = CASES_BY_ROUND[k]
    target_mean = float(np.mean(episode["target"][:cases]))
    a_mean = float(np.mean(episode["reference_a"]))
    b_mean = float(np.mean(episode["reference_b"]))
    if episode["high_reference_city"] == "A":
        high_mean, low_mean = a_mean, b_mean
    else:
        high_mean, low_mean = b_mean, a_mean
    no_context = base._mixture_posterior_mean(
        target_mean, cases, low_mean, high_mean, 0.5
    )
    # Numerically represent the certain direction without taking log(0).
    aligned_prior_high = 1.0 - 1e-12 if episode["cue_high"] else 1e-12
    aligned_context = base._mixture_posterior_mean(
        target_mean, cases, low_mean, high_mean, aligned_prior_high
    )
    return {
        "baseline": target_mean,
        "abc_no_context": no_context,
        "abc_context": aligned_context,
    }
