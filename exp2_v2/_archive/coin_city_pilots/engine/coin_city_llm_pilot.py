#!/usr/bin/env python3
"""Frozen design primitives for the bounded coin-to-city LLM pilot."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np


EXPERIMENT = "coin_city_llm_pilot_v1"
TASK_PREFIX = "coincityp1"
EPISODES = 40
ROUNDS = (1, 2, 3, 4, 5)
CASES_BY_ROUND = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}
REFERENCE_CASES = 6
PROMPT_ARMS = ("baseline", "abc_no_context", "abc_context")
MODELS = ("claude-opus-4-8", "DeepSeek-V4-Pro", "gpt-5.4")
HARD_CALL_CAP = 5_000
PLANNED_CALLS = EPISODES * len(ROUNDS) * len(PROMPT_ARMS) * len(MODELS)

STARTING_POLL = 50.0
NEWS = 8
LOW_ENDPOINT_MEAN = 52.0
HIGH_ENDPOINT_MEAN = 58.0
CITY_ENDPOINT_SD = 0.75
CASE_NOISE_SD = 4.5
CUE_RELIABILITY = 0.80
SEED_BASE = 310_000

SHARED_PROMPT_INTRO = (
    "You are forecasting a post-news local-election poll.\n"
    "Each table row is a separate polling case, not a time series. Every "
    "case began with a poll of 50.0. Net news +8 then occurred, and the "
    "poll at the end of that case was measured. End-of-case polls vary "
    "because polling and local opinion are not perfectly stable."
)
SHARED_PROMPT_CLOSING = (
    "A new City C case begins with a poll of 50.0 and has net news +8.\n"
    "What poll do you predict at the end of this new case?\n\n"
    "Think carefully about the expected poll, but do not provide step-by-step "
    "working. Respond with one JSON object only:\n"
    '{"rationale": "one short sentence", "predicted_poll": <number from 0 to 100>}'
)

REFERENCE_BACKGROUND_HIGH = (
    "Background: Residents generally followed national campaign developments "
    "through immediate alerts and direct candidate feeds, with little local filtering."
)
REFERENCE_BACKGROUND_LOW = (
    "Background: Residents generally encountered national campaign developments "
    "through neighborhood briefings where local organizers interpreted outside political news."
)
TARGET_CONTEXT_HIGH = (
    "Background: Most voters learned about national campaign events from live "
    "updates and direct campaign feeds rather than local intermediaries."
)
TARGET_CONTEXT_LOW = (
    "Background: Most voters learned about national campaign events from community "
    "briefings in which local organizers interpreted outside political news."
)
TARGET_CONTEXT_NONE = "No additional background information is available."


def prompt_sha256(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode()).hexdigest()


def _rounded(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.round(values, 1)]


def make_episode(index: int) -> dict[str, Any]:
    """Make one balanced, fresh episode using a seed disjoint from earlier runs."""
    if not 0 <= index < EPISODES:
        raise ValueError(index)
    block, replicate = divmod(index, 10)
    high_reference_city = "A" if block in (0, 1) else "B"
    target_high = block in (0, 2)
    cue_correct = replicate < 8
    cue_high = target_high if cue_correct else not target_high
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
        "cue_high": cue_high,
        "cue_correct": cue_correct,
        "reference_low_truth": float(low_truth),
        "reference_high_truth": float(high_truth),
        "target_truth": float(target_truth),
        "reference_a": references["A"],
        "reference_b": references["B"],
        "target": target_rows,
    }


def _table(values: list[float]) -> str:
    lines = [
        "| Case | Starting poll | Net news | Poll at end of case |",
        "|---:|---:|---:|---:|",
    ]
    for index, value in enumerate(values, 1):
        lines.append(f"| {index} | 50.0 | +8 | {value:.1f} |")
    return "\n".join(lines)


def reference_background(episode: dict[str, Any], city: str) -> str:
    is_high = episode["high_reference_city"] == city
    return REFERENCE_BACKGROUND_HIGH if is_high else REFERENCE_BACKGROUND_LOW


def target_background(episode: dict[str, Any], arm: str) -> str:
    if arm != "abc_context":
        return TARGET_CONTEXT_NONE
    return TARGET_CONTEXT_HIGH if episode["cue_high"] else TARGET_CONTEXT_LOW


def prompt_sections(episode: dict[str, Any], k: int, arm: str) -> str:
    if arm not in PROMPT_ARMS or k not in ROUNDS:
        raise ValueError((arm, k))
    sections: list[str] = []
    if arm != "baseline":
        sections.extend(
            [
                "CITY A\n"
                + reference_background(episode, "A")
                + "\n"
                + _table(episode["reference_a"]),
                "CITY B\n"
                + reference_background(episode, "B")
                + "\n"
                + _table(episode["reference_b"]),
            ]
        )
    cases = CASES_BY_ROUND[k]
    sections.append(
        "CITY C\n"
        + target_background(episode, arm)
        + "\n"
        + _table(episode["target"][:cases])
    )
    return "\n\n".join(sections)


def make_prompt(episode: dict[str, Any], k: int, arm: str) -> str:
    return (
        SHARED_PROMPT_INTRO
        + "\n\n"
        + prompt_sections(episode, k, arm)
        + "\n\n"
        + SHARED_PROMPT_CLOSING
    )


def task_id(episode: int, k: int) -> str:
    return f"{TASK_PREFIX}_{episode:04d}_k{k}"


def _mixture_posterior_mean(
    target_mean: float,
    cases: int,
    reference_low_mean: float,
    reference_high_mean: float,
    prior_high: float,
) -> float:
    component_prior_var = (
        2.0 * CITY_ENDPOINT_SD**2
        + CASE_NOISE_SD**2 / REFERENCE_CASES
    )
    target_sampling_var = CASE_NOISE_SD**2 / float(cases)
    marginal_var = component_prior_var + target_sampling_var
    posterior_var = 1.0 / (
        1.0 / component_prior_var + 1.0 / target_sampling_var
    )
    low_posterior_mean = posterior_var * (
        reference_low_mean / component_prior_var
        + target_mean / target_sampling_var
    )
    high_posterior_mean = posterior_var * (
        reference_high_mean / component_prior_var
        + target_mean / target_sampling_var
    )
    log_low = math.log(1.0 - prior_high) - (
        0.5 * (target_mean - reference_low_mean) ** 2 / marginal_var
    )
    log_high = math.log(prior_high) - (
        0.5 * (target_mean - reference_high_mean) ** 2 / marginal_var
    )
    normalizer = max(log_low, log_high)
    low_weight = math.exp(log_low - normalizer)
    high_weight = math.exp(log_high - normalizer)
    posterior_high = high_weight / (low_weight + high_weight)
    return (
        (1.0 - posterior_high) * low_posterior_mean
        + posterior_high * high_posterior_mean
    )


def baseline_estimates(episode: dict[str, Any], k: int) -> dict[str, float]:
    cases = CASES_BY_ROUND[k]
    target_mean = float(np.mean(episode["target"][:cases]))
    reference_a_mean = float(np.mean(episode["reference_a"]))
    reference_b_mean = float(np.mean(episode["reference_b"]))
    if episode["high_reference_city"] == "A":
        reference_high_mean, reference_low_mean = reference_a_mean, reference_b_mean
    else:
        reference_high_mean, reference_low_mean = reference_b_mean, reference_a_mean
    no_context = _mixture_posterior_mean(
        target_mean, cases, reference_low_mean, reference_high_mean, 0.5
    )
    cue_prior_high = CUE_RELIABILITY if episode["cue_high"] else 1.0 - CUE_RELIABILITY
    context = _mixture_posterior_mean(
        target_mean, cases, reference_low_mean, reference_high_mean, cue_prior_high
    )
    return {
        "baseline": target_mean,
        "abc_no_context": no_context,
        "abc_context": context,
    }


def validate_arm_prompt(prompt_text: str, arm: str) -> None:
    """Reject malformed prompts before a worker can call a model."""
    if arm not in PROMPT_ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if not prompt_text.startswith(SHARED_PROMPT_INTRO + "\n\n"):
        raise ValueError("shared introduction mismatch")
    if not prompt_text.endswith("\n\n" + SHARED_PROMPT_CLOSING):
        raise ValueError("shared closing mismatch")
    if prompt_text.count("CITY C\n") != 1:
        raise ValueError("City C block count mismatch")
    has_references = "CITY A\n" in prompt_text and "CITY B\n" in prompt_text
    if has_references != (arm != "baseline"):
        raise ValueError("reference-city inclusion mismatch")
    has_target_context = (
        TARGET_CONTEXT_HIGH in prompt_text or TARGET_CONTEXT_LOW in prompt_text
    )
    if has_target_context != (arm == "abc_context"):
        raise ValueError("City C context mismatch")
    if arm != "abc_context" and TARGET_CONTEXT_NONE not in prompt_text:
        raise ValueError("missing no-context sentence")
    forbidden = ("bayes", "prior", "regime", "matches city", "closer to city", "pool")
    lowered = prompt_text.lower()
    if any(token in lowered for token in forbidden):
        raise ValueError("prompt discloses the benchmark structure")
