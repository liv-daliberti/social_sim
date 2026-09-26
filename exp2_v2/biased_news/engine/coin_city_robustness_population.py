#!/usr/bin/env python3
"""Harder post-hoc Coin City generator population.

This module deliberately leaves the frozen original generator untouched.  It
reuses only prompt/rendering helpers whose behavior is invariant to the slope
population.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from engine.coin_city_stable_relationship_claude_n250 import (
    CASE_NOISE_SD,
    C_CASE_LEVELS,
    EPISODES,
    PROMPT_ARMS as SEMANTIC_ARMS,
    QUERY_NEWS_VALUES,
    _poll_start,
    _reference_rows,
    _target_rows,
    make_prompt as make_semantic_prompt,
    prompt_sha256,
    validate_arm_prompt as validate_semantic_prompt,
)
from engine.coin_city_symbol_context_arm import (
    ARM as SYMBOL_ARM,
    make_prompt as make_symbol_prompt,
    validate_arm_prompt as validate_symbol_prompt,
)

PROTOCOL_VERSION = "coin_city_robustness_v1"
EXPERIMENT = "coin_city_population_075_040_sd010_v1"
TASK_PREFIX = "coincityrobustv1"
SEED_BASE = 940_000
STRONG_SLOPE_MEAN = 0.75
WEAK_SLOPE_MEAN = 0.40
CITY_SLOPE_SD = 0.10
ARMS = ("abc_no_context", "abc_context", SYMBOL_ARM)


def make_episode(index: int) -> dict[str, Any]:
    """Generate one episode without truncating or reordering slope draws."""
    if not 0 <= index < EPISODES:
        raise ValueError(index)
    if index < 63:
        strong_reference_city, target_strong = "A", True
    elif index < 125:
        strong_reference_city, target_strong = "A", False
    elif index < 187:
        strong_reference_city, target_strong = "B", True
    else:
        strong_reference_city, target_strong = "B", False

    rng = np.random.default_rng(SEED_BASE + index)
    strong_slope = STRONG_SLOPE_MEAN + rng.normal(0.0, CITY_SLOPE_SD)
    weak_slope = WEAK_SLOPE_MEAN + rng.normal(0.0, CITY_SLOPE_SD)
    target_slope = (
        STRONG_SLOPE_MEAN if target_strong else WEAK_SLOPE_MEAN
    ) + rng.normal(0.0, CITY_SLOPE_SD)
    slopes = (
        {"A": strong_slope, "B": weak_slope}
        if strong_reference_city == "A"
        else {"A": weak_slope, "B": strong_slope}
    )
    return {
        "episode": index,
        "seed": SEED_BASE + index,
        "strong_reference_city": strong_reference_city,
        "target_strong": target_strong,
        "cue_strong": target_strong,
        "cue_correct": True,
        "reference_strong_slope": float(strong_slope),
        "reference_weak_slope": float(weak_slope),
        "target_slope": float(target_slope),
        "reference_a": _reference_rows(rng, slopes["A"]),
        "reference_b": _reference_rows(rng, slopes["B"]),
        "target": _target_rows(rng, target_slope),
        "query_starting_poll": (query_start := _poll_start(rng)),
        "query_net_news": (query_news := int(rng.choice(QUERY_NEWS_VALUES))),
        "gold_expected_poll": float(query_start + target_slope * query_news),
    }


def task_id(episode: int, c_cases: int) -> str:
    if not 0 <= episode < EPISODES or c_cases not in C_CASE_LEVELS:
        raise ValueError((episode, c_cases))
    return f"{TASK_PREFIX}_{episode:04d}_c{c_cases}"


def make_prompt(episode: dict[str, Any], c_cases: int, arm: str) -> str:
    if arm == SYMBOL_ARM:
        prompt = make_symbol_prompt(episode, c_cases)
        validate_symbol_prompt(prompt, arm)
    elif arm in SEMANTIC_ARMS:
        prompt = make_semantic_prompt(episode, c_cases, arm)
        validate_semantic_prompt(prompt, arm)
    else:
        raise ValueError(arm)
    return prompt
