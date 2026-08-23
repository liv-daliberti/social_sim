#!/usr/bin/env python3
"""Variable-predictor coin-to-city experiment with prompt-only regressions."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np


EXPERIMENT = "coin_city_variable_regression_v1"
TASK_PREFIX = "coincityvr1"
EPISODES = 50
C_CASE_LEVELS = tuple(range(0, 7))
# Compatibility alias for the existing worker interface; figures and prompts
# use the substantive City C case count directly and never display "round".
ROUNDS = C_CASE_LEVELS
CASES_BY_ROUND = {cases: cases for cases in C_CASE_LEVELS}
REFERENCE_CASES = 3
PROMPT_ARMS = ("baseline", "abc_no_context", "abc_context")
MODELS = ("claude-opus-4-8", "DeepSeek-V4-Pro", "gpt-5.4")
HARD_CALL_CAP = 5_000
PLANNED_CALLS = EPISODES * len(ROUNDS) * len(PROMPT_ARMS) * len(MODELS)
SEED_BASE = 640_000
SOURCE_EPISODES_PER_CELL = 25
LIVE_REPLICATES_BY_CELL = (
    tuple(range(0, SOURCE_EPISODES_PER_CELL, 2)),
    tuple(range(1, SOURCE_EPISODES_PER_CELL, 2)),
    tuple(range(1, SOURCE_EPISODES_PER_CELL, 2)),
    tuple(range(0, SOURCE_EPISODES_PER_CELL, 2)),
)
LIVE_EPISODES_BY_CELL = tuple(map(len, LIVE_REPLICATES_BY_CELL))
BASE_REFERENCE_CASES = 2
SOURCE_TARGET_SEQUENCE_CASES = 16
EXTRA_REFERENCE_SEED_BASE = 910_000

STARTING_POLL_RANGE = (35.0, 65.0)
NEWS_VALUES = tuple(range(-10, -3)) + tuple(range(4, 11))
STRONG_SLOPE_MEAN = 0.95
WEAK_SLOPE_MEAN = 0.05
CITY_SLOPE_SD = 0.04
CASE_NOISE_SD = 6.0

SHARED_PROMPT_INTRO = (
    "You are forecasting a post-news local-election poll.\n"
    "Each table row is a separate polling case, not a time series. Starting "
    "polls and net-news scores vary across cases. Positive net news favors "
    "the candidate and negative net news harms the candidate. The poll at "
    "the end of each case was then measured. End-of-case polls vary because "
    "polling and local opinion are not perfectly stable."
)

REFERENCE_BACKGROUND_STRONG = (
    "Background: Residents generally encountered campaign developments through "
    "national news coverage, where campaign stories received sustained attention "
    "and tended to translate into larger immediate poll movements."
)
REFERENCE_BACKGROUND_WEAK = (
    "Background: Residents generally encountered campaign developments through "
    "local news coverage, where campaign stories competed with local issues and "
    "tended to translate into smaller immediate poll movements."
)
TARGET_CONTEXT_STRONG = (
    "Background: City C residents generally encounter campaign developments "
    "through national news coverage."
)
TARGET_CONTEXT_WEAK = (
    "Background: City C residents generally encounter campaign developments "
    "through local news coverage."
)
TARGET_CONTEXT_NONE = "No additional background information is available."


def prompt_sha256(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode()).hexdigest()


def _sample_news(rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.choice(np.asarray(NEWS_VALUES, dtype=int), size=size, replace=True)


def _sample_rows(
    rng: np.random.Generator,
    slope: float,
    size: int,
    *,
    ensure_both_signs: bool = False,
) -> list[dict[str, float | int]]:
    starting = np.round(
        rng.uniform(STARTING_POLL_RANGE[0], STARTING_POLL_RANGE[1], size=size),
        1,
    )
    if ensure_both_signs:
        if size < 2:
            raise ValueError("sign-balanced sampling needs at least two rows")
        news = np.concatenate(
            [
                np.asarray(
                    [rng.integers(-10, -3), rng.integers(4, 11)], dtype=int
                ),
                _sample_news(rng, size - 2),
            ]
        )
        rng.shuffle(news)
    else:
        news = _sample_news(rng, size)
    ending = np.round(
        starting + slope * news + rng.normal(0.0, CASE_NOISE_SD, size=size),
        1,
    )
    return [
        {
            "starting_poll": float(start),
            "net_news": int(shock),
            "ending_poll": float(end),
        }
        for start, shock, end in zip(starting, news, ending)
    ]


def make_episode(index: int) -> dict[str, Any]:
    """Select alternating source replicates, with 12/13 cases per cell."""
    if not 0 <= index < EPISODES:
        raise ValueError(index)
    remaining = index
    for block, replicates in enumerate(LIVE_REPLICATES_BY_CELL):
        if remaining < len(replicates):
            replicate = replicates[remaining]
            break
        remaining -= len(replicates)
    else:
        raise AssertionError("live episode mapping exhausted")
    source_index = block * SOURCE_EPISODES_PER_CELL + replicate
    strong_reference_city = "A" if block in (0, 1) else "B"
    target_strong = block in (0, 2)
    rng = np.random.default_rng(SEED_BASE + source_index)
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
    reference_a = _sample_rows(
        rng, slopes["A"], BASE_REFERENCE_CASES, ensure_both_signs=True
    )
    reference_b = _sample_rows(
        rng, slopes["B"], BASE_REFERENCE_CASES, ensure_both_signs=True
    )
    target = _sample_rows(
        rng, target_slope, SOURCE_TARGET_SEQUENCE_CASES
    )
    query_start = float(
        np.round(rng.uniform(*STARTING_POLL_RANGE), 1)
    )
    query_news = int(_sample_news(rng, 1)[0])
    query_truth = float(query_start + target_slope * query_news)
    a_extra_rng = np.random.default_rng(
        EXTRA_REFERENCE_SEED_BASE + 100 * source_index + 1
    )
    b_extra_rng = np.random.default_rng(
        EXTRA_REFERENCE_SEED_BASE + 100 * source_index + 2
    )
    reference_a.extend(_sample_rows(a_extra_rng, slopes["A"], 1))
    reference_b.extend(_sample_rows(b_extra_rng, slopes["B"], 1))
    return {
        "episode": index,
        "source_episode": source_index,
        "seed": SEED_BASE + source_index,
        "strong_reference_city": strong_reference_city,
        "target_strong": target_strong,
        "cue_strong": target_strong,
        "cue_correct": True,
        "reference_strong_slope": float(strong_slope),
        "reference_weak_slope": float(weak_slope),
        "target_slope": float(target_slope),
        "reference_a": reference_a,
        "reference_b": reference_b,
        "target": target,
        "query_starting_poll": query_start,
        "query_net_news": query_news,
        "gold_expected_poll": query_truth,
    }


def fit_change_slope(rows: Iterable[dict[str, float | int]]) -> float:
    """OLS through the origin for poll change on net news."""
    materialized = list(rows)
    news = np.asarray([row["net_news"] for row in materialized], dtype=float)
    change = np.asarray(
        [row["ending_poll"] - row["starting_poll"] for row in materialized],
        dtype=float,
    )
    denominator = float(np.sum(news**2))
    if denominator <= 0:
        raise ValueError("net-news variation is required")
    return float(np.sum(news * change) / denominator)


def predict_from_slope(episode: dict[str, Any], slope: float) -> float:
    return float(
        episode["query_starting_poll"] + episode["query_net_news"] * slope
    )


def empirical_estimates(
    episode: dict[str, Any], c_cases: int
) -> dict[str, float | None]:
    """Compute both displayed estimators from prompt-visible numeric rows."""
    if c_cases not in C_CASE_LEVELS:
        raise ValueError(c_cases)
    target_rows = episode["target"][:c_cases]
    city_c = (
        None
        if c_cases == 0
        else predict_from_slope(episode, fit_change_slope(target_rows))
    )
    abc_rows = episode["reference_a"] + episode["reference_b"] + target_rows
    abc = predict_from_slope(episode, fit_change_slope(abc_rows))
    return {
        "baseline": city_c,
        "abc_no_context": abc,
        # No privileged context regression is displayed.  The context arm is
        # compared with the same prompt-only A/B/C regression reference.
        "abc_context": abc,
    }


def _table(rows: list[dict[str, float | int]]) -> str:
    lines = [
        "| Case | Starting poll | Net news | Poll at end of case |",
        "|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows, 1):
        news = int(row["net_news"])
        lines.append(
            f"| {index} | {row['starting_poll']:.1f} | {news:+d} | "
            f"{row['ending_poll']:.1f} |"
        )
    return "\n".join(lines)


def reference_background(episode: dict[str, Any], city: str) -> str:
    return (
        REFERENCE_BACKGROUND_STRONG
        if episode["strong_reference_city"] == city
        else REFERENCE_BACKGROUND_WEAK
    )


def target_background(episode: dict[str, Any], arm: str) -> str:
    if arm != "abc_context":
        return TARGET_CONTEXT_NONE
    return TARGET_CONTEXT_STRONG if episode["cue_strong"] else TARGET_CONTEXT_WEAK


def prompt_sections(episode: dict[str, Any], c_cases: int, arm: str) -> str:
    if arm not in PROMPT_ARMS or c_cases not in C_CASE_LEVELS:
        raise ValueError((arm, c_cases))
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
    sections.append(
        "CITY C\n"
        + target_background(episode, arm)
        + "\n"
        + _table(episode["target"][:c_cases])
    )
    return "\n\n".join(sections)


def shared_prompt_closing(episode: dict[str, Any]) -> str:
    news = int(episode["query_net_news"])
    return (
        f"A new City C case begins with a poll of "
        f"{episode['query_starting_poll']:.1f} and has net news {news:+d}.\n"
        "What poll do you predict at the end of this new case?\n\n"
        "Think carefully about the expected poll, but do not provide step-by-step "
        "working. Respond with one JSON object only:\n"
        '{"rationale": "one short sentence", "predicted_poll": <number from 0 to 100>}'
    )


def make_prompt(episode: dict[str, Any], c_cases: int, arm: str) -> str:
    return (
        SHARED_PROMPT_INTRO
        + "\n\n"
        + prompt_sections(episode, c_cases, arm)
        + "\n\n"
        + shared_prompt_closing(episode)
    )


def task_id(episode: int, c_cases: int) -> str:
    return f"{TASK_PREFIX}_{episode:04d}_c{c_cases}"


def validate_arm_prompt(prompt_text: str, arm: str) -> None:
    """Fail closed if a frozen prompt violates its condition contract."""
    if arm not in PROMPT_ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if not prompt_text.startswith(SHARED_PROMPT_INTRO + "\n\n"):
        raise ValueError("shared introduction mismatch")
    if prompt_text.count("CITY C\n") != 1:
        raise ValueError("City C block count mismatch")
    has_references = "CITY A\n" in prompt_text and "CITY B\n" in prompt_text
    if has_references != (arm != "baseline"):
        raise ValueError("reference-city inclusion mismatch")
    has_target_context = (
        TARGET_CONTEXT_STRONG in prompt_text or TARGET_CONTEXT_WEAK in prompt_text
    )
    if has_target_context != (arm == "abc_context"):
        raise ValueError("City C context mismatch")
    if arm != "abc_context" and TARGET_CONTEXT_NONE not in prompt_text:
        raise ValueError("missing no-context sentence")
    if prompt_text.count("| Case | Starting poll | Net news | Poll at end of case |") != (
        1 if arm == "baseline" else 3
    ):
        raise ValueError("table count mismatch")
    forbidden = (
        "matches city",
        "closer to city",
        "use city a",
        "use city b",
        "true slope",
        "latent",
    )
    lowered = prompt_text.lower()
    if any(token in lowered for token in forbidden):
        raise ValueError("prompt discloses the hidden city assignment")
