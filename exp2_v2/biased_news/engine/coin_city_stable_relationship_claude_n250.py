#!/usr/bin/env python3
"""Claude-only n=250 stable-response design with visible poll changes."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np


EXPERIMENT = "coin_city_stable_relationship_claude_n250_v4"
TASK_PREFIX = "coincityclaude250v4"
EPISODES = 250
C_CASE_LEVELS = tuple(range(0, 5))
REFERENCE_CASES = 4
PROMPT_ARMS = ("baseline", "abc_no_context", "abc_context")
MODELS = ("claude-opus-4-8",)
HARD_CALL_CAP = 5_000
FUTURE_CALLS = EPISODES * len(C_CASE_LEVELS) * len(PROMPT_ARMS) * len(MODELS)
SEED_BASE = 930_000

STARTING_POLL_RANGE = (35.0, 65.0)
NEWS_MAGNITUDES = tuple(range(6, 11))
QUERY_NEWS_VALUES = tuple(range(-10, -3)) + tuple(range(4, 11))
STRONG_SLOPE_MEAN = 0.90
WEAK_SLOPE_MEAN = 0.25
CITY_SLOPE_SD = 0.03
CASE_NOISE_SD = 5.0

SHARED_PROMPT_INTRO = (
    "You are forecasting a post-news local-election poll.\n"
    "Each table row is a separate polling case, not a time series. Positive "
    "net news favors the candidate and negative net news harms the candidate. "
    "Poll change is the poll at the end of the case minus the starting poll. "
    "Within a city, its typical responsiveness to net news is stable across "
    "cases, although individual end polls remain noisy. When two displayed "
    "rows share a Pair label, they are matched: they begin at the same poll "
    "and receive equal-sized net-news scores in opposite directions."
)

REFERENCE_BACKGROUND_STRONG = (
    "Background: Residents generally encounter campaign developments through "
    "national news coverage, where campaign stories receive sustained attention "
    "and tend to produce larger immediate poll movements."
)
REFERENCE_BACKGROUND_WEAK = (
    "Background: Residents generally encounter campaign developments through "
    "local news coverage, where campaign stories compete with local issues and "
    "tend to produce smaller immediate poll movements."
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


def _poll_start(rng: np.random.Generator) -> float:
    return float(np.round(rng.uniform(*STARTING_POLL_RANGE), 1))


def _ending_poll(
    rng: np.random.Generator, start: float, news: int, slope: float
) -> float:
    return float(np.round(start + slope * news + rng.normal(0.0, CASE_NOISE_SD), 1))


def _matched_pair(
    rng: np.random.Generator, slope: float, pair: int
) -> list[dict[str, float | int]]:
    start = _poll_start(rng)
    magnitude = int(rng.choice(NEWS_MAGNITUDES))
    first_sign = 1 if int(rng.integers(0, 2)) else -1
    rows = []
    for news in (first_sign * magnitude, -first_sign * magnitude):
        rows.append(
            {
                "pair": pair,
                "starting_poll": start,
                "net_news": news,
                "ending_poll": _ending_poll(rng, start, news, slope),
            }
        )
    return rows


def _reference_rows(
    rng: np.random.Generator, slope: float
) -> list[dict[str, float | int]]:
    return [
        row
        for pair in range(1, 3)
        for row in _matched_pair(rng, slope, pair)
    ]


def _target_rows(
    rng: np.random.Generator, slope: float
) -> list[dict[str, float | int]]:
    return [
        row
        for pair in range(1, 3)
        for row in _matched_pair(rng, slope, pair)
    ]


def make_episode(index: int) -> dict[str, Any]:
    """Generate the closest-balanced 250-episode 2x2 factorial design."""
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
    reference_a = _reference_rows(rng, slopes["A"])
    reference_b = _reference_rows(rng, slopes["B"])
    target = _target_rows(rng, target_slope)
    query_start = _poll_start(rng)
    query_news = int(rng.choice(QUERY_NEWS_VALUES))
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
        "reference_a": reference_a,
        "reference_b": reference_b,
        "target": target,
        "query_starting_poll": query_start,
        "query_net_news": query_news,
        "gold_expected_poll": float(query_start + target_slope * query_news),
    }


def fit_change_slope(rows: Iterable[dict[str, float | int]]) -> float:
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
    return float(episode["query_starting_poll"] + episode["query_net_news"] * slope)


def empirical_estimates(
    episode: dict[str, Any], c_cases: int
) -> dict[str, float | None]:
    if c_cases not in C_CASE_LEVELS:
        raise ValueError(c_cases)
    target_rows = episode["target"][:c_cases]
    city_c = None if not target_rows else predict_from_slope(
        episode, fit_change_slope(target_rows)
    )
    abc_rows = episode["reference_a"] + episode["reference_b"] + target_rows
    abc = predict_from_slope(episode, fit_change_slope(abc_rows))
    return {"baseline": city_c, "abc_no_context": abc, "abc_context": abc}


def _table(rows: list[dict[str, float | int]]) -> str:
    lines = [
        "| Case | Pair | Start | News | Change | End |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows, 1):
        change = float(row["ending_poll"] - row["starting_poll"])
        lines.append(
            f"| {index} | {row['pair']} | {row['starting_poll']:.1f} | "
            f"{int(row['net_news']):+d} | {change:+.1f} | "
            f"{row['ending_poll']:.1f} |"
        )
    return "\n".join(lines)


def reference_background(episode: dict[str, Any], city: str) -> str:
    return REFERENCE_BACKGROUND_STRONG if episode["strong_reference_city"] == city else REFERENCE_BACKGROUND_WEAK


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
                "CITY A\n" + reference_background(episode, "A") + "\n" + _table(episode["reference_a"]),
                "CITY B\n" + reference_background(episode, "B") + "\n" + _table(episode["reference_b"]),
            ]
        )
    sections.append(
        "CITY C\n" + target_background(episode, arm) + "\n" + _table(episode["target"][:c_cases])
    )
    return "\n\n".join(sections)


def shared_prompt_closing(episode: dict[str, Any]) -> str:
    return (
        f"A new City C case begins with a poll of {episode['query_starting_poll']:.1f} "
        f"and has net news {int(episode['query_net_news']):+d}.\n"
        "What poll do you predict at the end of this new case?\n\n"
        "Think carefully about the expected poll, but do not provide step-by-step "
        "working. Respond with one JSON object only:\n"
        '{"rationale": "one short sentence", "predicted_poll": <number from 0 to 100>}'
    )


def make_prompt(episode: dict[str, Any], c_cases: int, arm: str) -> str:
    return SHARED_PROMPT_INTRO + "\n\n" + prompt_sections(episode, c_cases, arm) + "\n\n" + shared_prompt_closing(episode)


def task_id(episode: int, c_cases: int) -> str:
    return f"{TASK_PREFIX}_{episode:04d}_c{c_cases}"


def validate_arm_prompt(prompt_text: str, arm: str) -> None:
    if arm not in PROMPT_ARMS:
        raise ValueError(arm)
    if not prompt_text.startswith(SHARED_PROMPT_INTRO + "\n\n"):
        raise ValueError("shared introduction mismatch")
    has_references = "CITY A\n" in prompt_text and "CITY B\n" in prompt_text
    if has_references != (arm != "baseline"):
        raise ValueError("reference-city inclusion mismatch")
    has_context = TARGET_CONTEXT_STRONG in prompt_text or TARGET_CONTEXT_WEAK in prompt_text
    if has_context != (arm == "abc_context"):
        raise ValueError("City C context mismatch")
    if arm != "abc_context" and TARGET_CONTEXT_NONE not in prompt_text:
        raise ValueError("missing no-context sentence")
    lowered = prompt_text.lower()
    for forbidden in ("regression", "coefficient", "weighted average", "calculate a slope", "closer to city"):
        if forbidden in lowered:
            raise ValueError(f"prompt prescribes estimator: {forbidden}")
