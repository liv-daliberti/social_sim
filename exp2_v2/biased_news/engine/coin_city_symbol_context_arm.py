#!/usr/bin/env python3
"""Arbitrary-symbol control for the frozen Coin City n=250 design.

The semantic national/local descriptions are replaced everywhere by two novel
codes.  Their mapping to the two response regimes reverses on alternating
episodes, so a code has no stable direction across the evaluation.  Within a
prompt, City A and B's numeric examples are the only evidence for decoding the
mapping before applying it to City C.
"""

from __future__ import annotations

from typing import Any

from engine.coin_city_stable_relationship_claude_n250 import (
    C_CASE_LEVELS,
    SHARED_PROMPT_INTRO,
    _table,
    shared_prompt_closing,
)


ARM = "abc_symbol_context"
PROMPT_ARMS = ("baseline", "abc_no_context", "abc_context", ARM)
CONTEXT_ALIGNMENT = "episode_randomized_arbitrary_symbol_mapping"
SYMBOLS = ("KIV", "ZOR")
SYMBOL_INSTRUCTIONS = (
    "KIV and ZOR are arbitrary labels created for this task. They have no "
    "inherent meaning or ordering. Within this task, cities with the same "
    "label have the same typical responsiveness to net news. Infer what each "
    "label denotes only from the displayed cases."
)


def strong_symbol(episode: dict[str, Any]) -> str:
    """Return the episode-local label assigned to the higher-response regime."""
    return SYMBOLS[int(episode["episode"]) % 2]


def weak_symbol(episode: dict[str, Any]) -> str:
    return SYMBOLS[1 - int(episode["episode"]) % 2]


def city_symbol(episode: dict[str, Any], city: str) -> str:
    if city == "A":
        is_strong = episode["strong_reference_city"] == "A"
    elif city == "B":
        is_strong = episode["strong_reference_city"] == "B"
    elif city == "C":
        is_strong = bool(episode["cue_strong"])
    else:
        raise ValueError(city)
    return strong_symbol(episode) if is_strong else weak_symbol(episode)


def symbol_background(episode: dict[str, Any], city: str) -> str:
    return f"Background label: {city_symbol(episode, city)}."


def make_prompt(episode: dict[str, Any], c_cases: int) -> str:
    if c_cases not in C_CASE_LEVELS:
        raise ValueError(c_cases)
    sections = [
        "CITY A\n"
        + symbol_background(episode, "A")
        + "\n"
        + _table(episode["reference_a"]),
        "CITY B\n"
        + symbol_background(episode, "B")
        + "\n"
        + _table(episode["reference_b"]),
        "CITY C\n"
        + symbol_background(episode, "C")
        + "\n"
        + _table(episode["target"][:c_cases]),
    ]
    return (
        SHARED_PROMPT_INTRO
        + "\n\n"
        + SYMBOL_INSTRUCTIONS
        + "\n\n"
        + "\n\n".join(sections)
        + "\n\n"
        + shared_prompt_closing(episode)
    )


def validate_arm_prompt(prompt_text: str, arm: str) -> None:
    if arm != ARM:
        from engine.coin_city_stable_relationship_claude_n250 import (
            validate_arm_prompt as frozen_validate,
        )

        frozen_validate(prompt_text, arm)
        return
    if not prompt_text.startswith(SHARED_PROMPT_INTRO + "\n\n" + SYMBOL_INSTRUCTIONS):
        raise ValueError("shared introduction or symbol instructions mismatch")
    for city in "ABC":
        if f"CITY {city}\nBackground label: " not in prompt_text:
            raise ValueError(f"City {city} symbol missing")
    if prompt_text.count("Background label: KIV.") + prompt_text.count(
        "Background label: ZOR."
    ) != 3:
        raise ValueError("expected exactly three city labels")
    lowered = prompt_text.lower()
    for forbidden in (
        "national news",
        "local news",
        "larger immediate",
        "smaller immediate",
        "strong regime",
        "weak regime",
        "strongly responsive",
        "weakly responsive",
        "regression",
        "coefficient",
        "weighted average",
        "calculate a slope",
        "closer to city",
    ):
        if forbidden in lowered:
            raise ValueError(f"prompt leaks semantics or estimator: {forbidden}")
