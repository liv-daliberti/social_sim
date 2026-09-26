#!/usr/bin/env python3
"""Additive misleading-cue arm for the frozen Coin City n=250 design.

The three frozen arms all carry a correct City C description, so they cannot
separate "uses the cue to identify the response regime" from "trusts the cue
unconditionally": both accounts predict the same data. This arm supplies the
*opposite* description --- a City C that is truly strongly responsive is
described as having local news coverage, and vice versa --- while leaving every
number in the prompt untouched.

Nothing here mutates the frozen design. The arm's prompts are generated from the
frozen ``episodes.jsonl`` using the frozen engine's own rendering helpers, and
the builder asserts that each prompt differs from its ``abc_context``
counterpart by exactly the one substituted context sentence.
"""

from __future__ import annotations

from typing import Any

from engine.coin_city_stable_relationship_claude_n250 import (
    SHARED_PROMPT_INTRO,
    TARGET_CONTEXT_NONE,
    TARGET_CONTEXT_STRONG,
    TARGET_CONTEXT_WEAK,
    C_CASE_LEVELS,
    _table,
    reference_background,
    shared_prompt_closing,
)


ARM = "abc_wrong_context"
PROMPT_ARMS = ("baseline", "abc_no_context", "abc_context", ARM)
CONTEXT_ALIGNMENT = "0_percent_correct_news_environment"


def target_background(episode: dict[str, Any]) -> str:
    """The description that contradicts City C's true response regime."""
    return TARGET_CONTEXT_WEAK if episode["cue_strong"] else TARGET_CONTEXT_STRONG


def truthful_background(episode: dict[str, Any]) -> str:
    return TARGET_CONTEXT_STRONG if episode["cue_strong"] else TARGET_CONTEXT_WEAK


def make_prompt(episode: dict[str, Any], c_cases: int) -> str:
    if c_cases not in C_CASE_LEVELS:
        raise ValueError(c_cases)
    sections = [
        "CITY A\n"
        + reference_background(episode, "A")
        + "\n"
        + _table(episode["reference_a"]),
        "CITY B\n"
        + reference_background(episode, "B")
        + "\n"
        + _table(episode["reference_b"]),
        "CITY C\n"
        + target_background(episode)
        + "\n"
        + _table(episode["target"][:c_cases]),
    ]
    return (
        SHARED_PROMPT_INTRO
        + "\n\n"
        + "\n\n".join(sections)
        + "\n\n"
        + shared_prompt_closing(episode)
    )


def validate_arm_prompt(prompt_text: str, arm: str) -> None:
    """Contract for the misleading arm; other arms defer to the frozen check."""
    if arm != ARM:
        from engine.coin_city_stable_relationship_claude_n250 import (
            validate_arm_prompt as frozen_validate,
        )

        frozen_validate(prompt_text, arm)
        return
    if not prompt_text.startswith(SHARED_PROMPT_INTRO + "\n\n"):
        raise ValueError("shared introduction mismatch")
    if "CITY A\n" not in prompt_text or "CITY B\n" not in prompt_text:
        raise ValueError("reference cities missing")
    has_context = (
        TARGET_CONTEXT_STRONG in prompt_text or TARGET_CONTEXT_WEAK in prompt_text
    )
    if not has_context:
        raise ValueError("City C context missing")
    if TARGET_CONTEXT_NONE in prompt_text:
        raise ValueError("no-context sentence present in a context arm")
    lowered = prompt_text.lower()
    for forbidden in (
        "regression",
        "coefficient",
        "weighted average",
        "calculate a slope",
        "closer to city",
    ):
        if forbidden in lowered:
            raise ValueError(f"prompt prescribes estimator: {forbidden}")
