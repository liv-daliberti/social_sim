"""Noisier identifiable continuous-profile world for the v12 rerun."""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, Mapping, Optional

from engine import three_city_c2_v9 as v9

PROMPT_ARMS = ("c_only", "abc", "abc_structural_clue")
SEED_OFFSET = 93_000
STARTING_POLL = 50.0
NEWS_VALUE = 8
REFERENCE_CASES = 8
TARGET_CASES = 16
PREFIX_LADDER = tuple(range(1, 6))
CASES_BY_PREFIX = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}

CASE_SIGMA = 6.0
CITY_DEVIATION_SD = 0.16
LOW_PROFILE_RANGE = (0.10, 0.30)
HIGH_PROFILE_RANGE = (0.70, 0.90)
TARGET_NEAR_ENDPOINT_RANGE = (0.08, 0.22)
EPISODE_INTERCEPT_RANGE = (0.20, 0.38)
EPISODE_PROFILE_SLOPE_RANGE = (1.55, 1.95)


def _generate_city(
    rng: random.Random,
    *,
    profile: float,
    response: float,
    cases: int,
) -> Dict[str, Any]:
    news = [NEWS_VALUE] * cases
    ending_polls = [
        round(
            max(
                0.0,
                min(
                    100.0,
                    STARTING_POLL
                    + response * NEWS_VALUE
                    + rng.gauss(0.0, CASE_SIGMA),
                ),
            ),
            1,
        )
        for _ in range(cases)
    ]
    return {
        "profile_index": round(100.0 * profile, 1),
        "starting_poll": STARTING_POLL,
        "news": news,
        "ending_polls": ending_polls,
        "response": response,
    }


def generate_triplet(
    seed: Optional[int] = None,
    *,
    high_reference: Optional[str] = None,
    target_near_high: Optional[bool] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    low_profile = rng.uniform(*LOW_PROFILE_RANGE)
    high_profile = rng.uniform(*HIGH_PROFILE_RANGE)
    if target_near_high is None:
        target_near_high = bool(rng.randrange(2))
    endpoint_offset = rng.uniform(*TARGET_NEAR_ENDPOINT_RANGE)
    target_position = (
        1.0 - endpoint_offset if target_near_high else endpoint_offset
    )
    target_profile = low_profile + target_position * (
        high_profile - low_profile
    )
    intercept = rng.uniform(*EPISODE_INTERCEPT_RANGE)
    profile_slope = rng.uniform(*EPISODE_PROFILE_SLOPE_RANGE)
    low_response = (
        intercept + profile_slope * low_profile
        + rng.gauss(0.0, CITY_DEVIATION_SD)
    )
    high_response = (
        intercept + profile_slope * high_profile
        + rng.gauss(0.0, CITY_DEVIATION_SD)
    )
    target_response = (
        intercept + profile_slope * target_profile
        + rng.gauss(0.0, CITY_DEVIATION_SD)
    )
    low_city = _generate_city(
        rng, profile=low_profile, response=low_response, cases=REFERENCE_CASES
    )
    high_city = _generate_city(
        rng, profile=high_profile, response=high_response, cases=REFERENCE_CASES
    )
    target_city = _generate_city(
        rng, profile=target_profile, response=target_response, cases=TARGET_CASES
    )
    if high_reference is None:
        high_reference = rng.choice(("A", "B"))
    if high_reference not in ("A", "B"):
        raise ValueError("high_reference must be 'A' or 'B'")
    reference_a, reference_b = (
        (high_city, low_city) if high_reference == "A" else (low_city, high_city)
    )
    target_nearer_reference = (
        high_reference
        if target_near_high
        else ("B" if high_reference == "A" else "A")
    )
    return {
        "seed": seed,
        "reference_a": reference_a,
        "reference_b": reference_b,
        "target": target_city,
        "high_reference": high_reference,
        "target_near_high": target_near_high,
        "target_nearer_reference": target_nearer_reference,
        "target_profile_position": target_position,
        "test_news": NEWS_VALUE,
        "sigma": CASE_SIGMA,
        "city_deviation_sd": CITY_DEVIATION_SD,
        "latent_intercept": intercept,
        "latent_profile_slope": profile_slope,
    }


def balanced_triplets(
    n: int,
    *,
    seed_offset: int = SEED_OFFSET,
) -> Iterable[Dict[str, Any]]:
    for index in range(n):
        yield generate_triplet(
            seed=seed_offset + index,
            high_reference="A" if index % 2 == 0 else "B",
            target_near_high=(index // 2) % 2 == 0,
        )


cases_for_prefix = v9.cases_for_prefix
estimate_response = v9.estimate_response
compute_target_only = v9.compute_target_only
compute_reference_interpolation = v9.compute_reference_interpolation
gold_expected_poll = v9.gold_expected_poll


def compute_abc_shrinkage(
    triplet: Mapping[str, Any],
    k: int,
) -> Dict[str, Any]:
    cases_seen = cases_for_prefix(k)
    prior = compute_reference_interpolation(triplet)
    interpolation = float(prior["interpolation_weight_b"])
    reference_factor = (1.0 - interpolation) ** 2 + interpolation ** 2
    prior_variance = (
        CITY_DEVIATION_SD ** 2 * (1.0 + reference_factor)
        + CASE_SIGMA ** 2
        / (REFERENCE_CASES * NEWS_VALUE ** 2)
        * reference_factor
    )
    target_variance = CASE_SIGMA ** 2 / (
        cases_seen * NEWS_VALUE ** 2
    )
    reference_weight = target_variance / (
        target_variance + prior_variance
    )
    target_response = estimate_response(triplet["target"], cases_seen)
    response_hat = (
        reference_weight * float(prior["response_hat"])
        + (1.0 - reference_weight) * target_response
    )
    return {
        "k": k,
        "cases_seen": cases_seen,
        "reference_weight": round(reference_weight, 6),
        "target_weight": round(1.0 - reference_weight, 6),
        "prior_variance": round(prior_variance, 8),
        "target_variance": round(target_variance, 8),
        "prior_response_hat": prior["response_hat"],
        "target_response_hat": round(target_response, 6),
        "response_hat": round(response_hat, 6),
        "predicted_poll": round(
            max(
                0.0,
                min(
                    100.0,
                    STARTING_POLL
                    + response_hat * float(triplet["test_news"]),
                ),
            ),
            3,
        ),
    }
