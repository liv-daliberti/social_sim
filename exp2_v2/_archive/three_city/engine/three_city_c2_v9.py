"""Natural continuous-profile three-city forecasting world.

This version removes discrete city types. Each episode samples two diverse
reference-city profile scores and one target score between them. Expected poll
responses follow a noisy continuous linear relationship with the profile score,
and every city has an independent deviation from that relationship.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Mapping, Optional

PROMPT_ARMS = ("c_only", "abc", "abc_relevance")
SEED_OFFSET = 91_000
STARTING_POLL = 50.0
NEWS_VALUE = 8
REFERENCE_CASES = 8
TARGET_CASES = 16
PREFIX_LADDER = tuple(range(1, 6))
CASES_BY_PREFIX = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}

CASE_SIGMA = 4.0
CITY_DEVIATION_SD = 0.18
LOW_PROFILE_RANGE = (0.10, 0.35)
HIGH_PROFILE_RANGE = (0.65, 0.90)
TARGET_PROFILE_MARGIN = 0.05
EPISODE_INTERCEPT_RANGE = (0.30, 0.45)
EPISODE_PROFILE_SLOPE_RANGE = (0.35, 0.65)


def _clip_poll(value: float) -> float:
    return max(0.0, min(100.0, value))


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
            _clip_poll(
                STARTING_POLL
                + response * NEWS_VALUE
                + rng.gauss(0.0, CASE_SIGMA)
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
) -> Dict[str, Any]:
    """Generate one episode from the continuous noisy city relationship."""
    rng = random.Random(seed)
    low_profile = rng.uniform(*LOW_PROFILE_RANGE)
    high_profile = rng.uniform(*HIGH_PROFILE_RANGE)
    target_profile = rng.uniform(
        low_profile + TARGET_PROFILE_MARGIN,
        high_profile - TARGET_PROFILE_MARGIN,
    )
    intercept = rng.uniform(*EPISODE_INTERCEPT_RANGE)
    profile_slope = rng.uniform(*EPISODE_PROFILE_SLOPE_RANGE)

    low_response = (
        intercept
        + profile_slope * low_profile
        + rng.gauss(0.0, CITY_DEVIATION_SD)
    )
    high_response = (
        intercept
        + profile_slope * high_profile
        + rng.gauss(0.0, CITY_DEVIATION_SD)
    )
    target_response = (
        intercept
        + profile_slope * target_profile
        + rng.gauss(0.0, CITY_DEVIATION_SD)
    )

    low_city = _generate_city(
        rng,
        profile=low_profile,
        response=low_response,
        cases=REFERENCE_CASES,
    )
    high_city = _generate_city(
        rng,
        profile=high_profile,
        response=high_response,
        cases=REFERENCE_CASES,
    )
    target_city = _generate_city(
        rng,
        profile=target_profile,
        response=target_response,
        cases=TARGET_CASES,
    )
    if high_reference is None:
        high_reference = rng.choice(("A", "B"))
    if high_reference not in ("A", "B"):
        raise ValueError("high_reference must be 'A' or 'B'")
    reference_a, reference_b = (
        (high_city, low_city)
        if high_reference == "A"
        else (low_city, high_city)
    )
    return {
        "seed": seed,
        "reference_a": reference_a,
        "reference_b": reference_b,
        "target": target_city,
        "high_reference": high_reference,
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
    """Yield episodes balanced only on whether A or B has the higher score."""
    for index in range(n):
        yield generate_triplet(
            seed=seed_offset + index,
            high_reference="A" if index % 2 == 0 else "B",
        )


def cases_for_prefix(k: int) -> int:
    try:
        return CASES_BY_PREFIX[int(k)]
    except (KeyError, ValueError):
        raise ValueError(f"unknown evidence round: {k!r}") from None


def _changes(city: Mapping[str, Any], k: int) -> List[float]:
    return [
        float(poll) - float(city["starting_poll"])
        for poll in city["ending_polls"][:k]
    ]


def estimate_response(
    city: Mapping[str, Any],
    k: Optional[int] = None,
) -> float:
    """Fit the through-origin response slope from displayed cases."""
    if k is None:
        k = len(city["news"])
    if k <= 0:
        raise ValueError("at least one case is required")
    news = city["news"][:k]
    denominator = sum(float(value) ** 2 for value in news)
    return sum(
        float(value) * change
        for value, change in zip(news, _changes(city, k))
    ) / denominator


def compute_target_only(
    triplet: Mapping[str, Any],
    k: int,
) -> Dict[str, Any]:
    cases_seen = cases_for_prefix(k)
    response_hat = estimate_response(triplet["target"], cases_seen)
    return {
        "k": k,
        "cases_seen": cases_seen,
        "response_hat": round(response_hat, 6),
        "predicted_poll": round(
            _clip_poll(
                STARTING_POLL
                + response_hat * float(triplet["test_news"])
            ),
            3,
        ),
    }


def compute_reference_interpolation(
    triplet: Mapping[str, Any],
) -> Dict[str, Any]:
    """Interpolate the fitted A/B slopes at City C's continuous score."""
    city_a = triplet["reference_a"]
    city_b = triplet["reference_b"]
    response_a = estimate_response(city_a)
    response_b = estimate_response(city_b)
    profile_a = float(city_a["profile_index"]) / 100.0
    profile_b = float(city_b["profile_index"]) / 100.0
    profile_target = float(triplet["target"]["profile_index"]) / 100.0
    span = profile_b - profile_a
    if abs(span) < 1e-12:
        raise ValueError("reference profile scores must differ")
    interpolation = (profile_target - profile_a) / span
    response_hat = response_a + interpolation * (response_b - response_a)
    return {
        "response_a": round(response_a, 6),
        "response_b": round(response_b, 6),
        "profile_index_a": round(100.0 * profile_a, 1),
        "profile_index_b": round(100.0 * profile_b, 1),
        "profile_index_target": round(100.0 * profile_target, 1),
        "interpolation_weight_b": round(interpolation, 6),
        "response_hat": round(response_hat, 6),
        "predicted_poll": round(
            _clip_poll(
                STARTING_POLL
                + response_hat * float(triplet["test_news"])
            ),
            3,
        ),
    }


def compute_abc_shrinkage(
    triplet: Mapping[str, Any],
    k: int,
) -> Dict[str, Any]:
    """Matched posterior mean using the continuous A/B interpolation prior."""
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
        **prior,
        "prior_variance": round(prior_variance, 6),
        "target_variance": round(target_variance, 6),
        "reference_weight": round(reference_weight, 6),
        "target_weight": round(1.0 - reference_weight, 6),
        "target_only_response": round(target_response, 6),
        "response_hat": round(response_hat, 6),
        "predicted_poll": round(
            _clip_poll(
                STARTING_POLL
                + response_hat * float(triplet["test_news"])
            ),
            3,
        ),
    }


def gold_expected_poll(triplet: Mapping[str, Any]) -> float:
    return round(
        _clip_poll(
            STARTING_POLL
            + float(triplet["target"]["response"])
            * float(triplet["test_news"])
        ),
        3,
    )
