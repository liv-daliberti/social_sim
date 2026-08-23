"""Two-regime structural-choice world for the C2 v13 experiment.

The neutral A/B/C estimator partially pools City C toward the symmetric A/B
average.  The structural estimator uses the same amount of pooling but centers
its prior on the reference selected by profile proximity.  Consequently the
estimators differ only in structural information, and both converge toward the
City-C-only estimate as City C evidence accumulates.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, Mapping, Optional

PROMPT_ARMS = ("c_only", "abc", "abc_structural_clue")
SEED_OFFSET = 105_000
STARTING_POLL = 50.0
TEST_NEWS = 8
REFERENCE_CASES = 2
TARGET_CASES = 16
PREFIX_LADDER = tuple(range(1, 6))
CASES_BY_PREFIX = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}

# Each observation supplies one unit of test-news-equivalent information.
CASE_SIGMA = 9.5
CITY_DEVIATION_SD = 0.10
PRIOR_EFFECTIVE_CASES = 2.0

LOW_PROFILE_RANGE = (0.12, 0.28)
HIGH_PROFILE_RANGE = (0.72, 0.88)
TARGET_NEAR_ENDPOINT_RANGE = (0.06, 0.18)
EPISODE_RESPONSE_CENTER_RANGE = (0.95, 1.15)
EPISODE_REGIME_GAP_RANGE = (1.70, 2.10)


def _clip_poll(value: float) -> float:
    return max(0.0, min(100.0, value))


def _news_pattern(cases: int) -> list[int]:
    """Alternate equally informative favorable and unfavorable news cases."""
    return [TEST_NEWS if index % 2 == 0 else -TEST_NEWS for index in range(cases)]


def _generate_city(
    rng: random.Random,
    *,
    profile: float,
    response: float,
    cases: int,
) -> Dict[str, Any]:
    news = _news_pattern(cases)
    ending_polls = [
        round(
            _clip_poll(
                STARTING_POLL
                + response * dose
                + rng.gauss(0.0, CASE_SIGMA)
            ),
            1,
        )
        for dose in news
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

    response_center = rng.uniform(*EPISODE_RESPONSE_CENTER_RANGE)
    regime_gap = rng.uniform(*EPISODE_REGIME_GAP_RANGE)
    low_regime_response = response_center - regime_gap / 2.0
    high_regime_response = response_center + regime_gap / 2.0
    low_response = low_regime_response + rng.gauss(0.0, CITY_DEVIATION_SD)
    high_response = high_regime_response + rng.gauss(0.0, CITY_DEVIATION_SD)
    target_regime_response = (
        high_regime_response if target_near_high else low_regime_response
    )
    target_response = target_regime_response + rng.gauss(
        0.0, CITY_DEVIATION_SD
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
    relevant_reference = (
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
        "relevant_reference": relevant_reference,
        "target_profile_position": target_position,
        "test_news": TEST_NEWS,
        "sigma": CASE_SIGMA,
        "city_deviation_sd": CITY_DEVIATION_SD,
        "latent_response_center": response_center,
        "latent_regime_gap": regime_gap,
    }


def balanced_triplets(
    n: int,
    *,
    seed_offset: int = SEED_OFFSET,
) -> Iterable[Dict[str, Any]]:
    """Balance A/B labels and the target's high/low response regime."""
    for index in range(n):
        yield generate_triplet(
            seed=seed_offset + index,
            high_reference="A" if index % 2 == 0 else "B",
            target_near_high=(index // 2) % 2 == 0,
        )


def cases_for_prefix(k: int) -> int:
    try:
        return CASES_BY_PREFIX[int(k)]
    except (KeyError, ValueError):
        raise ValueError(f"unknown evidence round: {k!r}") from None


def estimate_response(
    city: Mapping[str, Any],
    cases: Optional[int] = None,
) -> float:
    """Fit the through-origin news-response slope from displayed cases."""
    if cases is None:
        cases = len(city["news"])
    if cases <= 0:
        raise ValueError("at least one case is required")
    news = [float(value) for value in city["news"][:cases]]
    changes = [
        float(poll) - float(city["starting_poll"])
        for poll in city["ending_polls"][:cases]
    ]
    denominator = sum(value * value for value in news)
    return sum(value * change for value, change in zip(news, changes)) / denominator


def _prediction(response_hat: float, test_news: float = TEST_NEWS) -> float:
    return round(_clip_poll(STARTING_POLL + response_hat * test_news), 3)


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
        "predicted_poll": _prediction(response_hat, float(triplet["test_news"])),
    }


def compute_reference_estimates(triplet: Mapping[str, Any]) -> Dict[str, Any]:
    response_a = estimate_response(triplet["reference_a"])
    response_b = estimate_response(triplet["reference_b"])
    profile_a = float(triplet["reference_a"]["profile_index"])
    profile_b = float(triplet["reference_b"]["profile_index"])
    profile_c = float(triplet["target"]["profile_index"])
    selected = "A" if abs(profile_c - profile_a) < abs(profile_c - profile_b) else "B"
    return {
        "response_a": round(response_a, 6),
        "response_b": round(response_b, 6),
        "symmetric_prior_response": round((response_a + response_b) / 2.0, 6),
        "profile_selected_reference": selected,
        "selected_prior_response": round(response_a if selected == "A" else response_b, 6),
        "displayed_poll_separation": round(TEST_NEWS * abs(response_a - response_b), 6),
    }


def _compute_partial_pooling(
    triplet: Mapping[str, Any],
    k: int,
    *,
    structural_choice: bool,
) -> Dict[str, Any]:
    cases_seen = cases_for_prefix(k)
    references = compute_reference_estimates(triplet)
    prior_response = float(
        references[
            "selected_prior_response" if structural_choice else "symmetric_prior_response"
        ]
    )
    target_response = estimate_response(triplet["target"], cases_seen)
    prior_weight = PRIOR_EFFECTIVE_CASES / (PRIOR_EFFECTIVE_CASES + cases_seen)
    target_weight = 1.0 - prior_weight
    response_hat = prior_weight * prior_response + target_weight * target_response
    return {
        "k": k,
        "cases_seen": cases_seen,
        "prior_effective_cases": PRIOR_EFFECTIVE_CASES,
        "prior_weight": round(prior_weight, 6),
        "target_weight": round(target_weight, 6),
        "prior_kind": "profile_selected_reference" if structural_choice else "symmetric_ab_average",
        "profile_selected_reference": references["profile_selected_reference"],
        "prior_response_hat": round(prior_response, 6),
        "target_response_hat": round(target_response, 6),
        "response_hat": round(response_hat, 6),
        "predicted_poll": _prediction(response_hat, float(triplet["test_news"])),
    }


def compute_abc_no_structure(
    triplet: Mapping[str, Any],
    k: int,
) -> Dict[str, Any]:
    """Partial pooling centered on the A/B average, with no regime choice."""
    return _compute_partial_pooling(triplet, k, structural_choice=False)


def compute_abc_structural(
    triplet: Mapping[str, Any],
    k: int,
) -> Dict[str, Any]:
    """Same-strength pooling centered on the profile-selected reference."""
    return _compute_partial_pooling(triplet, k, structural_choice=True)


def gold_expected_poll(triplet: Mapping[str, Any]) -> float:
    return round(
        _clip_poll(
            STARTING_POLL
            + float(triplet["target"]["response"])
            * float(triplet["test_news"])
        ),
        3,
    )
