"""Figure-1-aligned three-city C2 world.

The numerical world and natural-language context families are the independently
audited v2 design. This module adds the two estimator-matched forecasters needed
by the replacement experiment:

* ``compute_target_only``: fit City C alone, with the demonstrated A/B midpoint
  as the explicit no-C-data fallback at k=0.
* ``compute_abc_shrinkage``: use the two displayed A/B response regressions as
  an empirical continuous prior, then update that prior with the same City C
  regression used by ``compute_target_only``.

The main structure curve never uses a target-type context cue. Context effects
are a separate crossed manipulation (none, orthogonal, cue-high, cue-low).
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional

from engine.three_city_c2_v2 import (
    CONTEXT_CONDITIONS,
    CONTEXT_RELIABILITY,
    HIGH_RESPONSE,
    HIGH_TYPE,
    LOW_RESPONSE,
    LOW_TYPE,
    NEWS_VALUE,
    ORTHOGONAL_CONTEXTS,
    REFERENCE_CASES,
    REFERENCE_CONTEXTS,
    RESPONSE_BY_TYPE,
    SIGMA,
    STARTING_POLL,
    TARGET_CONTEXTS,
    behavioral_response,
    behavioral_type_score,
    compute_bayes,
    generate_city,
    gold_expected_poll,
    other_type,
    target_context,
)

PROMPT_ARMS = ("blind", "hint", "strong_hint")
SEED_OFFSET = 80_000
TARGET_CASES = 16
PREFIX_LADDER = tuple(range(1, 6))
CASES_BY_PREFIX = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16}
PROFILE_CENTER_BY_TYPE = {HIGH_TYPE: 0.75, LOW_TYPE: 0.25}
PROFILE_SIGNAL_SD = 0.18
PROFILE_SEED_OFFSET = 500_000


def generate_triplet(
    seed: Optional[int] = None,
    *,
    target_type: Optional[str] = None,
    high_reference: Optional[str] = None,
    context_family: Optional[int] = None,
    sigma: float = SIGMA,
) -> Dict[str, Any]:
    """Generate five evidence rounds with up to sixteen City C cases.

    Cases 1--8 retain the audited v2 draw order; later cases extend the same
    seeded target stream. A separate seeded draw creates the graded profile
    signal without changing any polling case.
    """
    rng = random.Random(seed)
    if target_type is None:
        target_type = rng.choice((HIGH_TYPE, LOW_TYPE))
    if target_type not in RESPONSE_BY_TYPE:
        raise ValueError(f"unknown target_type: {target_type!r}")
    if high_reference is None:
        high_reference = rng.choice(("A", "B"))
    if high_reference not in ("A", "B"):
        raise ValueError("high_reference must be 'A' or 'B'")
    if context_family is None:
        context_family = rng.randrange(len(ORTHOGONAL_CONTEXTS))
    context_family = int(context_family) % len(ORTHOGONAL_CONTEXTS)

    type_a = HIGH_TYPE if high_reference == "A" else LOW_TYPE
    type_b = other_type(type_a)
    reference_a = generate_city(
        rng=rng,
        city_type=type_a,
        cases=REFERENCE_CASES,
        sigma=sigma,
        center_reference_sample=True,
    )
    reference_b = generate_city(
        rng=rng,
        city_type=type_b,
        cases=REFERENCE_CASES,
        sigma=sigma,
        center_reference_sample=True,
    )
    target = generate_city(
        rng=rng,
        city_type=target_type,
        cases=TARGET_CASES,
        sigma=sigma,
    )
    profile_seed = (
        None if seed is None else PROFILE_SEED_OFFSET + int(seed)
    )
    profile_rng = random.Random(profile_seed)
    profile_target = max(
        0.0,
        min(
            1.0,
            PROFILE_CENTER_BY_TYPE[target_type]
            + profile_rng.gauss(0.0, PROFILE_SIGNAL_SD),
        ),
    )
    return {
        "seed": seed,
        "reference_a": reference_a,
        "reference_b": reference_b,
        "target": target,
        "profile_index_a": 100.0 if type_a == HIGH_TYPE else 0.0,
        "profile_index_b": 100.0 if type_b == HIGH_TYPE else 0.0,
        "profile_index_target": round(100.0 * profile_target, 1),
        "target_matches": "A" if target_type == type_a else "B",
        "high_reference": high_reference,
        "test_news": NEWS_VALUE,
        "context_family": context_family,
        "reference_context_a": REFERENCE_CONTEXTS[type_a][context_family],
        "reference_context_b": REFERENCE_CONTEXTS[type_b][context_family],
        "target_context_high": TARGET_CONTEXTS[HIGH_TYPE][context_family],
        "target_context_low": TARGET_CONTEXTS[LOW_TYPE][context_family],
        "target_context_orthogonal": ORTHOGONAL_CONTEXTS[context_family],
        "sigma": float(sigma),
    }


def balanced_triplets(
    n: int,
    *,
    seed_offset: int = SEED_OFFSET,
) -> Iterable[Dict[str, Any]]:
    """Yield the balanced design with five evidence rounds and sixteen cases."""
    for index in range(n):
        yield generate_triplet(
            seed=seed_offset + index,
            target_type=HIGH_TYPE if index % 2 == 0 else LOW_TYPE,
            high_reference="A" if (index // 2) % 2 == 0 else "B",
            context_family=(index // 4) % len(ORTHOGONAL_CONTEXTS),
        )


def cases_for_prefix(k: int) -> int:
    """Return cumulative City C cases displayed in evidence round k."""
    try:
        return CASES_BY_PREFIX[int(k)]
    except (KeyError, ValueError):
        raise ValueError(f"unknown evidence round: {k!r}") from None


def _clip(value: float) -> float:
    return max(0.0, min(100.0, value))


def _changes(city: Mapping[str, Any], k: int) -> List[float]:
    return [
        float(poll) - float(city["starting_poll"])
        for poll in city["ending_polls"][:k]
    ]


def estimate_response(
    city: Mapping[str, Any],
    k: Optional[int] = None,
) -> float:
    """Through-origin response regression using displayed cases only."""
    if k is None:
        k = len(city["news"])
    if k <= 0:
        raise ValueError("at least one case is required")
    news = city["news"][:k]
    changes = _changes(city, k)
    denominator = sum(float(dose) ** 2 for dose in news)
    return sum(
        float(dose) * change
        for dose, change in zip(news, changes)
    ) / denominator


def estimate_reference_sigma(triplet: Mapping[str, Any]) -> float:
    """Estimate case noise after fitting one response slope per reference."""
    residuals: List[float] = []
    for key in ("reference_a", "reference_b"):
        city = triplet[key]
        response = estimate_response(city)
        residuals.extend(
            change - response * float(news)
            for news, change in zip(
                city["news"],
                _changes(city, len(city["news"])),
            )
        )
    degrees = max(1, len(residuals) - 2)
    return max(
        0.5,
        math.sqrt(sum(value * value for value in residuals) / degrees),
    )


def compute_reference_midpoint(
    triplet: Mapping[str, Any],
) -> Dict[str, Any]:
    """Midpoint of the two demonstrated response regressions."""
    response_a = estimate_response(triplet["reference_a"])
    response_b = estimate_response(triplet["reference_b"])
    response_hat = 0.5 * (response_a + response_b)
    return {
        "response_a": round(response_a, 6),
        "response_b": round(response_b, 6),
        "response_hat": round(response_hat, 6),
        "predicted_poll": round(
            _clip(
                STARTING_POLL
                + response_hat * float(triplet["test_news"])
            ),
            3,
        ),
    }


def compute_target_only(
    triplet: Mapping[str, Any],
    k: int,
) -> Dict[str, Any]:
    """Fit City C alone at the cumulative case count for evidence round k."""
    cases_seen = cases_for_prefix(k)
    response_hat = estimate_response(triplet["target"], cases_seen)
    return {
        "k": k,
        "cases_seen": cases_seen,
        "available": True,
        "fallback": None,
        "response_hat": round(response_hat, 6),
        "predicted_poll": round(
            _clip(
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
    """Continuous profile-informed shrinkage over the City C response.

    A and B define a prompt-visible linear relationship between their profile
    indices and fitted response slopes. The displayed City C index interpolates
    a continuous prior mean; it never selects an exact class. Prior variance is
    the variance of a uniform distribution spanning the two displayed slopes.
    The likelihood is the same City C regression used by
    ``compute_target_only``. As cumulative target evidence grows over rounds
    1--5, the A/B weight decreases and both estimators converge.
    """
    cases_seen = cases_for_prefix(k)
    response_a = estimate_response(triplet["reference_a"])
    response_b = estimate_response(triplet["reference_b"])
    profile_a = float(triplet["profile_index_a"]) / 100.0
    profile_b = float(triplet["profile_index_b"]) / 100.0
    profile_target = float(triplet["profile_index_target"]) / 100.0
    profile_span = profile_b - profile_a
    if abs(profile_span) < 1e-12:
        raise ValueError("reference profile indices must differ")
    interpolation = (profile_target - profile_a) / profile_span
    prior_mean = response_a + interpolation * (response_b - response_a)
    prior_variance = max(
        1e-6,
        (response_a - response_b) ** 2 / 12.0,
    )
    sigma_hat = estimate_reference_sigma(triplet)
    target_response = estimate_response(triplet["target"], cases_seen)
    target_information = sum(
        float(dose) ** 2
        for dose in triplet["target"]["news"][:cases_seen]
    ) / (sigma_hat * sigma_hat)
    prior_information = 1.0 / prior_variance
    target_weight = target_information / (
        target_information + prior_information
    )
    reference_weight = 1.0 - target_weight
    response_hat = (
        target_weight * target_response + reference_weight * prior_mean
    )
    return {
        "k": k,
        "cases_seen": cases_seen,
        "response_a": round(response_a, 6),
        "response_b": round(response_b, 6),
        "profile_index_a": round(100.0 * profile_a, 1),
        "profile_index_b": round(100.0 * profile_b, 1),
        "profile_index_target": round(100.0 * profile_target, 1),
        "prior_mean_response": round(prior_mean, 6),
        "prior_variance": round(prior_variance, 6),
        "sigma_hat": round(sigma_hat, 6),
        "target_only_response": round(target_response, 6),
        "reference_weight": round(reference_weight, 6),
        "target_weight": round(target_weight, 6),
        "response_hat": round(response_hat, 6),
        "predicted_poll": round(
            _clip(
                STARTING_POLL
                + response_hat * float(triplet["test_news"])
            ),
            3,
        ),
    }


def compute_context_oracle(
    triplet: Mapping[str, Any],
    k: int,
    *,
    condition: str,
) -> Dict[str, Any]:
    """Privileged rational context benchmark from the audited v2 design."""
    result = compute_bayes(
        triplet,
        cases_for_prefix(k),
        condition=condition,
        use_context=True,
        context_reliability=CONTEXT_RELIABILITY,
    )
    return {
        **result,
        "k": k,
        "cases_seen": cases_for_prefix(k),
        "privileged_context_reliability": CONTEXT_RELIABILITY,
    }
