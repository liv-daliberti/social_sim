"""Structure-blind three-city world using independent cases, v3 calibration.

v2 used HIGH_RESPONSE 1.0, LOW_RESPONSE 0.25, SIGMA 3.25. That put the
per-case discriminability d' = (gap x news) / sigma at 1.85, which saturated
identification by k=3: the ceiling reached 0.93 there and only 0.997 by k=8, so
five of the nine rungs carried no information and a model could not visibly fall
short of the ceiling.

v3 targets d' = 1.12 by widening the noise rather than narrowing the gain gap.
Difficulty scales as gap/sigma but the accuracy headroom over a plain target
average scales as sigma, so narrowing the gap would have graded the curve at the
cost of ~40% of the effect size. Raising sigma grades it for free. The gain
values also drop the two artificial-looking properties of v2: 1.0 was exact
pass-through, and 4x was a caricatured ratio. 0.90/0.30 is 3x.

Every displayed row is a separate case with the same known starting poll:

    ending_poll = 50 + city_response * net_news + error

The equation and recurring response classes are evaluator-only. The prompt tells
the model only the non-latent sampling fact that cases are independent and begin
at 50. This removes the cumulative-state interpretation required by v1.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

STARTING_POLL = 50.0
HIGH_TYPE = "open_information"
LOW_TYPE = "buffered_information"
HIGH_RESPONSE = 0.90
LOW_RESPONSE = 0.30
RESPONSE_BY_TYPE = {
    HIGH_TYPE: HIGH_RESPONSE,
    LOW_TYPE: LOW_RESPONSE,
}
NEWS_VALUE = 8
REFERENCE_CASES = 6
# City C's ladder runs to eight cases. That is an interim upper anchor: the
# target-only frequentist still trails the Bayes ceiling by roughly 0.9 poll
# points here, because a two-point prior collapses exponentially while a sample
# mean only improves as 1/sqrt(k). Closing that gap to ~0.3 points needs about
# 64 cases; see PILOT_V2_AUDIT.md.
TARGET_CASES = 8
# The reported ladder: dense where the reference cities still carry weight,
# then doubling. Adjacent full-ladder rungs differ by less than the sampling
# error at 120 episodes, so curves and design gates use these rungs.
PREFIX_LADDER = (0, 1, 2, 3, 4, 6, 8)
SIGMA = 4.30
CONTEXT_RELIABILITY = 0.80
CONTEXT_CONDITIONS = ("none", "orthogonal", "cue_high", "cue_low")

REFERENCE_CONTEXTS = {
    HIGH_TYPE: (
        "Residents received national campaign developments through immediate "
        "alerts and unedited feeds, with few local intermediaries shaping what "
        "they saw.",
        "National political coverage reached residents through live reports and "
        "direct subscriptions, without waiting for a local editorial cycle.",
        "Campaign information spread through real-time national broadcasts and "
        "candidate feeds, with little filtering by local organizations.",
    ),
    LOW_TYPE: (
        "Residents learned about national campaign developments through "
        "neighborhood briefings, where local organizers selected and interpreted "
        "outside political news.",
        "National political coverage reached residents through community "
        "newsletters that summarized outside stories in terms of established "
        "local concerns.",
        "Campaign information spread through local civic groups that selected, "
        "delayed, and contextualized reports before sharing them with residents.",
    ),
}

TARGET_CONTEXTS = {
    HIGH_TYPE: (
        "Most voters followed national campaign events through instant radio "
        "updates and direct candidate feeds rather than neighborhood channels.",
        "Residents relied on real-time national news services that delivered "
        "political developments without local editing or delay.",
        "Voters received campaign updates from live national outlets and direct "
        "alerts, with few community gatekeepers involved.",
    ),
    LOW_TYPE: (
        "Most voters encountered national campaign events through community "
        "meetings where neighborhood leaders interpreted outside political news.",
        "Residents relied on local civic bulletins that selected and "
        "contextualized national political developments before circulating them.",
        "Voters received campaign updates through neighborhood networks that "
        "absorbed outside news slowly and related it to local concerns.",
    ),
}

ORTHOGONAL_CONTEXTS = (
    "The city expanded weekend bus service, renovated its central library, and "
    "installed new lighting in several public parks.",
    "The city replaced aging water mains, added protected bicycle lanes, and "
    "opened two recreation centers near downtown.",
    "The city revised its recycling schedule, restored a historic courthouse, "
    "and planted trees along several commercial streets.",
)


def other_type(city_type: str) -> str:
    if city_type == HIGH_TYPE:
        return LOW_TYPE
    if city_type == LOW_TYPE:
        return HIGH_TYPE
    raise ValueError(f"unknown city type: {city_type!r}")


def _clip(value: float) -> float:
    return max(0.0, min(100.0, value))


def generate_city(
    *,
    rng: random.Random,
    city_type: str,
    cases: int,
    sigma: float = SIGMA,
    news_value: int = NEWS_VALUE,
    center_reference_sample: bool = False,
) -> Dict[str, Any]:
    response = RESPONSE_BY_TYPE[city_type]
    news = [int(news_value)] * cases
    errors = [rng.gauss(0.0, sigma) for _ in news]
    if center_reference_sample:
        # Reference examples are deliberately selected to be representative:
        # their residuals retain dispersion but average to zero. This makes the
        # demonstrated city means unambiguous without changing target noise.
        mean_error = sum(errors) / len(errors)
        errors = [error - mean_error for error in errors]
    ending_polls = [
        round(
            _clip(
                STARTING_POLL
                + response * float(dose)
                + error
            ),
            1,
        )
        for dose, error in zip(news, errors)
    ]
    return {
        "starting_poll": STARTING_POLL,
        "city_type": city_type,
        "response": response,
        "news": news,
        "ending_polls": ending_polls,
    }


def generate_triplet(
    seed: Optional[int] = None,
    *,
    target_type: Optional[str] = None,
    high_reference: Optional[str] = None,
    context_family: Optional[int] = None,
    sigma: float = SIGMA,
) -> Dict[str, Any]:
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
    return {
        "seed": seed,
        "reference_a": reference_a,
        "reference_b": reference_b,
        "target": target,
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


def target_context(triplet: Mapping[str, Any], condition: str) -> str:
    if condition == "none":
        return ""
    if condition == "orthogonal":
        return str(triplet["target_context_orthogonal"])
    if condition == "cue_high":
        return str(triplet["target_context_high"])
    if condition == "cue_low":
        return str(triplet["target_context_low"])
    raise ValueError(f"unknown context condition: {condition!r}")


def _changes(city: Mapping[str, Any], k: int) -> List[float]:
    return [
        float(poll) - float(city["starting_poll"])
        for poll in city["ending_polls"][:k]
    ]


def _log_likelihood(
    news: Sequence[int],
    changes: Sequence[float],
    response: float,
    sigma: float,
) -> float:
    variance = sigma * sigma
    constant = math.log(2.0 * math.pi * variance)
    return -0.5 * sum(
        constant + ((change - response * dose) ** 2) / variance
        for dose, change in zip(news, changes)
    )


def context_prior_high(
    condition: str,
    reliability: float = CONTEXT_RELIABILITY,
) -> float:
    if not 0.5 <= reliability <= 1.0:
        raise ValueError("context reliability must be between 0.5 and 1")
    if condition in ("none", "orthogonal"):
        return 0.5
    if condition == "cue_high":
        return reliability
    if condition == "cue_low":
        return 1.0 - reliability
    raise ValueError(f"unknown context condition: {condition!r}")


def compute_bayes(
    triplet: Mapping[str, Any],
    k: int,
    *,
    condition: str = "none",
    use_context: bool = True,
    context_reliability: float = CONTEXT_RELIABILITY,
) -> Dict[str, Any]:
    target = triplet["target"]
    if not 0 <= k <= len(target["news"]):
        raise ValueError(f"k must be between 0 and {len(target['news'])}")
    prior_high = (
        context_prior_high(condition, context_reliability)
        if use_context
        else 0.5
    )
    news = target["news"][:k]
    changes = _changes(target, k)
    sigma = float(triplet["sigma"])
    ll_high = _log_likelihood(
        news,
        changes,
        HIGH_RESPONSE,
        sigma,
    )
    ll_low = _log_likelihood(
        news,
        changes,
        LOW_RESPONSE,
        sigma,
    )
    log_high = math.log(prior_high) + ll_high
    log_low = math.log(1.0 - prior_high) + ll_low
    maximum = max(log_high, log_low)
    weight_high = math.exp(log_high - maximum)
    weight_low = math.exp(log_low - maximum)
    p_high = weight_high / (weight_high + weight_low)
    response_hat = (
        p_high * HIGH_RESPONSE + (1.0 - p_high) * LOW_RESPONSE
    )
    forecast = _clip(
        STARTING_POLL + response_hat * float(triplet["test_news"])
    )
    a_is_high = triplet["reference_a"]["city_type"] == HIGH_TYPE
    p_match_a = p_high if a_is_high else 1.0 - p_high
    return {
        "k": k,
        "condition": condition,
        "prior_high": round(prior_high, 6),
        "p_high": round(p_high, 6),
        "p_match_a": round(p_match_a, 6),
        "response_hat": round(response_hat, 6),
        "predicted_poll": round(forecast, 3),
    }


def estimate_response(city: Mapping[str, Any], k: Optional[int] = None) -> float:
    if k is None:
        k = len(city["news"])
    news = city["news"][:k]
    changes = _changes(city, k)
    denominator = sum(float(dose) ** 2 for dose in news)
    return sum(
        float(dose) * change
        for dose, change in zip(news, changes)
    ) / denominator


def compute_frequentist(
    triplet: Mapping[str, Any],
    k: int,
) -> Dict[str, Any]:
    """Target-only sample estimate; no forecast exists before a target case."""
    if k == 0:
        return {
            "k": k,
            "available": False,
            "response_hat": None,
            "predicted_poll": None,
        }
    response_hat = estimate_response(triplet["target"], k)
    forecast = _clip(
        STARTING_POLL + response_hat * float(triplet["test_news"])
    )
    return {
        "k": k,
        "available": True,
        "response_hat": round(response_hat, 6),
        "predicted_poll": round(forecast, 3),
    }


def compute_pooled_exemplar(
    triplet: Mapping[str, Any],
    k: int,
) -> Dict[str, Any]:
    cities = (triplet["reference_a"], triplet["reference_b"])
    news = [dose for city in cities for dose in city["news"]]
    changes = [
        change
        for city in cities
        for change in _changes(city, len(city["news"]))
    ]
    denominator = sum(float(dose) ** 2 for dose in news)
    response_hat = sum(
        float(dose) * change
        for dose, change in zip(news, changes)
    ) / denominator
    forecast = _clip(
        STARTING_POLL + response_hat * float(triplet["test_news"])
    )
    return {
        "k": k,
        "response_hat": round(response_hat, 6),
        "predicted_poll": round(forecast, 3),
    }


def gold_expected_poll(triplet: Mapping[str, Any], k: int) -> float:
    """The noiseless end-of-case poll City C's hidden response implies.

    This is the scoring target, not a forecaster. It does not depend on `k`;
    the argument is kept so the answer key records which prefix it was joined
    to.
    """
    del k
    response = float(triplet["target"]["response"])
    return round(
        _clip(STARTING_POLL + response * float(triplet["test_news"])),
        3,
    )


def behavioral_response(
    triplet: Mapping[str, Any],
    predicted_poll: float,
) -> float:
    return (
        float(predicted_poll) - STARTING_POLL
    ) / float(triplet["test_news"])


def score_type(
    triplet: Mapping[str, Any],
    p_match_a: float,
) -> Dict[str, float]:
    truth_a = triplet["target_matches"] == "A"
    p_correct = p_match_a if truth_a else 1.0 - p_match_a
    return {
        "p_correct_type": round(p_correct, 6),
        "type_brier": round((p_match_a - float(truth_a)) ** 2, 6),
    }


def behavioral_type_score(
    triplet: Mapping[str, Any],
    response: float,
) -> float:
    true_response = float(triplet["target"]["response"])
    false_response = float(
        RESPONSE_BY_TYPE[other_type(triplet["target"]["city_type"])]
    )
    true_distance = abs(response - true_response)
    false_distance = abs(response - false_response)
    if abs(true_distance - false_distance) < 1e-12:
        return 0.5
    return float(true_distance < false_distance)


def balanced_triplets(
    n: int,
    *,
    seed_offset: int = 10_000,
) -> Iterable[Dict[str, Any]]:
    """Full 2 × 2 × 3 balance repeats exactly when n is a multiple of 12."""
    for index in range(n):
        yield generate_triplet(
            seed=seed_offset + index,
            target_type=HIGH_TYPE if index % 2 == 0 else LOW_TYPE,
            high_reference="A" if (index // 2) % 2 == 0 else "B",
            context_family=(index // 4) % len(ORTHOGONAL_CONTEXTS),
        )
