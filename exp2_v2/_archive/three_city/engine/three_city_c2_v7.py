"""Paper-ready three-city C2 world.

The task deliberately keeps v3's easy arithmetic. Every displayed row is an
independent case with the same +8 net-news value:

    ending_poll = 50 + city_response * 8 + error

The target-only estimate is therefore just City C's mean ending poll. That is a
feature, not a shortcut: it makes the statistical alternative maximally easy so
the experiment isolates whether a forecaster combines City C's sparse evidence
with the two reference-city mechanisms.

The model is never told that there are two response regimes, that City C shares
one of the demonstrated responses, the response values, the equation, or a
prior. Mechanism-relevant target backgrounds are always truthful. Orthogonal
backgrounds are length-matched and carry no information.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

STARTING_POLL = 50.0
NEWS_VALUE = 8
HIGH_TYPE = "open_information"
LOW_TYPE = "buffered_information"
HIGH_RESPONSE = 0.90
LOW_RESPONSE = 0.30
RESPONSE_BY_TYPE = {
    HIGH_TYPE: HIGH_RESPONSE,
    LOW_TYPE: LOW_RESPONSE,
}
REFERENCE_CASES = 6
TARGET_CASES = 4
PREFIX_LADDER = (0, 1, 2, 4)
SIGMA = 4.30
CONTEXT_CONDITIONS = ("relevant", "none", "orthogonal")
CONTEXT_PSEUDO_CASES = 1.0
CONTEXT_SENSITIVITY_WEIGHTS = (0.0, 0.5, 1.0, 2.0)

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

# Frozen, analyst-coded semantic features. These operate only on the background
# sentences shown in the prompt. They keep the public hierarchical reference
# from consulting City C's hidden type or the evaluator's target-match label.
_DIRECT_CONTEXT_CUES = (
    "immediate alerts",
    "unedited feeds",
    "direct subscriptions",
    "live reports",
    "real-time",
    "national broadcasts",
    "candidate feeds",
    "instant radio",
    "direct candidate",
    "without local editing",
    "live national",
    "direct alerts",
    "few local intermediaries",
    "little filtering",
    "few community gatekeepers",
)
_BUFFERED_CONTEXT_CUES = (
    "neighborhood briefings",
    "local organizers",
    "selected and interpreted",
    "community newsletters",
    "summarized outside",
    "local civic groups",
    "selected, delayed",
    "community meetings",
    "neighborhood leaders",
    "local civic bulletins",
    "selected and contextualized",
    "neighborhood networks",
    "absorbed outside news slowly",
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
    center_reference_sample: bool = False,
) -> Dict[str, Any]:
    """Generate independent cases for one city.

    Reference demonstrations retain dispersion but are centered on their
    expected response. This makes A and B unambiguous exemplars while City C
    remains an ordinary noisy sample. The selection is recorded in the frozen
    manifest and must be disclosed in the paper.
    """
    response = RESPONSE_BY_TYPE[city_type]
    news = [NEWS_VALUE] * cases
    errors = [rng.gauss(0.0, sigma) for _ in news]
    if center_reference_sample:
        mean_error = sum(errors) / len(errors)
        errors = [error - mean_error for error in errors]
    ending_polls = [
        round(
            _clip(STARTING_POLL + response * float(dose) + error),
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
        raise ValueError(f"unknown target type: {target_type!r}")
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
    target_matches = "A" if target_type == type_a else "B"
    return {
        "seed": seed,
        "reference_a": reference_a,
        "reference_b": reference_b,
        "target": target,
        "target_matches": target_matches,
        "high_reference": high_reference,
        "test_news": NEWS_VALUE,
        "context_family": context_family,
        "reference_context_a": REFERENCE_CONTEXTS[type_a][context_family],
        "reference_context_b": REFERENCE_CONTEXTS[type_b][context_family],
        "target_context_relevant": TARGET_CONTEXTS[target_type][context_family],
        "target_context_orthogonal": ORTHOGONAL_CONTEXTS[context_family],
        "sigma": float(sigma),
    }


def target_context(triplet: Mapping[str, Any], condition: str) -> str:
    if condition == "relevant":
        return str(triplet["target_context_relevant"])
    if condition == "orthogonal":
        return str(triplet["target_context_orthogonal"])
    if condition == "none":
        return ""
    raise ValueError(f"unknown context condition: {condition!r}")


def infer_context_mechanism(background: str) -> str:
    """Classify a displayed background using a frozen public-text codebook."""
    lowered = " ".join(str(background).lower().split())
    direct_score = sum(cue in lowered for cue in _DIRECT_CONTEXT_CUES)
    buffered_score = sum(cue in lowered for cue in _BUFFERED_CONTEXT_CUES)
    if direct_score == buffered_score:
        raise ValueError(
            "background is ambiguous under the frozen semantic codebook"
        )
    return HIGH_TYPE if direct_score > buffered_score else LOW_TYPE


def infer_public_context_match(triplet: Mapping[str, Any]) -> str:
    """Match City C to A or B using only the three displayed backgrounds."""
    target_mechanism = infer_context_mechanism(
        str(triplet["target_context_relevant"])
    )
    reference_mechanisms = {
        "A": infer_context_mechanism(str(triplet["reference_context_a"])),
        "B": infer_context_mechanism(str(triplet["reference_context_b"])),
    }
    matches = [
        label
        for label, mechanism in reference_mechanisms.items()
        if mechanism == target_mechanism
    ]
    if len(matches) != 1:
        raise ValueError(
            "public backgrounds do not identify exactly one reference match"
        )
    return matches[0]


def _changes(city: Mapping[str, Any], k: int) -> List[float]:
    return [
        float(poll) - float(city["starting_poll"])
        for poll in city["ending_polls"][:k]
    ]


def estimate_response(
    city: Mapping[str, Any],
    k: Optional[int] = None,
) -> float:
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
    """Estimate case noise from only the prompt-visible A/B records."""
    residuals: List[float] = []
    for city_name in ("reference_a", "reference_b"):
        city = triplet[city_name]
        response = estimate_response(city)
        residuals.extend(
            change - response * float(dose)
            for dose, change in zip(
                city["news"],
                _changes(city, len(city["news"])),
            )
        )
    degrees = max(1, len(residuals) - 2)
    return max(0.5, math.sqrt(sum(x * x for x in residuals) / degrees))


def _log_likelihood(
    news: Sequence[int],
    changes: Sequence[float],
    response: float,
    sigma: float,
) -> float:
    variance = sigma * sigma
    return -0.5 * sum(
        ((change - response * float(dose)) ** 2) / variance
        for dose, change in zip(news, changes)
    )


def compute_naive(triplet: Mapping[str, Any], k: int) -> Dict[str, Any]:
    """Ignore all city-specific information and use the population midpoint."""
    del k
    response_hat = (HIGH_RESPONSE + LOW_RESPONSE) / 2.0
    return {
        "response_hat": round(response_hat, 6),
        "predicted_poll": round(
            _clip(STARTING_POLL + response_hat * triplet["test_news"]),
            3,
        ),
    }


def compute_target_only(
    triplet: Mapping[str, Any],
    k: int,
) -> Dict[str, Any]:
    """Use City C alone, with the population midpoint as the k=0 fallback."""
    if k == 0:
        result = compute_naive(triplet, k)
        return {"k": k, "available": False, **result}
    response_hat = estimate_response(triplet["target"], k)
    return {
        "k": k,
        "available": True,
        "response_hat": round(response_hat, 6),
        "predicted_poll": round(
            _clip(STARTING_POLL + response_hat * triplet["test_news"]),
            3,
        ),
    }


def compute_pooled_references(
    triplet: Mapping[str, Any],
    k: int,
) -> Dict[str, Any]:
    """Pool A and B, ignoring their distinct response patterns and City C."""
    del k
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
    return {
        "response_hat": round(response_hat, 6),
        "predicted_poll": round(
            _clip(STARTING_POLL + response_hat * triplet["test_news"]),
            3,
        ),
    }


def compute_empirical_hierarchical(
    triplet: Mapping[str, Any],
    k: int,
    *,
    condition: str,
    context_pseudo_cases: float = CONTEXT_PSEUDO_CASES,
) -> Dict[str, Any]:
    """Public-data two-prototype reference.

    A and B's responses and the residual scale are estimated only from displayed
    records. With no or orthogonal background, A and B begin equally weighted.
    A truthful relevant background contributes one transparent pseudo-case at
    the semantically matching reference response. This is a diagnostic
    same-information reference, not a claim that one pseudo-case is uniquely
    optimal.
    """
    if condition not in CONTEXT_CONDITIONS:
        raise ValueError(f"unknown context condition: {condition!r}")
    if context_pseudo_cases < 0:
        raise ValueError("context_pseudo_cases must be nonnegative")
    response_a = estimate_response(triplet["reference_a"])
    response_b = estimate_response(triplet["reference_b"])
    sigma = estimate_reference_sigma(triplet)
    target = triplet["target"]
    news = target["news"][:k]
    changes = _changes(target, k)
    log_a = _log_likelihood(news, changes, response_a, sigma)
    log_b = _log_likelihood(news, changes, response_b, sigma)

    if condition == "relevant" and context_pseudo_cases:
        match_a = infer_public_context_match(triplet) == "A"
        matched_response = response_a if match_a else response_b
        pseudo_change = matched_response * float(triplet["test_news"])
        pseudo_news = [int(triplet["test_news"])]
        pseudo_changes = [pseudo_change]
        log_a += context_pseudo_cases * _log_likelihood(
            pseudo_news,
            pseudo_changes,
            response_a,
            sigma,
        )
        log_b += context_pseudo_cases * _log_likelihood(
            pseudo_news,
            pseudo_changes,
            response_b,
            sigma,
        )

    maximum = max(log_a, log_b)
    weight_a = math.exp(log_a - maximum)
    weight_b = math.exp(log_b - maximum)
    p_match_a = weight_a / (weight_a + weight_b)
    response_hat = p_match_a * response_a + (1.0 - p_match_a) * response_b
    return {
        "k": k,
        "condition": condition,
        "context_pseudo_cases": context_pseudo_cases,
        "response_a": round(response_a, 6),
        "response_b": round(response_b, 6),
        "sigma_hat": round(sigma, 6),
        "p_match_a": round(p_match_a, 6),
        "response_hat": round(response_hat, 6),
        "predicted_poll": round(
            _clip(STARTING_POLL + response_hat * triplet["test_news"]),
            3,
        ),
    }


def gold_expected_poll(triplet: Mapping[str, Any], k: int) -> float:
    del k
    response = float(triplet["target"]["response"])
    return round(
        _clip(STARTING_POLL + response * float(triplet["test_news"])),
        3,
    )


def compute_dgp_oracle(triplet: Mapping[str, Any], k: int) -> Dict[str, Any]:
    """Privileged ceiling that knows City C's hidden response type."""
    return {
        "k": k,
        "privileged": True,
        "response_hat": float(triplet["target"]["response"]),
        "predicted_poll": gold_expected_poll(triplet, k),
    }


def behavioral_response(
    triplet: Mapping[str, Any],
    predicted_poll: float,
) -> float:
    return (
        float(predicted_poll) - STARTING_POLL
    ) / float(triplet["test_news"])


def balanced_triplets(
    n: int,
    *,
    seed_offset: int = 70_000,
) -> Iterable[Dict[str, Any]]:
    """Exact 2 × 2 × 3 balance when n is a multiple of twelve."""
    if n <= 0 or n % 12:
        raise ValueError("n must be a positive multiple of 12")
    for index in range(n):
        yield generate_triplet(
            seed=seed_offset + index,
            target_type=HIGH_TYPE if index % 2 == 0 else LOW_TYPE,
            high_reference="A" if (index // 2) % 2 == 0 else "B",
            context_family=(index // 4) % len(ORTHOGONAL_CONTEXTS),
        )
