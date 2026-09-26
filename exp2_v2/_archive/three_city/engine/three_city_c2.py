"""C2-valid three-city world: controlled recovery plus contextual induction.

This module deliberately simplifies the pilot world.  The current poll is the
fully observed state:

    poll_t = poll_(t-1) + g_z * news_t + epsilon_t

Only the stable city mechanism ``z`` is hidden.  Two reference cities illustrate
the recurring mechanisms; City C receives weak evidence first and diagnostic
evidence second.  The same numerical episode can be rendered with no context,
orthogonal context, a high-response cue, or a buffered-response cue.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

INITIAL_POLL = 50.0
HIGH_TYPE = "open_information"
LOW_TYPE = "buffered_information"
HIGH_GAIN = 1.0
LOW_GAIN = 0.25
GAIN_BY_TYPE = {HIGH_TYPE: HIGH_GAIN, LOW_TYPE: LOW_GAIN}
SIGMA_TARGET = 2.0
SIGMA_REFERENCE = 0.75
CONTEXT_RELIABILITY = 0.80
REFERENCE_NEWS = (10, -10, 12, -12, 8, -8)
TARGET_MAGNITUDES = (2, 12, 10)
CONTEXT_CONDITIONS = ("none", "orthogonal", "cue_high", "cue_low")

REFERENCE_CONTEXTS = {
    HIGH_TYPE: (
        "Most residents received campaign developments through real-time national "
        "alerts, and local outlets rebroadcast the same stories with little delay.",
        "National political coverage reached residents directly through live feeds; "
        "local media rarely altered or delayed those reports.",
        "The city's dominant news services distributed national campaign updates "
        "immediately and with minimal local filtering.",
    ),
    LOW_TYPE: (
        "Residents relied mainly on locally curated bulletins; national campaign "
        "stories were usually summarized after local editors had filtered them.",
        "Neighborhood news organizations interpreted outside political stories "
        "through local issues before most residents encountered them.",
        "National campaign coverage reached voters mostly through local civic media "
        "that selected, delayed, and reframed outside reports.",
    ),
}

TARGET_CONTEXTS = {
    HIGH_TYPE: (
        "Commuters mostly followed campaign events through live national radio and "
        "unedited candidate feeds rather than neighborhood intermediaries.",
        "Residents used real-time national news aggregators that delivered political "
        "developments without waiting for a local editorial cycle.",
        "Campaign information spread through direct subscriptions to national outlets "
        "and instant alerts, with few local gatekeepers.",
    ),
    LOW_TYPE: (
        "Civic associations and neighborhood newsletters were residents' main "
        "information channels, so outside campaign stories arrived through local interpretation.",
        "Most political information circulated through community briefings that "
        "selected and contextualized national stories for local audiences.",
        "Residents depended on local discussion networks that absorbed outside news "
        "slowly and interpreted it through established community concerns.",
    ),
}

ORTHOGONAL_CONTEXTS = (
    "The city recently expanded weekend bus service and renovated its central library.",
    "The municipal government introduced a new recycling schedule and opened two public parks.",
    "A downtown streetscape project added bike lanes and replaced aging water mains.",
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
    news: Sequence[int],
    sigma: float,
) -> Dict[str, Any]:
    gain = GAIN_BY_TYPE[city_type]
    polls: List[float] = []
    current = INITIAL_POLL
    for dose in news:
        current = _clip(current + gain * dose + rng.gauss(0.0, sigma))
        current = round(current, 1)
        polls.append(current)
    return {
        "initial_poll": INITIAL_POLL,
        "city_type": city_type,
        "gain": gain,
        "news": list(news),
        "polls": polls,
    }


def generate_triplet(
    seed: Optional[int] = None,
    *,
    target_type: Optional[str] = None,
    high_reference: Optional[str] = None,
    target_sign: Optional[int] = None,
    context_family: Optional[int] = None,
    sigma_target: float = SIGMA_TARGET,
    sigma_reference: float = SIGMA_REFERENCE,
) -> Dict[str, Any]:
    """Generate a label-balanced-capable three-city episode.

    Optional explicit arguments let a dataset builder stratify type, A/B label,
    shock sign, and context paraphrase rather than trusting random balance.
    """
    rng = random.Random(seed)
    if target_type is None:
        target_type = rng.choice((HIGH_TYPE, LOW_TYPE))
    if target_type not in GAIN_BY_TYPE:
        raise ValueError(f"unknown target_type: {target_type!r}")
    if high_reference is None:
        high_reference = rng.choice(("A", "B"))
    if high_reference not in ("A", "B"):
        raise ValueError("high_reference must be 'A' or 'B'")
    if target_sign is None:
        target_sign = rng.choice((-1, 1))
    if target_sign not in (-1, 1):
        raise ValueError("target_sign must be -1 or 1")
    if context_family is None:
        context_family = rng.randrange(len(ORTHOGONAL_CONTEXTS))
    context_family = int(context_family) % len(ORTHOGONAL_CONTEXTS)

    type_a = HIGH_TYPE if high_reference == "A" else LOW_TYPE
    type_b = other_type(type_a)
    target_news = [
        target_sign * TARGET_MAGNITUDES[0],
        target_sign * TARGET_MAGNITUDES[1],
        -target_sign * TARGET_MAGNITUDES[2],
    ]
    ref_news_a = list(REFERENCE_NEWS)
    ref_news_b = list(reversed(REFERENCE_NEWS))
    if rng.random() < 0.5:
        ref_news_a = [-x for x in ref_news_a]
    if rng.random() < 0.5:
        ref_news_b = [-x for x in ref_news_b]

    reference_a = generate_city(
        rng=rng,
        city_type=type_a,
        news=ref_news_a,
        sigma=sigma_reference,
    )
    reference_b = generate_city(
        rng=rng,
        city_type=type_b,
        news=ref_news_b,
        sigma=sigma_reference,
    )
    target = generate_city(
        rng=rng,
        city_type=target_type,
        news=target_news,
        sigma=sigma_target,
    )
    target_matches = "A" if target_type == type_a else "B"
    test_news = -target_sign * 10

    return {
        "seed": seed,
        "reference_a": reference_a,
        "reference_b": reference_b,
        "target": target,
        "target_matches": target_matches,
        "high_reference": high_reference,
        "test_news": test_news,
        "context_family": context_family,
        "reference_context_a": REFERENCE_CONTEXTS[type_a][context_family],
        "reference_context_b": REFERENCE_CONTEXTS[type_b][context_family],
        "target_context_high": TARGET_CONTEXTS[HIGH_TYPE][context_family],
        "target_context_low": TARGET_CONTEXTS[LOW_TYPE][context_family],
        "target_context_orthogonal": ORTHOGONAL_CONTEXTS[context_family],
        "sigma_target": float(sigma_target),
        "sigma_reference": float(sigma_reference),
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


def _increments(city: Mapping[str, Any], k: int) -> List[float]:
    previous = float(city["initial_poll"])
    out = []
    for poll in city["polls"][:k]:
        out.append(float(poll) - previous)
        previous = float(poll)
    return out


def _log_likelihood(
    news: Sequence[int],
    increments: Sequence[float],
    gain: float,
    sigma: float,
) -> float:
    variance = sigma * sigma
    constant = math.log(2.0 * math.pi * variance)
    return -0.5 * sum(
        constant + ((change - gain * dose) ** 2) / variance
        for dose, change in zip(news, increments)
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
    """Two-type Bayes forecast using target evidence and an optional context cue."""
    target = triplet["target"]
    if not 0 <= k <= len(target["news"]):
        raise ValueError(f"k must be between 0 and {len(target['news'])}")
    prior_high = (
        context_prior_high(condition, context_reliability)
        if use_context
        else 0.5
    )
    increments = _increments(target, k)
    news = target["news"][:k]
    ll_high = _log_likelihood(
        news,
        increments,
        HIGH_GAIN,
        float(triplet["sigma_target"]),
    )
    ll_low = _log_likelihood(
        news,
        increments,
        LOW_GAIN,
        float(triplet["sigma_target"]),
    )
    log_high = math.log(prior_high) + ll_high
    log_low = math.log(1.0 - prior_high) + ll_low
    maximum = max(log_high, log_low)
    weight_high = math.exp(log_high - maximum)
    weight_low = math.exp(log_low - maximum)
    p_high = weight_high / (weight_high + weight_low)
    gain_hat = p_high * HIGH_GAIN + (1.0 - p_high) * LOW_GAIN
    current_poll = (
        float(target["polls"][k - 1]) if k else float(target["initial_poll"])
    )
    forecast = _clip(current_poll + gain_hat * float(triplet["test_news"]))
    a_is_high = triplet["reference_a"]["city_type"] == HIGH_TYPE
    p_match_a = p_high if a_is_high else 1.0 - p_high
    return {
        "k": k,
        "condition": condition,
        "prior_high": round(prior_high, 6),
        "p_high": round(p_high, 6),
        "p_match_a": round(p_match_a, 6),
        "gain_hat": round(gain_hat, 6),
        "predicted_poll": round(forecast, 3),
    }


def compute_frequentist(
    triplet: Mapping[str, Any],
    k: int,
) -> Dict[str, Any]:
    """Unregularized through-origin regression using City C only."""
    if k == 0:
        return {
            "k": k,
            "available": False,
            "gain_hat": None,
            "predicted_poll": None,
        }
    target = triplet["target"]
    news = target["news"][:k]
    changes = _increments(target, k)
    denominator = sum(float(dose) ** 2 for dose in news)
    gain_hat = (
        sum(float(dose) * change for dose, change in zip(news, changes))
        / denominator
    )
    current_poll = float(target["polls"][k - 1])
    forecast = _clip(current_poll + gain_hat * float(triplet["test_news"]))
    return {
        "k": k,
        "available": True,
        "gain_hat": round(gain_hat, 6),
        "predicted_poll": round(forecast, 3),
    }


def compute_naive(triplet: Mapping[str, Any], k: int) -> Dict[str, Any]:
    """No-structure baseline: ignore the announced news and repeat the poll."""
    gain_hat = 0.0
    target = triplet["target"]
    current_poll = (
        float(target["polls"][k - 1]) if k else float(target["initial_poll"])
    )
    return {
        "k": k,
        "gain_hat": gain_hat,
        "predicted_poll": round(
            _clip(current_poll + gain_hat * float(triplet["test_news"])),
            3,
        ),
    }


def compute_pooled_exemplar(
    triplet: Mapping[str, Any],
    k: int,
) -> Dict[str, Any]:
    """Estimate one response coefficient by pooling both reference cities."""
    news = (
        list(triplet["reference_a"]["news"])
        + list(triplet["reference_b"]["news"])
    )
    changes = (
        _increments(
            triplet["reference_a"],
            len(triplet["reference_a"]["news"]),
        )
        + _increments(
            triplet["reference_b"],
            len(triplet["reference_b"]["news"]),
        )
    )
    denominator = sum(float(dose) ** 2 for dose in news)
    gain_hat = sum(
        float(dose) * change
        for dose, change in zip(news, changes)
    ) / denominator
    target = triplet["target"]
    current_poll = (
        float(target["polls"][k - 1]) if k else float(target["initial_poll"])
    )
    return {
        "k": k,
        "gain_hat": round(gain_hat, 6),
        "predicted_poll": round(
            _clip(current_poll + gain_hat * float(triplet["test_news"])),
            3,
        ),
    }


def compute_two_prototype_exemplar(
    triplet: Mapping[str, Any],
    k: int,
) -> Dict[str, Any]:
    """Observation-matched statistical baseline inferred from A and B.

    It estimates one slope per reference city and a pooled residual scale, then
    weights the two estimated prototypes by City C's likelihood.  It is never
    handed the true gains, type names, or noise level.
    """
    ref_a = triplet["reference_a"]
    ref_b = triplet["reference_b"]
    gain_a = reference_ols_gain(ref_a)
    gain_b = reference_ols_gain(ref_b)
    residuals = []
    for city, gain in ((ref_a, gain_a), (ref_b, gain_b)):
        changes = _increments(city, len(city["news"]))
        residuals.extend(
            change - gain * dose
            for dose, change in zip(city["news"], changes)
        )
    sigma_hat = math.sqrt(
        sum(residual * residual for residual in residuals)
        / max(1, len(residuals) - 2)
    )
    sigma_hat = max(0.5, sigma_hat)

    target = triplet["target"]
    news = target["news"][:k]
    changes = _increments(target, k)
    ll_a = _log_likelihood(news, changes, gain_a, sigma_hat)
    ll_b = _log_likelihood(news, changes, gain_b, sigma_hat)
    maximum = max(ll_a, ll_b)
    weight_a = math.exp(ll_a - maximum)
    weight_b = math.exp(ll_b - maximum)
    p_match_a = weight_a / (weight_a + weight_b)
    gain_hat = p_match_a * gain_a + (1.0 - p_match_a) * gain_b
    current_poll = (
        float(target["polls"][k - 1]) if k else float(target["initial_poll"])
    )
    return {
        "k": k,
        "reference_a_gain_hat": round(gain_a, 6),
        "reference_b_gain_hat": round(gain_b, 6),
        "sigma_hat": round(sigma_hat, 6),
        "p_match_a": round(p_match_a, 6),
        "gain_hat": round(gain_hat, 6),
        "predicted_poll": round(
            _clip(current_poll + gain_hat * float(triplet["test_news"])),
            3,
        ),
    }


def compute_clairvoyant(triplet: Mapping[str, Any], k: int) -> Dict[str, Any]:
    target = triplet["target"]
    gain = float(target["gain"])
    current_poll = (
        float(target["polls"][k - 1]) if k else float(target["initial_poll"])
    )
    return {
        "k": k,
        "gain_hat": gain,
        "p_match_a": 1.0 if triplet["target_matches"] == "A" else 0.0,
        "predicted_poll": round(
            _clip(current_poll + gain * float(triplet["test_news"])),
            3,
        ),
    }


def gold_expected_poll(triplet: Mapping[str, Any], k: int) -> float:
    return compute_clairvoyant(triplet, k)["predicted_poll"]


def score_type(triplet: Mapping[str, Any], p_match_a: float) -> Dict[str, float]:
    truth_a = triplet["target_matches"] == "A"
    p_correct = p_match_a if truth_a else 1.0 - p_match_a
    return {
        "p_correct_type": round(p_correct, 6),
        "type_brier": round((p_match_a - float(truth_a)) ** 2, 6),
    }


def behavioral_gain(
    triplet: Mapping[str, Any],
    k: int,
    predicted_poll: float,
) -> float:
    target = triplet["target"]
    current_poll = (
        float(target["polls"][k - 1]) if k else float(target["initial_poll"])
    )
    return (
        float(predicted_poll) - current_poll
    ) / float(triplet["test_news"])


def belief_forecast_gap(
    triplet: Mapping[str, Any],
    p_match_a: float,
    behavior_gain: float,
) -> float:
    gain_a = float(triplet["reference_a"]["gain"])
    gain_b = float(triplet["reference_b"]["gain"])
    belief_gain = p_match_a * gain_a + (1.0 - p_match_a) * gain_b
    return abs(behavior_gain - belief_gain)


def behavioral_type_correct(
    triplet: Mapping[str, Any],
    behavior_gain: float,
) -> bool:
    true_gain = float(triplet["target"]["gain"])
    false_gain = float(GAIN_BY_TYPE[other_type(triplet["target"]["city_type"])])
    return abs(behavior_gain - true_gain) < abs(behavior_gain - false_gain)


def reference_ols_gain(city: Mapping[str, Any]) -> float:
    changes = _increments(city, len(city["news"]))
    denominator = sum(float(dose) ** 2 for dose in city["news"])
    return sum(
        float(dose) * change
        for dose, change in zip(city["news"], changes)
    ) / denominator


def balanced_triplets(
    n: int,
    *,
    seed_offset: int = 0,
) -> Iterable[Dict[str, Any]]:
    """Yield a deterministically stratified episode set."""
    for index in range(n):
        yield generate_triplet(
            seed=seed_offset + index,
            target_type=HIGH_TYPE if index % 2 == 0 else LOW_TYPE,
            high_reference="A" if (index // 2) % 2 == 0 else "B",
            target_sign=1 if (index // 4) % 2 == 0 else -1,
            context_family=(index // 8) % len(ORTHOGONAL_CONTEXTS),
        )
