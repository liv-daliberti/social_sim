"""Three-city latent-type world for Experiment 2 v2.

Each episode contains two completed reference cities and one target city:

* one reference has a strong news-response gain (default ``g=1.0``);
* the other has a sticky news-response gain (default ``g=0.25``);
* the target is equally likely to share either reference city's exact type.

The forecaster sees both reference histories, then predicts the target repeatedly:
before seeing any target evidence and after each new target poll.  The design
therefore separates two operations that the original one-city task combined:

1. infer the two possible response types from examples in other cities; and
2. use within-target evidence to decide which type generated the target city.

The matched oracle below performs hierarchical Bayesian inference.  It learns a
posterior over each reference city's gain, compares the target history under
those two posteriors, and predicts by averaging over the remaining uncertainty.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

INITIAL_OPINION = 50.0
PHI = 0.90
SIGMA_OPINION = 1.0
SIGMA_SURVEY = 2.0
STRONG_GAIN = 1.0
WEAK_GAIN = 0.25
REFERENCE_WEEKS = 6
TARGET_WEEKS = 5
PROBE_SHOCKS = (-10, -5, 5, 10)

# This support deliberately includes 0.0 so the same code covers the user's
# "completely sticky" (100% muted) variant via weak_gain=0.
GAIN_GRID = tuple(round(i / 40, 4) for i in range(41))
_LOG2PI = math.log(2.0 * math.pi)


def _clip_poll(value: float) -> float:
    return max(0.0, min(100.0, value))


def _diagnostic_schedule(rng: random.Random, weeks: int) -> List[int]:
    """Large, balanced shocks that make the two response types identifiable."""
    base = [12, -12, 10, -10, 8, -8]
    out = [base[i % len(base)] for i in range(weeks)]
    rng.shuffle(out)
    if rng.random() < 0.5:
        out = [-x for x in out]
    return out


def _ambiguous_schedule(rng: random.Random, weeks: int) -> List[int]:
    """Small shocks: a control in which an early poll need not identify the type."""
    base = [1, -1, 2, -2, 1, -1]
    out = [base[i % len(base)] for i in range(weeks)]
    rng.shuffle(out)
    if rng.random() < 0.5:
        out = [-x for x in out]
    return out


def _random_schedule(rng: random.Random, weeks: int) -> List[int]:
    return [
        (1 if rng.random() < 0.5 else -1) * rng.randint(3, 12)
        for _ in range(weeks)
    ]


def make_news_schedule(
    rng: random.Random,
    weeks: int,
    mode: str = "diagnostic",
) -> List[int]:
    """Create a signed-news sequence for one city."""
    if mode == "diagnostic":
        return _diagnostic_schedule(rng, weeks)
    if mode == "ambiguous":
        return _ambiguous_schedule(rng, weeks)
    if mode == "random":
        return _random_schedule(rng, weeks)
    raise ValueError(f"unknown evidence mode: {mode!r}")


def generate_city(
    *,
    rng: random.Random,
    gain: float,
    news: Sequence[int],
    sigma_opinion: float = SIGMA_OPINION,
    sigma_survey: float = SIGMA_SURVEY,
    phi: float = PHI,
) -> Dict[str, Any]:
    """Generate one city from a fixed baseline of 50.

    Week zero is an observed, noise-free baseline poll of 50.  This is important:
    a single large Week-1 shock can genuinely diagnose the target type instead of
    being confounded with an unknown starting level.
    """
    opinion = INITIAL_OPINION
    opinions = [opinion]
    polls: List[int] = []

    for dose in news:
        opinion = _clip_poll(
            INITIAL_OPINION
            + phi * (opinion - INITIAL_OPINION)
            + gain * dose
            + rng.gauss(0.0, sigma_opinion)
        )
        poll = int(round(_clip_poll(opinion + rng.gauss(0.0, sigma_survey))))
        opinions.append(opinion)
        polls.append(poll)

    return {
        "initial_poll": int(INITIAL_OPINION),
        "gain": float(gain),
        "news": list(news),
        "polls": polls,
        "opinions": opinions,
    }


def generate_triplet(
    seed: Optional[int] = None,
    *,
    reference_weeks: int = REFERENCE_WEEKS,
    target_weeks: int = TARGET_WEEKS,
    strong_gain: float = STRONG_GAIN,
    weak_gain: float = WEAK_GAIN,
    target_evidence: str = "diagnostic",
    sigma_opinion: float = SIGMA_OPINION,
    sigma_survey: float = SIGMA_SURVEY,
    phi: float = PHI,
) -> Dict[str, Any]:
    """Generate two reference cities and a 50/50 target city.

    Which reference is strong is randomized, as is which reference the target
    matches.  City labels therefore cannot be used as a shortcut.
    """
    if not 0.0 <= weak_gain < strong_gain <= 1.0:
        raise ValueError("expected 0 <= weak_gain < strong_gain <= 1")
    if reference_weeks < 1 or target_weeks < 1:
        raise ValueError("reference_weeks and target_weeks must both be positive")

    rng = random.Random(seed)
    a_is_strong = rng.random() < 0.5
    gain_a = strong_gain if a_is_strong else weak_gain
    gain_b = weak_gain if a_is_strong else strong_gain
    target_matches = "A" if rng.random() < 0.5 else "B"
    target_gain = gain_a if target_matches == "A" else gain_b

    ref_a = generate_city(
        rng=rng,
        gain=gain_a,
        news=make_news_schedule(rng, reference_weeks, "diagnostic"),
        sigma_opinion=sigma_opinion,
        sigma_survey=sigma_survey,
        phi=phi,
    )
    ref_b = generate_city(
        rng=rng,
        gain=gain_b,
        news=make_news_schedule(rng, reference_weeks, "diagnostic"),
        sigma_opinion=sigma_opinion,
        sigma_survey=sigma_survey,
        phi=phi,
    )
    target_news = make_news_schedule(rng, target_weeks, target_evidence)
    if target_evidence == "diagnostic":
        # Put the largest available shock first.  This creates the intended case
        # where one City C measurement can be genuinely decisive.
        largest = max(range(len(target_news)), key=lambda i: abs(target_news[i]))
        target_news[0], target_news[largest] = target_news[largest], target_news[0]
    target = generate_city(
        rng=rng,
        gain=target_gain,
        news=target_news,
        sigma_opinion=sigma_opinion,
        sigma_survey=sigma_survey,
        phi=phi,
    )

    return {
        "seed": seed,
        "reference_a": ref_a,
        "reference_b": ref_b,
        "target": target,
        "target_matches": target_matches,
        "strong_reference": "A" if a_is_strong else "B",
        "strong_gain": float(strong_gain),
        "weak_gain": float(weak_gain),
        "target_evidence": target_evidence,
        "sigma_opinion": float(sigma_opinion),
        "sigma_survey": float(sigma_survey),
        "phi": float(phi),
    }


def _loglik_and_state(
    news: Sequence[int],
    polls: Sequence[int],
    gain: float,
    *,
    sigma_opinion: float,
    sigma_survey: float,
    phi: float,
) -> Tuple[float, float, float]:
    """Kalman log likelihood and final opinion state under a fixed gain."""
    if len(news) != len(polls):
        raise ValueError("news and polls must have equal lengths")

    # The initial poll is explicitly fixed at 50 in both the DGP and prompt.
    mu, var, ll = INITIAL_OPINION, 0.0, 0.0
    so2, ss2 = sigma_opinion**2, sigma_survey**2
    for dose, poll in zip(news, polls):
        mu_pred = INITIAL_OPINION + phi * (mu - INITIAL_OPINION) + gain * dose
        var_pred = phi * phi * var + so2
        innovation_var = var_pred + ss2
        innovation = poll - mu_pred
        ll += -0.5 * (
            _LOG2PI
            + math.log(innovation_var)
            + innovation * innovation / innovation_var
        )
        kalman_gain = var_pred / innovation_var
        mu = mu_pred + kalman_gain * innovation
        var = (1.0 - kalman_gain) * var_pred
    return ll, mu, var


def _logsumexp(values: Sequence[float]) -> float:
    m = max(values)
    return m + math.log(sum(math.exp(value - m) for value in values))


def _normalise_logs(values: Sequence[float]) -> List[float]:
    z = _logsumexp(values)
    return [math.exp(value - z) for value in values]


def _gain_posterior(
    city: Mapping[str, Any],
    *,
    gain_grid: Sequence[float],
    sigma_opinion: float,
    sigma_survey: float,
    phi: float,
) -> Dict[str, Any]:
    """Posterior over a reference city's shared type gain."""
    log_weights: List[float] = []
    state_means: List[float] = []
    state_vars: List[float] = []
    for gain in gain_grid:
        ll, mu, var = _loglik_and_state(
            city["news"],
            city["polls"],
            gain,
            sigma_opinion=sigma_opinion,
            sigma_survey=sigma_survey,
            phi=phi,
        )
        log_weights.append(ll)  # uniform grid prior
        state_means.append(mu)
        state_vars.append(var)
    weights = _normalise_logs(log_weights)
    mean_gain = sum(w * g for w, g in zip(weights, gain_grid))
    return {
        "weights": weights,
        "log_weights": [math.log(max(w, 1e-300)) for w in weights],
        "mean_gain": mean_gain,
        "state_means": state_means,
        "state_vars": state_vars,
    }


def _target_likelihoods(
    news: Sequence[int],
    polls: Sequence[int],
    *,
    gain_grid: Sequence[float],
    sigma_opinion: float,
    sigma_survey: float,
    phi: float,
) -> Tuple[List[float], List[float], List[float]]:
    lls, mus, variances = [], [], []
    for gain in gain_grid:
        ll, mu, var = _loglik_and_state(
            news,
            polls,
            gain,
            sigma_opinion=sigma_opinion,
            sigma_survey=sigma_survey,
            phi=phi,
        )
        lls.append(ll)
        mus.append(mu)
        variances.append(var)
    return lls, mus, variances


def compute_hierarchical_bayes(
    triplet: Mapping[str, Any],
    k: int,
    *,
    shocks: Iterable[int] = PROBE_SHOCKS,
    gain_grid: Sequence[float] = GAIN_GRID,
) -> Dict[str, Any]:
    """Matched hierarchical oracle after ``k`` target-city observations.

    The reference gains are not handed to the oracle.  They are inferred from
    the same two example histories shown to the language model.
    """
    target = triplet["target"]
    if not 0 <= k <= len(target["news"]):
        raise ValueError(f"k must be between 0 and {len(target['news'])}")
    if not gain_grid:
        raise ValueError("gain_grid cannot be empty")

    sigma_opinion = float(triplet["sigma_opinion"])
    sigma_survey = float(triplet["sigma_survey"])
    phi = float(triplet["phi"])

    ref_a = _gain_posterior(
        triplet["reference_a"],
        gain_grid=gain_grid,
        sigma_opinion=sigma_opinion,
        sigma_survey=sigma_survey,
        phi=phi,
    )
    ref_b = _gain_posterior(
        triplet["reference_b"],
        gain_grid=gain_grid,
        sigma_opinion=sigma_opinion,
        sigma_survey=sigma_survey,
        phi=phi,
    )

    target_news = target["news"][:k]
    target_polls = target["polls"][:k]
    target_ll, target_mu, _ = _target_likelihoods(
        target_news,
        target_polls,
        gain_grid=gain_grid,
        sigma_opinion=sigma_opinion,
        sigma_survey=sigma_survey,
        phi=phi,
    )

    # If C matches A, its exact gain is shared with A; likewise for B.
    log_joint_a = [
        ref_a["log_weights"][i] + target_ll[i]
        for i in range(len(gain_grid))
    ]
    log_joint_b = [
        ref_b["log_weights"][i] + target_ll[i]
        for i in range(len(gain_grid))
    ]
    log_evidence_a = _logsumexp(log_joint_a)
    log_evidence_b = _logsumexp(log_joint_b)
    p_match_a = _normalise_logs([log_evidence_a, log_evidence_b])[0]
    conditional_a = _normalise_logs(log_joint_a)
    conditional_b = _normalise_logs(log_joint_b)
    gain_weights = [
        p_match_a * conditional_a[i] + (1.0 - p_match_a) * conditional_b[i]
        for i in range(len(gain_grid))
    ]

    gain_hat = sum(w * g for w, g in zip(gain_weights, gain_grid))
    gain_sd = math.sqrt(
        max(
            0.0,
            sum(w * (g - gain_hat) ** 2 for w, g in zip(gain_weights, gain_grid)),
        )
    )
    forecasts: Dict[str, float] = {}
    for shock in shocks:
        pred = sum(
            gain_weights[i]
            * (
                INITIAL_OPINION
                + phi * (target_mu[i] - INITIAL_OPINION)
                + gain_grid[i] * shock
            )
            for i in range(len(gain_grid))
        )
        forecasts[str(int(shock))] = round(_clip_poll(pred), 3)

    return {
        "k": k,
        "p_match_a": round(p_match_a, 6),
        "p_match_b": round(1.0 - p_match_a, 6),
        "gain_hat": round(gain_hat, 6),
        "gain_sd": round(gain_sd, 6),
        "reference_a_gain_hat": round(ref_a["mean_gain"], 6),
        "reference_b_gain_hat": round(ref_b["mean_gain"], 6),
        "forecasts": forecasts,
        "gain_posterior": [
            [float(gain), round(weight, 8)]
            for gain, weight in zip(gain_grid, gain_weights)
        ],
    }


def compute_target_only_bayes(
    triplet: Mapping[str, Any],
    k: int,
    *,
    shocks: Iterable[int] = PROBE_SHOCKS,
    gain_grid: Sequence[float] = GAIN_GRID,
) -> Dict[str, Any]:
    """Bayes baseline that ignores the two reference cities."""
    target = triplet["target"]
    sigma_opinion = float(triplet["sigma_opinion"])
    sigma_survey = float(triplet["sigma_survey"])
    phi = float(triplet["phi"])
    lls, mus, _ = _target_likelihoods(
        target["news"][:k],
        target["polls"][:k],
        gain_grid=gain_grid,
        sigma_opinion=sigma_opinion,
        sigma_survey=sigma_survey,
        phi=phi,
    )
    weights = _normalise_logs(lls)
    gain_hat = sum(w * g for w, g in zip(weights, gain_grid))
    forecasts = {}
    for shock in shocks:
        pred = sum(
            weights[i]
            * (
                INITIAL_OPINION
                + phi * (mus[i] - INITIAL_OPINION)
                + gain_grid[i] * shock
            )
            for i in range(len(gain_grid))
        )
        forecasts[str(int(shock))] = round(_clip_poll(pred), 3)
    return {
        "k": k,
        "gain_hat": round(gain_hat, 6),
        "forecasts": forecasts,
    }


def compute_pooled_baseline(
    triplet: Mapping[str, Any],
    k: int,
    *,
    shocks: Iterable[int] = PROBE_SHOCKS,
    gain_grid: Sequence[float] = GAIN_GRID,
) -> Dict[str, Any]:
    """Never classify C; permanently average the two reference-city gains."""
    sigma_opinion = float(triplet["sigma_opinion"])
    sigma_survey = float(triplet["sigma_survey"])
    phi = float(triplet["phi"])
    ref_a = _gain_posterior(
        triplet["reference_a"],
        gain_grid=gain_grid,
        sigma_opinion=sigma_opinion,
        sigma_survey=sigma_survey,
        phi=phi,
    )
    ref_b = _gain_posterior(
        triplet["reference_b"],
        gain_grid=gain_grid,
        sigma_opinion=sigma_opinion,
        sigma_survey=sigma_survey,
        phi=phi,
    )
    assumed_gain = 0.5 * (ref_a["mean_gain"] + ref_b["mean_gain"])
    target = triplet["target"]
    _, mu, _ = _loglik_and_state(
        target["news"][:k],
        target["polls"][:k],
        assumed_gain,
        sigma_opinion=sigma_opinion,
        sigma_survey=sigma_survey,
        phi=phi,
    )
    forecasts = {
        str(int(shock)): round(
            _clip_poll(
                INITIAL_OPINION
                + phi * (mu - INITIAL_OPINION)
                + assumed_gain * shock
            ),
            3,
        )
        for shock in shocks
    }
    return {
        "k": k,
        "gain_hat": round(assumed_gain, 6),
        "forecasts": forecasts,
    }


def compute_naive_baseline(
    triplet: Mapping[str, Any],
    k: int,
    *,
    shocks: Iterable[int] = PROBE_SHOCKS,
) -> Dict[str, Any]:
    """Fixed 50/50-average response that never learns City C's type.

    Unlike ``compute_pooled_baseline``, this baseline does not even re-estimate
    the two reference gains from their noisy histories.  It always uses the
    population midpoint implied by the experiment configuration.
    """
    target = triplet["target"]
    if not 0 <= k <= len(target["news"]):
        raise ValueError(f"k must be between 0 and {len(target['news'])}")
    assumed_gain = 0.5 * (
        float(triplet["strong_gain"]) + float(triplet["weak_gain"])
    )
    _, mu, _ = _loglik_and_state(
        target["news"][:k],
        target["polls"][:k],
        assumed_gain,
        sigma_opinion=float(triplet["sigma_opinion"]),
        sigma_survey=float(triplet["sigma_survey"]),
        phi=float(triplet["phi"]),
    )
    forecasts = {
        str(int(shock)): round(
            _clip_poll(
                INITIAL_OPINION
                + float(triplet["phi"]) * (mu - INITIAL_OPINION)
                + assumed_gain * shock
            ),
            3,
        )
        for shock in shocks
    }
    return {
        "k": k,
        "gain_hat": round(assumed_gain, 6),
        "forecasts": forecasts,
    }


def compute_frequentist_baseline(
    triplet: Mapping[str, Any],
    k: int,
    *,
    shocks: Iterable[int] = PROBE_SHOCKS,
) -> Dict[str, Any]:
    """Unregularized target-only gain regression.

    For each observed target week, regress the persistence-adjusted poll change
    on signed news through the origin:

        poll_t - [50 + phi * (poll_{t-1} - 50)] = g * news_t + noise.

    It sees no reference-city data and uses no prior or gain clipping.  At k=0
    there is no frequentist estimate, so the method is reported as unavailable.
    """
    target = triplet["target"]
    if not 0 <= k <= len(target["news"]):
        raise ValueError(f"k must be between 0 and {len(target['news'])}")
    if k == 0:
        return {
            "k": k,
            "available": False,
            "gain_hat": None,
            "forecasts": {},
        }

    phi = float(triplet["phi"])
    previous_poll = float(target["initial_poll"])
    numerator = 0.0
    denominator = 0.0
    for dose, poll in zip(target["news"][:k], target["polls"][:k]):
        expected_without_news = (
            INITIAL_OPINION + phi * (previous_poll - INITIAL_OPINION)
        )
        adjusted_change = float(poll) - expected_without_news
        numerator += float(dose) * adjusted_change
        denominator += float(dose) ** 2
        previous_poll = float(poll)

    gain_hat = numerator / denominator if denominator else 0.0
    forecasts = {
        str(int(shock)): round(
            _clip_poll(
                INITIAL_OPINION
                + phi * (previous_poll - INITIAL_OPINION)
                + gain_hat * shock
            ),
            3,
        )
        for shock in shocks
    }
    return {
        "k": k,
        "available": True,
        "gain_hat": round(gain_hat, 6),
        "forecasts": forecasts,
    }


def compute_clairvoyant_ceiling(
    triplet: Mapping[str, Any],
    k: int,
    *,
    shocks: Iterable[int] = PROBE_SHOCKS,
) -> Dict[str, Any]:
    """Literal knows-everything ceiling: true target type, gain, and opinion."""
    truth_a = triplet["target_matches"] == "A"
    return {
        "k": k,
        "p_match_a": 1.0 if truth_a else 0.0,
        "p_match_b": 0.0 if truth_a else 1.0,
        "gain_hat": float(triplet["target"]["gain"]),
        "forecasts": counterfactual_gold(triplet, k, shocks),
    }


def counterfactual_gold(
    triplet: Mapping[str, Any],
    k: int,
    shocks: Iterable[int] = PROBE_SHOCKS,
) -> Dict[str, float]:
    """True expected next poll for each probe shock at target prefix ``k``."""
    target = triplet["target"]
    opinion = float(target["opinions"][k])
    gain = float(target["gain"])
    phi = float(triplet["phi"])
    return {
        str(int(shock)): round(
            _clip_poll(
                INITIAL_OPINION
                + phi * (opinion - INITIAL_OPINION)
                + gain * shock
            ),
            3,
        )
        for shock in shocks
    }


def type_score(triplet: Mapping[str, Any], p_match_a: float) -> Dict[str, float]:
    """Probability/accuracy scores for target-type identification."""
    truth_a = triplet["target_matches"] == "A"
    p_correct = p_match_a if truth_a else 1.0 - p_match_a
    return {
        "p_correct_type": round(p_correct, 6),
        "type_correct": float((p_match_a >= 0.5) == truth_a),
        "type_brier": round((p_match_a - float(truth_a)) ** 2, 6),
    }


def position_score(triplet: Mapping[str, Any], gain_hat: float) -> Dict[str, float]:
    """How far the estimate has moved from the 50/50 midpoint to the right type.

    ``adaptation`` is 0 at the exact midpoint, 1 at the correct prototype, and
    negative if the estimate moves toward the wrong prototype.
    """
    gain_a = float(triplet["reference_a"]["gain"])
    gain_b = float(triplet["reference_b"]["gain"])
    true_gain = float(triplet["target"]["gain"])
    midpoint = 0.5 * (gain_a + gain_b)
    span = abs(gain_a - gain_b)
    midpoint_error = abs(true_gain - midpoint)
    gain_error = abs(gain_hat - true_gain)
    adaptation = (
        1.0 - gain_error / midpoint_error
        if midpoint_error > 0
        else float("nan")
    )
    return {
        "gain_error": round(gain_error, 6),
        "normalised_gain_error": round(gain_error / span, 6) if span else 0.0,
        "adaptation": round(adaptation, 6),
    }
