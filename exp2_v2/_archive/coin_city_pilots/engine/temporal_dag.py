"""
Biased-News Election World — temporal sequential DGP (parameterized).

One "episode" = one city.  Each city has a hidden *responsiveness gain* g drawn
once.  At each step the model observes an event and a noisy survey, and must
predict the next survey.

Latent structure
----------------
  gain g    ~ prior over a gain support      # never revealed to the model
  Opinion_0 ~ Uniform(40, 60)

  At each step t = 1 … T+1:
      effect_t   = base_effect · g · sign(polarity(Event_t))
                   sign = {negative: -1, positive: +1, neutral: 0}
      Opinion_t  = clip(Opinion_{t-1} + effect_t + ε_t, 0, 100),  ε_t ~ N(0, σ_opinion)
      Survey_t   = round(clip(Opinion_t + δ_t, 0, 100)),          δ_t ~ N(0, σ_survey)

The model sees (Event_1, Survey_1) … (Event_T, Survey_T) and predicts Survey_{T+1}.
The gold label is E[Survey_{T+1}] = Opinion_T (after the T+1-th event step).

Why a "gain"?
-------------
The only thing hidden is *how strongly events move opinion*.  A neutral city has
gain 1.0 (full effect); a biased city has a small gain (events barely move the
published surveys).  Making the gain a free parameter unifies three regimes:

  * binary   : gain support {1.0, biased_gain}              (the canonical world)
  * graded   : a handful of discrete gain levels
  * continuous: a fine grid approximating a continuum

The optimal (Bayes) forecaster runs a Kalman filter under *each* gain hypothesis
and posterior-weights them — the ceiling for any forecaster using the correct
generative model.  A *latent-blind* forecaster uses the prior-mean gain and never
infers it from the history; the Bayes-minus-blind gap is the pure value of latent
inference, which is what Experiment 2 is trying to measure.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

_LOG2PI = math.log(2 * math.pi)

# ── Parameters (defaults; all overridable per-call) ───────────────────────────

# Clean canonical world (chosen 2026-06-17 for a minimal, easy-to-explain paper
# story): a polar event moves support ±10 in a neutral-media city; a biased-media
# city mutes the same news 5× (±2). Low survey noise keeps the picture crisp;
# support saturates at the 0/100 poll bounds only rarely (~2% of steps).
BASE_EFFECT   = 10.0   # neutral-media magnitude of a polar event (±10)
BIASED_GAIN   = 0.20   # biased-media gain ×base_effect → ±2 (5× muted)
SIGMA_OPINION = 2.0    # step-wise random-walk noise on latent opinion
SIGMA_SURVEY  = 2.0    # measurement noise on the survey
T_RANGE       = (6, 12)
OPINION_INIT  = (40.0, 60.0)

POLARITY_SIGN: Dict[str, float] = {"negative": -1.0, "positive": +1.0, "neutral": 0.0}

# Back-compat: the default binary world as an EFFECT[polarity][bias] table.
EFFECT: Dict[str, Dict[str, float]] = {
    pol: {
        "neutral": BASE_EFFECT * 1.0         * POLARITY_SIGN[pol],
        "biased":  BASE_EFFECT * BIASED_GAIN * POLARITY_SIGN[pol],
    }
    for pol in POLARITY_SIGN
}

# A gain support is a list of (gain, prior_weight); weights need not be normalised.
GainSupport = List[Tuple[float, float]]


def effect_of(polarity: str, gain: float, base_effect: float = BASE_EFFECT) -> float:
    """Signed opinion effect of an event of the given polarity at the given gain."""
    return base_effect * gain * POLARITY_SIGN[polarity]


def binary_gains(biased_gain: float = BIASED_GAIN) -> GainSupport:
    """The canonical 2-point support: neutral (1.0) vs biased (biased_gain), 50/50."""
    return [(1.0, 0.5), (float(biased_gain), 0.5)]


def graded_gains(n: int = 5, lo: Optional[float] = None, hi: float = 1.0,
                 biased_gain: float = BIASED_GAIN) -> GainSupport:
    """n equally-weighted gain levels in [lo, hi] (lo defaults to biased_gain)."""
    lo = biased_gain if lo is None else lo
    if n <= 1:
        return [(hi, 1.0)]
    step = (hi - lo) / (n - 1)
    w = 1.0 / n
    return [(round(lo + step * i, 6), w) for i in range(n)]


def _norm(support: GainSupport) -> GainSupport:
    tot = sum(w for _, w in support) or 1.0
    return [(g, w / tot) for g, w in support]


def prior_mean_gain(support: GainSupport) -> float:
    """Expected gain under the prior — the best fixed guess for a blind forecaster."""
    s = _norm(support)
    return sum(g * w for g, w in s)


# ── Event vocabulary ──────────────────────────────────────────────────────────

_W = round(1 / 9, 4)  # equal weight per event (3 negative + 3 positive + 3 neutral)

EVENTS: List[Dict[str, Any]] = [
    {"id": "neg_scandal",     "polarity": "negative", "weight": _W,
     "text": "A corruption scandal involving the candidate was reported in local media."},
    {"id": "neg_gaffe",       "polarity": "negative", "weight": _W,
     "text": "The candidate made a significant gaffe during a major public speech."},
    {"id": "neg_poll_drop",   "polarity": "negative", "weight": _W,
     "text": "Internal party polling showed unexpected weakness in key demographic groups."},
    {"id": "pos_endorsement", "polarity": "positive", "weight": _W,
     "text": "A prominent national figure publicly endorsed the candidate."},
    {"id": "pos_rally",       "polarity": "positive", "weight": _W,
     "text": "A large and enthusiastic rally was held, drawing record turnout in a key district."},
    {"id": "pos_policy",      "polarity": "positive", "weight": _W,
     "text": "The candidate unveiled a popular policy platform that resonated strongly with voters."},
    {"id": "neutral_debate",  "polarity": "neutral",  "weight": _W,
     "text": "The candidates debated without a clear winner emerging."},
    {"id": "neutral_nothing", "polarity": "neutral",  "weight": _W,
     "text": "No major campaign events occurred during this period."},
    {"id": "neutral_ad",      "polarity": "neutral",  "weight": _W,
     "text": "A new campaign advertisement aired, receiving mixed reactions from viewers."},
]

_EVENT_WEIGHTS = [e["weight"] for e in EVENTS]


# ── Context-variant definitions ───────────────────────────────────────────────

VARIANTS: Dict[str, str] = {
    "V0": "",
    "V1": (
        "Surveys are conducted by CityView Polling. "
        "Note: one major polling firm in this region is known to produce surveys with a "
        "'house effect' — their numbers move more slowly than those of independent pollsters, "
        "though the firm's identity and exact methodology are not publicly confirmed."
    ),
    "V2": (
        "Surveys are conducted by Hartfield Analytics. "
        "This firm has been documented in academic literature as producing 'sticky' poll readings: "
        "their surveys systematically underreport week-to-week opinion change in both directions, "
        "regardless of whether events are favorable or unfavorable to the candidate. "
        "Independent analysts estimate Hartfield surveys capture only about 20% of the true "
        "opinion shift that other pollsters would report for the same events."
    ),
}


# ── Core simulation ───────────────────────────────────────────────────────────

def _sample_event(rng: random.Random) -> Dict[str, Any]:
    return rng.choices(EVENTS, weights=_EVENT_WEIGHTS, k=1)[0]


def generate_episode(
    seed: Optional[int] = None,
    T: Optional[int] = None,
    bias: Optional[str] = None,
    *,
    gain: Optional[float] = None,
    base_effect: float   = BASE_EFFECT,
    biased_gain: float   = BIASED_GAIN,
    sigma_opinion: float = SIGMA_OPINION,
    sigma_survey: float  = SIGMA_SURVEY,
    T_range: Tuple[int, int] = T_RANGE,
) -> Dict[str, Any]:
    """
    Generate one city episode.

    Latent gain resolution (in priority order):
      * explicit ``gain``           → continuous/graded exploration
      * ``bias`` ("biased"/"neutral") → gain = biased_gain / 1.0
      * otherwise                   → random binary bias (50/50)

    A binary ``bias`` label is always derived (gain < 1.0 ⇒ "biased") so the
    viewer/eval can keep splitting by condition.
    """
    rng = random.Random(seed)

    if T is None:
        T = rng.randint(*T_range)

    # Resolve the latent gain and a binary bias label.
    if gain is None:
        if bias is None:
            bias = rng.choice(["biased", "neutral"])
        gain = 1.0 if bias == "neutral" else float(biased_gain)
    else:
        gain = float(gain)
        if bias is None:
            bias = "neutral" if gain >= 1.0 else "biased"

    opinion = rng.uniform(*OPINION_INIT)
    opinion_traj = [opinion]
    events_drawn: List[Dict[str, Any]] = []
    surveys: List[int] = []

    for _ in range(T + 1):  # T observed + 1 prediction target
        evt = _sample_event(rng)
        eff = effect_of(evt["polarity"], gain, base_effect)
        opinion = max(0.0, min(100.0, opinion + eff + rng.gauss(0, sigma_opinion)))
        survey  = int(round(max(0.0, min(100.0, opinion + rng.gauss(0, sigma_survey)))))
        events_drawn.append(evt)
        opinion_traj.append(opinion)
        surveys.append(survey)

    return {
        "bias":          bias,
        "gain":          round(gain, 6),
        "base_effect":   base_effect,
        "sigma_opinion": sigma_opinion,
        "sigma_survey":  sigma_survey,
        "opinion_init":  opinion_traj[0],
        "opinion_traj":  opinion_traj,
        "events":        events_drawn,
        "surveys":       surveys,
        "T":             T,
        "gold_survey":   surveys[T],
        "gold_opinion":  opinion_traj[T],
    }


def compute_exact_next_survey(
    opinion_at_T: float,
    next_event_polarity: str,
    bias: Optional[str] = None,
    *,
    gain: Optional[float] = None,
    base_effect: float = BASE_EFFECT,
    biased_gain: float = BIASED_GAIN,
) -> float:
    """E[Survey_{T+1}] = clip(opinion_at_T + effect, 0, 100) given the true latent."""
    if gain is None:
        gain = 1.0 if bias == "neutral" else float(biased_gain)
    return max(0.0, min(100.0, opinion_at_T + effect_of(next_event_polarity, gain, base_effect)))


# ── Inference: Kalman per gain hypothesis ─────────────────────────────────────

def _kalman_ll_mu(
    events: Sequence[Dict[str, Any]],
    surveys: Sequence[int],
    gain: float,
    base_effect: float,
    sigma_opinion: float,
    sigma_survey: float,
    mu0: float = 50.0,
    var0: float = (20.0 ** 2) / 12.0,
) -> Tuple[float, float]:
    """Return (log-likelihood of surveys, posterior mean opinion) under a fixed gain."""
    mu, var, ll = mu0, var0, 0.0
    so2, ss2 = sigma_opinion ** 2, sigma_survey ** 2
    for evt, survey in zip(events, surveys):
        mu_pred  = mu + effect_of(evt["polarity"], gain, base_effect)
        var_pred = var + so2
        innov_var = var_pred + ss2
        innov     = survey - mu_pred
        ll += -0.5 * (_LOG2PI + math.log(innov_var) + innov ** 2 / innov_var)
        K   = var_pred / innov_var
        mu  = mu_pred + K * innov
        var = (1.0 - K) * var_pred
    return ll, mu


def compute_bayes_forecast(
    events:              List[Dict[str, Any]],
    surveys:             List[int],
    next_event_polarity: str,
    sigma_opinion:       float = SIGMA_OPINION,
    sigma_survey:        float = SIGMA_SURVEY,
    *,
    gain_grid:   Optional[GainSupport] = None,
    base_effect: float = BASE_EFFECT,
    biased_gain: float = BIASED_GAIN,
) -> Dict[str, Any]:
    """
    Optimal Bayesian forecast of E[Survey_{T+1}], marginalising the latent gain
    over ``gain_grid`` (defaults to the binary {1.0, biased_gain} support).

    The theoretical ceiling for any forecaster that uses the correct generative
    model.  ``p_biased`` is the posterior mass on gains below 1.0.
    """
    grid = _norm(gain_grid if gain_grid is not None else binary_gains(biased_gain))

    log_post, final_mu = [], []
    for g, prior in grid:
        ll, mu = _kalman_ll_mu(events, surveys, g, base_effect, sigma_opinion, sigma_survey)
        log_post.append(ll + math.log(prior) if prior > 0 else -math.inf)
        final_mu.append(mu)

    m = max(log_post)
    w = [math.exp(lp - m) for lp in log_post]
    tot = sum(w) or 1.0
    post = [x / tot for x in w]

    preds = [
        max(0.0, min(100.0, final_mu[i] + effect_of(next_event_polarity, grid[i][0], base_effect)))
        for i in range(len(grid))
    ]
    optimal = sum(post[i] * preds[i] for i in range(len(grid)))
    p_biased = sum(post[i] for i in range(len(grid)) if grid[i][0] < 1.0)

    return {
        "predicted_survey": round(optimal, 2),
        "p_biased":         round(p_biased, 4),
        "p_neutral":        round(1.0 - p_biased, 4),
        "posterior":        [[round(grid[i][0], 4), round(post[i], 4)] for i in range(len(grid))],
        "post_mean_gain":   round(sum(post[i] * grid[i][0] for i in range(len(grid))), 4),
    }


def compute_blind_forecast(
    events:              List[Dict[str, Any]],
    surveys:             List[int],
    next_event_polarity: str,
    sigma_opinion:       float = SIGMA_OPINION,
    sigma_survey:        float = SIGMA_SURVEY,
    *,
    assumed_gain: Optional[float] = None,
    gain_grid:    Optional[GainSupport] = None,
    base_effect:  float = BASE_EFFECT,
    biased_gain:  float = BIASED_GAIN,
) -> Dict[str, Any]:
    """
    Latent-BLIND forecast: tracks opinion with a Kalman filter but uses a single
    *fixed* gain (default = the prior-mean gain) and never infers the latent from
    the history.  Bayes-minus-blind on biased cities = the value of latent inference.
    """
    grid = gain_grid if gain_grid is not None else binary_gains(biased_gain)
    if assumed_gain is None:
        assumed_gain = prior_mean_gain(grid)
    _, mu = _kalman_ll_mu(events, surveys, assumed_gain, base_effect, sigma_opinion, sigma_survey)
    pred = max(0.0, min(100.0, mu + effect_of(next_event_polarity, assumed_gain, base_effect)))
    return {"predicted_survey": round(pred, 2), "assumed_gain": round(assumed_gain, 4)}


def compute_bayes_trajectory(
    events:        List[Dict[str, Any]],
    surveys:       List[int],
    sigma_opinion: float = SIGMA_OPINION,
    sigma_survey:  float = SIGMA_SURVEY,
    *,
    gain_grid:   Optional[GainSupport] = None,
    base_effect: float = BASE_EFFECT,
    biased_gain: float = BIASED_GAIN,
) -> List[Dict[str, Any]]:
    """
    Step-by-step posterior for the chart: after each observation, the posterior
    P(biased) and the posterior-weighted opinion estimate.
    """
    grid = _norm(gain_grid if gain_grid is not None else binary_gains(biased_gain))
    so2, ss2 = sigma_opinion ** 2, sigma_survey ** 2
    state = [{"mu": 50.0, "var": (20.0 ** 2) / 12.0, "ll": 0.0} for _ in grid]

    trajectory = []
    for t, (evt, survey) in enumerate(zip(events, surveys)):
        for i, (g, _) in enumerate(grid):
            s = state[i]
            mu_pred  = s["mu"] + effect_of(evt["polarity"], g, base_effect)
            var_pred = s["var"] + so2
            innov_var = var_pred + ss2
            innov     = survey - mu_pred
            s["ll"]  += -0.5 * (_LOG2PI + math.log(innov_var) + innov ** 2 / innov_var)
            K = var_pred / innov_var
            s["mu"]  = mu_pred + K * innov
            s["var"] = (1.0 - K) * var_pred

        log_post = [state[i]["ll"] + math.log(grid[i][1]) if grid[i][1] > 0 else -math.inf
                    for i in range(len(grid))]
        m = max(log_post)
        w = [math.exp(lp - m) for lp in log_post]
        tot = sum(w) or 1.0
        post = [x / tot for x in w]

        p_biased = sum(post[i] for i in range(len(grid)) if grid[i][0] < 1.0)
        bayes_mu = sum(post[i] * state[i]["mu"] for i in range(len(grid)))
        # back-compat: report the two extreme-gain means as mu_neutral / mu_biased
        mu_neutral = max((state[i]["mu"], grid[i][0]) for i in range(len(grid)))[0]
        mu_biased  = min((state[i]["mu"], grid[i][0]) for i in range(len(grid)))[0]

        trajectory.append({
            "step":         t + 1,
            "mu_biased":    round(mu_biased, 2),
            "mu_neutral":   round(mu_neutral, 2),
            "p_biased":     round(p_biased, 4),
            "bayes_survey": round(bayes_mu, 2),
        })
    return trajectory


# ── Task builder ──────────────────────────────────────────────────────────────

def build_task(
    episode: Dict[str, Any],
    variant: str,
    city_id: int,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert a raw episode into the jsonl task format used by training/eval."""
    T = episode["T"]
    history = [
        {"step": t + 1, "event": episode["events"][t]["text"], "survey": episode["surveys"][t]}
        for t in range(T)
    ]
    next_event = episode["events"][T]
    return {
        "task_id":            f"city_{city_id:05d}_{variant}_T{T}",
        "variant":            variant,
        "context_prefix":     VARIANTS[variant],
        "history":            history,
        "next_event":         next_event["text"],
        "question": (
            f"Given the campaign history above and this week's event (Week {T + 1}), "
            f"what survey result (an integer from 0 to 100) do you expect?"
        ),
        "settlement_value":   round(episode["gold_opinion"], 4),
        "settlement_rounded": episode["gold_survey"],
        "_hidden_bias":               episode["bias"],
        "_hidden_gain":               episode.get("gain"),
        "_hidden_opinion_init":       round(episode["opinion_init"], 4),
        "_hidden_opinion_traj":       [round(o, 4) for o in episode["opinion_traj"]],
        "_hidden_true_opinion_at_T1": round(episode["gold_opinion"], 4),
        "_hidden_T":                  T,
        "_hidden_events":             [e["id"] for e in episode["events"]],
        "_hidden_next_event_id":      next_event["id"],
    }


def generate_tasks(
    n_cities: int,
    variants: Optional[List[str]] = None,
    seed: int = 42,
    T_range: Tuple[int, int] = T_RANGE,
    **world: Any,
) -> List[Dict[str, Any]]:
    """Generate n_cities × len(variants) tasks (same city, different context prefixes).

    Extra keyword args (base_effect, biased_gain, sigma_opinion, sigma_survey) are
    forwarded to ``generate_episode`` so a whole dataset can be drawn from a chosen
    world regime.
    """
    if variants is None:
        variants = ["V0", "V1", "V2"]
    master_rng = random.Random(seed)
    tasks: List[Dict[str, Any]] = []
    for city_id in range(n_cities):
        city_seed = master_rng.randint(0, 2 ** 31)
        T = master_rng.randint(*T_range)
        episode = generate_episode(seed=city_seed, T=T, **world)
        for variant in variants:
            tasks.append(build_task(episode, variant=variant, city_id=city_id, seed=city_seed))
    return tasks


# ── Statistics ────────────────────────────────────────────────────────────────

def episode_stats(n: int = 10_000, seed: int = 0, **world: Any) -> Dict[str, Any]:
    """Monte-Carlo opinion statistics by bias condition (parameter sanity-check)."""
    import statistics
    rng = random.Random(seed)
    by_bias: Dict[str, List[List[float]]] = {"biased": [], "neutral": []}
    for _ in range(n):
        ep = generate_episode(seed=rng.randint(0, 2 ** 31), T=10, **world)
        by_bias[ep["bias"]].append(ep["opinion_traj"])

    stats: Dict[str, Any] = {}
    for b, trajs in by_bias.items():
        if not trajs:
            continue
        step_means = [round(statistics.mean(tr[t] for tr in trajs), 2) for t in range(len(trajs[0]))]
        stats[b] = {
            "n":          len(trajs),
            "step_means": step_means,
            "final_mean": round(statistics.mean(tr[-1] for tr in trajs), 2),
            "final_std":  round(statistics.stdev(tr[-1] for tr in trajs), 2) if len(trajs) > 1 else 0.0,
        }
    return stats
