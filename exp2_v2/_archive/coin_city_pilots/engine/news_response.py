"""
Biased-News Election World v2 — "recover the city's news-responsiveness".

Clean, controlled redesign for Experiment 2. One episode = one city/race with a
single hidden parameter: a news-responsiveness gain g.

    g  ∈ [G_LO, G_HI]      (hidden, drawn once per city)
       low  g → biased/sticky coverage: news barely moves the polls
       high g → straight coverage:      polls track the news one-for-one

Each week t the model sees an EXPLICIT signed news dose news_t (e.g. +8), and the
poll responds in proportion to g:

    opinion_t = opinion_{t-1} + g · news_t + ε_t      ε_t ~ N(0, σ_opinion)
    poll_t    = round(opinion_t + δ_t)                δ_t ~ N(0, σ_survey)

The model observes (news_1, poll_1) … (news_T, poll_T), then a TEST event
news_{T+1} (a deliberately large shock), and predicts poll_{T+1}.

The point of the experiment is NOT raw forecast error — it is LATENT RECOVERY:
from any forecaster's prediction we read its *implied* gain

    ĝ = (poll_pred − poll_T) / news_{T+1}

and compare it to the true g. A forecaster that inferred the city's responsiveness
lands on the ĝ = g diagonal; one that takes the news at face value sits flat at
ĝ ≈ 1; one that ignores the news sits flat at ĝ ≈ 0.

`compute_bayes` is the optimal estimator: a grid over g, a Kalman filter on opinion
under each g, posterior-weighted — the ceiling any forecaster using the correct
model could reach. It exposes both the predicted poll and the recovered ĝ (posterior
mean) plus the full posterior (for belief-convergence-over-time plots).
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

_LOG2PI = math.log(2 * math.pi)

# ── parameters ────────────────────────────────────────────────────────────────
G_LO, G_HI       = 0.1, 1.0     # responsiveness range (continuous latent)
SIGMA_OPINION    = 1.0          # process noise on opinion
SIGMA_SURVEY     = 2.0          # poll measurement noise
T_RANGE          = (6, 10)      # observed weeks
OPINION_INIT     = 50.0
PHI              = 0.90          # opinion mean-reverts toward OPINION_INIT each week, so the
                                 # latent stays in the observable band (g, the news response, unchanged)
NEWS_MIN, NEWS_MAX = 3, 12       # |news| magnitude per week (signed)
TEST_NEWS_MAG    = 10           # magnitude of the held-out test shock (sign random)

# inference grid over g (fine → ~continuous)
G_GRID = [round(G_LO + (G_HI - G_LO) * i / 40, 4) for i in range(41)]


def sample_news(rng: random.Random) -> int:
    mag = rng.randint(NEWS_MIN, NEWS_MAX)
    return mag if rng.random() < 0.5 else -mag


def generate_episode(
    seed: Optional[int] = None,
    T: Optional[int] = None,
    g: Optional[float] = None,
    *,
    sigma_opinion: float = SIGMA_OPINION,
    sigma_survey: float = SIGMA_SURVEY,
    test_news: Optional[int] = None,
    T_range: Tuple[int, int] = T_RANGE,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    if T is None:
        T = rng.randint(*T_range)
    if g is None:
        g = round(rng.uniform(G_LO, G_HI), 4)

    opinion = OPINION_INIT
    opinion_traj = [opinion]
    news: List[int] = []
    polls: List[int] = []
    for _ in range(T):
        nt = sample_news(rng)
        opinion = max(0.0, min(100.0, OPINION_INIT + PHI * (opinion - OPINION_INIT) + g * nt + rng.gauss(0, sigma_opinion)))
        poll = int(round(max(0.0, min(100.0, opinion + rng.gauss(0, sigma_survey)))))
        news.append(nt); opinion_traj.append(opinion); polls.append(poll)

    # held-out test shock (large, so implied-ĝ is well-conditioned)
    if test_news is None:
        test_news = TEST_NEWS_MAG if rng.random() < 0.5 else -TEST_NEWS_MAG
    gold_opinion = max(0.0, min(100.0, OPINION_INIT + PHI * (opinion - OPINION_INIT) + g * test_news))
    gold_poll = int(round(max(0.0, min(100.0, gold_opinion + rng.gauss(0, sigma_survey)))))

    return {
        "g":            g,
        "T":            T,
        "news":         news,           # observed doses, len T
        "polls":        polls,          # observed polls, len T
        "opinion_traj": opinion_traj,   # hidden, len T+1
        "test_news":    test_news,      # the T+1 shock (shown to the model)
        "gold_opinion": round(gold_opinion, 3),
        "gold_poll":    gold_poll,      # E[poll_{T+1}] target
        "last_poll":    polls[-1],
        "sigma_opinion": sigma_opinion,
        "sigma_survey":  sigma_survey,
    }


# ── optimal estimator: grid over g, Kalman filter on opinion ──────────────────
def _loglik_and_mu(news: Sequence[int], polls: Sequence[int], g: float,
                   so: float, ss: float,
                   mu0: float = OPINION_INIT, var0: float = 25.0) -> Tuple[float, float]:
    mu, var, ll = mu0, var0, 0.0
    so2, ss2 = so * so, ss * ss
    for nt, p in zip(news, polls):
        mu_pred = OPINION_INIT + PHI * (mu - OPINION_INIT) + g * nt
        var_pred = PHI * PHI * var + so2
        iv = var_pred + ss2
        innov = p - mu_pred
        ll += -0.5 * (_LOG2PI + math.log(iv) + innov * innov / iv)
        K = var_pred / iv
        mu = mu_pred + K * innov
        var = (1.0 - K) * var_pred
    return ll, mu


def compute_bayes(news: Sequence[int], polls: Sequence[int], test_news: float,
                  *, sigma_opinion: float = SIGMA_OPINION, sigma_survey: float = SIGMA_SURVEY,
                  g_grid: Sequence[float] = G_GRID) -> Dict[str, Any]:
    """Posterior over g (uniform prior on the grid), the recovered ĝ (posterior mean),
    and the posterior-weighted poll prediction for the test event."""
    lls, mus = [], []
    for g in g_grid:
        ll, mu = _loglik_and_mu(news, polls, g, sigma_opinion, sigma_survey)
        lls.append(ll); mus.append(mu)
    m = max(lls)
    w = [math.exp(l - m) for l in lls]
    tot = sum(w) or 1.0
    post = [x / tot for x in w]
    g_hat = sum(post[i] * g_grid[i] for i in range(len(g_grid)))
    # posterior-weighted prediction of the next poll (same mean-reverting transition)
    pred = sum(post[i] * (OPINION_INIT + PHI * (mus[i] - OPINION_INIT) + g_grid[i] * test_news)
               for i in range(len(g_grid)))
    pred = max(0.0, min(100.0, pred))
    g_sd = math.sqrt(max(0.0, sum(post[i] * (g_grid[i] - g_hat) ** 2 for i in range(len(g_grid)))))
    return {
        "g_hat":           round(g_hat, 4),
        "g_sd":            round(g_sd, 4),
        "predicted_poll":  round(pred, 2),
        "posterior":       [[g_grid[i], round(post[i], 5)] for i in range(len(g_grid))],
    }


def bayes_g_trajectory(news: Sequence[int], polls: Sequence[int],
                       *, sigma_opinion: float = SIGMA_OPINION, sigma_survey: float = SIGMA_SURVEY,
                       g_grid: Sequence[float] = G_GRID) -> List[float]:
    """Posterior-mean ĝ after each observed week (for belief-convergence plots)."""
    out = []
    for t in range(1, len(news) + 1):
        out.append(compute_bayes(news[:t], polls[:t], 0.0,
                                 sigma_opinion=sigma_opinion, sigma_survey=sigma_survey,
                                 g_grid=g_grid)["g_hat"])
    return out


# ── shortcut baselines (fixed, non-inferring) ─────────────────────────────────
def predict_fixed_gain(last_poll: float, test_news: float, assumed_g: float) -> float:
    """Forecast assuming a fixed gain (no inference). assumed_g=1 → take news at
    face value; assumed_g=0 → ignore the news (predict no change)."""
    return max(0.0, min(100.0, last_poll + assumed_g * test_news))


def implied_g(predicted_poll: float, last_poll: float, test_news: float) -> float:
    """The gain a forecaster's prediction implies. The recovery axis."""
    if test_news == 0:
        return float("nan")
    return (predicted_poll - last_poll) / test_news
