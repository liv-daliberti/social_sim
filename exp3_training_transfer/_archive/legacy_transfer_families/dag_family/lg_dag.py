"""lg_dag.py — a family of small linear-Gaussian DAG worlds for Experiment 3.

Each structure is a linear-Gaussian state-space model with ONE hidden scalar (the per-episode
"gain" g, drawn once and never revealed). Observed exogenous drivers u_t (news-like) drive latent
states s_t (opinion-like, mean-reverting to a baseline), which emit observed outputs y_t
(poll-like, noisy):

    s_t = base + A (s_{t-1} - base) + B(g) u_t + w_t,   w_t ~ N(0, sigma_s^2 I)
    y_t = C s_t + v_t,                                  v_t ~ N(0, sigma_y^2 I)

Structures differ in TOPOLOGY (state / input / output dims, chain depth, confounding, extra
channels, and which single edge is hidden). The shared task across all structures is: FORECAST the
next output y_{t+1} given the history and the next inputs. The optimal forecaster is a per-structure
Kalman filter that marginalizes the hidden g over a grid -- computable for ANY linear-Gaussian DAG,
which is the whole reason we stay linear-Gaussian.

exp2's biased-news world is exactly the `chain1_gain` structure here (d=m=p=1, A=[[phi]], B=[[g]],
C=[[1]], base=50), so Experiment 3 can hold it out and ask whether training on OTHER structures
improves recovery on the Experiment-2 world.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# ── shared world constants (match biased-news so chain1_gain == the exp2 world) ─────────────────
G_LO, G_HI = 0.1, 1.0
G_GRID = np.round(np.linspace(G_LO, G_HI, 41), 4)      # inference grid over the hidden gain
BASE = 50.0
NEWS_MIN, NEWS_MAX = 3, 12                             # |driver| magnitude per week
SIGMA_S, SIGMA_Y = 1.0, 2.0                            # process / observation noise
T_DEFAULT = 10
SHOCKS = (-10, -5, 5, 10)                              # held-out recovery shocks (as in exp2)


@dataclass
class LGStructure:
    """A linear-Gaussian DAG world. A, B, C are the KNOWN structure; `hidden` names the single
    slot (in B or A) filled with the per-episode gain g. `recover=True` marks structures for which
    a clean one-step input->output gain recovery is defined (the driver moves the tracked output in
    a single step); deeper/delayed structures set it False and are scored on forecast error only."""
    name: str
    A: np.ndarray                       # d x d  transition (mean-reversion + inter-latent coupling)
    B: np.ndarray                       # d x m  input matrix (one slot is the hidden g)
    C: np.ndarray                       # p x d  observation
    hidden: Tuple[str, int, int]        # ('B', i, j) or ('A', i, j)
    sigma_s: float = SIGMA_S
    sigma_y: float = SIGMA_Y
    baseline: float = BASE
    probe_input: int = 0                # driver channel the recovery shocks perturb
    probe_output: int = 0               # output channel the recovery reads
    recover: bool = True                # is a clean one-step gain recovery defined?
    note: str = ""                      # short human description of the topology

    @property
    def d(self) -> int: return self.A.shape[0]
    @property
    def m(self) -> int: return self.B.shape[1]
    @property
    def p(self) -> int: return self.C.shape[0]

    def with_gain(self, g: float) -> Tuple[np.ndarray, np.ndarray]:
        A, B = self.A.astype(float).copy(), self.B.astype(float).copy()
        kind, i, j = self.hidden
        (B if kind == "B" else A)[i, j] = g
        return A, B


# ── simulation ──────────────────────────────────────────────────────────────────────────────────
def _sample_inputs(rng: np.random.Generator, m: int) -> np.ndarray:
    mags = rng.integers(NEWS_MIN, NEWS_MAX + 1, size=m).astype(float)
    signs = np.where(rng.random(m) < 0.5, 1.0, -1.0)
    return mags * signs


def generate_episode(struct: LGStructure, seed: Optional[int] = None, T: int = T_DEFAULT,
                     g: Optional[float] = None) -> dict:
    """Simulate one episode (one 'city') of a structure. Returns observed drivers U (T x m),
    observed outputs Y (T x p, clipped to [0,100] and integer, poll-like), the hidden latent
    trajectory S (T x d), and the drawn gain g."""
    rng = np.random.default_rng(seed)
    if g is None:
        g = round(float(rng.uniform(G_LO, G_HI)), 4)
    A, B = struct.with_gain(g)
    base = np.full(struct.d, struct.baseline)
    s = base.copy()
    U, Y, S = [], [], []
    for _ in range(T):
        u = _sample_inputs(rng, struct.m)
        s = base + A @ (s - base) + B @ u + rng.normal(0.0, struct.sigma_s, struct.d)
        y = struct.C @ s + rng.normal(0.0, struct.sigma_y, struct.p)
        U.append(u); S.append(s.copy()); Y.append(np.clip(np.round(y), 0, 100))
    return {"name": struct.name, "g": g, "T": T,
            "U": np.array(U), "Y": np.array(Y), "S": np.array(S), "baseline": struct.baseline}


def true_next_output(struct: LGStructure, S: np.ndarray, k: int, g: float, u_next: np.ndarray) -> np.ndarray:
    """Ground-truth EXPECTED next output E[y_{k+1} | state_k, u_next] (noise-free) -- the
    Bayes-optimal forecast target. S[k-1] is the true latent after k weeks; S[-1]==base for k=0."""
    A, B = struct.with_gain(g)
    base = np.full(struct.d, struct.baseline)
    s_k = S[k - 1] if k >= 1 else base
    s_next = base + A @ (s_k - base) + B @ u_next
    return np.clip(struct.C @ s_next, 0, 100)


# ── horizon-aware probe (2026-07-08 audit fix) ───────────────────────────────────────────────────
# A fixed one-step probe is DEGENERATE in every delayed world: news at week k+1 cannot reach the
# poll at week k+1, so all four shock targets coincide, the slope target is 0, and a constant answer
# is exactly optimal -- the probe carries no information about g. Probing each structure at its OWN
# first-response horizon h* restores a g-bearing slope wherever the gain sits on an input edge.
def response_at_horizon(struct: LGStructure, g: float, h: int) -> float:
    """(C A^{h-1} B)[probe_out, probe_in]: how far the probed poll moves h weeks after a unit pulse
    on the probed driver. h=1 recovers the one-step gain (C B)."""
    A, B = struct.with_gain(g)
    M = np.linalg.matrix_power(A, h - 1) @ B
    return float((struct.C @ M)[struct.probe_output, struct.probe_input])


def first_response_horizon(struct: LGStructure, hmax: int = 8, tol: float = 1e-9) -> int:
    """Smallest h at which the probed driver actually moves the probed poll (checked at both ends of
    the gain range, since g may sit in A)."""
    for h in range(1, hmax + 1):
        if max(abs(response_at_horizon(struct, G_LO, h)),
               abs(response_at_horizon(struct, G_HI, h))) > tol:
            return h
    raise ValueError(f"{struct.name}: no probe response within {hmax} weeks")


def encodes_gain(struct: LGStructure, h: int, tol: float = 1e-9) -> bool:
    """Does the response at horizon h vary with g?  True when g sits on an input edge (the response
    IS the gain); False when g hides in a persistence -- then the input->output response is a known
    constant and no amount of probing identifies g from it."""
    return abs(response_at_horizon(struct, G_HI, h) - response_at_horizon(struct, G_LO, h)) > tol


def true_output_at_horizon(struct: LGStructure, S: np.ndarray, k: int, g: float,
                           u_pulse: np.ndarray, h: int) -> np.ndarray:
    """Ground-truth E[y_{k+h}] given a single news pulse at week k+1 and NO news thereafter:
        s_{k+h} = base + A^h (s_k - base) + A^{h-1} B u_pulse.
    h=1 reduces to true_next_output."""
    A, B = struct.with_gain(g)
    base = np.full(struct.d, struct.baseline)
    s_k = S[k - 1] if k >= 1 else base
    s = base + np.linalg.matrix_power(A, h) @ (s_k - base) \
        + np.linalg.matrix_power(A, h - 1) @ B @ u_pulse
    return np.clip(struct.C @ s, 0, 100)


# ── per-structure Kalman oracle (marginalize the hidden gain) ─────────────────────────────────────
def _kalman_ll_state(struct: LGStructure, A: np.ndarray, B: np.ndarray,
                     U: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray]:
    """Log-likelihood of the outputs and the filtered final state, for known (A, B)."""
    d, p = struct.d, struct.p
    Qs = struct.sigma_s ** 2 * np.eye(d)
    Ry = struct.sigma_y ** 2 * np.eye(p)
    base = np.full(d, struct.baseline)
    mu, P = base.copy(), (5.0 ** 2) * np.eye(d)
    ll = 0.0
    _LOG2PI = np.log(2 * np.pi)
    for u, y in zip(U, Y):
        mu_pred = base + A @ (mu - base) + B @ u
        P_pred = A @ P @ A.T + Qs
        S = struct.C @ P_pred @ struct.C.T + Ry
        Sinv = np.linalg.inv(S)
        innov = y - struct.C @ mu_pred
        sign, logdet = np.linalg.slogdet(S)
        ll += -0.5 * (p * _LOG2PI + logdet + innov @ Sinv @ innov)
        K = P_pred @ struct.C.T @ Sinv
        mu = mu_pred + K @ innov
        P = (np.eye(d) - K @ struct.C) @ P_pred
    return ll, mu


def filter_all(struct: LGStructure, U: np.ndarray, Y: np.ndarray, g_grid: np.ndarray = G_GRID):
    """Filter the history under every gain hypothesis. Returns (post, mus, ABs, base): the posterior
    over g, the per-g filtered final state, the per-g (A,B), and the baseline. This is the expensive
    part and is INDEPENDENT of any future input, so it is computed once and reused to forecast under
    many next-inputs (the 4 recovery shocks, the real next week, etc.)."""
    base = np.full(struct.d, struct.baseline)
    lls, mus, ABs = [], [], []
    for g in g_grid:
        A, B = struct.with_gain(g)
        ll, mu = _kalman_ll_state(struct, A, B, U, Y) if len(U) else (0.0, base.copy())
        lls.append(ll); mus.append(mu); ABs.append((A, B))
    lls = np.asarray(lls)
    w = np.exp(lls - lls.max()); post = w / w.sum()
    return post, mus, ABs, base


def forecast_from(struct: LGStructure, filt, u_next: np.ndarray) -> np.ndarray:
    """Posterior-weighted next-output forecast under a given next input, reusing a `filter_all`."""
    post, mus, ABs, base = filt
    preds = np.zeros(struct.p)
    for i in range(len(post)):
        A, B = ABs[i]
        preds += post[i] * np.clip(struct.C @ (base + A @ (mus[i] - base) + B @ u_next), 0, 100)
    return preds


def forecast_at_horizon(struct: LGStructure, filt, u_pulse: np.ndarray, h: int) -> np.ndarray:
    """Posterior-weighted forecast of y_{k+h} after a single pulse at week k+1 and no news after.
    The h=1 case is exactly `forecast_from`."""
    post, mus, ABs, base = filt
    preds = np.zeros(struct.p)
    for i in range(len(post)):
        A, B = ABs[i]
        s = base + np.linalg.matrix_power(A, h) @ (mus[i] - base) \
            + np.linalg.matrix_power(A, h - 1) @ B @ u_pulse
        preds += post[i] * np.clip(struct.C @ s, 0, 100)
    return preds


def compute_bayes(struct: LGStructure, U: np.ndarray, Y: np.ndarray, u_next: np.ndarray,
                  g_grid: np.ndarray = G_GRID) -> dict:
    """Optimal forecaster: grid over the hidden g, Kalman-filter each hypothesis, posterior-weight.
    Returns g_hat (posterior mean of g), the posterior-weighted next-output forecast, the posterior,
    and the reusable `filt` tuple."""
    filt = filter_all(struct, U, Y, g_grid)
    post = filt[0]
    return {"g_hat": float(post @ np.asarray(g_grid)), "pred": forecast_from(struct, filt, u_next),
            "post": post, "filt": filt}


# ── recovery probe (only where `recover`): implied one-step input->output gain via 4 shocks ───────
def oracle_recovery(struct: LGStructure, U: np.ndarray, Y: np.ndarray,
                    filt=None) -> Optional[float]:
    """The gain the oracle IMPLIES for the probe channel: slope of its forecast probe-output vs. a
    shock on the probe-input, matching exp2's 4-shock predict->slope readout. Filters the history
    ONCE (or reuses a passed-in `filt`) and forecasts the 4 shocks cheaply. None if not `recover`."""
    if not struct.recover:
        return None
    if filt is None:
        filt = filter_all(struct, U, Y)
    xs = np.asarray(SHOCKS, float); ys = []
    for s in SHOCKS:
        u = np.zeros(struct.m); u[struct.probe_input] = s
        ys.append(forecast_from(struct, filt, u)[struct.probe_output])
    ys = np.asarray(ys)
    return float(np.cov(xs, ys, bias=True)[0, 1] / np.var(xs))


def true_recovery_gain(struct: LGStructure, g: float) -> Optional[float]:
    """The TRUE one-step input->output sensitivity (C B)[probe_output, probe_input] at the drawn g
    -- the target the implied gain is scored against. For chain1_gain this is exactly g."""
    if not struct.recover:
        return None
    _, B = struct.with_gain(g)
    return float((struct.C @ B)[struct.probe_output, struct.probe_input])
