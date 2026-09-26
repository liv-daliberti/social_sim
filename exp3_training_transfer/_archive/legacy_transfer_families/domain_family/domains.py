"""domains.py -- the COIN-* domain family for Experiment 3 (domain-transfer arm).

Experiment 3's original synthetic arm varied TOPOLOGY (12 LG-DAG structures, 8 train / 4 held out)
inside a single cover story: news -> opinion -> poll. That tests whether a model can use a
described-but-unseen graph. It does NOT test whether the forecasting skill is tied to the surface
domain it was trained on, because every world was an election.

This module holds the graph FIXED (the `direct` world = the Exp-2 world: driver -> latent -> output,
one hidden gain g) and varies the DOMAIN instead. Train on some domains, evaluate on held-out ones:

    TRAIN pool   CoinCity        net news        -> poll support
                 CoinFishing     sea-state index -> catch
                 CoinFarm        rainfall anomaly-> yield
    HELD OUT     CoinBasketball  roster-health   -> points scored
                 CoinClinic      staffing delta  -> patient wait time

Every domain shares ONE abstract task -- infer a hidden per-episode sensitivity g from a short
history, then forecast the observable under four probe interventions -- so a model that has learned
the skill rather than the vocabulary should carry it into a domain it never trained on.

WHY AN AFFINE SKIN, NOT A NEW SIMULATOR
---------------------------------------
If the domains were the same numbers with different nouns, "domain transfer" would only demonstrate
robustness to renaming. So each domain gets its own observable scale, offset, input scale, noise, and
hence its own IMPLIED sensitivity range -- a model cannot carry over a memorized "g is about 0.5".

We get that without touching lg_dag.py's verified stability/clipping gates. The canonical world runs
exactly as before (baseline 50, outputs clipped to [0,100], g ~ U[0.1, 1.0]); each domain then maps
canonical quantities onto its own units:

    displayed input       x = in_scale * u
    displayed observable  y = offset + obs_scale * (y_canonical - 50)
    implied sensitivity   g_disp = (obs_scale / in_scale) * g_canonical

The map is affine and invertible, so reward targets and the slope target transform exactly the same
way and the Kalman oracle stays optimal in canonical space. Noise is varied INDEPENDENTLY of scale
by giving each domain its own sigma_s/sigma_y on its LGStructure (displayed noise is then
obs_scale * sigma_y), so scale and noise are not confounded.

The resulting displayed ranges are deliberately spread apart, and the two held-out domains sit
INSIDE the convex hull of the training domains' sensitivity ranges -- this is an interpolation test
of domain generality, not an extrapolation test.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

# reuse the verified LG machinery from the structure-transfer arm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dag_family"))
from lg_dag import G_HI, G_LO, LGStructure  # noqa: E402


@dataclass(frozen=True)
class Domain:
    """One cover story over the canonical `direct` world.

    The tuple (offset, obs_scale, in_scale) is the display skin; (sigma_s, sigma_y) is the
    canonical-space noise, so scale and noise vary independently across domains.
    """
    name: str                      # registry key, e.g. "coin_city"
    label: str                     # display name, e.g. "CoinCity"
    entity: str                    # "city", "fishing ground", ...
    entity_plural: str
    period: str                    # "week", "trip", "season", "game", "day"
    input_name: str                # column header for the driver
    input_desc: str                # one clause explaining the driver's sign convention
    output_name: str               # column header for the observable
    output_desc: str               # noun phrase for the observable, used in the question
    output_unit: str               # "poll points", "crates", ...
    latent_name: str               # "public opinion", "fish stock", ...
    offset: float                  # displayed value at canonical baseline 50
    obs_scale: float               # displayed units per canonical point
    in_scale: float                # displayed driver units per canonical driver unit
    sigma_s: float                 # canonical latent noise
    sigma_y: float                 # canonical observation noise
    decimals: int = 0              # display precision for the observable
    in_decimals: int = 0           # display precision for the driver

    # ---- display range / sensitivity range (derived) ----
    @property
    def out_lo(self) -> float:
        return self.offset - 50.0 * self.obs_scale

    @property
    def out_hi(self) -> float:
        return self.offset + 50.0 * self.obs_scale

    @property
    def gain_ratio(self) -> float:
        """Displayed observable units per displayed driver unit, per unit of canonical g."""
        return self.obs_scale / self.in_scale

    @property
    def g_disp_range(self) -> Tuple[float, float]:
        return (G_LO * self.gain_ratio, G_HI * self.gain_ratio)

    @property
    def sigma_y_disp(self) -> float:
        return self.sigma_y * self.obs_scale

    # ---- the structure this domain instantiates ----
    def structure(self) -> LGStructure:
        """The canonical `direct` world (driver -> latent -> observable, hidden input gain g)
        carrying THIS domain's noise. A/B/C are identical across domains by construction: the
        graph is held fixed so that only the domain varies."""
        return LGStructure(
            name=self.name,
            A=np.array([[0.9]], float),
            B=np.array([[0.5]], float),      # the 0.5 slot is overwritten by the drawn g
            C=np.array([[1.0]], float),
            hidden=("B", 0, 0),
            sigma_s=self.sigma_s,
            sigma_y=self.sigma_y,
            recover=True,
            note=f"{self.label}: {self.input_name} -> {self.latent_name} -> {self.output_name}",
        )

    # ---- the affine display skin ----
    def show_output(self, y_canonical) -> np.ndarray:
        """Canonical observable (baseline 50) -> displayed units."""
        return self.offset + self.obs_scale * (np.asarray(y_canonical, float) - 50.0)

    def show_input(self, u_canonical) -> np.ndarray:
        return self.in_scale * np.asarray(u_canonical, float)

    def hide_input(self, x_displayed) -> np.ndarray:
        """Displayed driver -> canonical driver (inverse of show_input)."""
        return np.asarray(x_displayed, float) / self.in_scale

    def show_slope(self, slope_canonical: float) -> float:
        """Canonical d(output)/d(input) -> displayed d(output)/d(input)."""
        return float(slope_canonical) * self.gain_ratio

    def fmt_output(self, v: float) -> str:
        return f"{v:,.{self.decimals}f}"

    def fmt_input(self, v: float) -> str:
        return f"{v:+,.{self.in_decimals}f}"


# ── the five domains ────────────────────────────────────────────────────────────────────────────
# Displayed ranges and implied sensitivities are deliberately spread apart so that no single
# memorized numeric regime covers the family. CoinCity is the identity skin, so it reproduces the
# Experiment-2 world exactly and keeps the two arms commensurable.
COIN_CITY = Domain(
    name="coin_city", label="CoinCity",
    entity="city", entity_plural="cities", period="week",
    input_name="news", input_desc="positive = good for the candidate, negative = bad",
    output_name="poll", output_desc="poll support", output_unit="poll points",
    latent_name="public opinion",
    offset=50.0, obs_scale=1.0, in_scale=1.0, sigma_s=1.0, sigma_y=2.0,
)
COIN_FISHING = Domain(
    name="coin_fishing", label="CoinFishing",
    entity="fishing ground", entity_plural="fishing grounds", period="trip",
    input_name="sea_state", input_desc="positive = favourable water, negative = rough water",
    output_name="catch", output_desc="the catch", output_unit="crates",
    latent_name="fish stock",
    offset=200.0, obs_scale=4.0, in_scale=2.0, sigma_s=1.5, sigma_y=3.0,
)
COIN_FARM = Domain(
    name="coin_farm", label="CoinFarm",
    entity="plot", entity_plural="plots", period="season",
    input_name="rainfall", input_desc="positive = above-normal rain, negative = drought",
    output_name="yield", output_desc="the yield", output_unit="bushels per acre",
    latent_name="soil condition",
    offset=120.0, obs_scale=2.4, in_scale=1.5, sigma_s=0.8, sigma_y=1.6, decimals=1,
)
COIN_BASKETBALL = Domain(
    name="coin_basketball", label="CoinBasketball",
    entity="team", entity_plural="teams", period="game",
    input_name="roster", input_desc="positive = key players healthy, negative = injuries",
    output_name="points", output_desc="points scored", output_unit="points",
    latent_name="team form",
    offset=100.0, obs_scale=0.6, in_scale=0.5, sigma_s=1.2, sigma_y=2.4, in_decimals=1,
)
COIN_CLINIC = Domain(
    name="coin_clinic", label="CoinClinic",
    entity="clinic", entity_plural="clinics", period="day",
    input_name="staffing", input_desc="positive = extra staff on shift, negative = short-staffed",
    output_name="wait", output_desc="the patient wait time", output_unit="minutes",
    latent_name="clinic congestion",
    offset=90.0, obs_scale=1.8, in_scale=1.0, sigma_s=1.0, sigma_y=2.2,
)

# Ordered training pool: arm Dk trains on the FIRST k domains, so D1 c D2 c D3 is a nested ladder
# and the only thing that changes between arms is domain diversity (row count is held fixed by
# make_dataset_domains.py).
TRAIN_POOL = [COIN_CITY, COIN_FISHING, COIN_FARM]
HELD_OUT = [COIN_BASKETBALL, COIN_CLINIC]
ALL_DOMAINS = TRAIN_POOL + HELD_OUT
BY_NAME = {d.name: d for d in ALL_DOMAINS}


def train_domains(n: int) -> Sequence[Domain]:
    """The first `n` domains of the nested training ladder (n = 1, 2, 3)."""
    if not 1 <= n <= len(TRAIN_POOL):
        raise ValueError(f"n must be in 1..{len(TRAIN_POOL)}, got {n}")
    return TRAIN_POOL[:n]


# ── health gates: fail fast on a bad edit rather than poisoning a training run ──────────────────
def _check() -> None:
    names = [d.name for d in ALL_DOMAINS]
    assert len(names) == len(set(names)), f"duplicate domain names: {names}"
    assert not (set(d.name for d in TRAIN_POOL) & set(d.name for d in HELD_OUT)), \
        "a domain appears in both the training pool and the held-out set"

    for d in ALL_DOMAINS:
        assert d.obs_scale > 0 and d.in_scale > 0, f"{d.name}: scales must be positive"
        assert d.sigma_s > 0 and d.sigma_y > 0, f"{d.name}: noise must be positive"
        # the affine skin must be an exact round trip on the driver
        u = np.array([-12.0, -3.0, 5.0, 11.0])
        assert np.allclose(d.hide_input(d.show_input(u)), u), f"{d.name}: input skin not invertible"
        # the canonical structure must still pass lg_dag's stability gate at both ends of g
        st = d.structure()
        for g in (G_LO, G_HI):
            A, _ = st.with_gain(g)
            rho = float(np.max(np.abs(np.linalg.eigvals(A))))
            assert rho < 1.0, f"{d.name}: unstable dynamics, spectral radius {rho:.3f}"
        assert 25.0 <= (st.C @ np.full(st.d, st.baseline)).item() <= 75.0, \
            f"{d.name}: canonical baseline too close to the clip bounds"

    # The held-out sensitivity ranges must lie inside the hull of the training ranges: this is an
    # interpolation test of domain generality. If a future edit breaks this, the paper's claim
    # changes from interpolation to extrapolation, so fail loudly.
    lo = min(d.g_disp_range[0] for d in TRAIN_POOL)
    hi = max(d.g_disp_range[1] for d in TRAIN_POOL)
    for d in HELD_OUT:
        g0, g1 = d.g_disp_range
        assert lo <= g0 and g1 <= hi, (
            f"{d.name}: implied sensitivity range [{g0:.3f}, {g1:.3f}] escapes the training hull "
            f"[{lo:.3f}, {hi:.3f}] -- transfer would become an extrapolation claim")

    # Domains must be numerically DISTINGUISHABLE, else this is a renaming study. Require that no
    # two domains share both an observable range and an implied sensitivity range.
    for i, a in enumerate(ALL_DOMAINS):
        for b in ALL_DOMAINS[i + 1:]:
            same_out = np.isclose(a.out_lo, b.out_lo) and np.isclose(a.out_hi, b.out_hi)
            same_g = np.allclose(a.g_disp_range, b.g_disp_range)
            assert not (same_out and same_g), \
                f"{a.name} and {b.name} are numerically identical -- vary scale or noise"


_check()


def summary_table() -> str:
    """Human-readable roster, also used by the preflight audit and the paper's methods table."""
    hdr = (f"{'domain':<16}{'role':<7}{'observable':<22}{'range':>16}"
           f"{'sigma_y':>9}{'implied g':>16}{'period':>9}")
    lines = [hdr, "-" * len(hdr)]
    for d in ALL_DOMAINS:
        role = "train" if d in TRAIN_POOL else "HELD"
        g0, g1 = d.g_disp_range
        lines.append(
            f"{d.label:<16}{role:<7}{d.output_name + ' (' + d.output_unit + ')':<22}"
            f"{d.fmt_output(d.out_lo) + '-' + d.fmt_output(d.out_hi):>16}"
            f"{d.sigma_y_disp:>9.2f}{f'{g0:.2f}-{g1:.2f}':>16}{d.period:>9}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary_table())
