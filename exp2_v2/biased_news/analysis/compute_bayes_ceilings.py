#!/usr/bin/env python3
"""What is achievable on Coin City: probe-target ceilings and forecast-MAE floors.

Two kinds of score in this experiment are read against a latent generator
quantity, so before treating either as a fact about a model we have to know what
the displayed evidence makes achievable. This module derives both from the
data-generating process in
``engine/coin_city_stable_relationship_claude_n250.py``; it makes no model call
and reads no response file.

The forecast floor is the MAE a Bayes-optimal reader of each prompt arm attains
at each evidence depth. It is what the reported MAE curves should be read
against: an arm whose floor is 2.27 and an arm whose floor is 0.17 are not
comparably hard, and a deployment sitting at 1.6 is near-optimal in the first and
an order of magnitude short in the second.

The three targets behave very differently.

  REGIME        Binary strong/weak. A correct semantic cue names it, so the
                ceiling is 1.000. Under arbitrary labels it must be induced from
                the two reference cities, whose fitted slopes are themselves
                noisy, and the ceiling falls well below 1.
  FULL SLOPE    g_C. Regime membership explains essentially all of its variance,
                so wherever the regime is recoverable the ceiling is near 1.
  RESIDUAL      g_C minus its regime mean. Generated with standard deviation
                CITY_SLOPE_SD = .03, and estimable only from City C's own rows,
                whose through-origin fit carries a standard deviation an order of
                magnitude larger. At k=0 City C displays no rows and the residual
                is unidentifiable: the ceiling is exactly zero.

For a Bayes posterior mean the held-out R^2 ceiling against a target with prior
variance tau^2 observed through noise sigma^2 is tau^2 / (tau^2 + sigma^2); the
Monte Carlo path re-derives the same numbers from simulated episodes built with
the generator's own row logic, including its 0.1 rounding, and the two are
required to agree.

Outputs:
  <data-root>/analysis/bayes_ceilings.json
  paper/tables/exp2_probe_target_ceilings.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
CODE_ROOT = _HERE.parent
REPO_ROOT = CODE_ROOT.parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from engine.coin_city_stable_relationship_claude_n250 import (  # noqa: E402
    CASE_NOISE_SD,
    QUERY_NEWS_VALUES,
    CITY_SLOPE_SD,
    EXPERIMENT,
    NEWS_MAGNITUDES,
    STRONG_SLOPE_MEAN,
    WEAK_SLOPE_MEAN,
)

DATA_ROOT = CODE_ROOT / "data" / EXPERIMENT
OUT_JSON = DATA_ROOT / "analysis" / "bayes_ceilings.json"
OUT_TEX = REPO_ROOT / "paper" / "tables" / "exp2_probe_target_ceilings.tex"

DEFAULT_DRAWS = 400_000
DEFAULT_SEED = 20260826
ROUNDING = 0.1  # engine rounds every displayed ending poll to one decimal


def _matched_pair_news(rng: np.random.Generator, n: int, pairs: int) -> np.ndarray:
    """News values for `pairs` matched equal-and-opposite pairs, as the engine draws them."""
    magnitude = rng.choice(NEWS_MAGNITUDES, size=(n, pairs))
    sign = rng.choice((-1, 1), size=(n, pairs))
    signed = magnitude * sign
    return np.stack([v for pair in range(pairs) for v in (signed[:, pair], -signed[:, pair])],
                    axis=1)


def _fit_through_origin(news: np.ndarray, slope: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """Through-origin OLS slope from displayed rows, with generator noise and rounding."""
    noise = rng.normal(0.0, CASE_NOISE_SD, news.shape)
    quantization = rng.uniform(-ROUNDING / 2, ROUNDING / 2, news.shape)
    delta = slope[:, None] * news + noise + quantization
    return (news * delta).sum(1) / (news ** 2).sum(1)


def analytic_ceilings() -> dict:
    """Closed-form R^2 ceilings for the two continuous targets."""
    tau2 = CITY_SLOPE_SD ** 2
    regime_var = ((STRONG_SLOPE_MEAN - WEAK_SLOPE_MEAN) / 2.0) ** 2
    # Expected 1/sum(x^2) over the matched-pair news draw, for k rows.
    magnitudes = np.array(NEWS_MAGNITUDES, dtype=float)
    out = {"residual_slope": {}, "full_slope": {}}
    for k in (0, 1, 2, 3, 4):
        if k == 0:
            out["residual_slope"]["k=0"] = 0.0
            out["full_slope"]["k=0"] = regime_var / (regime_var + tau2)
            continue
        # sum(x^2) over k rows drawn as matched pairs from NEWS_MAGNITUDES
        pairs_complete, leftover = divmod(k, 2)
        sq = np.zeros(1)
        for _ in range(pairs_complete):
            sq = (sq[:, None] + 2 * magnitudes[None, :] ** 2).ravel()
        if leftover:
            sq = (sq[:, None] + magnitudes[None, :] ** 2).ravel()
        sigma2 = float(np.mean(CASE_NOISE_SD ** 2 / sq))
        shrink = tau2 / (tau2 + sigma2)
        out["residual_slope"][f"k={k}"] = shrink
        out["full_slope"][f"k={k}"] = (regime_var + tau2 * shrink) / (regime_var + tau2)
    return out


def monte_carlo(draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    strong_is_a = rng.integers(0, 2, draws).astype(bool)
    c_is_strong = rng.integers(0, 2, draws).astype(bool)

    mu_a = np.where(strong_is_a, STRONG_SLOPE_MEAN, WEAK_SLOPE_MEAN)
    mu_b = np.where(strong_is_a, WEAK_SLOPE_MEAN, STRONG_SLOPE_MEAN)
    mu_c = np.where(c_is_strong, STRONG_SLOPE_MEAN, WEAK_SLOPE_MEAN)
    g_a = rng.normal(mu_a, CITY_SLOPE_SD)
    g_b = rng.normal(mu_b, CITY_SLOPE_SD)
    g_c = rng.normal(mu_c, CITY_SLOPE_SD)

    ref_news = _matched_pair_news(rng, draws, pairs=2)
    b_a = _fit_through_origin(ref_news, g_a, rng)
    b_b = _fit_through_origin(_matched_pair_news(rng, draws, pairs=2), g_b, rng)

    result: dict = {}

    # --- Does a reference fit carry information about City C beyond the regime? --
    matched_ref = np.where(c_is_strong == strong_is_a, b_a, b_b)
    matched_mu = np.where(c_is_strong == strong_is_a, mu_a, mu_b)
    result["reference_fit_sd_within_regime"] = float((matched_ref - matched_mu).std())
    result["corr_reference_residual_with_city_c_residual"] = float(
        np.corrcoef(matched_ref - matched_mu, g_c - mu_c)[0, 1])
    result["corr_reference_fit_with_city_c_slope"] = float(
        np.corrcoef(matched_ref, g_c)[0, 1])
    result["corr_regime_indicator_with_city_c_slope"] = float(
        np.corrcoef(mu_c, g_c)[0, 1])

    # --- Regime AUC ceiling under an arbitrary label -----------------------------
    # The label tells which reference shares City C's regime; that reference's
    # regime must be read off the two noisy fits. Bayes-optimal score for
    # "City C is strong" is the posterior that the label-matched reference is the
    # strong one, which is monotone in the signed difference of the fits.
    label_matched_is_a = (c_is_strong == strong_is_a)
    score = np.where(label_matched_is_a, b_a - b_b, b_b - b_a)
    order = np.argsort(score)
    ranks = np.empty(draws, float)
    ranks[order] = np.arange(1, draws + 1)
    pos, neg = c_is_strong.sum(), (~c_is_strong).sum()
    result["regime_auc_ceiling_arbitrary_label_k0"] = float(
        (ranks[c_is_strong].sum() - pos * (pos + 1) / 2) / (pos * neg))
    result["regime_accuracy_ceiling_arbitrary_label_k0"] = float(
        ((score > 0) == c_is_strong).mean())
    result["regime_auc_ceiling_correct_semantic_cue"] = 1.0

    # --- Residual and full-slope R^2 ceilings from City C's own rows ------------
    result["r2_ceiling"] = {"residual_slope": {}, "full_slope": {}}
    target_news_full = _matched_pair_news(rng, draws, pairs=2)
    for k in (0, 1, 2, 3, 4):
        if k == 0:
            result["r2_ceiling"]["residual_slope"]["k=0"] = 0.0
            # regime known (correct cue) and nothing else
            resid_hat = np.zeros(draws)
        else:
            news_k = target_news_full[:, :k]
            b_c = _fit_through_origin(news_k, g_c, rng)
            sigma2 = CASE_NOISE_SD ** 2 / (news_k ** 2).sum(1)
            tau2 = CITY_SLOPE_SD ** 2
            resid_hat = (tau2 / (tau2 + sigma2)) * (b_c - mu_c)
            resid = g_c - mu_c
            result["r2_ceiling"]["residual_slope"][f"k={k}"] = float(
                1 - ((resid - resid_hat) ** 2).mean() / (resid ** 2).mean())
        slope_hat = mu_c + resid_hat
        result["r2_ceiling"]["full_slope"][f"k={k}"] = float(
            1 - ((g_c - slope_hat) ** 2).mean() / ((g_c - g_c.mean()) ** 2).mean())
    return result


def forecast_mae_floor(draws: int, seed: int) -> dict:
    """Bayes-optimal forecast MAE for each prompt arm at each evidence depth.

    ``cued`` is the correct-context arm, where the sentence names City C's regime
    outright. ``blind`` covers the target-only and no-context arms, which display
    no statement of City C's regime and must infer it, if at all, from City C's
    own rows; at k=0 they have nothing and the optimal forecast is a constant.
    The misleading arm shares the cued floor, since a reader who knew the cue was
    inverted would simply invert it.
    """
    rng = np.random.default_rng(seed)
    strong = rng.integers(0, 2, draws).astype(bool)
    mu = np.where(strong, STRONG_SLOPE_MEAN, WEAK_SLOPE_MEAN)
    g = rng.normal(mu, CITY_SLOPE_SD)
    x = rng.choice(np.array(QUERY_NEWS_VALUES, dtype=float), draws)
    rows = _matched_pair_news(rng, draws, pairs=2)
    tau2 = CITY_SLOPE_SD ** 2
    out = {"cued": {}, "blind": {}}
    for k in range(5):
        if k == 0:
            # Nothing identifies City C's regime, so the MAE-optimal forecast is a
            # constant; every value between the two regime means scores the same.
            blind = np.abs((0.5 * (STRONG_SLOPE_MEAN + WEAK_SLOPE_MEAN) - g) * x).mean()
            cued = np.abs((mu - g) * x).mean()
        else:
            news = rows[:, :k]
            b = _fit_through_origin(news, g, rng)
            s2 = CASE_NOISE_SD ** 2 / (news ** 2).sum(1)
            shrink = tau2 / (tau2 + s2)
            cued = np.abs((mu + shrink * (b - mu) - g) * x).mean()
            # Blind: posterior over the two regimes from City C's rows alone,
            # then shrink within the modal regime.
            dens = [np.exp(-0.5 * (b - m) ** 2 / (tau2 + s2))
                    for m in (WEAK_SLOPE_MEAN, STRONG_SLOPE_MEAN)]
            modal = np.where(dens[0] > dens[1], WEAK_SLOPE_MEAN, STRONG_SLOPE_MEAN)
            blind = np.abs((modal + shrink * (b - modal) - g) * x).mean()
        out["cued"][f"k={k}"] = float(cued)
        out["blind"][f"k={k}"] = float(blind)
    return out


def write_table(analytic: dict, mc: dict) -> None:
    rows = [
        (r"Regime, correct semantic cue", "AUC", "1.000", "1.000"),
        (r"Regime, arbitrary label",      "AUC",
         f"{mc['regime_auc_ceiling_arbitrary_label_k0']:.3f}", "---"),
        (r"Full slope $g_C$, known regime", "$R^2$",
         f"{analytic['full_slope']['k=0']:.3f}", f"{analytic['full_slope']['k=4']:.3f}"),
        (r"Residual slope $g_C-\mu$, known regime", "$R^2$",
         f"{analytic['residual_slope']['k=0']:.3f}",
         f"{analytic['residual_slope']['k=4']:.3f}"),
    ]
    body = "\n".join(f"{name} & {unit} & {k0} & {k4} \\\\" for name, unit, k0, k4 in rows)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(
        "% Generated by exp2_v2/biased_news/analysis/compute_bayes_ceilings.py;"
        " do not edit.\n"
        "\\begin{tabular}{llcc}\n\\toprule\n"
        "\\textbf{Probe target} & & $k=0$ & $k=4$ \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    analytic = analytic_ceilings()
    mc = monte_carlo(args.draws, args.seed)
    floor = forecast_mae_floor(args.draws, args.seed + 1)

    # Fail closed if the closed form and the simulated generator disagree.
    for target in ("residual_slope", "full_slope"):
        for k, value in analytic[target].items():
            simulated = mc["r2_ceiling"][target][k]
            if abs(simulated - value) > 5e-3:
                raise SystemExit(
                    f"ceiling mismatch for {target} {k}: "
                    f"analytic {value:.4f} vs Monte Carlo {simulated:.4f}")

    result = {
        "analysis": "coin_city_bayes_ceilings",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT,
        "makes_model_calls": False,
        "generator_constants": {
            "strong_slope_mean": STRONG_SLOPE_MEAN,
            "weak_slope_mean": WEAK_SLOPE_MEAN,
            "city_slope_sd": CITY_SLOPE_SD,
            "case_noise_sd": CASE_NOISE_SD,
            "news_magnitudes": list(NEWS_MAGNITUDES),
            "displayed_poll_rounding": ROUNDING,
        },
        "monte_carlo": {"draws": args.draws, "seed": args.seed},
        "analytic_r2_ceiling": analytic,
        "simulated": mc,
        "forecast_mae_floor": floor,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    write_table(analytic, mc)

    print("R^2 ceiling (analytic / Monte Carlo)")
    for target in ("full_slope", "residual_slope"):
        for k in ("k=0", "k=1", "k=2", "k=3", "k=4"):
            print(f"  {target:15s} {k}: {analytic[target][k]:.4f} / "
                  f"{mc['r2_ceiling'][target][k]:.4f}")
    print()
    print(f"regime AUC ceiling, correct semantic cue      : 1.000")
    print(f"regime AUC ceiling, arbitrary label at k=0    : "
          f"{mc['regime_auc_ceiling_arbitrary_label_k0']:.3f}")
    print(f"regime accuracy ceiling, arbitrary label k=0  : "
          f"{mc['regime_accuracy_ceiling_arbitrary_label_k0']:.3f}")
    print()
    print(f"reference fit sd within regime                : "
          f"{mc['reference_fit_sd_within_regime']:.3f} "
          f"(generator slope sd {CITY_SLOPE_SD})")
    print(f"corr(reference residual, City C residual)     : "
          f"{mc['corr_reference_residual_with_city_c_residual']:+.4f}")
    print(f"corr(reference fit, City C slope)             : "
          f"{mc['corr_reference_fit_with_city_c_slope']:.3f}")
    print(f"corr(regime indicator, City C slope)          : "
          f"{mc['corr_regime_indicator_with_city_c_slope']:.3f}")
    print()
    print()
    print("Bayes-optimal forecast MAE (poll points)")
    print(f"  {'k':>2} {'blind arms':>11} {'cued arm':>9}")
    for k in range(5):
        print(f"  {k:>2} {floor['blind'][f'k={k}']:11.3f} {floor['cued'][f'k={k}']:9.3f}")
    print()
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT_TEX.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
