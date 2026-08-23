#!/usr/bin/env python3
"""Build OAT-format train/eval datasets for the LG DAG family (Experiment 3, transfer arm).

2026-07-08 audit redesign — MULTI-SHOCK rows with a slope-aware reward target. Each row is one
(structure, episode, prefix) and asks for next week's poll under FOUR probe shocks jointly
(scenarios A..D, prompt.build_prompt_multi). The reference carries the four noise-free level
targets AND the true one-step input->output slope, so the reward can score the RESPONSE, not just
the level: a constant 'ignore the news' policy provably leaves slope reward on the table in every
world, per city. (The old single-shock rows let a global shrinkage policy collect ~90% of the
achievable reward with zero per-episode inference — that is what the Jul-7 runs learned: rho
collapsed 0.58->0.26 on the held-out direct world while fmae fell.)

Row schema (mirrors the biased-news schema; run_biased_news_rl.py branches on `targets`):

  * input      – prompt.build_prompt_multi: revealed structure + news/poll history + 4 scenarios.
  * reference  – JSON {targets:[4], shocks:[4], slope_target, last_poll, g, structure, seed, k,
                  recover, true_gain, control}.  slope_target = (C@B)[probe_out, probe_in] at the
                  drawn g (the true immediate response; 0 for delayed chains, a KNOWN constant
                  where the hidden edge is a persistence). true_gain kept where `recover` for the
                  ε readout.

TRAIN rows come from the 8 TRAINING structures; the held-out OAT eval split is the 4 TEST
structures (direct = the Exp-2 world, chain4, mediators, two_news_feedback) at prefixes
ks=(1,2,3,5,7,9) — deep prefixes included because chain4/mediators only become identifiable at
k>=5-7 (oracle-vs-blind margin scan) — read by transfer_report.py into the ε/ρ/π transfer table.

CONTROL (--control): structureless training data for the C3 discriminating test. Same episodes,
same prompt format, but the DISPLAYED news is resampled independently of the polls (zero mutual
information), every scenario's level target is the true expected next poll under ZERO input, and
the slope target is 0. Training on this can teach format, level-tracking, and news-suppression —
but no news->poll structure. The eval split is IDENTICAL (real worlds): if control-trained ≈
family-trained on held-out ε/ρ/π, the family "transfer" was format adaptation; if family-trained
wins, that is the C3 evidence.

ARMS (--mode):
  family        real news; targets at the city's TRUE g.       structure + per-city inference
  structureless news decoupled from polls; targets under zero input, slope 0.
                                                               format + level-tracking only
  prior_mean    real news; targets = the PRIOR-BLIND forecast (this structure's Kalman filter with g
                frozen at the range midpoint). Uses the disclosed structure, never infers the city's
                gain -- the best structure-free policy.
                family - structureless  isolates "did training use structure at all?"
                family - prior_mean     isolates "did training infer the PER-CITY gain?"
  The eval split is IDENTICAL (real worlds) for every arm.

Usage:
    python make_dataset_dag.py --mode family        # -> data/rl_multishock
    python make_dataset_dag.py --mode structureless # -> data/rl_multishock_control
    python make_dataset_dag.py --mode prior_mean    # -> data/rl_multishock_priormean
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lg_dag import (SHOCKS, encodes_gain, filter_all, first_response_horizon, forecast_at_horizon,
                    generate_episode, response_at_horizon, true_output_at_horizon)
from catalog import TRAIN, TEST
from prompt import build_prompt_multi

PRIOR_MEAN = 0.55        # midpoint of the gain range; the prior-mean control freezes g here
MODES = ("family", "structureless", "prior_mean")
OUT_DIR = {"family": "rl_multishock", "structureless": "rl_multishock_control",
           "prior_mean": "rl_multishock_priormean"}

# Each structure is probed at its OWN first-response horizon h*: the earliest week a news pulse can
# move the probed poll. Probing a delayed world at h=1 is degenerate (all four scenario targets
# coincide, slope_target=0, a constant answer is optimal), so the fixed one-step probe carried no
# information about g in 5 of 8 training and 2 of 4 held-out worlds. With h* it carries g wherever
# the gain sits on an input edge: 6/8 training and 4/4 held-out.
HORIZON = {}      # struct.name -> h*
ENCODES = {}      # struct.name -> does the response at h* vary with g?

TRAIN_SEED_BASE = 100_000
EVAL_SEED_BASE = 90_000_000
# training shock quadruples are sampled (2 negative + 2 positive) from these pools so the model
# learns the response across a range rather than memorizing the four eval points
TRAIN_SHOCK_POOL_NEG = (-15, -12, -10, -8, -6, -5, -3, -2)
TRAIN_SHOCK_POOL_POS = (2, 3, 5, 6, 8, 10, 12, 15)
K_RANGE = (1, 9)                                       # observed-weeks prefix sampled per row (T=10)
EVAL_KS = (1, 2, 3, 5, 7, 9)


def _horizon(struct) -> int:
    if struct.name not in HORIZON:
        h = first_response_horizon(struct)
        HORIZON[struct.name] = h
        ENCODES[struct.name] = encodes_gain(struct, h)
    return HORIZON[struct.name]


def _row(struct, ep, seed, k, shocks, mode: str = "family", rng=None):
    """mode:
      family        real news, targets at the city's TRUE g          (structure + per-city inference)
      structureless news decoupled from polls, targets under zero input, slope 0
                                                                     (format + level only)
      prior_mean    real news, targets = the PRIOR-BLIND forecast: this structure's Kalman filter
                    with g frozen at the range midpoint. It uses the disclosed structure but never
                    infers the city's gain -- the best structure-free policy. family MINUS this
                    isolates PER-CITY inference, which family MINUS structureless does not.
    """
    U, Y, S, g = ep["U"], ep["Y"], ep["S"], ep["g"]
    po = struct.probe_output
    h = _horizon(struct)
    shocks = [float(s) for s in shocks]
    if mode == "structureless":
        # display news that is INDEPENDENT of the polls; targets ignore the displayed scenarios
        U_shown = np.stack([rng.integers(3, 13, size=struct.m).astype(float)
                            * np.where(rng.random(struct.m) < 0.5, 1.0, -1.0)
                            for _ in range(len(U))])
        tgt0 = float(true_output_at_horizon(struct, S, k, g, np.zeros(struct.m), h)[po])
        targets = [tgt0] * len(shocks)
        slope_t = 0.0
    elif mode == "prior_mean":
        U_shown = U
        filt = filter_all(struct, U[:k], Y[:k], g_grid=np.asarray([PRIOR_MEAN]))
        targets = []
        for s in shocks:
            u = np.zeros(struct.m); u[struct.probe_input] = s
            targets.append(float(forecast_at_horizon(struct, filt, u, h)[po]))
        x = np.asarray(shocks); y = np.asarray(targets)
        slope_t = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
    else:
        U_shown = U
        targets, slope_t = [], response_at_horizon(struct, g, h)
        for s in shocks:
            u = np.zeros(struct.m); u[struct.probe_input] = s
            targets.append(float(true_output_at_horizon(struct, S, k, g, u, h)[po]))
    # recovery is defined iff the response at h* actually varies with g (i.e. g sits on an input
    # edge). Where g hides in a persistence the response is a known constant and identifies nothing.
    rec = ENCODES[struct.name]
    ref = {
        "targets": [round(t, 3) for t in targets],     # noise-free reward targets, one per scenario
        "shocks": shocks,
        "slope_target": round(slope_t, 4),             # true response at h*: the slope to be matched
        "horizon": int(h),
        "last_poll": float(Y[k - 1][po]) if k >= 1 else float(struct.baseline),
        "g": float(g),
        "structure": struct.name,
        "seed": int(seed),
        "k": int(k),
        "recover": bool(rec),                          # is a gain recovery defined at h*?
        "true_gain": (round(response_at_horizon(struct, g, h), 4) if rec else None),
        "control": bool(mode != "family"),
        "mode": mode,
    }
    return {"input": build_prompt_multi(struct, U_shown[:k], Y[:k], shocks, horizon=h),
            "reference": json.dumps(ref)}


def build_train(n_per: int, mode: str = "family") -> list[dict]:
    """One multi-shock row per (train structure, episode): random prefix k, sampled shock quadruple."""
    rows = []
    for si, struct in enumerate(TRAIN):
        for i in range(n_per):
            seed = TRAIN_SEED_BASE + si * 1_000_000 + i
            rng = np.random.default_rng(seed ^ 0x5EED)
            ep = generate_episode(struct, seed=seed, T=10)
            k = int(rng.integers(K_RANGE[0], K_RANGE[1] + 1))
            neg = rng.choice(TRAIN_SHOCK_POOL_NEG, size=2, replace=False)
            pos = rng.choice(TRAIN_SHOCK_POOL_POS, size=2, replace=False)
            shocks = sorted(int(x) for x in (*neg, *pos))
            rows.append(_row(struct, ep, seed, k, shocks, mode=mode, rng=rng))
    rng = np.random.default_rng(0xDA6)
    rng.shuffle(rows)
    return rows


def build_eval(n_per: int, ks=EVAL_KS) -> list[dict]:
    """Held-out TEST structures, canonical four-shock probe, one row per (city, prefix). ALWAYS the
    real (non-control) worlds: the control arm is trained differently but tested identically."""
    rows = []
    for si, struct in enumerate(TEST):
        for i in range(n_per):
            seed = EVAL_SEED_BASE + si * 1_000_000 + i
            ep = generate_episode(struct, seed=seed, T=10)
            for k in ks:
                rows.append(_row(struct, ep, seed, k, list(SHOCKS)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train-per", type=int, default=600, help="episodes per train structure")
    ap.add_argument("--n-eval-per", type=int, default=60, help="episodes per held-out structure")
    ap.add_argument("--mode", choices=MODES, default="family",
                    help="family | structureless (news decoupled) | prior_mean (best structure-free)")
    ap.add_argument("--control", action="store_true",
                    help="[deprecated] alias for --mode structureless")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    mode = "structureless" if args.control else args.mode
    out = args.out or Path(__file__).resolve().parent / "data" / OUT_DIR[mode]

    from datasets import Dataset, DatasetDict

    train = build_train(args.n_train_per, mode=mode)
    held = build_eval(args.n_eval_per)
    out.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": Dataset.from_list(train)}).save_to_disk(str(out / "train"))
    DatasetDict({"train": Dataset.from_list(held)}).save_to_disk(str(out / "heldout"))

    print(f"wrote {len(train)} {mode.upper()} train rows ({len(TRAIN)} structures) -> {out/'train'}")
    print(f"wrote {len(held)} held-out rows ({len(TEST)} structs x {args.n_eval_per} cities x ks={EVAL_KS}) -> {out/'heldout'}")
    print("\n--- example prompt ---\n" + train[0]["input"][:900])
    print("\n--- example reference ---\n" + train[0]["reference"])


if __name__ == "__main__":
    main()
