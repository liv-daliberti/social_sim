#!/usr/bin/env python3
"""Build OAT-format train/held-out datasets for the Experiment-3 DOMAIN-transfer arm.

The question is whether forecasting skill learned in some domains carries into a domain the model
never trained on. Arms form a NESTED LADDER over the training pool:

    d1            CoinCity                                   1 training domain
    d2            CoinCity + CoinFishing                     2 training domains
    d3            CoinCity + CoinFishing + CoinFarm          3 training domains
    structureless same three domains, driver decoupled       format + level only (control)

Every arm is evaluated on the SAME held-out domains (CoinBasketball, CoinClinic), which appear in no
arm's training data.

TWO CONFOUNDS THIS BUILDER CONTROLS
-----------------------------------
1. QUANTITY. Every arm gets exactly MAX_TRAIN rows, split evenly across its domains (d1: 4800 from
   one domain; d3: 1600 from each of three). Without this, "more domains" would also mean "more
   data" and the ladder would measure the wrong thing.

2. REWARD DIFFICULTY. Domains differ in observable scale, so a fixed points-valued tolerance would
   make CoinFishing (0-400 crates) roughly four times harder than CoinCity (0-100 poll points) for
   the same relative accuracy. Each row therefore carries its own `reward_scale` and `slope_scale`,
   scaled by the domain's display skin, plus the `clip` range the oracle should use.

Episodes are drawn per domain from disjoint seed blocks, so no canonical episode is ever re-skinned
into two domains -- otherwise CoinFishing would literally be CoinCity's numbers times four, and a
model could transfer by spotting the ratio rather than by inferring a sensitivity.

The probed structure is `direct` for every domain, whose first-response horizon is 1, so the
canonical slope target is exactly the drawn g and the displayed slope target is g * gain_ratio.

Usage:
    python make_dataset_domains.py --arm d1             # -> data/domain_d1
    python make_dataset_domains.py --arm d2             # -> data/domain_d2
    python make_dataset_domains.py --arm d3             # -> data/domain_d3
    python make_dataset_domains.py --arm structureless  # -> data/domain_structureless
    python make_dataset_domains.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dag_family"))
from lg_dag import SHOCKS, generate_episode, true_output_at_horizon  # noqa: E402

from domains import ALL_DOMAINS, HELD_OUT, TRAIN_POOL, Domain, train_domains  # noqa: E402
from prompt_domain import build_prompt_domain  # noqa: E402

ARMS = ("d1", "d2", "d3", "structureless")
OUT_DIR = {a: f"domain_{a}" for a in ARMS}

# Total training rows per arm -- held FIXED across the ladder so only diversity varies.
MAX_TRAIN = 4800
N_EVAL_PER = 120                 # episodes per held-out domain
EVAL_KS = (1, 2, 3, 5, 7, 9)     # 2 domains x 120 x 6 = 1440 held-out rows
K_RANGE = (1, 9)
T = 10
HORIZON = 1                      # `direct` responds immediately, so h* = 1 for every domain

# Base reward tolerances, in CANONICAL units. Each row scales these by its domain's skin so the
# reward is equally forgiving in relative terms across domains. These must stay in step with the
# launcher's --reward_scale_pts / --slope_scale_g defaults, which apply to legacy dag_family rows.
REWARD_SCALE_BASE = 15.0
SLOPE_SCALE_BASE = 0.5

# Disjoint seed blocks: domain index picks the block, train and eval never overlap.
TRAIN_SEED_BASE = 200_000_000
EVAL_SEED_BASE = 700_000_000
DOMAIN_SEED_STRIDE = 10_000_000

TRAIN_SHOCK_POOL_NEG = (-15, -12, -10, -8, -6, -5, -3, -2)
TRAIN_SHOCK_POOL_POS = (2, 3, 5, 6, 8, 10, 12, 15)


def _domain_index(dom: Domain) -> int:
    return [d.name for d in ALL_DOMAINS].index(dom.name)


def _row(dom: Domain, ep: dict, seed: int, k: int, shocks_can, structureless: bool = False,
         rng: np.random.Generator | None = None) -> dict:
    """One multi-scenario row. `shocks_can` are canonical; everything shown and every target is
    mapped through the domain skin."""
    struct = dom.structure()
    U, Y, S, g = ep["U"], ep["Y"], ep["S"], ep["g"]
    shocks_can = [float(s) for s in shocks_can]

    if structureless:
        # Displayed driver is resampled independently of the observable (zero mutual information).
        # Targets are the expected next observable under ZERO input and the slope target is 0, so
        # this arm can learn format, units, and level-tracking but no driver -> observable structure.
        U_shown = (rng.integers(3, 13, size=(len(U), 1)).astype(float)
                   * np.where(rng.random((len(U), 1)) < 0.5, 1.0, -1.0))
        tgt0 = float(true_output_at_horizon(struct, S, k, g, np.zeros(1), HORIZON)[0])
        targets_can = [tgt0] * len(shocks_can)
        slope_can = 0.0
    else:
        U_shown = U
        targets_can = [float(true_output_at_horizon(struct, S, k, g, np.array([s]), HORIZON)[0])
                       for s in shocks_can]
        slope_can = float(g)          # h* = 1 on `direct`, so the response IS the drawn gain

    targets_disp = [float(v) for v in dom.show_output(targets_can)]
    shocks_disp = [float(v) for v in dom.show_input(shocks_can)]

    ref = {
        "targets": [round(t, 3) for t in targets_disp],
        "shocks": [round(s, 3) for s in shocks_disp],
        "slope_target": round(dom.show_slope(slope_can), 4),
        "clip": [dom.out_lo, dom.out_hi],
        "reward_scale": round(REWARD_SCALE_BASE * dom.obs_scale, 4),
        "slope_scale": round(SLOPE_SCALE_BASE * dom.gain_ratio, 4),
        "domain": dom.name,
        "g": float(g),                                  # canonical drawn gain
        "true_gain": round(dom.show_slope(g), 4),       # displayed sensitivity: the recovery target
        "horizon": HORIZON,
        "last_obs": float(dom.show_output(Y[k - 1][0])) if k >= 1 else float(dom.offset),
        "seed": int(seed),
        "k": int(k),
        "recover": True,
        "control": bool(structureless),
        "mode": "structureless" if structureless else "domain",
    }
    return {"input": build_prompt_domain(dom, U_shown[:k, 0], Y[:k, 0], shocks_can),
            "reference": json.dumps(ref)}


def build_train(doms, structureless: bool = False, total: int = MAX_TRAIN) -> list[dict]:
    """`total` rows split evenly across `doms`, so every arm sees the same amount of data."""
    per = total // len(doms)
    rows = []
    for dom in doms:
        block = TRAIN_SEED_BASE + _domain_index(dom) * DOMAIN_SEED_STRIDE
        struct = dom.structure()
        for i in range(per):
            seed = block + i
            rng = np.random.default_rng(seed ^ 0x5EED)
            ep = generate_episode(struct, seed=seed, T=T)
            k = int(rng.integers(K_RANGE[0], K_RANGE[1] + 1))
            neg = rng.choice(TRAIN_SHOCK_POOL_NEG, size=2, replace=False)
            pos = rng.choice(TRAIN_SHOCK_POOL_POS, size=2, replace=False)
            shocks = sorted(int(x) for x in (*neg, *pos))
            rows.append(_row(dom, ep, seed, k, shocks, structureless=structureless, rng=rng))
    np.random.default_rng(20260811).shuffle(rows)
    return rows


def build_eval(n_per: int = N_EVAL_PER, ks=EVAL_KS) -> list[dict]:
    """Held-out domains at the canonical four-shock probe. ALWAYS the real (non-decoupled) worlds:
    the structureless arm trains differently but is tested identically."""
    rows = []
    for dom in HELD_OUT:
        block = EVAL_SEED_BASE + _domain_index(dom) * DOMAIN_SEED_STRIDE
        struct = dom.structure()
        for i in range(n_per):
            seed = block + i
            ep = generate_episode(struct, seed=seed, T=T)
            for k in ks:
                rows.append(_row(dom, ep, seed, k, list(SHOCKS)))
    return rows


def arm_domains(arm: str):
    if arm == "structureless":
        return TRAIN_POOL, True
    return train_domains(int(arm[1:])), False


def build_arm(arm: str, out_root: Path, n_eval_per: int = N_EVAL_PER) -> Path:
    from datasets import Dataset, DatasetDict

    doms, structureless = arm_domains(arm)
    train = build_train(doms, structureless=structureless)
    held = build_eval(n_eval_per)
    out = out_root / OUT_DIR[arm]
    out.mkdir(parents=True, exist_ok=True)
    DatasetDict({"train": Dataset.from_list(train)}).save_to_disk(str(out / "train"))
    DatasetDict({"train": Dataset.from_list(held)}).save_to_disk(str(out / "heldout"))
    names = ", ".join(d.label for d in doms)
    print(f"[{arm:<13}] {len(train):>5} train rows over {len(doms)} domain(s) ({names})"
          f"{' [decoupled]' if structureless else ''}")
    print(f"{'':<15} {len(held):>5} held-out rows over "
          f"{', '.join(d.label for d in HELD_OUT)} -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--all", action="store_true", help="build every arm")
    ap.add_argument("--n-eval-per", type=int, default=N_EVAL_PER)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data")
    args = ap.parse_args()
    if not args.arm and not args.all:
        ap.error("pass --arm or --all")

    for arm in (ARMS if args.all else [args.arm]):
        out = build_arm(arm, args.out, args.n_eval_per)

    from datasets import load_from_disk
    ds = load_from_disk(str(out / "train"))["train"]
    print("\n--- example prompt ---\n" + ds[0]["input"][:1000])
    print("\n--- example reference ---\n" + ds[0]["reference"])


if __name__ == "__main__":
    main()
