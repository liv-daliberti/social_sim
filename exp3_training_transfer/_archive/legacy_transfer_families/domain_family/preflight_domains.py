#!/usr/bin/env python3
"""Fail-closed preflight for the Experiment-3 DOMAIN-transfer campaign (exp3d).

Everything here is a gate, not a report: any violation raises before a GPU is booked. The previous
campaign burned 24 GPU-hours on jobs that never took a training step, so the rule is that nothing is
submitted until the data has been proven sound.

Gates
-----
LEAKAGE     no held-out domain appears in any arm's training split; train and eval seed blocks are
            disjoint; the scored held-out content is byte-identical across arms (so the arms differ
            only in what they trained on, never in what they are graded against)
MATCHING    every arm has exactly the same number of training rows, so the ladder measures domain
            DIVERSITY rather than data quantity
SCALE       every target lies inside its row's declared clip range, and each row's reward tolerance
            matches its domain's display skin -- otherwise a wide-scale domain is silently harder
SIGNAL      on training rows, corr(true displayed sensitivity, slope target) is ~1 for the real arms
            and ~0 for the structureless control, which is what makes the control a control
IDENTIFIABILITY on held-out rows, an ORACLE that knows g must earn materially more reward than a
            CONSTANT 'ignore the driver' policy, else the probe cannot detect the skill it claims to
            measure

Usage:
    python preflight_domains.py
    python preflight_domains.py --write-manifest protocol/exp3d_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "dag_family"))

from domains import ALL_DOMAINS, HELD_OUT, TRAIN_POOL, BY_NAME  # noqa: E402
from make_dataset_domains import (ARMS, EVAL_SEED_BASE, MAX_TRAIN, OUT_DIR,  # noqa: E402
                                  REWARD_SCALE_BASE, SLOPE_SCALE_BASE, TRAIN_SEED_BASE,
                                  arm_domains)


class PreflightError(AssertionError):
    pass


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise PreflightError(msg)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_hashes(path: Path) -> dict[str, str]:
    return {str(p.relative_to(ROOT)): _sha256(p)
            for p in sorted(path.rglob("*")) if p.is_file()}


def _refs(split_dir: Path) -> list[dict[str, Any]]:
    from datasets import load_from_disk
    ds = load_from_disk(str(split_dir))["train"]
    return [json.loads(r) for r in ds["reference"]]


def _inputs(split_dir: Path) -> list[str]:
    from datasets import load_from_disk
    return list(load_from_disk(str(split_dir))["train"]["input"])


def _corr(xs, ys) -> float | None:
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _reward(preds, ref: dict) -> float:
    """Mirror of BiasedNewsOracle._reward_multi, so the preflight scores exactly what training will."""
    targets = np.asarray(ref["targets"], float)
    shocks = np.asarray(ref["shocks"], float)
    lo, hi = ref["clip"]
    preds = np.clip(np.asarray(preds, float), lo, hi)
    level = float(np.mean(np.maximum(0.0, 1.0 - np.abs(preds - targets) / ref["reward_scale"])))
    slope = float(np.cov(shocks, preds, bias=True)[0, 1] / np.var(shocks))
    slope_r = max(0.0, 1.0 - abs(slope - ref["slope_target"]) / ref["slope_scale"])
    return 0.5 * level + 0.5 * slope_r


def audit() -> dict[str, Any]:
    data = ROOT / "data"
    arms: dict[str, Any] = {}
    eval_fingerprints: dict[str, str] = {}
    held_names = {d.name for d in HELD_OUT}
    train_pool_names = {d.name for d in TRAIN_POOL}

    for arm in ARMS:
        base = data / OUT_DIR[arm]
        _require(base.exists(), f"{arm}: dataset missing at {base} -- run make_dataset_domains.py")
        tr, ev = _refs(base / "train"), _refs(base / "heldout")
        doms, structureless = arm_domains(arm)
        expect = {d.name for d in doms}

        # ---- MATCHING ------------------------------------------------------------------------
        _require(len(tr) == MAX_TRAIN,
                 f"{arm}: {len(tr)} training rows, expected {MAX_TRAIN} (arms must be row-matched)")
        seen = {}
        for r in tr:
            seen[r["domain"]] = seen.get(r["domain"], 0) + 1
        _require(set(seen) == expect, f"{arm}: trained on {sorted(seen)}, expected {sorted(expect)}")
        _require(len(set(seen.values())) == 1,
                 f"{arm}: uneven domain split {seen} -- domains must contribute equally")

        # ---- LEAKAGE -------------------------------------------------------------------------
        _require(not (set(seen) & held_names),
                 f"{arm}: held-out domain(s) {sorted(set(seen) & held_names)} appear in TRAINING")
        ev_names = {r["domain"] for r in ev}
        _require(ev_names == held_names,
                 f"{arm}: evaluated on {sorted(ev_names)}, expected {sorted(held_names)}")
        _require(not (ev_names & train_pool_names),
                 f"{arm}: evaluation contains a training-pool domain")
        tr_seeds = {r["seed"] for r in tr}
        ev_seeds = {r["seed"] for r in ev}
        _require(not (tr_seeds & ev_seeds), f"{arm}: {len(tr_seeds & ev_seeds)} shared train/eval seeds")
        _require(min(ev_seeds) >= EVAL_SEED_BASE, f"{arm}: eval seed below the eval block")
        _require(max(tr_seeds) < EVAL_SEED_BASE, f"{arm}: train seed inside the eval block")

        # the held-out split must be byte-identical across arms: same graded content, always
        fp = hashlib.sha256(
            json.dumps([json.dumps(r, sort_keys=True) for r in ev]).encode()).hexdigest()
        eval_fingerprints[arm] = fp

        # ---- SCALE ---------------------------------------------------------------------------
        for r in tr + ev:
            dom = BY_NAME[r["domain"]]
            lo, hi = r["clip"]
            _require(np.isclose(lo, dom.out_lo) and np.isclose(hi, dom.out_hi),
                     f"{arm}/{r['domain']}: clip {r['clip']} != domain range")
            _require(all(lo - 1e-6 <= t <= hi + 1e-6 for t in r["targets"]),
                     f"{arm}/{r['domain']}: target outside clip range: {r['targets']}")
            _require(np.isclose(r["reward_scale"], REWARD_SCALE_BASE * dom.obs_scale),
                     f"{arm}/{r['domain']}: reward_scale not skin-scaled")
            _require(np.isclose(r["slope_scale"], SLOPE_SCALE_BASE * dom.gain_ratio),
                     f"{arm}/{r['domain']}: slope_scale not skin-scaled")

        # ---- SIGNAL --------------------------------------------------------------------------
        c = _corr([r["true_gain"] for r in tr], [r["slope_target"] for r in tr])
        if structureless:
            _require(c is None or abs(c) < 0.05,
                     f"{arm}: control still carries sensitivity signal (corr={c})")
            _require(all(r["slope_target"] == 0.0 for r in tr), f"{arm}: control slope targets nonzero")
            _require(all(len(set(r["targets"])) == 1 for r in tr),
                     f"{arm}: control targets vary with the driver")
        else:
            _require(c is not None and c > 0.99,
                     f"{arm}: slope target does not track the true sensitivity (corr={c})")

        arms[arm] = {
            "domains": sorted(expect),
            "structureless": structureless,
            "train_rows": len(tr),
            "rows_per_domain": seen,
            "heldout_rows": len(ev),
            "heldout_domains": sorted(ev_names),
            "corr_true_gain_vs_slope_target": c,
            "heldout_fingerprint": fp,
            "dataset_hashes": _tree_hashes(base),
        }

    # ---- LEAKAGE (cross-arm) -------------------------------------------------------------------
    _require(len(set(eval_fingerprints.values())) == 1,
             f"held-out splits differ across arms: {eval_fingerprints}")

    # the ladder must be nested and strictly growing in diversity
    ladder = [set(arms[f"d{i}"]["domains"]) for i in (1, 2, 3)]
    for a, b in zip(ladder, ladder[1:]):
        _require(a < b, f"training ladder is not nested/growing: {a} vs {b}")

    # ---- IDENTIFIABILITY -----------------------------------------------------------------------
    # Score an ORACLE (knows g, answers the noise-free targets) against a CONSTANT policy (ignores
    # the driver, answers the last observation). If the probe cannot separate them, no amount of
    # training could show the skill.
    ev = _refs(data / OUT_DIR["d3"] / "heldout")
    margins = {}
    for dom in HELD_OUT:
        rows = [r for r in ev if r["domain"] == dom.name]
        orc = float(np.mean([_reward(r["targets"], r) for r in rows]))
        con = float(np.mean([_reward([r["last_obs"]] * len(r["targets"]), r) for r in rows]))
        _require(orc > con + 0.10,
                 f"{dom.name}: oracle {orc:.3f} vs constant {con:.3f} -- probe cannot detect the skill")
        margins[dom.name] = {"oracle_reward": round(orc, 4),
                             "constant_reward": round(con, 4),
                             "margin": round(orc - con, 4)}

    return {
        "protocol": "exp3d_domain_transfer_v1",
        "status": "PASS",
        "arms": arms,
        "domains": {d.name: {"label": d.label, "role": "train" if d in TRAIN_POOL else "held_out",
                             "observable": f"{d.output_name} ({d.output_unit})",
                             "range": [d.out_lo, d.out_hi],
                             "sigma_y_displayed": round(d.sigma_y_disp, 3),
                             "implied_g_range": [round(v, 4) for v in d.g_disp_range]}
                    for d in ALL_DOMAINS},
        "identifiability": margins,
        "audits": {
            "row_matched_across_arms": True,
            "heldout_identical_across_arms": True,
            "no_heldout_domain_in_training": True,
            "train_eval_seeds_disjoint": True,
            "targets_within_clip": True,
            "reward_tolerances_skin_scaled": True,
            "training_ladder_nested": True,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-manifest", type=Path)
    args = ap.parse_args()
    try:
        result = audit()
    except PreflightError as exc:
        print(f"PREFLIGHT FAILED\n  {exc}", file=sys.stderr)
        raise SystemExit(1)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.write_manifest:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(rendered, encoding="utf-8")
        print(f"manifest -> {args.write_manifest}")
    for arm in ARMS:
        a = result["arms"][arm]
        print(f"[{arm:<13}] {a['train_rows']} rows over {len(a['domains'])} domain(s) "
              f"{a['domains']} corr={a['corr_true_gain_vs_slope_target']}")
    print("\nidentifiability on held-out domains (oracle vs constant reward):")
    for name, m in result["identifiability"].items():
        print(f"  {name:<18} oracle {m['oracle_reward']:.3f}  constant {m['constant_reward']:.3f}"
              f"  margin +{m['margin']:.3f}")
    print("\nPREFLIGHT PASS")


if __name__ == "__main__":
    main()
