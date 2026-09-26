#!/usr/bin/env python3
"""Adversarial audit of the Experiment 3/4 claim surface.

This is deliberately independent of ``make_paper_outputs.py`` and the Polymarket
renderers: it re-derives every headline quantity from the raw score records so a
bug in the reporting path cannot hide behind itself.  Checks are grouped by the
claim they attack.  Exit status is non-zero when a fail-closed check trips.

    python exp3_training_transfer/red_team_audit.py [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
COIN = ROOT / "coin_city_structural" / "reports"
POLY = ROOT / "polymarket" / "reports"
POLY_DATA = ROOT / "polymarket" / "data"
SEEDS = (42, 43, 44)
CELLS = (
    ("coin_city", "direct_a"),
    ("coin_city", "mediated_b"),
    ("coin_harbor", "direct_a"),
    ("coin_harbor", "mediated_b"),
)
TCRIT_DF2 = 4.302652729911275  # t_{.975, df = 2}
COPY_TOL = 5e-3  # a forecast this close to the displayed price is an echo

findings: list[dict] = []


def record(check: str, status: str, detail: str, **extra) -> None:
    findings.append({"check": check, "status": status, "detail": detail, **extra})
    mark = {"pass": "ok  ", "warn": "WARN", "fail": "FAIL"}[status]
    print(f"[{mark}] {check}: {detail}")


# --------------------------------------------------------------------------
# Coin City
# --------------------------------------------------------------------------
def load_coin() -> tuple[list[tuple], dict]:
    registered = json.loads((COIN / "registered_results.json").read_text())
    rows = []
    for rel in sorted(registered["score_file_sha256"]):
        with open(rel) as fh:
            for line in fh:
                r = json.loads(line)
                rows.append(
                    (
                        r["model"], r["arm"], int(r.get("seed", -1)), r["decode"],
                        r["domain"], r["target_structure"], r["cue"], int(r["k"]),
                        r["task_id"], bool(r["parsed"]), float(r["response_mae"]),
                    )
                )
    return rows, registered


M, A, S, D, DOM, ST, CUE, K, TASK, PARSED, MAE = range(11)


def episode_means(rows, sel) -> dict[int, dict[str, float]]:
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if all(r[i] == v for i, v in sel.items()):
            g[r[S]][r[TASK]].append(r[MAE])
    return {s: {e: float(np.mean(v)) for e, v in d.items()} for s, d in g.items()}


def cell_mean(rows, sel, parsed_only: bool = False) -> float:
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if all(r[i] == v for i, v in sel.items()):
            if parsed_only and not r[PARSED]:
                continue
            g[r[S]][r[TASK]].append(r[MAE])
    per = [np.mean([np.mean(v) for v in d.values()]) for d in g.values() if d]
    return float(np.mean(per)) if per else float("nan")


def contrast(rows, model, dom, st, cue="correct", decode="stochastic", bseed=0):
    """Prior-minus-matched with the registered bootstrap and two alternatives."""
    common = {M: model, D: decode, DOM: dom, ST: st, CUE: cue}
    left = episode_means(rows, {**common, A: "population_prior"})
    right = episode_means(rows, {**common, A: "causal"})
    diffs = {}
    for s in SEEDS:
        keys = sorted(set(left.get(s, {})) & set(right.get(s, {})))
        diffs[s] = np.array([left[s][e] - right[s][e] for e in keys])
    seed_means = np.array([diffs[s].mean() for s in SEEDS])
    est = float(seed_means.mean())
    rng = np.random.default_rng(bseed)
    draws = [
        np.mean(
            [
                rng.choice(diffs[int(x)], len(diffs[int(x)]), replace=True).mean()
                for x in rng.choice(np.array(SEEDS), 3, replace=True)
            ]
        )
        for _ in range(5000)
    ]
    blo, bhi = (float(x) for x in np.quantile(draws, (0.025, 0.975)))
    se = float(seed_means.std(ddof=1)) / np.sqrt(3)
    return {
        "estimate": est,
        "boot95": [blo, bhi],
        "t95_seed": [est - TCRIT_DF2 * se, est + TCRIT_DF2 * se],
        "seed_means": [float(x) for x in seed_means],
        "all_seeds_same_sign": bool(np.all(seed_means > 0) or np.all(seed_means < 0)),
    }


def audit_coin() -> None:
    rows, registered = load_coin()

    bad = [
        p for p, h in registered["score_file_sha256"].items()
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h
    ]
    record(
        "coin/score-file-hashes",
        "fail" if bad else "pass",
        f"{len(registered['score_file_sha256']) - len(bad)}/"
        f"{len(registered['score_file_sha256'])} recorded digests reproduce",
        mismatched=bad,
    )
    record(
        "coin/row-count",
        "pass" if len(rows) == registered["row_draws"] else "fail",
        f"{len(rows)} scored draws vs {registered['row_draws']} registered",
    )

    # -- Claim: temperature-zero endpoints "parse completely". --------------
    greedy = [r for r in rows if r[D] == "greedy"]
    unparsed = [r for r in greedy if not r[PARSED]]
    # Informational, not fail-closed: the manuscript now documents this rate.
    # Fail-closed integrity lives in the hash, row-count and disjointness checks.
    record(
        "coin/greedy-parse-rate",
        "pass" if not unparsed else "warn",
        f"{len(unparsed)}/{len(greedy)} temperature-zero draws fail to parse "
        f"({100 * (1 - len(unparsed) / len(greedy)):.2f}% parse rate)",
        by_arm={
            f"{m}/{a}": n
            for (m, a), n in sorted(collections.Counter((r[M], r[A]) for r in unparsed).items())
        },
    )

    # -- Is the greedy "non-reproduction" a penalty artefact? ---------------
    for model in ("qwen3_8b", "llama3_1_8b"):
        sel = {M: model, D: "greedy", DOM: "coin_harbor", ST: "mediated_b", CUE: "correct"}
        arms = ("base", "causal", "population_prior")
        allm = {a: cell_mean(rows, {**sel, A: a}) for a in arms}
        par = {a: cell_mean(rows, {**sel, A: a}, True) for a in arms}
        c_all = allm["population_prior"] - allm["causal"]
        c_par = par["population_prior"] - par["causal"]
        # Does the penalty invert either the contrast or trained-vs-base ordering?
        flips = ((c_all < 0) != (c_par < 0)) or (
            (allm["causal"] > allm["base"]) != (par["causal"] > par["base"])
        )
        record(
            f"coin/greedy-penalty-artefact/{model}",
            "warn" if flips else "pass",
            f"joint-cell levels base/matched/prior = "
            f"{allm['base']:.2f}/{allm['causal']:.2f}/{allm['population_prior']:.2f} "
            f"with the parse penalty but "
            f"{par['base']:.2f}/{par['causal']:.2f}/{par['population_prior']:.2f} on parsed "
            f"draws; contrast {c_all:+.3f} -> {c_par:+.3f}"
            + (" -- penalty inverts the reported ordering" if flips else ""),
            all_draws=allm,
            parsed_only=par,
        )

    # -- Claim: matched training has the lowest mean in every cell. ---------
    for model in ("qwen3_8b", "qwen3_4b", "llama3_1_8b"):
        losses = []
        for dom, st in CELLS:
            sel = {M: model, D: "stochastic", DOM: dom, ST: st, CUE: "correct"}
            arms = {a: cell_mean(rows, {**sel, A: a}) for a in ("base", "causal", "population_prior")}
            if min(arms, key=arms.get) != "causal":
                losses.append(f"{dom}/{st}: lowest arm is {min(arms, key=arms.get)} "
                              f"(base {arms['base']:.3f}, matched {arms['causal']:.3f}, "
                              f"prior {arms['population_prior']:.3f})")
        record(
            f"coin/matched-lowest-in-every-cell/{model}",
            "pass" if not losses else "warn",
            "matched arm is lowest in all four cells" if not losses
            else "; ".join(losses),
        )

    # -- Full prespecified factorial, reported and unreported alike. --------
    table = {}
    for model in ("qwen3_8b", "qwen3_4b", "llama3_1_8b"):
        wrong_way = []
        for i, (dom, st) in enumerate(CELLS):
            c = contrast(rows, model, dom, st, bseed=20260818 + i)
            table[f"{model}/{dom}/{st}"] = c
            lo, hi = c["boot95"]
            if hi < 0:
                wrong_way.append(f"{dom}/{st} {c['estimate']:+.3f} [{lo:+.3f},{hi:+.3f}]")
        record(
            f"coin/factorial-against-hypothesis/{model}",
            "pass" if not wrong_way else "warn",
            "no cell significantly favours the control arm" if not wrong_way
            else f"control arm significantly better in {len(wrong_way)} cell(s): "
                 + "; ".join(wrong_way),
        )

    # -- Does significance survive a small-cluster-honest interval? ---------
    fragile = [
        name for name, c in table.items()
        if (c["boot95"][0] > 0 or c["boot95"][1] < 0)
        and not (c["t95_seed"][0] > 0 or c["t95_seed"][1] < 0)
    ]
    record(
        "coin/three-cluster-bootstrap-fragility",
        "warn" if fragile else "pass",
        f"{len(fragile)}/{len(table)} intervals exclude zero under the registered "
        f"3-cluster percentile bootstrap but not under a t interval on the three "
        f"seed means",
        fragile=fragile,
    )

    # -- The distribution-free fallback that does survive. ------------------
    for model in ("qwen3_8b", "qwen3_4b", "llama3_1_8b"):
        vals = [table[f"{model}/{d}/{s}"]["seed_means"] for d, s in CELLS]
        pos = sum(v > 0 for cell in vals for v in cell)
        record(
            f"coin/seed-sign-consistency/{model}",
            "pass" if pos in (0, 12) else "warn",
            f"{pos}/12 seed x cell contrasts favour episode-matched training",
        )

    # -- How much of the training gain is episode matching? -----------------
    for model in ("qwen3_8b", "qwen3_4b"):
        sel = {M: model, D: "stochastic", DOM: "coin_harbor", ST: "mediated_b", CUE: "correct"}
        arms = {a: cell_mean(rows, {**sel, A: a}) for a in ("base", "causal", "population_prior")}
        parsed_base = cell_mean(rows, {**sel, A: "base"}, True)
        total = arms["base"] - arms["causal"]
        share = (arms["population_prior"] - arms["causal"]) / total
        total_p = parsed_base - arms["causal"]
        share_p = (arms["population_prior"] - arms["causal"]) / total_p
        record(
            f"coin/episode-matching-share/{model}",
            "pass",
            f"episode matching supplies {100 * share:.0f}% of the {total:.2f}-point gain over "
            f"base ({100 * share_p:.0f}% of the {total_p:.2f}-point gain on parsed base draws)",
        )


# --------------------------------------------------------------------------
# Polymarket
# --------------------------------------------------------------------------
def audit_polymarket() -> None:
    reg = POLY_DATA / "exp3b_registered"
    splits = {n: [json.loads(l) for l in open(reg / f"{n}.tasks.jsonl")]
              for n in ("train", "dev", "test")}
    splits["scale_holdout"] = [
        json.loads(l) for l in open(POLY_DATA / "exp4_scale_registered" / "test.tasks.jsonl")
    ]

    leaks = []
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for key in ("task_id", "market_id", "event_id", "template_key"):
                A = {r.get(key) for r in splits[a] if r.get(key)}
                B = {r.get(key) for r in splits[b] if r.get(key)}
                if A & B:
                    leaks.append(f"{a}/{b} share {len(A & B)} {key}")
    record(
        "poly/family-disjointness",
        "fail" if leaks else "pass",
        "train/dev/test/holdout are disjoint on task, market, event and template"
        if not leaks else "; ".join(leaks),
    )

    # -- Is the comparison baseline printed in the model's own prompt? ------
    exposed = {
        name: sum(1 for r in rows if str(r.get("market_yes_prob")) and
                  f"{r['market_yes_prob']:.4f}" in r.get("input", ""))
        for name, rows in splits.items()
    }
    record(
        "poly/market-price-in-prompt",
        "warn" if any(exposed.values()) else "pass",
        "the market probability used as the comparison baseline is printed as the "
        "final price line of the prompt in " + ", ".join(
            f"{k} ({v}/{len(splits[k])})" for k, v in exposed.items()),
    )

    # -- Copy rate and residual skill. --------------------------------------
    endpoints = {
        "exp4-holdout/qwen3_1_7b": "exp4_scale_qwen3_1_7b_stochastic_j30890110.jsonl",
        "exp4-holdout/qwen3_4b": "exp4_scale_qwen3_4b_stochastic_j30870546.jsonl",
        "exp4-holdout/qwen3_8b": "exp4_scale_qwen3_8b_stochastic_j30870547.jsonl",
        "exp4-holdout/qwen3_14b": "exp4_scale_qwen3_14b_stochastic_j30870548.jsonl",
        "exp4-holdout/llama3_1_8b": "exp4_scale_llama3_1_8b_stochastic_j30889436.jsonl",
        "exp4-holdout/llama3_2_3b": "exp4_scale_llama3_2_3b_stochastic_j30890115.jsonl",
        "exp4-locked/qwen3_8b": "exp3b_qwen3_8b_locked_test_j30856250.jsonl",
        "exp4-locked/original": "exp3b_locked_test_j30505540.jsonl",
        "exp4-locked/llama3_1_8b": "exp3b_llama3_1_8b_locked_test_j30856254.jsonl",
    }
    copy_table = {}
    for name, fn in endpoints.items():
        path = POLY / fn
        if not path.exists():
            continue
        rows = [json.loads(l) for l in open(path)]
        arms = [a for a in rows[0]["forecasts"] if a == "base" or a.startswith("seed_")]
        per_arm = {}
        for arm in arms:
            usable = [r for r in rows
                      if r["forecasts"].get(arm) is not None
                      and r.get("market_yes_prob") is not None]
            if not usable:
                continue
            copies, dev = [], []
            for r in usable:
                echo = abs(r["forecasts"][arm] - r["market_yes_prob"]) < COPY_TOL
                (copies if echo else dev).append(r)
            outcome = [1.0 if r["settlement_yes"] else 0.0 for r in dev]
            per_arm[arm] = {
                "copy_rate": len(copies) / len(usable),
                "n_deviating": len(dev),
                "model_brier_on_deviations": statistics.mean(
                    (r["forecasts"][arm] - y) ** 2 for r, y in zip(dev, outcome)
                ) if dev else None,
                "market_brier_on_deviations": statistics.mean(
                    (r["market_yes_prob"] - y) ** 2 for r, y in zip(dev, outcome)
                ) if dev else None,
            }
        copy_table[name] = per_arm
        base = per_arm.get("base", {}).get("copy_rate")
        trained = [v["copy_rate"] for k, v in per_arm.items() if k.startswith("seed_")]
        if base is None or not trained:
            continue
        edge = [
            v["model_brier_on_deviations"] - v["market_brier_on_deviations"]
            for k, v in per_arm.items()
            if k.startswith("seed_") and v["n_deviating"]
        ]
        record(
            f"poly/echo-rate/{name}",
            "warn",
            f"echoes the displayed price on {100 * base:.0f}% of tasks untrained -> "
            f"{100 * statistics.mean(trained):.0f}% trained; on the tasks where the trained "
            f"model does deviate it is {statistics.mean(edge):+.5f} Brier "
            f"{'worse' if statistics.mean(edge) > 0 else 'better'} than the price it was shown",
        )

    findings.append({"check": "poly/echo-table", "status": "pass",
                     "detail": "per-arm echo rates", "table": copy_table})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    print("=== Coin City (Experiment 3) ===")
    audit_coin()
    print("\n=== Polymarket (Experiment 4) ===")
    audit_polymarket()
    fails = [f for f in findings if f["status"] == "fail"]
    warns = [f for f in findings if f["status"] == "warn"]
    print(f"\n{len(fails)} fail, {len(warns)} warn, "
          f"{len(findings) - len(fails) - len(warns)} pass")
    if args.json:
        args.json.write_text(json.dumps(findings, indent=2, sort_keys=True))
        print(f"wrote {args.json}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
