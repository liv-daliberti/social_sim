#!/usr/bin/env python3
"""Seed-level power and robustness for the mechanism-family arm contrast.

Applies the four estimators fixed in the Coin City eight-seed amendment
(hierarchical seed-first paired bootstrap; Student-t on seed-level means; wild
cluster bootstrap with Webb weights imposing the null; exact one-sided
seed-level sign test) to the registered mechanism-family endpoints, and
projects the Student-t interval forward at larger seed counts.

Endpoints are resolved through `reports/c3_mechanism_full_aggregate.json`, the
registered score-file list -- **never** by globbing `reports/`. There are up to
seven attempt directories per cell (retries, recoveries, backfills), and a glob
silently picks superseded runs: doing so moved the Qwen3-4B disclosed estimate
from +0.036 spanning zero to its true +0.323 excluding zero.

Reads only registered artifacts and writes nothing.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
AGGREGATE = ROOT / "reports/c3_mechanism_full_aggregate.json"

MODELS = ("qwen3_4b", "qwen3_8b", "llama3_1_8b")
DISCLOSURES = ("disclosed", "undisclosed")
SEEDS = (42, 43, 44)
# Seeds 45/46 postdate the registered aggregate, so they are resolved from the
# five-seed submission ledgers and then required to map to exactly one report
# directory -- never globbed blind, since cells carry up to seven attempts.
EXTENSION_RUNS = REPO / "exp3_training_transfer/five_seed_extension/runs"
BOOTSTRAP_SEED = 20260818
REPETITIONS = 5000
WCB_REPETITIONS = 9999
WEBB = np.array([-math.sqrt(1.5), -1.0, -math.sqrt(0.5),
                 math.sqrt(0.5), 1.0, math.sqrt(1.5)])
TCRIT = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582,
         6: 2.446912, 7: 2.364624, 8: 2.306004, 9: 2.262157, 10: 2.228139,
         11: 2.200985, 12: 2.178813, 13: 2.160369, 14: 2.144787, 19: 2.093024,
         24: 2.063899, 29: 2.045230, 39: 2.022691, 49: 2.009575}


def tcrit(df: int) -> float:
    if df in TCRIT:
        return TCRIT[df]
    below = max(k for k in TCRIT if k < df)
    return TCRIT[below]


def added_seed_endpoints(model: str) -> dict[tuple, Path]:
    """{(disclosure, arm, model, seed): score file} for ledger-recorded additions."""
    out: dict[tuple, Path] = {}
    for ledger in sorted(EXTENSION_RUNS.glob("mechanism_*_five_seed_*.json")):
        payload = json.loads(ledger.read_text(encoding="utf-8"))
        if payload.get("model_key") != model:
            continue
        for job in payload.get("training_jobs", []):
            if not job.get("job_id"):
                continue
            pattern = (f"{job['disclosure']}_{job['arm']}_{model}"
                       f"_s{job['seed']}_*_j{job['job_id']}")
            found = sorted((ROOT / "reports").glob(pattern))
            if len(found) != 1:
                raise SystemExit(f"{pattern}: expected one report root, got {found}")
            path = found[0] / "stochastic_n5.scores.jsonl"
            if not path.is_file():
                raise SystemExit(f"missing endpoint: {path}")
            out[(job["disclosure"], job["arm"], model, int(job["seed"]))] = path
    return out


def registered_endpoints() -> dict[tuple, Path]:
    payload = json.loads(AGGREGATE.read_text())
    out: dict[tuple, Path] = {}
    for rel in payload["score_files"]:
        if "stochastic" not in rel:
            continue
        m = re.search(r"reports/(disclosed|undisclosed)_"
                      r"(causal_family|population_prior|structureless)_"
                      r"(qwen3_4b|qwen3_8b|llama3_1_8b)_s(\d+)_", rel)
        if m:
            out[(m.group(1), m.group(2), m.group(3), int(m.group(4)))] = REPO / rel
            continue
        b = re.search(r"reports/base_(disclosed|undisclosed)_"
                      r"(qwen3_4b|qwen3_8b|llama3_1_8b)_", rel)
        if b:
            out[(b.group(1), "base", b.group(2), 0)] = REPO / rel
    return out


def episode_means(path: Path) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            acc[row["task_id"]].append(float(row["response_mae"]))
    return {task: float(np.mean(v)) for task, v in acc.items()}


def hierarchical(differences: dict[int, np.ndarray], seeds, *, seed: int):
    estimate = float(np.mean([np.mean(v) for v in differences.values()]))
    rng = np.random.default_rng(seed)
    arr = np.asarray(seeds)
    draws = []
    for _ in range(REPETITIONS):
        means = []
        for chosen in rng.choice(arr, len(arr), replace=True):
            v = differences[int(chosen)]
            means.append(float(np.mean(rng.choice(v, len(v), replace=True))))
        draws.append(float(np.mean(means)))
    low, high = np.quantile(draws, (0.025, 0.975))
    return estimate, float(low), float(high)


def wild_cluster(seed_means: np.ndarray, *, seed: int) -> float:
    n = len(seed_means)
    sd = float(np.std(seed_means, ddof=1))
    if sd == 0.0:
        return 0.0
    t_obs = float(np.mean(seed_means)) / (sd / math.sqrt(n))
    rng = np.random.default_rng(seed)
    starred = rng.choice(WEBB, size=(WCB_REPETITIONS, n)) * seed_means
    with np.errstate(divide="ignore", invalid="ignore"):
        t_star = starred.mean(axis=1) / (starred.std(axis=1, ddof=1) / math.sqrt(n))
    t_star = t_star[np.isfinite(t_star)]
    return float((np.abs(t_star) >= abs(t_obs)).sum() + 1) / (len(t_star) + 1)


def sign_p(seed_means: np.ndarray) -> tuple[int, float]:
    n = len(seed_means)
    pos = int((seed_means > 0).sum())
    return pos, sum(math.comb(n, k) for k in range(pos, n + 1)) / 2 ** n


def min_seeds(mean: float, sd: float, limit: int = 60) -> int | None:
    if mean == 0:
        return None
    for n in range(3, limit + 1):
        if abs(mean) - tcrit(n - 1) * sd / math.sqrt(n) > 0:
            return n
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=int, nargs="+", default=(5, 8, 10))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--model", action="append", dest="models",
                        help="restrict to these models; repeatable")
    args = parser.parse_args()
    globals()["SEEDS"] = tuple(sorted(set(args.seeds)))
    reg = registered_endpoints()
    for m in (args.models or MODELS):
        reg |= added_seed_endpoints(m)
    print(f"resolved {len(reg)} registered stochastic endpoints from the aggregate\n")

    for model in (args.models or MODELS):
        for disc in DISCLOSURES:
            diffs: dict[int, np.ndarray] = {}
            for s in SEEDS:
                c = reg.get((disc, "causal_family", model, s))
                p = reg.get((disc, "population_prior", model, s))
                if not c or not p:
                    break
                cm, pm = episode_means(c), episode_means(p)
                tasks = sorted(set(cm) & set(pm))
                diffs[s] = np.asarray([pm[t] - cm[t] for t in tasks])
            if len(diffs) != len(SEEDS):
                print(f"{model} / {disc}: incomplete registered roster\n")
                continue

            seed_means = np.asarray([float(np.mean(diffs[s])) for s in SEEDS])
            mean, sd = float(seed_means.mean()), float(seed_means.std(ddof=1))
            cell_seed = (BOOTSTRAP_SEED + MODELS.index(model) * 2
                         + DISCLOSURES.index(disc))
            est, lo, hi = hierarchical(diffs, SEEDS, seed=cell_seed)
            half = tcrit(len(SEEDS) - 1) * sd / math.sqrt(len(SEEDS))
            pos, sp = sign_p(seed_means)

            print(f"=== {model} / {disc} ===")
            print("  per-seed: " + ", ".join(f"s{s}={v:+.4f}" for s, v in zip(SEEDS, seed_means)))
            print(f"  mean {mean:+.4f}  sd {sd:.4f}")
            print(f"  hierarchical bootstrap  {est:+.4f} [{lo:+.4f},{hi:+.4f}]"
                  f"  {'excl 0' if lo * hi > 0 else 'spans 0'}")
            print(f"  student-t df={len(SEEDS)-1}          {mean:+.4f} [{mean-half:+.4f},{mean+half:+.4f}]"
                  f"  {'excl 0' if (mean-half)*(mean+half) > 0 else 'spans 0'}")
            print(f"  wild cluster (Webb)     p = {wild_cluster(seed_means, seed=cell_seed):.4f}")
            floor = 2.0 ** -len(SEEDS)
            print(f"  sign test               {pos}/{len(SEEDS)} positive, "
                  f"p = {sp:.4f} (floor {floor:.4f} at G={len(SEEDS)})")
            for g in args.project:
                h = tcrit(g - 1) * sd / math.sqrt(g)
                print(f"  projected G={g:<3}         [{mean-h:+.4f},{mean+h:+.4f}]"
                      f"  {'excl 0' if (mean-h)*(mean+h) > 0 else 'spans 0'}")
            need = min_seeds(mean, sd)
            print(f"  smallest G excluding zero at this sd: {need if need else '>60'}\n")


if __name__ == "__main__":
    main()
