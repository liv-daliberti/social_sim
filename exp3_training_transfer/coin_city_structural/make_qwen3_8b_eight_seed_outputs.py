#!/usr/bin/env python3
"""Render the Qwen3-8B Coin City transfer endpoint at an arbitrary seed roster.

Implements the four analyses pre-specified in
`five_seed_extension/QWEN3_8B_EIGHT_SEED_AMENDMENT.md`:

  1. hierarchical seed-first paired bootstrap (the parent analysis, unchanged);
  2. Student-t interval on the seed-level means;
  3. wild cluster bootstrap with Webb six-point weights, imposing the null;
  4. exact one-sided seed-level sign test.

The seed roster is a parameter so the same code and the same bootstrap seeds
serve the G=7 interim look and the G=8 primary analysis. Given `--seeds 42 43
44` it must reproduce the published three-seed estimates in
`reports/registered_results.json`; `--self-test` asserts exactly that.

    python make_qwen3_8b_eight_seed_outputs.py --self-test
    python make_qwen3_8b_eight_seed_outputs.py --seeds 42 43 44 45 46 47 49 \
        --label g7_interim
    python make_qwen3_8b_eight_seed_outputs.py --seeds 42 43 44 45 46 47 48 49 \
        --label eight_seed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from make_interim_outputs import CELLS, KEYS
from make_paper_outputs import read_jsonl, validate_rows

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs"
EXTENSION_RUNS = REPO / "exp3_training_transfer/five_seed_extension/runs"

# Module-level default; overridden by --model. The script is model-agnostic
# apart from this constant and the offline snapshot each roster pins.
MODEL = "qwen3_8b"
ARMS = ("causal", "population_prior")
PARENT_SEEDS = (42, 43, 44)
BOOTSTRAP_SEED = 20260818
REPETITIONS = 5000
WCB_REPETITIONS = 9999

# Webb six-point weights: the few-clusters remedy from Cameron, Gelbach and
# Miller (2008). Rademacher weights admit only 2^G distinct draws, which at
# G=7 floors the two-sided p-value at 1/64.
WEBB = np.array([-math.sqrt(1.5), -1.0, -math.sqrt(0.5),
                 math.sqrt(0.5), 1.0, math.sqrt(1.5)])

TCRIT = {1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582,
         6: 2.446912, 7: 2.364624, 8: 2.306004, 9: 2.262157, 10: 2.228139}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# roster assembly
# --------------------------------------------------------------------------

def parent_ledger() -> dict:
    for path in sorted(RUNS.glob("coin_city_structural_*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("protocol") == "coin_city_structural_transfer_v1":
            return payload | {"__path__": str(path)}
    raise SystemExit("no effective Coin City structural-transfer ledger found")


def added_seed_jobs() -> dict[tuple[str, int], str]:
    """{(arm, seed): job_id} for every added seed recorded in a submission ledger."""
    jobs: dict[tuple[str, int], str] = {}
    for path in sorted(EXTENSION_RUNS.glob("qwen3_8b_eight_seed_submission_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("training_jobs", []):
            if row.get("job_id"):
                # A later ledger is a recorded retry and supersedes the earlier
                # attempt for that cell; report-root resolution below is what
                # actually decides, and rejects any residual ambiguity.
                jobs[(row["arm"], int(row["seed"]))] = str(row["job_id"])
    return jobs


def resolve_report_root(arm: str, seed: int, job_id: str | None) -> Path:
    roots = sorted(REPORTS.glob(f"{arm}_{MODEL}_s{seed}_*"))
    if job_id is not None:
        preferred = [p for p in roots if p.name.endswith(f"_j{job_id}")]
        roots = preferred or roots
    if len(roots) != 1:
        raise AssertionError(
            f"{arm} s{seed}: expected exactly one report root, found "
            f"{[p.name for p in roots]}; move failed attempts under "
            f"reports/failed_attempts/")
    return roots[0]


def build_roster(seeds: tuple[int, ...]) -> list[dict]:
    ledger = parent_ledger()
    parent_rows = {
        (row["arm"], int(row["seed"])): str(row["job_id"])
        for row in ledger.get("full_training", [])
        if row.get("model") == MODEL and row.get("kind") == "confirmatory"
    }
    added = added_seed_jobs()

    roster: list[dict] = []
    for arm in ARMS:
        for seed in seeds:
            key = (arm, seed)
            if seed in PARENT_SEEDS:
                if key not in parent_rows:
                    raise AssertionError(f"parent ledger lacks {arm} s{seed}")
                job_id = parent_rows[key]
            else:
                if key not in added:
                    raise AssertionError(
                        f"no submission ledger records {arm} s{seed}")
                job_id = added[key]
            roster.append({"model": MODEL, "arm": arm, "seed": seed,
                           "kind": "confirmatory", "job_id": job_id})

    bases = [row for row in ledger.get("base_evaluations", [])
             if row.get("model") == MODEL]
    if len(bases) != 1:
        raise AssertionError("parent ledger does not hold exactly one Qwen3-8B base")
    roster.append({"model": MODEL, "arm": None, "seed": None,
                   "kind": "base", "job_id": str(bases[0]["job_id"])})
    return roster


def load_rows(roster: list[dict]) -> tuple[list[dict], list[Path]]:
    rows: list[dict] = []
    paths: list[Path] = []
    universe = None
    for item in roster:
        if item["kind"] == "base":
            root = sorted(REPORTS.glob(f"base_{MODEL}_j{item['job_id']}"))
            if len(root) != 1:
                raise AssertionError(f"base job {item['job_id']}: report root not unique")
            path = root[0] / "stochastic_n5.scores.jsonl"
        else:
            path = resolve_report_root(item["arm"], item["seed"],
                                       item["job_id"]) / "stochastic_n5.scores.jsonl"
        if not path.is_file():
            raise AssertionError(f"job {item['job_id']}: missing {path}")
        scored = read_jsonl(path)
        seen = validate_rows(scored, item, "stochastic")
        if universe is None:
            universe = seen
        elif seen != universe:
            raise AssertionError(f"job {item['job_id']}: endpoint task universe differs")
        rows.extend(scored)
        paths.append(path)

    expected = len(roster) * 1440 * 5
    if len(rows) != expected:
        raise AssertionError(f"expected {expected} draws, found {len(rows)}")
    return rows, paths


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------

def episode_values(rows: list[dict], selectors: dict) -> dict[int, dict[str, float]]:
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if all(row.get(name) == value for name, value in selectors.items()):
            grouped[int(row["seed"])][str(row["task_id"])].append(
                float(row["response_mae"]))
    return {seed: {task: float(np.mean(v)) for task, v in tasks.items()}
            for seed, tasks in grouped.items()}


def mean_cell(rows: list[dict], selectors: dict) -> float:
    values = episode_values(rows, selectors)
    means = [np.mean(list(e.values())) for e in values.values() if e]
    return float(np.mean(means)) if means else float("nan")


def seed_differences(rows: list[dict], left: dict, right: dict,
                     seeds: tuple[int, ...]) -> dict[int, np.ndarray]:
    left_values = episode_values(rows, left)
    right_values = episode_values(rows, right)
    out: dict[int, np.ndarray] = {}
    for seed in seeds:
        common = sorted(set(left_values.get(seed, {})) & set(right_values.get(seed, {})))
        if not common:
            raise AssertionError(f"no paired episodes for seed {seed}")
        out[seed] = np.asarray([left_values[seed][e] - right_values[seed][e]
                                for e in common])
    return out


def hierarchical_interval(differences: dict[int, np.ndarray], seeds: tuple[int, ...],
                          *, repetitions: int, seed: int) -> tuple[float, float, float]:
    """Seed-first, paired-episode bootstrap; identical algorithm to the parent."""
    estimate = float(np.mean([np.mean(v) for v in differences.values()]))
    rng = np.random.default_rng(seed)
    seed_array = np.asarray(seeds)
    draws = []
    for _ in range(repetitions):
        sampled = rng.choice(seed_array, len(seed_array), replace=True)
        means = []
        for selected in sampled:
            values = differences[int(selected)]
            means.append(float(np.mean(rng.choice(values, len(values), replace=True))))
        draws.append(float(np.mean(means)))
    low, high = np.quantile(draws, (0.025, 0.975))
    return estimate, float(low), float(high)


def student_t_interval(seed_means: np.ndarray) -> tuple[float, float, float]:
    n = len(seed_means)
    mean = float(np.mean(seed_means))
    sd = float(np.std(seed_means, ddof=1))
    half = TCRIT[n - 1] * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def wild_cluster_p(seed_means: np.ndarray, *, repetitions: int,
                   seed: int) -> dict[str, float]:
    """Wild cluster bootstrap-t for H0: mean = 0, Webb weights, null imposed."""
    n = len(seed_means)
    mean = float(np.mean(seed_means))
    sd = float(np.std(seed_means, ddof=1))
    if sd == 0.0:
        return {"t": float("inf"), "p_two_sided": 0.0, "repetitions": repetitions}
    t_observed = mean / (sd / math.sqrt(n))
    # Null-imposed residuals: the restricted estimate of the mean is zero.
    rng = np.random.default_rng(seed)
    weights = rng.choice(WEBB, size=(repetitions, n))
    starred = weights * seed_means  # d*_g = w_g * u_g, u_g = d_g under H0
    star_mean = starred.mean(axis=1)
    star_sd = starred.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_star = star_mean / (star_sd / math.sqrt(n))
    t_star = t_star[np.isfinite(t_star)]
    p = float((np.abs(t_star) >= abs(t_observed)).sum() + 1) / (len(t_star) + 1)
    return {"t": float(t_observed), "p_two_sided": p, "repetitions": int(len(t_star))}


def sign_test(seed_means: np.ndarray) -> dict[str, float]:
    """Exact one-sided binomial test in the pre-registered direction (mean > 0)."""
    n = len(seed_means)
    positive = int((seed_means > 0).sum())
    p = sum(math.comb(n, k) for k in range(positive, n + 1)) / (2 ** n)
    return {"seeds": n, "positive": positive, "p_one_sided": float(p),
            "unanimous": bool(positive == n or positive == 0)}


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def analyze(rows: list[dict], seeds: tuple[int, ...]) -> list[dict]:
    results = []
    for index, (label, domain, structure) in enumerate(CELLS):
        common = {"model": MODEL, "decode": "stochastic", "domain": domain,
                  "target_structure": structure, "cue": "correct"}
        means = {arm: mean_cell(rows, {**common, "arm": arm})
                 for arm in ("base", "population_prior", "causal")}
        differences = seed_differences(
            rows, {**common, "arm": "population_prior"},
            {**common, "arm": "causal"}, seeds)
        seed_means = np.asarray([float(np.mean(differences[s])) for s in seeds])
        cell_seed = BOOTSTRAP_SEED + index

        estimate, low, high = hierarchical_interval(
            differences, seeds, repetitions=REPETITIONS, seed=cell_seed)
        t_estimate, t_low, t_high = student_t_interval(seed_means)
        results.append({
            "label": label, "key": KEYS[label], "domain": domain,
            "target_structure": structure, **means,
            "seed_level_contrasts": {str(s): float(v)
                                     for s, v in zip(seeds, seed_means)},
            "hierarchical_bootstrap": {
                "estimate": estimate, "ci95_low": low, "ci95_high": high,
                "repetitions": REPETITIONS, "bootstrap_seed": cell_seed},
            "student_t": {
                "estimate": t_estimate, "ci95_low": t_low, "ci95_high": t_high,
                "df": len(seeds) - 1},
            "wild_cluster_bootstrap": wild_cluster_p(
                seed_means, repetitions=WCB_REPETITIONS, seed=cell_seed),
            "sign_test": sign_test(seed_means),
        })
    return results


def self_test(rows: list[dict]) -> None:
    """The parent three-seed estimates must come back byte-for-byte."""
    published = json.loads((REPORTS / "registered_results.json").read_text())
    want = {
        (entry["domain"], entry["target_structure"]): entry
        for entry in published["registered_estimates"]["transfer_cells"]
        if entry.get("model") == MODEL and entry.get("decode") == "stochastic"
    }
    got = analyze(rows, PARENT_SEEDS)
    checked = 0
    for cell in got:
        reference = want.get((cell["domain"], cell["target_structure"]))
        if reference is None:
            raise SystemExit(
                f"published results lack {MODEL} cell "
                f"{cell['domain']}/{cell['target_structure']}")
        mine = cell["hierarchical_bootstrap"]
        pairs = (
            ("estimate", reference["population_prior_minus_causal"], mine["estimate"]),
            ("ci95_low", reference["ci95_low"], mine["ci95_low"]),
            ("ci95_high", reference["ci95_high"], mine["ci95_high"]),
        )
        for name, expected, value in pairs:
            if abs(expected - value) > 5e-9:
                raise SystemExit(
                    f"self-test FAILED for {cell['label']} {name}: "
                    f"published {expected!r}, recomputed {value!r}")
        checked += 1
    if checked != len(CELLS):
        raise SystemExit(f"self-test only matched {checked} of {len(CELLS)} cells")
    print(f"self-test OK: reproduced all {checked} published three-seed "
          f"hierarchical intervals to 5e-9")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL,
                        choices=("qwen3_8b", "qwen3_4b", "llama3_1_8b"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(PARENT_SEEDS))
    parser.add_argument("--label", default="eight_seed",
                        help="output filename discriminator")
    parser.add_argument("--self-test", action="store_true",
                        help="reproduce the published three-seed estimates and exit")
    args = parser.parse_args()
    globals()["MODEL"] = args.model

    seeds = tuple(sorted(set(args.seeds)))
    roster = build_roster(seeds)
    rows, paths = load_rows(roster)
    print(f"validated {len(paths)} endpoint files, {len(rows)} draws, "
          f"G={len(seeds)} seeds {list(seeds)}")

    if args.self_test:
        parent_roster = build_roster(PARENT_SEEDS)
        parent_rows, _ = load_rows(parent_roster)
        self_test(parent_rows)
        return

    payload = {
        "analysis": f"{MODEL}_round300_five_draw_endpoint_G{len(seeds)}",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "exp3_qwen3_8b_coin_city_eight_seed_v1",
        "model": MODEL,
        "seeds": list(seeds),
        "G": len(seeds),
        "arms": list(ARMS),
        "draws_per_task": 5,
        "validated_jobs": len(roster),
        "validated_tasks_per_job": 1440,
        "row_draws": len(rows),
        "variance_unit": "training seed; paired task within seed; draws averaged",
        "analysis_code_sha256": sha256(Path(__file__)),
        "roster": roster,
        "score_file_sha256": {
            str(p.relative_to(REPO)): sha256(p) for p in paths},
        "transfer_cells": analyze(rows, seeds),
    }
    out = REPORTS / f"{MODEL}_step300_stochastic_{args.label}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
