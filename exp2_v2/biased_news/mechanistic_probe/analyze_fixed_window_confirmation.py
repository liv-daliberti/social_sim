#!/usr/bin/env python3
"""Score the fixed-window patch confirmation exactly as protocol.json froze it.

The estimand, sign-flip null and bootstrap are imported from the registered
analyzer. The two primary tests per model are H1 (mid window moves the forecast
at k=0) and H2 (the mid window moves it more than the late window, paired by
episode), Holm-adjusted across both models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze_symbol_relational_probe import bootstrap_mean, exact_sign_flip  # noqa: E402
from patch_fixed_window_confirmation import ARMS, load_frozen  # noqa: E402
from freeze_fixed_window_confirmation import RUN_DIR  # noqa: E402

CROSS = ("cross_mid_window", "cross_late_window", "cross_embedding_only")


def episode_shifts(by_key: dict, episodes: list[int], target: dict, condition: str, k: int) -> dict[int, float]:
    shifts = {}
    for e in episodes:
        need = [by_key.get((e, k, "abc_context", "unpatched")),
                by_key.get((e, k, "abc_wrong_context", "unpatched")),
                by_key.get((e, k, "abc_wrong_context", condition)),
                by_key.get((e, k, "abc_context", condition))]
        if not all(r is not None and r["parsed"] for r in need):
            continue
        correct_base, wrong_base, wrong_patched, correct_patched = need
        sign = 1.0 if target[e] else -1.0
        forward = sign * (wrong_patched["implied_slope"] - wrong_base["implied_slope"])
        reverse = sign * (correct_base["implied_slope"] - correct_patched["implied_slope"])
        shifts[e] = 0.5 * (forward + reverse)
    return shifts


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    result = exact_sign_flip(arr)
    result["bootstrap_ci95"] = bootstrap_mean(arr)
    return result


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for rank, name in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    a = ap.parse_args()
    protocol, tasks = load_frozen(a.run_dir)
    target = {int(r["episode"]): bool(r["target_strong"]) for r in tasks}
    n_episodes = protocol["episodes"]["sealed_test_episodes"]
    expected = n_episodes * len(protocol["depths"]) * len(ARMS) * len(protocol["conditions"])

    results, primary_p = {"study": protocol["study"], "models": {}}, {}
    for key in protocol["models"]:
        rows = [json.loads(l) for f in sorted((a.run_dir / key).glob("shard*.jsonl"))
                for l in f.read_text().splitlines() if l.strip()]
        by_key = {(r["episode"], r["c_cases"], r["recipient_arm"], r["condition"]): r for r in rows}
        if len(by_key) != len(rows):
            raise ValueError(f"{key}: duplicate generation keys")
        if len(rows) != expected:
            raise ValueError(f"{key}: {len(rows)} of {expected} generations; run is incomplete")
        episodes = sorted({r["episode"] for r in rows})

        self_pairs = [by_key[(e, k, arm, "unpatched")]["predicted_poll"]
                      == by_key[(e, k, arm, "self_mid_window")]["predicted_poll"]
                      for e in episodes for k in protocol["depths"] for arm in ARMS
                      if by_key[(e, k, arm, "unpatched")]["parsed"]
                      and by_key[(e, k, arm, "self_mid_window")]["parsed"]]
        out = {"self_patch": {"parsed_pairs": len(self_pairs),
                              "exact_matches": int(sum(self_pairs)),
                              "expected_pairs": n_episodes * len(protocol["depths"]) * len(ARMS)},
               "conditions": {}}
        shifts = {}
        for condition in CROSS:
            out["conditions"][condition] = {}
            for k in protocol["depths"]:
                shifts[(condition, k)] = episode_shifts(by_key, episodes, target, condition, k)
                out["conditions"][condition][f"k{k}"] = summarize(list(shifts[(condition, k)].values()))

        k0 = protocol["primary_depth"]
        mid, late = shifts[("cross_mid_window", k0)], shifts[("cross_late_window", k0)]
        paired = [mid[e] - late[e] for e in sorted(set(mid) & set(late))]
        out["H1_mid_positive"] = out["conditions"]["cross_mid_window"][f"k{k0}"]
        out["H2_mid_exceeds_late"] = summarize(paired)
        gates = protocol["gates"]
        out["gates"] = {
            "complete_episodes_H1": out["H1_mid_positive"]["n"],
            "complete_episodes_H2": out["H2_mid_exceeds_late"]["n"],
            "complete_episode_floor": int(n_episodes * gates["min_complete_episode_fraction"]),
            "self_patch_floor": int(out["self_patch"]["expected_pairs"]
                                    * gates["min_self_patch_exact_match_fraction"]),
        }
        out["gates"]["passed"] = bool(
            min(out["gates"]["complete_episodes_H1"], out["gates"]["complete_episodes_H2"])
            >= out["gates"]["complete_episode_floor"]
            and out["self_patch"]["exact_matches"] >= out["gates"]["self_patch_floor"])
        primary_p[f"{key}:H1"] = out["H1_mid_positive"]["one_sided_p"]
        primary_p[f"{key}:H2"] = out["H2_mid_exceeds_late"]["one_sided_p"]
        results["models"][key] = out

    results["holm_adjusted_p"] = holm(primary_p)
    results["protocol_sha256"] = __import__("hashlib").sha256(
        (a.run_dir / "protocol.json").read_bytes()).hexdigest()
    (a.run_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    for key, out in results["models"].items():
        c = out["conditions"]
        print(f"{key}: k0 mid {c['cross_mid_window']['k0']['mean']:.3f} "
              f"late {c['cross_late_window']['k0']['mean']:.3f} "
              f"emb {c['cross_embedding_only']['k0']['mean']:.3f} | "
              f"H1 p={out['H1_mid_positive']['one_sided_p']:.4g} "
              f"H2 p={out['H2_mid_exceeds_late']['one_sided_p']:.4g} gates={out['gates']['passed']}")
    print("Holm:", results["holm_adjusted_p"])


if __name__ == "__main__":
    main()
