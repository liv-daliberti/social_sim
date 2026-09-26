#!/usr/bin/env python3
"""Emit every Experiment 3 number the manuscript cites, as LaTeX macros.

Nothing in the rewritten Experiment 3 section is typed by hand. Each figure in
the prose resolves to a macro defined here and computed from a registered
artifact, so a stale number is a build error rather than a silent transcription
mistake. The week that produced this restructure included one analysis whose
estimate moved from +0.036 to +0.323 purely because a glob picked a superseded
report directory; sourcing every number from the same validated path is the
cheapest defence against a repeat.

Sources, all read-only:
  coin_city_structural/reports/qwen3_{8b,4b}_step300_stochastic_eight_seed.json
  coin_city_structural/reports/registered_results.json      (published G=3)
  coin_city_structural/reports/sel_* and selbase_*          (pilot endpoints)
  mechanism_family/reports/ via the five-seed submission ledgers

    python exp3_training_transfer/render_exp3_paper_macros.py
    python ... --out paper/tables/exp3_rebuilt_data.tex
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
COIN = REPO / "exp3_training_transfer/coin_city_structural"
MECH = REPO / "exp3_training_transfer/mechanism_family"
DEFAULT_OUT = REPO / "paper/tables/exp3_rebuilt_data.tex"

CELLS = (("ID", "coin_city", "direct_a"), ("Domain", "coin_harbor", "direct_a"),
         ("Structure", "coin_city", "mediated_b"), ("Joint", "coin_harbor", "mediated_b"))
TCRIT = {2: 4.302653, 4: 2.776445, 7: 2.364624}
WORDS = {"ID": "Id", "Domain": "Domain", "Structure": "Structure", "Joint": "Joint"}


def fmt(x: float, places: int = 3) -> str:
    """Leading-zero-stripped, matching the manuscript's existing convention."""
    s = f"{x:.{places}f}"
    return s.replace("0.", ".", 1) if s.startswith("0.") else s.replace("-0.", "-.", 1)


def interval(low: float, high: float) -> str:
    return f"[{fmt(low)},{fmt(high)}]"


def load_scores(pattern: str, root: Path = COIN / "reports") -> list[dict]:
    dirs = [d for d in glob.glob(str(root / pattern)) if "failed_attempts" not in d]
    if len(dirs) != 1:
        raise SystemExit(f"{pattern}: expected one report root, found {dirs}")
    path = Path(dirs[0]) / "stochastic_n5.scores.jsonl"
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def cue_index(rows: list[dict], domain: str, labels: str) -> tuple[float, float, float]:
    """Cue-following index D, plus the two persistence levels it differences.

    Ratio of means rather than mean of ratios: a per-episode ratio explodes when
    a predicted horizon-1 response is near zero, which produced a spurious
    -1.2e6 on the untrained base the first time this was computed.
    """
    out = {}
    for want in ("mediated_b", "direct_a"):
        h1, h3 = [], []
        for r in rows:
            parts = r["task_id"].split(":")
            dom, lab, struct = parts[1], parts[2], parts[3]
            cue = parts[-1].replace("cue-", "")
            if cue == "none" or dom != domain or lab != labels:
                continue
            if not r.get("predicted_response"):
                continue
            shown = struct if cue == "correct" else (
                "mediated_b" if struct == "direct_a" else "direct_a")
            if shown != want:
                continue
            p = np.asarray(r["predicted_response"])
            h1.append(np.mean(np.abs(p[:4])))
            h3.append(np.mean(np.abs(p[4:])))
        out[want] = float(np.mean(h3) / max(np.mean(h1), 1e-9)) if h1 else float("nan")
    return out["mediated_b"] - out["direct_a"], out["mediated_b"], out["direct_a"]


def episode_mae(rows: list[dict], domain: str, structure: str,
                parsed_only: bool = False) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["domain"] != domain or r["target_structure"] != structure:
            continue
        if r["cue"] != "correct":
            continue
        if parsed_only and not r.get("parsed"):
            continue
        acc[r["task_id"]].append(float(r["response_mae"]))
    return {t: float(np.mean(v)) for t, v in acc.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    lines: list[str] = []

    def macro(name: str, value: str) -> None:
        # providecommand, not newcommand: the macro file is input by both the
        # body section and the appendix, and either may be built alone.
        lines.append(f"\\providecommand{{\\e{name}}}{{{value}}}")

    # ---- Coin City eight-seed endpoints -----------------------------------
    for tag, model in (("Eight", "qwen3_8b"), ("Fourb", "qwen3_4b")):
        path = COIN / f"reports/{model}_step300_stochastic_eight_seed.json"
        payload = json.loads(path.read_text())
        macro(f"{tag}G", str(payload["G"]))
        for cell in payload["transfer_cells"]:
            k = WORDS[cell["key"]]
            h, t = cell["hierarchical_bootstrap"], cell["student_t"]
            macro(f"{tag}{k}Base", fmt(cell["base"], 2))
            macro(f"{tag}{k}Prior", fmt(cell["population_prior"], 2))
            macro(f"{tag}{k}Matched", fmt(cell["causal"], 2))
            macro(f"{tag}{k}Est", fmt(h["estimate"]))
            macro(f"{tag}{k}Ci", interval(h["ci95_low"], h["ci95_high"]))
            macro(f"{tag}{k}Tci", interval(t["ci95_low"], t["ci95_high"]))
            macro(f"{tag}{k}Wild", fmt(cell["wild_cluster_bootstrap"]["p_two_sided"], 4))
            macro(f"{tag}{k}Sign", f"{cell['sign_test']['positive']}/{cell['sign_test']['seeds']}")
            macro(f"{tag}{k}SignP", fmt(cell["sign_test"]["p_one_sided"], 4))

    # ---- published three-seed values, for the before/after -----------------
    pub = json.loads((COIN / "reports/registered_results.json").read_text())
    for entry in pub["registered_estimates"]["transfer_cells"]:
        if entry.get("decode") != "stochastic":
            continue
        key = next((w for w, d, s in CELLS
                    if d == entry["domain"] and s == entry["target_structure"]), None)
        if key is None:
            continue
        short = {"qwen3_8b": "Eight", "qwen3_4b": "Fourb", "llama3_1_8b": "Llama"}.get(
            entry["model"])
        if short:
            macro(f"Pub{short}{WORDS[key]}Est", fmt(entry["population_prior_minus_causal"]))
            macro(f"Pub{short}{WORDS[key]}Ci",
                  interval(entry["ci95_low"], entry["ci95_high"]))

    # ---- base-versus-trained, fail-closed and parsed-only ------------------
    for tag, model, seeds in (("Eight", "qwen3_8b", range(42, 50)),
                              ("Fourb", "qwen3_4b", range(42, 50))):
        base = load_scores(f"base_{model}_j*")
        arms = {a: {s: load_scores(f"{a}_{model}_s{s}_*") for s in seeds}
                for a in ("causal", "population_prior")}
        for key, domain, structure in CELLS:
            for suffix, parsed in (("", False), ("Parsed", True)):
                b = episode_mae(base, domain, structure, parsed)
                per = []
                for s in seeds:
                    c = episode_mae(arms["causal"][s], domain, structure, parsed)
                    p = episode_mae(arms["population_prior"][s], domain, structure, parsed)
                    common = sorted(set(b) & set(c) & set(p))
                    per.append(float(np.mean([b[t] - (c[t] + p[t]) / 2 for t in common])))
                v = np.asarray(per)
                m, sd = float(v.mean()), float(v.std(ddof=1))
                h = TCRIT[len(v) - 1] * sd / math.sqrt(len(v))
                macro(f"{tag}{WORDS[key]}BaseGain{suffix}", fmt(m))
                macro(f"{tag}{WORDS[key]}BaseGain{suffix}Ci", interval(m - h, m + h))
        macro(f"{tag}BaseParse", fmt(100 * np.mean([bool(r.get("parsed")) for r in base]), 2))

    # ---- structure-selection pilot ----------------------------------------
    try:
        selbase = load_scores("selbase_qwen3_8b_j*")
        pilot = {s: load_scores(f"sel_causal_qwen3_8b_s{s}_*") for s in (42, 43, 44)}
    except SystemExit:
        print("pilot endpoints incomplete; skipping pilot macros", file=sys.stderr)
    else:
        for domain, labels, name in (("coin_city", "semantic", "CitySem"),
                                     ("coin_harbor", "semantic", "HarborSem"),
                                     ("coin_city", "arbitrary", "CityArb"),
                                     ("coin_harbor", "arbitrary", "HarborArb")):
            d_base, bm, bd = cue_index(selbase, domain, labels)
            macro(f"Sel{name}Base", fmt(d_base))
            v = np.asarray([cue_index(pilot[s], domain, labels)[0] for s in (42, 43, 44)])
            m, sd = float(v.mean()), float(v.std(ddof=1))
            h = TCRIT[2] * sd / math.sqrt(3)
            macro(f"Sel{name}Est", fmt(m))
            macro(f"Sel{name}Ci", interval(m - h, m + h))
        _, bm, bd = cue_index(selbase, "coin_city", "semantic")
        macro("SelBasePersistMed", fmt(bm, 2))
        macro("SelBasePersistDir", fmt(bd, 2))
        levels = [cue_index(pilot[s], "coin_city", "semantic")[1] for s in (42, 43, 44)]
        macro("SelTrainedPersistLo", fmt(min(levels), 2))
        macro("SelTrainedPersistHi", fmt(max(levels), 2))

    # ---- mechanism family -------------------------------------------------
    # Import the validated analyzer and call it, rather than scraping its
    # printed output: the first version of this parsed stdout and silently
    # emitted no macros at all when the format did not match.
    sys.path.insert(0, str(MECH))
    try:
        import project_mechanism_seed_power as mech
    except ImportError as exc:
        print(f"mechanism analyzer unavailable ({exc}); skipping", file=sys.stderr)
    else:
        mech_seeds = (42, 43, 44, 45, 46)
        mech.SEEDS = mech_seeds
        reg = mech.registered_endpoints() | mech.added_seed_endpoints("qwen3_4b")
        for disc, tag in (("disclosed", "Disc"), ("undisclosed", "Undisc")):
            diffs = {}
            ok = True
            for s in mech_seeds:
                c = reg.get((disc, "causal_family", "qwen3_4b", s))
                p_ = reg.get((disc, "population_prior", "qwen3_4b", s))
                if not c or not p_:
                    ok = False
                    break
                cm, pm = mech.episode_means(c), mech.episode_means(p_)
                tasks = sorted(set(cm) & set(pm))
                diffs[s] = np.asarray([pm[t] - cm[t] for t in tasks])
            if not ok:
                print(f"mechanism {disc}: incomplete roster; skipping", file=sys.stderr)
                continue
            means = np.asarray([float(np.mean(diffs[s])) for s in mech_seeds])
            m, sd = float(means.mean()), float(means.std(ddof=1))
            cell_seed = mech.BOOTSTRAP_SEED + ("disclosed", "undisclosed").index(disc)
            _, lo, hi = mech.hierarchical(diffs, mech_seeds, seed=cell_seed)
            half = TCRIT[len(mech_seeds) - 1] * sd / math.sqrt(len(mech_seeds))
            pos, sp = mech.sign_p(means)
            macro(f"Mech{tag}Est", fmt(m))
            macro(f"Mech{tag}Ci", interval(lo, hi))
            macro(f"Mech{tag}Tci", interval(m - half, m + half))
            macro(f"Mech{tag}Wild", fmt(mech.wild_cluster(means, seed=cell_seed), 4))
            macro(f"Mech{tag}Sign", f"{pos}/{len(mech_seeds)}")
            macro(f"Mech{tag}SignP", fmt(sp, 4))
            macro(f"Mech{tag}Sd", fmt(sd))
        macro("MechG", str(len(mech_seeds)))

    # ---- mechanism-family shuffled-target control -------------------------
    # Distribution-matched: identical prompts, budget and target multiset;
    # only the episode-to-target assignment differs. Partial rosters are
    # refused outright so no incomplete cell can reach the prose.
    shuf = MECH.parent / "overnight_diagnostics_20260921/shuffled_control"
    results_path, status_path = shuf / "results.json", shuf / "finalization_status.json"
    if not (results_path.is_file() and status_path.is_file()):
        print("shuffled-target control: no finalized results; skipping", file=sys.stderr)
    else:
        res = json.loads(results_path.read_text())
        status = json.loads(status_path.read_text())
        if not (res.get("complete") and status.get("complete")
                and status.get("analysis_exit_code") == 0 and not res.get("incomplete_cells")):
            print("shuffled-target control: incomplete roster; skipping", file=sys.stderr)
        else:
            tags = {("disclosed", "stochastic"): "DiscStoch",
                    ("disclosed", "greedy"): "DiscGreedy",
                    ("undisclosed", "stochastic"): "UndiscStoch",
                    ("undisclosed", "greedy"): "UndiscGreedy"}
            seen = set()
            for c in res["contrasts"]:
                if c["scope"] != "overall":
                    continue
                tag = tags.get((c["disclosure"], c["decode"]))
                if tag is None:
                    continue
                if not c.get("complete_five_seed_roster"):
                    raise SystemExit(f"shuffled control {tag}: partial roster in a complete result")
                suffix = "" if c["metric"] == "response_mae" else "Level"
                macro(f"Shuf{tag}{suffix}Est", fmt(c["effect"]))
                macro(f"Shuf{tag}{suffix}Tci", interval(*c["student_t_95ci"]))
                macro(f"Shuf{tag}{suffix}Ci", interval(*c["hierarchical_seed_trajectory_95ci"]))
                macro(f"Shuf{tag}{suffix}Wild", fmt(c["wild_cluster_webb_p_two_sided"], 4))
                macro(f"Shuf{tag}{suffix}Sign",
                      f"{c['sign_test']['positive']}/{c['sign_test']['seeds']}")
                macro(f"Shuf{tag}{suffix}SignP", fmt(c["sign_test"]["p_one_sided"], 4))
                seen.add((tag, suffix))
                macro("ShufG", str(c["training_seeds"]))
            if len(seen) != 8:
                raise SystemExit(f"shuffled control: expected 8 overall contrasts, got {len(seen)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = ("% Generated by exp3_training_transfer/render_exp3_paper_macros.py\n"
              "% Do not edit. Every Experiment 3 number in the prose comes from here.\n")
    args.out.write_text(header + "\n".join(sorted(set(lines))) + "\n")
    print(f"wrote {len(set(lines))} macros -> {args.out}")


if __name__ == "__main__":
    main()
