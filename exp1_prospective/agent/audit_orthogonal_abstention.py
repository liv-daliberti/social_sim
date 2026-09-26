#!/usr/bin/env python3
"""Decompose the Experiment 1 sensitivity ratio into abstention and magnitude.

Source of record: the frozen June 17 consistency report, the same enumeration
used by ``audit_selectivity_robustness.py``.  Each scored run contributes its
stored ``delta_yes_prob``; no collection directory is read.

The published sensitivity ratio is a ratio of mean absolute revisions.  Because
a revision of exactly zero enters that mean, the ratio mixes two behaviours a
reader would want separated: how often a deployment declines to move at all,
and how far it moves when it does.  Writing A for the share of runs whose
revision is exactly zero and M for mean absolute revision among the rest,

    E|dp| = (1 - A) * M

so the ratio factors exactly:

    Sensitivity = M_dir / M_orth  *  (1 - A_dir) / (1 - A_orth)
                  \_____________/    \_______________________/
                   magnitude term        abstention term

Both factors are reported per deployment with market-clustered percentile
intervals matching ``evaluate_clustered_uncertainty.py`` (20,000 draws, whole
markets resampled so every packet and repeated initial run stays together).

Qwen2.5-7B is excluded throughout, as in every other magnitude comparison:
its archived updates mix 0-1 and 0-100 scales.

Outputs:
  data/results/orthogonal_abstention.json
  paper/tables/exp1_abstention_decomposition.tex
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
REPO_ROOT = ROOT.parent
FROZEN = ROOT / "data" / "results" / "consistency_report_2026-06-17.json"
OUT_JSON = ROOT / "data" / "results" / "orthogonal_abstention.json"
OUT_TEX = REPO_ROOT / "paper" / "tables" / "exp1_abstention_decomposition.tex"

DEFAULT_REPETITIONS = 20_000
DEFAULT_SEED = 20_260_822

ARMS = ("pro_H1", "anti_H1", "orthogonal")
# A stored revision counts as "unchanged" when the updated headline probability
# reproduces the displayed prior. The frozen deltas are exact zeros rather than
# small residuals; the tolerance guards float representation only.
ZERO_TOL_PP = 1e-9

MODEL_ORDER = (
    "claude-opus-4-8", "gpt-5.4", "DeepSeek-V4-Pro",
    "qwen2.5:72b", "llama3.3:70b", "llama3.1:70b",
    "qwen2.5:32b", "qwen2.5:14b", "llama3.1:8b",
)
HOSTED = {"claude-opus-4-8", "gpt-5.4", "DeepSeek-V4-Pro"}
LABELS = {
    "claude-opus-4-8": "Claude Opus~4.8", "gpt-5.4": "GPT-5.4",
    "DeepSeek-V4-Pro": "DeepSeek V4-Pro", "qwen2.5:72b": "Qwen2.5-72B",
    "llama3.3:70b": "Llama-3.3-70B", "llama3.1:70b": "Llama-3.1-70B",
    "qwen2.5:32b": "Qwen2.5-32B", "qwen2.5:14b": "Qwen2.5-14B",
    "llama3.1:8b": "Llama-3.1-8B",
}


def load_market_arrays(report: dict) -> dict[str, dict[str, np.ndarray]]:
    """One row per market per arm: record count, unchanged count, movement sum."""
    acc: dict[str, dict[tuple[str, str], list[float]]] = {
        m: defaultdict(lambda: [0.0, 0.0, 0.0]) for m in MODEL_ORDER
    }
    for model, model_report in report["per_model"].items():
        if model not in acc:
            continue
        for market in model_report["per_market"]:
            task_id = market.get("task_id")
            for packet in market.get("cf_results", []):
                direction = packet.get("direction")
                if direction not in ARMS:
                    continue
                arm = "orth" if direction == "orthogonal" else "dir"
                for run in packet.get("runs", []):
                    delta = run.get("delta_yes_prob")
                    if delta is None:
                        continue
                    moved_pp = abs(delta) * 100.0
                    cell = acc[model][(task_id, arm)]
                    cell[0] += 1.0
                    cell[1] += 1.0 if moved_pp <= ZERO_TOL_PP else 0.0
                    cell[2] += moved_pp

    out: dict[str, dict[str, np.ndarray]] = {}
    for model in MODEL_ORDER:
        markets = sorted({task for (task, _) in acc[model]})
        index = {task: i for i, task in enumerate(markets)}
        block = {
            f"{arm}_{field}": np.zeros(len(markets))
            for arm in ("dir", "orth")
            for field in ("n", "zero", "sum")
        }
        for (task, arm), (n, zero, total) in acc[model].items():
            i = index[task]
            block[f"{arm}_n"][i] = n
            block[f"{arm}_zero"][i] = zero
            block[f"{arm}_sum"][i] = total
        block["markets"] = np.array(markets, dtype=object)
        out[model] = block
    return out


def statistics(block: dict[str, np.ndarray], rows: np.ndarray | None = None) -> dict:
    """Point estimands from a (possibly resampled) set of market rows."""
    take = (lambda key: block[key][rows]) if rows is not None else (lambda key: block[key])
    stats = {}
    for arm in ("dir", "orth"):
        n = take(f"{arm}_n").sum()
        zero = take(f"{arm}_zero").sum()
        total = take(f"{arm}_sum").sum()
        moved = n - zero
        stats[f"{arm}_n"] = n
        stats[f"{arm}_abstention"] = zero / n if n else np.nan
        stats[f"{arm}_mean_abs"] = total / n if n else np.nan
        stats[f"{arm}_mean_abs_given_move"] = total / moved if moved else np.nan
    stats["sensitivity"] = (
        stats["dir_mean_abs"] / stats["orth_mean_abs"]
        if stats["orth_mean_abs"] else np.nan
    )
    stats["sensitivity_given_move"] = (
        stats["dir_mean_abs_given_move"] / stats["orth_mean_abs_given_move"]
        if stats["orth_mean_abs_given_move"] else np.nan
    )
    stats["abstention_factor"] = (
        (1 - stats["dir_abstention"]) / (1 - stats["orth_abstention"])
        if stats["orth_abstention"] < 1 else np.nan
    )
    return stats


REPORTED = (
    "orth_abstention", "dir_abstention",
    "orth_mean_abs", "orth_mean_abs_given_move", "dir_mean_abs_given_move",
    "sensitivity", "sensitivity_given_move", "abstention_factor",
)


def analyze(block: dict[str, np.ndarray], repetitions: int, seed: int) -> dict:
    point = statistics(block)
    n_markets = len(block["markets"])
    rng = np.random.default_rng(seed)
    draws = {key: np.empty(repetitions) for key in REPORTED}
    for r in range(repetitions):
        rows = rng.integers(0, n_markets, n_markets)
        sample = statistics(block, rows)
        for key in REPORTED:
            draws[key][r] = sample[key]
    result = {"n_markets": int(n_markets),
              "orthogonal_records": int(point["orth_n"]),
              "directional_records": int(point["dir_n"])}
    for key in REPORTED:
        finite = draws[key][np.isfinite(draws[key])]
        result[key] = {
            "estimate": float(point[key]),
            "ci_lower": float(np.percentile(finite, 2.5)),
            "ci_upper": float(np.percentile(finite, 97.5)),
        }
    return result


def _fmt_ratio(cell: dict) -> str:
    return (f"${cell['estimate']:.1f}\\times$ "
            f"{{\\scriptsize[{cell['ci_lower']:.1f},\\,{cell['ci_upper']:.1f}]}}")


def _fmt_pct(cell: dict) -> str:
    return (f"{100 * cell['estimate']:.1f}\\% "
            f"{{\\scriptsize[{100 * cell['ci_lower']:.1f},\\,"
            f"{100 * cell['ci_upper']:.1f}]}}")


def write_table(per_model: dict) -> None:
    lines = []
    for model in MODEL_ORDER:
        r = per_model[model]
        lines.append(
            f"{LABELS[model]:16s} & {_fmt_pct(r['orth_abstention'])} & "
            f"{r['orth_mean_abs_given_move']['estimate']:.1f}\\,pp & "
            f"{r['dir_mean_abs_given_move']['estimate']:.1f}\\,pp & "
            f"{_fmt_ratio(r['sensitivity_given_move'])} & "
            f"{_fmt_ratio(r['sensitivity'])} \\\\"
        )
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(
        "% Generated by exp1_prospective/agent/audit_orthogonal_abstention.py;"
        " do not edit.\n"
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        "& \\textbf{Orthogonal} & \\multicolumn{2}{c}{\\textbf{Mean $|\\Delta\\hat p|$"
        " given a move}} & \\textbf{Sens.} & \\textbf{Sens.} \\\\\n"
        "\\cmidrule(lr){3-4}\n"
        "\\textbf{Model} & \\textbf{abstention} & orthogonal & directional"
        " & \\textbf{given a move} & \\textbf{unconditional} \\\\\n\\midrule\n"
        + "\n".join(lines[:3]) + "\n\\midrule\n" + "\n".join(lines[3:])
        + "\n\\bottomrule\n\\end{tabular}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    report = json.loads(FROZEN.read_text())
    blocks = load_market_arrays(report)
    per_model = {m: analyze(blocks[m], args.repetitions, args.seed)
                 for m in MODEL_ORDER}

    # Group contrast on the abstention rate, macro-averaged within group so a
    # deployment with more runs does not dominate.
    def group_rate(models):
        return float(np.mean([per_model[m]["orth_abstention"]["estimate"]
                              for m in models]))

    result = {
        "analysis": "exp1_orthogonal_abstention",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_source": FROZEN.name,
        "excluded_models": ["qwen2.5:7b"],
        "zero_tolerance_pp": ZERO_TOL_PP,
        "bootstrap": {"repetitions": args.repetitions, "seed": args.seed,
                      "unit": "market", "interval": "percentile 95%"},
        "identity": "sensitivity = sensitivity_given_move * abstention_factor",
        "group_orthogonal_abstention": {
            "hosted_macro_mean": group_rate([m for m in MODEL_ORDER if m in HOSTED]),
            "open_weight_macro_mean": group_rate(
                [m for m in MODEL_ORDER if m not in HOSTED]),
        },
        "per_model": {m: {"label": LABELS[m], **per_model[m]} for m in MODEL_ORDER},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    write_table(per_model)

    hdr = (f"{'model':17s} {'orth n':>7s} {'abstain':>9s} {'orth|move':>10s} "
           f"{'dir|move':>9s} {'sens|move':>10s} {'abst.fac':>9s} {'sens':>7s} "
           f"{'check':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for model in MODEL_ORDER:
        r = per_model[model]
        check = (r["sensitivity_given_move"]["estimate"]
                 * r["abstention_factor"]["estimate"])
        print(f"{LABELS[model].replace('~',' '):17s} "
              f"{r['orthogonal_records']:7d} "
              f"{100 * r['orth_abstention']['estimate']:8.1f}% "
              f"{r['orth_mean_abs_given_move']['estimate']:10.2f} "
              f"{r['dir_mean_abs_given_move']['estimate']:9.2f} "
              f"{r['sensitivity_given_move']['estimate']:9.1f}x "
              f"{r['abstention_factor']['estimate']:9.2f} "
              f"{r['sensitivity']['estimate']:6.1f}x "
              f"{check:6.1f}x")
    print()
    print(f"hosted orthogonal abstention (macro mean):      "
          f"{100 * result['group_orthogonal_abstention']['hosted_macro_mean']:.1f}%")
    print(f"open-weight orthogonal abstention (macro mean): "
          f"{100 * result['group_orthogonal_abstention']['open_weight_macro_mean']:.1f}%")
    print()
    print(f"wrote {OUT_JSON.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT_TEX.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
