#!/usr/bin/env python3
"""Plot deterministic design checks for the independent-case C2 v2 task."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from engine.three_city_c2_v2 import PREFIX_LADDER

_DATA = _ROOT / "data" / "three_city_c2_v2"


def _read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _mae(records, method, k):
    errors = []
    for record in records:
        if record["condition"] != "none" or record["k"] != k:
            continue
        prediction = record["baselines"][method]["predicted_poll"]
        if prediction is not None:
            errors.append(
                abs(prediction - record["gold"]["expected_poll"])
            )
    return statistics.mean(errors) if errors else None


def _p_correct(record):
    truth_a = record["gold"]["target_matches"] == "A"
    p_a = record["baselines"]["context_oracle"]["p_match_a"]
    return p_a if truth_a else 1.0 - p_a


def _context_curve(records, role, k):
    values = []
    for record in records:
        if record["k"] != k:
            continue
        target_high = record["gold"]["target_response"] == 1.0
        if role == "none":
            wanted = record["condition"] == "none"
        elif role == "aligned":
            wanted = (
                (target_high and record["condition"] == "cue_high")
                or (not target_high and record["condition"] == "cue_low")
            )
        elif role == "misleading":
            wanted = (
                (target_high and record["condition"] == "cue_low")
                or (not target_high and record["condition"] == "cue_high")
            )
        else:
            raise ValueError(role)
        if wanted:
            values.append(_p_correct(record))
    return statistics.mean(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=_DATA / "answer_key_c2_v2.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DATA / "design_diagnostics.png",
    )
    args = parser.parse_args()

    records = _read_jsonl(args.answer_key)
    ks = list(PREFIX_LADDER)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    methods = {
        "pooled_exemplar": ("Pooled examples (references only)", "--", "#9a7b37"),
        "frequentist": ("Frequentist: City C only", "-.", "#222222"),
        "target_only_bayes": (
            "Structure-informed ceiling",
            "-",
            "#b03a6a",
        ),
    }
    mae = {
        method: [_mae(records, method, k) for k in ks] for method in methods
    }
    for method, (label, linestyle, color) in methods.items():
        axes[0].plot(
            ks,
            mae[method],
            marker="o",
            linestyle=linestyle,
            linewidth=2,
            color=color,
            label=label,
        )
    # Shade what the reference cities and the two-point prior are still worth
    # over averaging City C alone. It narrows along the ladder without closing.
    axes[0].fill_between(
        ks[1:],
        mae["target_only_bayes"][1:],
        mae["frequentist"][1:],
        color="#2b8c6b",
        alpha=0.18,
        linewidth=0,
        label="Headroom over target-only averaging",
        zorder=0,
    )
    axes[0].annotate(
        f"still {mae['frequentist'][-1] - mae['target_only_bayes'][-1]:.2f} pts "
        f"at k={ks[-1]}",
        xy=(ks[-1], mae["frequentist"][-1]),
        xytext=(-6, 10),
        textcoords="offset points",
        ha="right",
        fontsize=7,
        color="#1d6049",
    )
    axes[0].set_title("A. Forecast information curve")
    axes[0].set_ylabel("Mean absolute error (poll points)")
    axes[0].set_xlabel("Completed City C cases (k)")
    axes[0].set_xticks(ks)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    context_styles = {
        "aligned": ("Aligned background", "#2b8c6b"),
        "none": ("No background", "#37679a"),
        "misleading": ("Misleading background", "#d65f45"),
    }
    for role, (label, color) in context_styles.items():
        axes[1].plot(
            ks,
            [_context_curve(records, role, k) for k in ks],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
    axes[1].set_title("B. Evidence eventually dominates context")
    axes[1].set_ylabel("Hidden-oracle probability on correct pattern")
    axes[1].set_xlabel("Completed City C cases (k)")
    axes[1].set_xticks(ks)
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "Independent-case three-city v2 — offline design checks only",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
