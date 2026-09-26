#!/usr/bin/env python3
"""Aggregate Experiment 2 v2 three-city JSONL results."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    return statistics.mean(clean) if clean else None


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def load_records(path: Path) -> list[Dict[str, Any]]:
    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}") from exc
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def aggregate(records: list[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    by_k: Dict[int, list[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        for point in record["curve"]:
            by_k[int(point["k"])].append(point)

    result: Dict[int, Dict[str, Any]] = {}
    for k, points in sorted(by_k.items()):
        row: Dict[str, Any] = {"k": k, "n": len(points)}
        for method in (
            "oracle",
            "clairvoyant",
            "frequentist",
            "naive",
            "target_only",
            "pooled",
            "lm",
        ):
            entries = [point.get(method) for point in points]
            entries = [entry for entry in entries if entry is not None]
            if not entries:
                continue
            row[method] = {
                "n": len(entries),
                "n_type": sum(
                    entry.get("p_correct_type") is not None for entry in entries
                ),
                "n_gain": sum(
                    entry.get("gain_error") is not None for entry in entries
                ),
                "n_forecast": sum(
                    entry.get("forecast_mae") is not None for entry in entries
                ),
                "p_correct_type": _mean(
                    entry.get("p_correct_type") for entry in entries
                ),
                "type_accuracy": _mean(
                    entry.get("type_correct") for entry in entries
                ),
                "type_brier": _mean(
                    entry.get("type_brier") for entry in entries
                ),
                "gain_error": _mean(
                    entry.get("gain_error") for entry in entries
                ),
                "normalised_gain_error": _mean(
                    entry.get("normalised_gain_error") for entry in entries
                ),
                "adaptation": _mean(
                    entry.get("adaptation") for entry in entries
                ),
                "forecast_mae": _mean(
                    entry.get("forecast_mae") for entry in entries
                ),
            }
        result[k] = row
    return result


def markdown_table(summary: Dict[int, Dict[str, Any]]) -> str:
    has_lm = any("lm" in row for row in summary.values())
    headers = [
        "k",
        "n",
        "Oracle P(correct)",
        "Oracle gain MAE",
        "Frequentist gain MAE",
        "Naive gain MAE",
        "Oracle forecast MAE",
        "Frequentist forecast MAE",
        "Naive forecast MAE",
    ]
    if has_lm:
        headers += [
            "LM P(correct)",
            "LM gain MAE",
            "LM forecast MAE",
        ]

    align = ["---:", "---:"] + ["---:"] * (len(headers) - 2)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(align) + "|",
    ]
    for k, row in summary.items():
        oracle = row["oracle"]
        frequentist = row["frequentist"]
        naive = row["naive"]
        values = [
            str(k),
            str(row["n"]),
            _fmt(oracle["p_correct_type"]),
            _fmt(oracle["gain_error"]),
            _fmt(frequentist["gain_error"]),
            _fmt(naive["gain_error"]),
            _fmt(oracle["forecast_mae"]),
            _fmt(frequentist["forecast_mae"]),
            _fmt(naive["forecast_mae"]),
        ]
        if has_lm:
            lm = row.get("lm", {})
            values += [
                _fmt(lm.get("p_correct_type")),
                _fmt(lm.get("gain_error")),
                _fmt(lm.get("forecast_mae")),
            ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def make_figure(summary: Dict[int, Dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --figure") from exc

    ks = list(summary)
    has_lm = any("lm" in row for row in summary.values())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    axes[0].plot(
        ks,
        [summary[k]["oracle"]["p_correct_type"] for k in ks],
        marker="o",
        linewidth=2.2,
        label="Hierarchical Bayes",
    )
    if has_lm:
        axes[0].plot(
            ks,
            [summary[k].get("lm", {}).get("p_correct_type") for k in ks],
            marker="s",
            linewidth=2.2,
            label="Language model",
        )
    axes[0].axhline(0.5, color="0.55", linestyle="--", linewidth=1.2)
    axes[0].set(
        xlabel="Target-city polls observed (k)",
        ylabel="Probability assigned to correct type",
        ylim=(0.45, 1.02),
        title="A. Identifies which reference City C matches",
    )
    axes[0].legend(frameon=False)

    styles = {
        "oracle": ("Hierarchical Bayes", "o", 2.2),
        "frequentist": ("Target-only frequentist", "^", 1.8),
        "naive": ("Fixed 50/50 average", "x", 1.8),
    }
    if has_lm:
        styles["lm"] = ("Language model", "s", 2.2)
    for method, (label, marker, width) in styles.items():
        axes[1].plot(
            ks,
            [
                summary[k].get(method, {}).get("normalised_gain_error")
                for k in ks
            ],
            marker=marker,
            linewidth=width,
            label=label,
        )
    axes[1].set(
        xlabel="Target-city polls observed (k)",
        ylabel="Gain error / distance between city types",
        title="B. Moves from the average to the right city type",
    )
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        axis.set_xticks(ks)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    if path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args()

    records = load_records(args.results)
    summary = aggregate(records)
    table = markdown_table(summary)
    print(table)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(table + "\n")
        print(f"\nWrote {args.summary_out}")
    if args.figure is not None:
        make_figure(summary, args.figure)
        print(f"Wrote {args.figure} and {args.figure.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
