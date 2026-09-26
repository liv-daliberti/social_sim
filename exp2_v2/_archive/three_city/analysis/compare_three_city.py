#!/usr/bin/env python3
"""Compare multiple frontier models on shared three-city triplets."""

from __future__ import annotations

import argparse
from pathlib import Path

from analyze_three_city import aggregate, load_records


DISPLAY_NAMES = {
    "claude-opus-4-8": "Claude Opus 4.8",
    "DeepSeek-V4-Pro": "DeepSeek V4 Pro",
    "gpt-5.4": "GPT-5.4",
}


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def load_runs(paths):
    runs = {}
    for path in paths:
        records = load_records(path)
        model = records[0]["model"]
        if any(record["model"] != model for record in records):
            raise ValueError(f"mixed model labels in {path}")
        runs[model] = aggregate(records)
    return runs


def make_table(runs):
    first = max(
        runs.values(),
        key=lambda summary: summary[min(summary)]["n"],
    )
    models = list(runs)
    lines = [
        "| k | Method | P(correct type) | Gain MAE | Forecast MAE | Valid n |",
        "|---:|:-------|----------------:|---------:|-------------:|--------:|",
    ]
    for k in first:
        for method, label in (
            ("oracle", "Bayes oracle"),
            ("frequentist", "Frequentist"),
            ("naive", "Naive"),
            ("clairvoyant", "Clairvoyant ceiling"),
        ):
            values = first[k][method]
            valid_n = values["n_gain"]
            lines.append(
                f"| {k} | {label} | {_fmt(values['p_correct_type'])} | "
                f"{_fmt(values['gain_error'])} | {_fmt(values['forecast_mae'])} | "
                f"{valid_n} |"
            )
        for model in models:
            values = runs[model][k].get("lm", {})
            valid_n = values.get("n_gain", 0)
            lines.append(
                f"| {k} | {DISPLAY_NAMES.get(model, model)} | "
                f"{_fmt(values.get('p_correct_type'))} | "
                f"{_fmt(values.get('gain_error'))} | "
                f"{_fmt(values.get('forecast_mae'))} | {valid_n} |"
            )
    return "\n".join(lines)


def make_figure(runs, path):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --figure") from exc

    first = max(
        runs.values(),
        key=lambda summary: summary[min(summary)]["n"],
    )
    ks = list(first)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = {
        "claude-opus-4-8": "#7b2cbf",
        "DeepSeek-V4-Pro": "#d95f02",
        "gpt-5.4": "#1b9e77",
    }

    axes[0].plot(
        ks,
        [first[k]["oracle"]["p_correct_type"] for k in ks],
        color="black",
        marker="o",
        linewidth=2.3,
        label="Bayes oracle",
    )
    for model, summary in runs.items():
        model_n = summary[min(summary)].get("lm", {}).get("n_gain", 0)
        axes[0].plot(
            ks,
            [summary[k].get("lm", {}).get("p_correct_type") for k in ks],
            marker="o",
            linewidth=2,
            color=colors.get(model),
            label=f"{DISPLAY_NAMES.get(model, model)} (n={model_n})",
        )
    axes[0].axhline(0.5, color="0.55", linestyle="--", linewidth=1.2)
    axes[0].set(
        title="A. Identifies City C's type",
        xlabel="City C polls observed (k)",
        ylabel="P(correct reference type)",
        ylim=(0.45, 1.02),
    )

    baseline_styles = {
        "oracle": ("Bayes oracle", "black", "o"),
        "frequentist": ("Frequentist", "#377eb8", "^"),
        "naive": ("Naive", "#777777", "x"),
    }
    for method, (label, color, marker) in baseline_styles.items():
        axes[1].plot(
            ks,
            [first[k][method]["gain_error"] for k in ks],
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
        )
        axes[2].plot(
            ks,
            [first[k][method]["forecast_mae"] for k in ks],
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
        )
    for model, summary in runs.items():
        model_n = summary[min(summary)].get("lm", {}).get("n_gain", 0)
        label = f"{DISPLAY_NAMES.get(model, model)} (n={model_n})"
        color = colors.get(model)
        axes[1].plot(
            ks,
            [summary[k].get("lm", {}).get("gain_error") for k in ks],
            label=label,
            color=color,
            marker="o",
            linewidth=2,
        )
        axes[2].plot(
            ks,
            [summary[k].get("lm", {}).get("forecast_mae") for k in ks],
            label=label,
            color=color,
            marker="o",
            linewidth=2,
        )

    axes[1].set(
        title="B. Recovers the response gain",
        xlabel="City C polls observed (k)",
        ylabel="Absolute gain error",
    )
    axes[2].set(
        title="C. Forecasts held-out shocks",
        xlabel="City C polls observed (k)",
        ylabel="Next-poll MAE",
    )
    for axis in axes:
        axis.set_xticks(ks)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--table-out", type=Path, default=None)
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args()

    runs = load_runs(args.results)
    table = make_table(runs)
    print(table)
    if args.table_out is not None:
        args.table_out.parent.mkdir(parents=True, exist_ok=True)
        args.table_out.write_text(table + "\n")
        print(f"\nWrote {args.table_out}")
    if args.figure is not None:
        make_figure(runs, args.figure)
        print(f"Wrote {args.figure} and {args.figure.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
