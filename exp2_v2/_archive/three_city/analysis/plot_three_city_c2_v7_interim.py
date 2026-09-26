#!/usr/bin/env python3
"""Watermarked operational snapshot of an incomplete v7 confirmatory run.

This script is deliberately separate from the frozen confirmatory analyzer. It
must not overwrite confirmatory outputs or inform changes to the frozen design.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

import analyze_three_city_c2_v7_confirmatory as confirmatory

_DATA = _ROOT / "data" / "three_city_c2_v7"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=_DATA / "answer_key_c2_v7.jsonl",
    )
    parser.add_argument(
        "--responses-dir",
        type=Path,
        default=_DATA / "confirmatory",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=_DATA / "interim",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    args = parser.parse_args()

    confirmatory._BOOTSTRAP = args.bootstrap_draws
    keys = {
        row["task_id"]: row
        for row in confirmatory._read_jsonl(args.answer_key)
    }
    responses = confirmatory._latest_responses(
        sorted(args.responses_dir.glob("responses_*.jsonl"))
    )
    rows = confirmatory._score_rows(responses, keys)
    models = [
        model
        for model in confirmatory._MODEL_ORDER
        if any(row["model"] == model for row in rows)
    ]
    if not models:
        raise SystemExit("no parsed responses available")

    frozen_draw_mae_panel = confirmatory._draw_mae_panel

    def draw_interim_mae_panel(axis, **kwargs):
        frozen_draw_mae_panel(axis, **kwargs)
        model = kwargs["model"]
        counts = [
            len(
                confirmatory._paired_rows(
                    rows,
                    model=model,
                    condition=confirmatory._PRIMARY_CONDITION,
                    k=k,
                )
            )
            for k in confirmatory.PREFIX_LADDER
        ]
        if len(set(counts)) == 1:
            count_label = f"paired n={counts[0]} at each k"
        else:
            count_label = f"paired n={min(counts)}–{max(counts)} across k"
        axis.set_title(
            f"{confirmatory._DISPLAY[model]}\n{count_label}",
            fontsize=8.3,
            pad=4,
        )

    confirmatory._draw_mae_panel = draw_interim_mae_panel

    timestamp = datetime.now(timezone.utc)
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    args.outdir.mkdir(parents=True, exist_ok=True)
    raw_path = args.outdir / f"interim_{stamp}_raw.png"
    final_path = args.outdir / f"interim_{stamp}.png"
    confirmatory._make_figure(
        raw_path,
        rows=rows,
        keys=keys,
        models=models,
    )

    raster = plt.imread(raw_path)
    figure = plt.figure(figsize=(7.25, 5.88), facecolor="white")
    axis = figure.add_axes([0.0, 0.0, 1.0, 0.94])
    axis.imshow(raster)
    axis.set_axis_off()
    figure.text(
        0.5,
        0.975,
        "INTERIM — INCOMPLETE COLLECTION — NOT CONFIRMATORY",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#A61B1B",
    )
    figure.savefig(final_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    primary = {}
    for model_index, model in enumerate(models):
        paired = confirmatory._paired_rows(
            rows,
            model=model,
            condition=confirmatory._PRIMARY_CONDITION,
            k=confirmatory._PRIMARY_K,
        )
        primary[model] = confirmatory._bootstrap_primary(
            paired,
            seed=confirmatory._BOOTSTRAP_SEED + 1000 * model_index,
        )
    parsed = {
        model: {
            arm: sum(
                row["model"] == model and row["arm"] == arm
                for row in rows
            )
            for arm in ("blind", "hint")
        }
        for model in models
    }
    metadata = {
        "status": "interim_incomplete_not_confirmatory",
        "created_at": timestamp.isoformat(),
        "bootstrap_draws": args.bootstrap_draws,
        "models": models,
        "parsed": parsed,
        "primary_snapshot": primary,
        "watermarked_figure": final_path.name,
        "raw_figure": raw_path.name,
    }
    metadata_path = args.outdir / f"interim_{stamp}.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(parsed, indent=2, sort_keys=True))
    print(f"Wrote {final_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
