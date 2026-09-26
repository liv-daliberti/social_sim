#!/usr/bin/env python3
"""Tabulate the Exp4 price ablation next to the reported with-price results."""

from __future__ import annotations

import glob
import json
from pathlib import Path

REPORTS = Path(__file__).resolve().parents[1] / "reports"
REPORTED_318 = {
    "qwen3_1_7b": "exp4_scale_qwen3_1_7b_stochastic_j30890110",
    "llama3_2_3b": "exp4_scale_llama3_2_3b_stochastic_j30890115",
    "qwen3_14b": "exp4_scale_qwen3_14b_stochastic_j30870548",
    "qwen3_32b": "exp4_scale_qwen3_32b_stochastic_j30913533",
}


def fmt(c: dict) -> str:
    return f"{c['estimate']:+.4f} [{c['ci95_low']:+.4f},{c['ci95_high']:+.4f}]"


def trained_mean(models: dict) -> float:
    return sum(models[f"seed_{s}"]["brier"] for s in (42, 43, 44)) / 3


def main() -> None:
    lines = []
    for model, reported in REPORTED_318.items():
        s = json.loads((REPORTS / f"{reported}.summary.json").read_text())
        lines.append(
            f"{model:12s} scale318  five_draw  with_price(reported)  "
            f"base={s['models']['base']['brier']:.4f} trained={trained_mean(s['models']):.4f} "
            f"market={s['baselines']['market']['brier']:.4f} "
            f"trained-base={fmt(s['comparisons_brier']['trained_seed_mean_minus_base'])}"
        )
        for path in sorted(glob.glob(str(REPORTS / f"exp4_price_ablation_{model}_*.summary.json"))):
            a = json.loads(Path(path).read_text())
            c = a["comparisons_brier"]
            cover = min(m["parse_coverage"] for m in a["models"].values())
            lines.append(
                f"{model:12s} {a['holdout']:9s} {a['decoder']:10s} {a['condition']:21s} "
                f"base={a['models']['base']['brier']:.4f} trained={trained_mean(a['models']):.4f} "
                f"clim={a['baselines']['train_climatology']['brier']:.4f} "
                f"market={a['baselines']['market_last_price']['brier']:.4f} "
                f"min_parse={cover:.3f}\n"
                f"{'':12s}   trained-base={fmt(c['trained_seed_mean_minus_base'])} "
                f"trained-clim={fmt(c['trained_seed_mean_minus_climatology'])} "
                f"base-clim={fmt(c['base_minus_climatology'])}"
            )
        lines.append("")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
