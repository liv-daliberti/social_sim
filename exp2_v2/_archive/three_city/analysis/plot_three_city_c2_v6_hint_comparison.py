#!/usr/bin/env python3
"""Blind arm versus relevance-hint arm: capability or default?

Both arms answer byte-identical tables and share task IDs and answer key. The
only difference is the closing instruction, which in the hint arm adds that the
two earlier cities may be informative and that City C's measurements are noisy,
and permits extended reasoning.

The hint arm is a NON-BLIND control. It is not evidence about spontaneous
inference and must not be pooled with the blind arm. Its job is to separate
"cannot combine the examples with the target" from "does not, unprompted".

Three panels, all on the no-context condition:

1. forecast error --- does the hint actually help?
2. distance from City C's own average --- does the hint move them off it?
3. distance from the two-prototype forecaster, the information-matched agent
   that does combine both sources --- does the hint move them toward it?
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from engine.three_city_c2_v6 import PREFIX_LADDER, STARTING_POLL

_DATA = _ROOT / "data"
_BLIND = _DATA / "three_city_c2_v6" / "pilot_v6"
_HINT = _DATA / "three_city_c2_v6_hint" / "pilot_v6_hint"
_KEY = _DATA / "three_city_c2_v6" / "answer_key_c2_v6.jsonl"
_DISPLAY = {
    "claude-opus-4-8": "Claude Opus 4.8",
    "DeepSeek-V4-Pro": "DeepSeek V4 Pro",
    "gpt-5.4": "GPT-5.4",
}
_COLORS = {
    "claude-opus-4-8": "#7b4ab5",
    "DeepSeek-V4-Pro": "#168a86",
    "gpt-5.4": "#d65f45",
}


def _read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _estimate(city):
    changes = [p - city["starting_poll"] for p in city["ending_polls"]]
    return sum(d * c for d, c in zip(city["news"], changes)) / sum(
        d * d for d in city["news"]
    )


def _target_only_forecast(public, k):
    """City C's own response, estimated from City C alone.

    v6 splits each city's cases evenly between +m and -m, so the raw column
    mean sits near 50 and estimates nothing. The estimator is the difference of
    the two group means divided by twice the magnitude. This is what the ported
    v3 script got wrong: it compared models against the column mean, a statistic
    nobody in v6 is using.
    """
    target = public["target"]
    news = target["news"][:k]
    polls = target["ending_polls"][:k]
    if not news:
        return None
    plus = [p for d, p in zip(news, polls) if d > 0]
    minus = [p for d, p in zip(news, polls) if d < 0]
    if plus and minus:
        magnitude = abs(news[0])
        response = (
            statistics.mean(plus) - statistics.mean(minus)
        ) / (2 * magnitude)
    else:
        # An odd prefix can be one-signed; fall back to the through-origin fit.
        changes = [p - target["starting_poll"] for p in polls]
        response = sum(d * c for d, c in zip(news, changes)) / sum(
            d * d for d in news
        )
    return STARTING_POLL + response * public["test_news"]


def _two_prototype(public, k):
    """Information-matched: uses only what the prompt prints."""
    a, b = public["reference_a"], public["reference_b"]
    ra, rb = _estimate(a), _estimate(b)
    residuals = [
        (p - city["starting_poll"]) - r * d
        for city, r in ((a, ra), (b, rb))
        for d, p in zip(city["news"], city["ending_polls"])
    ]
    sd = max(
        0.5,
        math.sqrt(sum(x * x for x in residuals) / max(1, len(residuals) - 2)),
    )
    target = public["target"]
    news = target["news"][:k]
    changes = [p - target["starting_poll"] for p in target["ending_polls"][:k]]
    if not news:
        return STARTING_POLL + 0.5 * (ra + rb) * public["test_news"]

    def ll(r):
        return -0.5 * sum(
            ((c - r * d) ** 2) / (sd * sd) for d, c in zip(news, changes)
        )

    la, lb = ll(ra), ll(rb)
    top = max(la, lb)
    wa, wb = math.exp(la - top), math.exp(lb - top)
    pa = wa / (wa + wb)
    return STARTING_POLL + (pa * ra + (1 - pa) * rb) * public["test_news"]


def _load(directory: Path, keys):
    """Return {(model, k): [row, ...]} for the no-context condition."""
    out = {}
    for path in sorted(directory.glob("responses_*.jsonl")):
        for record in _read_jsonl(path):
            if record.get("predicted_poll") is None:
                continue
            key = keys.get(record["task_id"])
            if key is None or key["condition"] != "none":
                continue
            public = key["public"]
            target_only = _target_only_forecast(public, int(key["k"]))
            out.setdefault((record["model"], int(key["k"])), []).append(
                {
                    "episode": key["episode_id"],
                    "pred": float(record["predicted_poll"]),
                    "gold": float(key["gold"]["expected_poll"]),
                    "target_only": target_only,
                    "two_proto": _two_prototype(public, int(key["k"])),
                }
            )
    return out


def _paired(blind, hint, model, k, field):
    """Mean |pred - field| over episodes both arms completed."""
    left = {r["episode"]: r for r in blind.get((model, k), [])}
    right = {r["episode"]: r for r in hint.get((model, k), [])}
    shared = sorted(set(left) & set(right))
    if not shared:
        return (None, None, 0)
    if field is None:
        pick = lambda r: abs(r["pred"] - r["gold"])
    else:
        pick = lambda r: (
            None if r[field] is None else abs(r["pred"] - r[field])
        )
    lv = [pick(left[e]) for e in shared]
    rv = [pick(right[e]) for e in shared]
    if any(v is None for v in lv + rv):
        return (None, None, len(shared))
    return (statistics.mean(lv), statistics.mean(rv), len(shared))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind", type=Path, default=_BLIND)
    parser.add_argument("--hint", type=Path, default=_HINT)
    parser.add_argument("--answer-key", type=Path, default=_KEY)
    parser.add_argument(
        "--out",
        type=Path,
        default=_HINT / "hint_vs_blind.png",
    )
    args = parser.parse_args()

    keys = {r["task_id"]: r for r in _read_jsonl(args.answer_key)}
    blind = _load(args.blind, keys)
    hint = _load(args.hint, keys)
    if not hint:
        raise SystemExit("no hint-arm responses found yet")

    models = [
        m
        for m in _DISPLAY
        if any(key[0] == m for key in hint) and any(key[0] == m for key in blind)
    ]
    panels = (
        (None, "1. Does the hint improve accuracy?", "Forecast error (poll points)"),
        (
            "target_only",
            "2. Does it move them off the target-only estimate?",
            "Distance from City C's own data alone",
        ),
        (
            "two_proto",
            "3. Toward the agent that uses both sources?",
            "Distance from the two-prototype forecaster",
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    shared_n = 0
    for axis, (field, title, ylabel) in zip(axes, panels):
        ks = [k for k in PREFIX_LADDER if field is None or k > 0]
        for model in models:
            pairs = [_paired(blind, hint, model, k, field) for k in ks]
            shared_n = max(shared_n, max((p[2] for p in pairs), default=0))
            axis.plot(
                ks,
                [p[0] for p in pairs],
                marker="o",
                linewidth=2.2,
                color=_COLORS[model],
                label=f"{_DISPLAY[model]} — blind",
            )
            axis.plot(
                ks,
                [p[1] for p in pairs],
                marker="s",
                markersize=5,
                linestyle="--",
                linewidth=2.0,
                color=_COLORS[model],
                markerfacecolor="white",
                label=f"{_DISPLAY[model]} — hint",
            )
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.set_xlabel("City C cases the forecaster has seen (k)")
        axis.set_xticks(ks)
        axis.grid(alpha=0.25)
        axis.axhline(0, color="black", linewidth=0.8)
    axes[0].legend(fontsize=7.5, ncol=1, loc="upper right")

    fig.suptitle(
        "Relevance hint versus structure-blind, same numbers throughout   "
        f"({shared_n} paired episodes, no-context condition)   "
        "—  HINT ARM IS A NON-BLIND CONTROL",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    fig.savefig(args.out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {args.out}")

    print()
    print("Mean distance in poll points, paired episodes, no-context:")
    for field, label in (
        (None, "error vs truth"),
        ("target_only", "distance from the target-only estimate"),
        ("two_proto", "distance from two-prototype"),
    ):
        print(f"\n  {label}")
        print("    model              k     blind     hint")
        for model in models:
            for k in (2, 4, 8):
                b, h, n = _paired(blind, hint, model, k, field)
                if b is None:
                    continue
                print(
                    f"    {_DISPLAY[model]:18s} {k}    {b:6.3f}   {h:6.3f}   (n={n})"
                )


if __name__ == "__main__":
    main()
