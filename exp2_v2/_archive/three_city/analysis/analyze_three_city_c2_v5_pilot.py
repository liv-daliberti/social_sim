#!/usr/bin/env python3
"""Score and plot the independent-case three-city C2 v5 pilot.

The model runner cannot read the answer key. This post-hoc analysis is the only
place where saved model forecasts are joined to hidden conditions and gold
values.

Because every v2 case starts at the same known poll, the readout is a single
implied response, `(predicted_poll - 50) / net_news`. That removes the
baseline-versus-current-poll ambiguity diagnosed in the v1 pilot.

Context effects are measured as *paired* movements against the same episode's
no-context prompt, always as levels shown alongside the Bayes oracle's own
level. Nothing subtracts one agent from another: the oracle becomes unmovable at
high k, and a difference would hide that its benchmark had vanished. The three
questions are kept independent — how far the background moves the forecast,
whether irrelevant prose moves it too, and whether the movement runs in the
direction the wording implies.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from engine.three_city_c2_v5 import (
    HIGH_RESPONSE,
    HIGH_TYPE,
    LOW_RESPONSE,
    LOW_TYPE,
    PREFIX_LADDER,
)

_DATA = _ROOT / "data" / "three_city_c2_v5"
_PILOT = _DATA / "pilot_v5"
_ANSWER_KEY = _DATA / "answer_key_c2_v5.jsonl"
_CONDITIONS = ("none", "orthogonal", "cue_high", "cue_low")
_BOOTSTRAP = 2000
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
# Two information-matched anchors plus the structure-informed ceiling.
# Because every case carries the same news value, the "frequentist" estimate
# collapses to the plain average of City C's own completed cases, so it is
# labelled as what it is rather than by method name.
_BASELINES = {
    "frequentist": ("City C's own cases only (fitted slope)", "-.", "#222222"),
    "pooled_exemplar": ("City A+B cases only (fitted slope)", ":", "#9a7b37"),
    "target_only_bayes": ("Bayes ceiling", (0, (5, 2)), "#b03a6a"),
}
# The rational-agent reference for context panels. Under `none` it coincides
# with target_only_bayes, so it is only used for paired context contrasts.
_RATIONAL = "context_oracle"


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.mean(clean) if clean else None


def _bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int = 0,
) -> tuple[Optional[float], Optional[float]]:
    """Percentile CI over episodes. Small-n honest, deterministic."""
    clean = [float(value) for value in values if value is not None]
    if len(clean) < 3:
        return (None, None)
    rng = random.Random(seed)
    means = []
    size = len(clean)
    for _ in range(_BOOTSTRAP):
        means.append(
            statistics.mean(clean[rng.randrange(size)] for _ in range(size))
        )
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(len(means) - 1, int(0.975 * len(means)))]
    return (lo, hi)


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _fmt_ci(
    value: Optional[float],
    ci: tuple[Optional[float], Optional[float]],
    digits: int = 3,
) -> str:
    if value is None:
        return "—"
    if ci[0] is None:
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _load_latest_responses(paths: Iterable[Path]):
    latest: Dict[tuple[str, str], Dict[str, Any]] = {}
    failures: Dict[tuple[str, str], Dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for record in _read_jsonl(path):
            key = (record["model"], record["task_id"])
            if record.get("predicted_poll") is not None:
                latest[key] = record
            elif key not in latest:
                failures[key] = record
    return latest, failures


def _implied_response(key: Mapping[str, Any], predicted_poll: float) -> float:
    return (
        float(predicted_poll) - float(key["public"]["test_starting_poll"])
    ) / float(key["public"]["test_news"])


def _type_score(response: float, true_response: float) -> float:
    false_response = (
        LOW_RESPONSE if true_response == HIGH_RESPONSE else HIGH_RESPONSE
    )
    true_distance = abs(response - true_response)
    false_distance = abs(response - false_response)
    if abs(true_distance - false_distance) < 1e-12:
        return 0.5
    return float(true_distance < false_distance)


def _context_role(key: Mapping[str, Any]) -> str:
    condition = key["condition"]
    if condition in ("none", "orthogonal"):
        return condition
    target_high = key["gold"]["target_type"] == HIGH_TYPE
    aligned = (
        (condition == "cue_high" and target_high)
        or (condition == "cue_low" and not target_high)
    )
    return "aligned" if aligned else "misleading"


def _score_prediction(
    key: Mapping[str, Any],
    predicted_poll: Optional[float],
) -> Dict[str, Any]:
    if predicted_poll is None:
        return {
            "predicted_poll": None,
            "mae": None,
            "response": None,
            "response_error": None,
            "type_accuracy": None,
            "outside_demonstrated_band": None,
        }
    predicted_poll = float(predicted_poll)
    response = _implied_response(key, predicted_poll)
    true_response = float(key["gold"]["target_response"])
    low, high = sorted((LOW_RESPONSE, HIGH_RESPONSE))
    return {
        "predicted_poll": predicted_poll,
        "mae": abs(predicted_poll - float(key["gold"]["expected_poll"])),
        "response": response,
        "response_error": abs(response - true_response),
        "type_accuracy": _type_score(response, true_response),
        # The band is never described as a bound, so this is a descriptive
        # diagnostic of exemplar anchoring, not a correctness criterion.
        "outside_demonstrated_band": float(
            response < low - 1e-9 or response > high + 1e-9
        ),
    }


def _row(key: Mapping[str, Any], source: str, predicted, rationale=""):
    return {
        "source": source,
        "episode_id": key["episode_id"],
        "task_id": key["task_id"],
        "k": int(key["k"]),
        "condition": key["condition"],
        "role": _context_role(key),
        "target_type": key["gold"]["target_type"],
        "rationale": rationale,
        **_score_prediction(key, predicted),
    }


def _model_rows(responses, keys) -> list[Dict[str, Any]]:
    return [
        _row(
            keys[task_id],
            model,
            response.get("predicted_poll"),
            response.get("rationale", ""),
        )
        for (model, task_id), response in responses.items()
        if task_id in keys
    ]


def _baseline_rows(keys, selected_task_ids) -> list[Dict[str, Any]]:
    methods = tuple(_BASELINES) + (_RATIONAL,)
    return [
        _row(
            keys[task_id],
            method,
            keys[task_id]["baselines"][method].get("predicted_poll"),
        )
        for task_id in sorted(selected_task_ids)
        for method in methods
    ]


def _group_mean(rows, source, k, field, *, role=None, target_type=None):
    return _mean(
        row[field]
        for row in rows
        if row["source"] == source
        and row["k"] == k
        and (role is None or row["role"] == role)
        and (target_type is None or row["target_type"] == target_type)
    )


def _type_separation(rows, source, k, *, role="none"):
    """Implied response on open targets minus that on buffered targets.

    The identification curve: how far apart the two hidden kinds of City C are
    pushed as City C's own cases accumulate.
    """
    high = _group_mean(
        rows, source, k, "response", role=role, target_type=HIGH_TYPE
    )
    low = _group_mean(
        rows, source, k, "response", role=role, target_type=LOW_TYPE
    )
    if high is None or low is None:
        return None
    return high - low


def _by_episode(rows, source, k):
    """Group one forecaster's rows for one prefix by episode and role."""
    grouped: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for row in rows:
        if row["source"] == source and row["k"] == k:
            grouped[row["episode_id"]][row["role"]] = row
    return grouped


def _paired_values(rows, source, k, metric) -> list[float]:
    """Per-episode paired contrasts. Requires all four conditions present."""
    values = []
    for conditions in _by_episode(rows, source, k).values():
        if not {"none", "orthogonal", "aligned", "misleading"} <= set(
            conditions
        ):
            continue
        none = conditions["none"]
        if none["response"] is None:
            continue
        if metric == "orthogonal_movement":
            other = conditions["orthogonal"]
            if other["response"] is None:
                continue
            values.append(abs(other["response"] - none["response"]))
        elif metric == "directional_consistency":
            # Does the forecast move the way the wording implies — a
            # high-transmission background raising the implied response above
            # what a buffered one gives? Independent of how far it moves.
            by_condition = {
                row["condition"]: row for row in conditions.values()
            }
            if not {"cue_high", "cue_low"} <= set(by_condition):
                continue
            high = by_condition["cue_high"]["response"]
            low = by_condition["cue_low"]["response"]
            if high is None or low is None:
                continue
            values.append(1.0 if high > low else (0.5 if high == low else 0.0))
        elif metric == "relevant_movement":
            # Step-matched against orthogonal_movement: both measure how far
            # ADDING one background moves the forecast away from adding none.
            # The old cue_high-minus-cue_low swing spanned two conditions and so
            # carried roughly twice the leverage, which inflated the ratio.
            high = conditions["aligned"]
            low = conditions["misleading"]
            if high["response"] is None or low["response"] is None:
                continue
            values.append(
                (
                    abs(high["response"] - none["response"])
                    + abs(low["response"] - none["response"])
                )
                / 2.0
            )
        else:
            raise ValueError(metric)
    return values


def _paired_mean(rows, source, k, metric):
    return _mean(_paired_values(rows, source, k, metric))


def _paired_ci(rows, source, k, metric):
    return _bootstrap_ci(_paired_values(rows, source, k, metric))


# Deliberately no "excess susceptibility" helper any more. Subtracting the
# oracle's misleading-cue penalty from a model's looked like a calibration
# measure but degenerates: the oracle's no-context posterior reaches 0.997 by
# k=8, so an 80%-reliable cue cannot move it and its penalty collapses to 0.003.
# The difference then equals the model's own raw penalty while still being
# labelled "excess over a rational agent". Panels now plot both agents' levels
# side by side instead of subtracting, so a vanishing benchmark stays visible.


_SELECTIVITY_FLOOR = 0.05


def _selectivity(rows, source, k):
    """Relevant movement divided by orthogonal movement, step-matched.

    Both terms measure how far adding one background moves the forecast away
    from adding none, so 1.0 genuinely means "irrelevant prose moves the
    forecast as much as mechanism-relevant prose does". Undefined once the
    relevant movement itself is below resolution: a ratio of two near-zero
    movements carries no information about selectivity.
    """
    relevant = _paired_mean(rows, source, k, "relevant_movement")
    orthogonal = _paired_mean(rows, source, k, "orthogonal_movement")
    if relevant is None or orthogonal is None:
        return None
    if relevant < _SELECTIVITY_FLOOR:
        return None
    if orthogonal < 1e-6:
        return float("inf")
    return relevant / orthogonal


def _shrinkage_gap(rows, keys, source, k, method, *, role="none"):
    return _mean(
        abs(
            row["response"]
            - keys[row["task_id"]]["baselines"][method]["response_hat"]
        )
        for row in rows
        if row["source"] == source
        and row["k"] == k
        and row["role"] == role
        and row["response"] is not None
        and keys[row["task_id"]]["baselines"][method]["response_hat"]
        is not None
    )


def _write_summary(
    path: Path,
    *,
    keys,
    model_rows,
    baseline_rows,
    models,
    prefixes,
    selected_count,
    completed_by_model,
    failure_counts,
) -> None:
    lines = [
        "# Three-city C2 v5 pilot diagnostics",
        "",
        "> Independent-case design, City C ladder to eight cases. This is a",
        "> design check, not a model comparison or a paper result.",
        "",
        "Every case starts at 50.0 and receives the same net news, so the",
        "implied response is `(predicted_poll - 50) / net_news`. The two",
        f"demonstrated responses are {LOW_RESPONSE:.2f} and {HIGH_RESPONSE:.2f}.",
        "Bracketed intervals are 2.5–97.5 percentile bootstrap intervals over",
        "episodes.",
        "",
        "## Completion",
        "",
        f"Scored on the {selected_count} prompts every model answered, across "
        f"{len({row['episode_id'] for row in model_rows})} episodes.",
        "",
        "| Model | Scored | Unparsed |",
        "|---|---:|---:|",
    ]
    for model in models:
        lines.append(
            f"| {_DISPLAY.get(model, model)} | "
            f"{completed_by_model.get(model, 0)} | "
            f"{failure_counts.get(model, 0)} |"
        )

    lines += [
        "",
        "## No-context recovery",
        "",
        "| Model | k | Forecast MAE | Implied response | Response error | "
        "Pattern accuracy | Outside band |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        for k in prefixes:
            lines.append(
                f"| {_DISPLAY.get(model, model)} | {k} | "
                f"{_fmt(_group_mean(model_rows, model, k, 'mae', role='none'))} | "
                f"{_fmt(_group_mean(model_rows, model, k, 'response', role='none'))} | "
                f"{_fmt(_group_mean(model_rows, model, k, 'response_error', role='none'))} | "
                f"{_fmt(_group_mean(model_rows, model, k, 'type_accuracy', role='none'))} | "
                f"{_fmt(_group_mean(model_rows, model, k, 'outside_demonstrated_band', role='none'))} |"
            )

    lines += [
        "",
        "## Identification: implied response by hidden City C kind (no context)",
        "",
        "Separation is the open-target mean minus the buffered-target mean. The",
        f"full demonstrated gap is {HIGH_RESPONSE - LOW_RESPONSE:.2f}.",
        "",
        "| Forecaster | k | Open target | Buffered target | Separation |",
        "|---|---:|---:|---:|---:|",
    ]
    for source, label in [
        (model, _DISPLAY.get(model, model)) for model in models
    ] + [
        (method, _BASELINES[method][0])
        for method in ("frequentist", "target_only_bayes")
    ]:
        rows = model_rows if source in models else baseline_rows
        for k in prefixes:
            lines.append(
                f"| {label} | {k} | "
                f"{_fmt(_group_mean(rows, source, k, 'response', role='none', target_type=HIGH_TYPE))} | "
                f"{_fmt(_group_mean(rows, source, k, 'response', role='none', target_type=LOW_TYPE))} | "
                f"{_fmt(_type_separation(rows, source, k))} |"
            )

    lines += [
        "",
        "## Where the forecasts sit: target sample mean or reference cities",
        "",
        "Mean absolute distance between a model's implied response and each",
        "reference forecaster's. Sitting on the target-only slope means City C's own",
        "cases are being fitted without help from the demonstrated cities.",
        "",
        "| Model | k | Distance to City C-only slope | Distance to A+B slope |",
        "|---|---:|---:|---:|",
    ]
    for model in models:
        for k in prefixes:
            if k == 0:
                continue
            lines.append(
                f"| {_DISPLAY.get(model, model)} | {k} | "
                f"{_fmt(_shrinkage_gap(model_rows, keys, model, k, 'frequentist'))} | "
                f"{_fmt(_shrinkage_gap(model_rows, keys, model, k, 'pooled_exemplar'))} |"
            )

    lines += [
        "",
        "## Direction: is the background read the right way round?",
        "",
        "Fraction of episodes where the implied response is higher under the",
        "high-transmission background than under the buffered one. This is about",
        "direction only, independent of how far the forecast moves. 0.5 is a coin",
        f"flip. Where relevant movement falls below {_SELECTIVITY_FLOOR:.2f} the",
        "direction is unresolved rather than wrong.",
        "",
        "| Forecaster | " + " | ".join(f"k={k}" for k in prefixes) + " |",
        "|---" * (len(prefixes) + 1) + "|",
    ]
    for source, label in [
        (model, _DISPLAY.get(model, model)) for model in models
    ] + [(_RATIONAL, "Bayes oracle (rational benchmark)")]:
        rows = model_rows if source in models else baseline_rows
        cells = []
        for k in prefixes:
            value = _paired_mean(rows, source, k, "directional_consistency")
            movement = _paired_mean(rows, source, k, "relevant_movement") or 0.0
            marker = "" if movement >= _SELECTIVITY_FLOOR else " *"
            cells.append(_fmt(value, 3) + marker)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "`*` marks an unresolved rung: the cue barely moves the forecast there.",
        "The figure shows the same rungs with hollow rather than filled markers.",
        "",
        "### Withdrawn framings",
        "",
        "Two earlier readouts were removed as unsound rather than merely unclear.",
        "",
        "*Aligned benefit and misleading cost* — the paired change in response",
        "error when the background happens to be right, and when it happens to be",
        "wrong. For any forecaster that shifts by some amount `d` on the",
        "high-transmission wording and `-d` on the buffered wording, the aligned",
        "change is `-d` and the misleading change is `+d` **whatever `d` is**. The",
        "reassuring negative-then-positive shape was therefore arithmetically",
        "forced by using the cue at all, not evidence of using it well, and the",
        "pair carried exactly one number: the movement magnitude now in panel D.",
        "Confirmed empirically — at `k=0` half the cost-minus-benefit spread",
        "reproduced panel D's movement to within 0.01 for every model.",
        "",
        "*Excess susceptibility* — the model's misleading cost minus the oracle's.",
        "The oracle's no-context posterior reaches 0.997 by `k=8`, so an",
        "80%-reliable cue cannot move it and its penalty collapses from 0.225 to",
        "0.003. The difference then just restates the model's own penalty while",
        "still being labelled as excess over a rational agent. Every panel now",
        "plots both agents' levels instead of subtracting, so a vanishing",
        "benchmark stays visible.",
        "",
        "A net-value readout — mean error across both cue conditions against no",
        "context — was also considered and rejected: on a design that renders both",
        "wordings for every episode, a symmetric shift cancels to first order, and",
        "it measured 0.00 to 0.09 for every forecaster including the oracle.",
    ]

    lines += [
        "",
        "## Selectivity: does only mechanism-relevant wording move the forecast",
        "",
        "Both movements are step-matched: each is how far ADDING one background",
        "moves the implied response away from adding none. Relevant movement",
        "averages the two mechanism-relevant backgrounds; orthogonal movement uses",
        "the equally detailed but irrelevant one. Selectivity is their ratio, so",
        "1.0 means irrelevant prose moves the forecast as much as relevant prose",
        "does. It is",
        f"left blank once the relevant swing falls below {_SELECTIVITY_FLOOR:.2f},",
        "because the ratio of two near-zero movements is uninformative.",
        "",
        "| Forecaster | k | Relevant movement | Orthogonal movement | Selectivity |",
        "|---|---:|---:|---:|---:|",
    ]
    for source, label in [
        (model, _DISPLAY.get(model, model)) for model in models
    ] + [(_RATIONAL, "Bayes oracle (rational benchmark)")]:
        rows = model_rows if source in models else baseline_rows
        for k in prefixes:
            selectivity = _selectivity(rows, source, k)
            lines.append(
                f"| {label} | {k} | "
                f"{_fmt_ci(_paired_mean(rows, source, k, 'relevant_movement'), _paired_ci(rows, source, k, 'relevant_movement'))} | "
                f"{_fmt_ci(_paired_mean(rows, source, k, 'orthogonal_movement'), _paired_ci(rows, source, k, 'orthogonal_movement'))} | "
                + ("∞" if selectivity == float("inf") else _fmt(selectivity, 1))
                + " |"
            )

    lines += [
        "",
        "## Reference forecasters (no context)",
        "",
        "Every case carries the same news value, so the target-only estimate is",
        "simply the **average of City C's own completed ending polls** — no",
        "reference cities, no background, undefined at `k=0`. The pooled exemplar",
        "is the mirror image: the two demonstrated cities only, never City C, so",
        "it is flat in `k`. The Bayes ceiling additionally knows the private",
        "generative process.",
        "",
        "The target-only slope and the ceiling give *identical* pattern accuracy by",
        "construction: the posterior mean is monotone in that same sample mean,",
        "so \"which demonstrated pattern is closer\" is the same cut on it. They",
        "differ in forecast MAE, not in classification.",
        "",
        "These are computed on the same episodes the models answered, so they",
        "are small subsamples and will not match the 120-episode design values",
        "in `../validation_c2_v5.json`.",
        "",
        "| Forecaster | k | Forecast MAE | Pattern accuracy |",
        "|---|---:|---:|---:|",
    ]
    for method, (label, _, _color) in _BASELINES.items():
        for k in prefixes:
            lines.append(
                f"| {label} | {k} | "
                f"{_fmt(_group_mean(baseline_rows, method, k, 'mae', role='none'))} | "
                f"{_fmt(_group_mean(baseline_rows, method, k, 'type_accuracy', role='none'))} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _plot(
    path: Path,
    *,
    model_rows,
    baseline_rows,
    models,
    prefixes,
    completed_by_model,
    selected_count,
) -> None:
    # Panel D asks two questions with opposite desired signs, so it gets two
    # stacked sub-panels on a common scale rather than eight lines in one axes.
    fig = plt.figure(figsize=(17, 10))
    grid = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.24)
    axes = {
        "A": fig.add_subplot(grid[0, 0]),
        "B": fig.add_subplot(grid[0, 1]),
        "C": fig.add_subplot(grid[0, 2]),
        "D": fig.add_subplot(grid[1, 0]),
        "E": fig.add_subplot(grid[1, 1]),
        "F": fig.add_subplot(grid[1, 2]),
    }
    ks = list(prefixes)

    def model_label(model: str) -> str:
        return (
            f"{_DISPLAY.get(model, model)} "
            f"({completed_by_model.get(model, 0)}/{selected_count})"
        )

    def band(axis, values, cis, color):
        lows = [ci[0] for ci in cis]
        highs = [ci[1] for ci in cis]
        if any(low is None for low in lows):
            return
        axis.fill_between(ks, lows, highs, color=color, alpha=0.12, linewidth=0)

    # --- Panel A shading: what the demonstrated cities and a two-point prior
    # are worth over averaging City C alone. Landing inside it is good.
    baseline_mae = {
        method: [
            _group_mean(baseline_rows, method, k, "mae", role="none")
            for k in ks
        ]
        for method in _BASELINES
    }
    shade_ks = [
        k
        for k, frequentist in zip(ks, baseline_mae["frequentist"])
        if frequentist is not None
    ]
    if len(shade_ks) > 1:
        offset = len(ks) - len(shade_ks)
        axes["A"].fill_between(
            shade_ks,
            baseline_mae["target_only_bayes"][offset:],
            baseline_mae["frequentist"][offset:],
            color="#2b8c6b",
            alpha=0.18,
            linewidth=0,
            label="Headroom over using City C alone",
            zorder=0,
        )

    # --- Row 1: recovery, unpaired, no-context only -----------------------
    for model in models:
        axes["A"].plot(
            ks,
            [_group_mean(model_rows, model, k, "mae", role="none") for k in ks],
            marker="o",
            linewidth=2,
            color=_COLORS.get(model),
            label=model_label(model),
        )
        axes["B"].plot(
            ks,
            [
                _group_mean(model_rows, model, k, "type_accuracy", role="none")
                for k in ks
            ],
            marker="o",
            linewidth=2,
            color=_COLORS.get(model),
            label=model_label(model),
        )
        axes["C"].plot(
            ks,
            [_type_separation(model_rows, model, k) for k in ks],
            marker="o",
            linewidth=2,
            color=_COLORS.get(model),
            label=model_label(model),
        )
    for method, (label, linestyle, color) in _BASELINES.items():
        axes["A"].plot(
            ks,
            baseline_mae[method],
            linestyle=linestyle,
            linewidth=1.5,
            color=color,
            label=label,
        )
        # City C's average and the Bayes ceiling classify identically here: the
        # posterior mean is a monotone function of that same sample mean, and
        # the "which demonstrated pattern is closer" threshold maps to the same
        # cut on it. Plotting both would hide one under the other and read as an
        # empirical coincidence, so only the ceiling is drawn.
        if method == "frequentist":
            continue
        axes["B"].plot(
            ks,
            [
                _group_mean(
                    baseline_rows, method, k, "type_accuracy", role="none"
                )
                for k in ks
            ],
            linestyle=linestyle,
            linewidth=1.5,
            color=color,
            label=label,
        )
        if method != "pooled_exemplar":
            axes["C"].plot(
                ks,
                [_type_separation(baseline_rows, method, k) for k in ks],
                linestyle=linestyle,
                linewidth=1.5,
                color=color,
                label=label,
            )

    # --- Bottom row: three independent questions about background use -----
    # Every panel plots LEVELS for the models and for the rational benchmark.
    # None subtracts one agent from the other, so a benchmark that collapses to
    # zero stays visible as a zero line rather than hiding inside a difference.
    for name, metric in (
        ("D", "relevant_movement"),
        ("E", "orthogonal_movement"),
    ):
        axis = axes[name]
        for model in models:
            values = [_paired_mean(model_rows, model, k, metric) for k in ks]
            cis = [_paired_ci(model_rows, model, k, metric) for k in ks]
            axis.plot(
                ks,
                values,
                marker="o",
                markersize=4,
                linewidth=2,
                color=_COLORS.get(model),
                label=model_label(model),
            )
            band(axis, values, cis, _COLORS.get(model))
        axis.plot(
            ks,
            [_paired_mean(baseline_rows, _RATIONAL, k, metric) for k in ks],
            linestyle=(0, (4, 2)),
            linewidth=1.8,
            color="#555555",
            label="Bayes oracle (rational benchmark)",
        )
        axis.axhline(0, color="black", linewidth=0.8)

    # Panel F: direction, not magnitude. Hollow markers mark rungs where the
    # cue moves the forecast less than the resolution floor, so the reader does
    # not over-read a decay that is really a coin flip on rounding noise.
    for model in models:
        values = [
            _paired_mean(model_rows, model, k, "directional_consistency")
            for k in ks
        ]
        resolved = [
            (_paired_mean(model_rows, model, k, "relevant_movement") or 0.0)
            >= _SELECTIVITY_FLOOR
            for k in ks
        ]
        axes["F"].plot(
            ks,
            values,
            linewidth=2,
            color=_COLORS.get(model),
            label=model_label(model),
            zorder=2,
        )
        for k, value, is_resolved in zip(ks, values, resolved):
            axes["F"].plot(
                [k],
                [value],
                marker="o",
                markersize=5,
                color=_COLORS.get(model),
                markerfacecolor=(
                    _COLORS.get(model) if is_resolved else "white"
                ),
                zorder=3,
            )
    axes["F"].plot(
        ks,
        [
            _paired_mean(baseline_rows, _RATIONAL, k, "directional_consistency")
            for k in ks
        ],
        linestyle=(0, (4, 2)),
        linewidth=1.8,
        color="#555555",
        label="Bayes oracle (rational benchmark)",
    )
    axes["F"].axhline(0.5, color="black", linewidth=0.8)
    axes["F"].set_ylim(0.35, 1.05)

    for name, axis in axes.items():
        axis.set_xticks(ks)
        axis.grid(alpha=0.25)
        axis.set_xlabel("City C completed cases (k)")

    axes["A"].set_title("A. No-context forecast error")
    axes["A"].set_ylabel("Mean absolute error (poll points)")
    axes["B"].set_title("B. No-context pattern recovery")
    axes["B"].set_ylabel("Behavioral pattern accuracy")
    axes["B"].set_ylim(-0.05, 1.05)
    axes["C"].set_title("C. Identification of City C's hidden kind")
    axes["C"].set_ylabel("Implied response: open − buffered targets")
    axes["C"].axhline(0, color="black", linewidth=0.8)
    axes["C"].axhline(
        HIGH_RESPONSE - LOW_RESPONSE,
        color="black",
        linewidth=0.8,
        linestyle="--",
        alpha=0.5,
    )
    axes["C"].annotate(
        "full demonstrated gap",
        xy=(ks[0], HIGH_RESPONSE - LOW_RESPONSE),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=7,
        color="#444444",
    )

    axes["D"].set_title("D. How much does the background move the forecast?")
    axes["D"].set_ylabel("Paired |Δ implied response| vs no context")
    axes["E"].set_title("E. Does irrelevant prose move it too?")
    axes["E"].set_ylabel("Paired |Δ implied response| vs no context")
    axes["F"].set_title("F. Does it read the background the right way round?")
    axes["F"].set_ylabel(
        "P(implied response is higher under the\nhigh-transmission background)"
    )

    # Merge the D-panel oracle reference into the shared legend, deduped.
    handles, labels = axes["A"].get_legend_handles_labels()
    for handle, label in zip(*axes["D"].get_legend_handles_labels()):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        "Three-city C2 v5 pilot — independent cases, City C ladder to k=8; "
        "shaded bands are bootstrap CIs over episodes",
        fontsize=13,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.115)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-key", type=Path, default=_ANSWER_KEY)
    parser.add_argument("--responses", type=Path, nargs="*", default=None)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="score the common subset even when models disagree on "
        "coverage; for interim plots only",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=_PILOT / "pilot_v5_diagnostics.png",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=_PILOT / "pilot_v5_summary.md",
    )
    parser.add_argument(
        "--rows-out",
        type=Path,
        default=_PILOT / "pilot_v5_rows.jsonl",
    )
    args = parser.parse_args()

    response_paths = args.responses or sorted(
        _PILOT.glob("responses_*.jsonl")
    )
    keys = {
        record["task_id"]: record for record in _read_jsonl(args.answer_key)
    }
    responses, failures = _load_latest_responses(response_paths)
    if not responses:
        raise SystemExit("no parseable model responses found")

    # Compare models only on prompts every model answered. Otherwise an
    # interrupted run silently shifts one model's sample and the paired context
    # contrasts stop being like-for-like.
    by_model: Dict[str, set] = defaultdict(set)
    for model, task_id in responses:
        by_model[model].add(task_id)
    common = set.intersection(*by_model.values()) if by_model else set()
    dropped = {
        model: len(task_ids - common) for model, task_ids in by_model.items()
    }
    if any(dropped.values()):
        detail = ", ".join(
            f"{model}={count}"
            for model, count in sorted(dropped.items())
            if count
        )
        message = (
            f"models disagree on coverage; {len(common)} prompts are common to "
            f"all. Would drop per model: {detail}."
        )
        if not args.allow_partial:
            # Fail closed. Silently restricting hides a model that died partway
            # through, and hides entirely a model that produced nothing at all
            # (it is simply absent from `by_model`, so from the comparison too).
            raise SystemExit(
                message
                + "\n\nRefusing to score a hole. Fill it with:\n"
                "  python eval/check_arm_completeness.py --tasks <task file> "
                "--responses <pilot dir> --resubmit --slurm <slurm script>\n"
                "or pass --allow-partial to score the common subset anyway "
                "(interim plots only)."
            )
        print("WARNING (--allow-partial): " + message)
    responses = {
        key: value for key, value in responses.items() if key[1] in common
    }
    selected_task_ids = set(common)
    model_rows = _model_rows(responses, keys)
    baseline_rows = _baseline_rows(keys, selected_task_ids)
    models = [
        model
        for model in _DISPLAY
        if any(row["source"] == model for row in model_rows)
    ]
    prefixes = [
        k for k in PREFIX_LADDER if any(row["k"] == k for row in model_rows)
    ]
    completed_by_model = {
        model: sum(1 for source, _ in responses if source == model)
        for model in models
    }
    failure_counts = {
        model: sum(1 for source, _ in failures if source == model)
        for model in models
    }
    selected_count = max(completed_by_model.values())
    _write_summary(
        args.summary,
        keys=keys,
        model_rows=model_rows,
        baseline_rows=baseline_rows,
        models=models,
        prefixes=prefixes,
        selected_count=selected_count,
        completed_by_model=completed_by_model,
        failure_counts=failure_counts,
    )
    _plot(
        args.figure,
        model_rows=model_rows,
        baseline_rows=baseline_rows,
        models=models,
        prefixes=prefixes,
        completed_by_model=completed_by_model,
        selected_count=selected_count,
    )
    if args.rows_out:
        args.rows_out.parent.mkdir(parents=True, exist_ok=True)
        with args.rows_out.open("w") as handle:
            for row in sorted(
                model_rows + baseline_rows,
                key=lambda row: (row["source"], row["task_id"]),
            ):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"Wrote {args.rows_out}")
    print(f"Wrote {args.summary}")
    print(f"Wrote {args.figure}")


if __name__ == "__main__":
    main()
