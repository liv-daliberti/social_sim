#!/usr/bin/env python3
"""Score Experiment 1 initial forecasts on strictly settled markets.

The June 9 roster was frozen while every market was unresolved.  This script
joins that roster to a dated, lean snapshot of Polymarket's public Gamma API and
admits an outcome only when the market is closed and its YES/NO token prices are
exactly [1, 0] or [0, 1].  Invalid 50/50 settlements and merely proposed or UMA-
resolved markets are excluded.

For each agent, the point forecast is the median of its parseable repeated
initial forecasts, matching the maintained viewer.  The contemporaneous crowd
benchmark is the YES price stored in the frozen June 9 roster.  Accuracy is a
descriptive thresholded diagnostic; Brier score and log loss are the proper-
scoring-rule comparisons.

Examples (from the repository root):

    python exp1_prospective/agent/evaluate_resolved_outcomes.py --fetch
    python exp1_prospective/agent/evaluate_resolved_outcomes.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROSTER = (
    EXPERIMENT_ROOT / "data/selected_markets/diverse_2026-06-09.jsonl"
)
DEFAULT_SNAPSHOT = (
    EXPERIMENT_ROOT / "data/results/resolved_market_snapshot_2026-08-24.json"
)
DEFAULT_OUTPUT = (
    EXPERIMENT_ROOT / "data/results/resolved_outcome_evaluation_2026-08-24.json"
)
DEFAULT_LATEX_OUTPUT = (
    EXPERIMENT_ROOT.parent / "paper/tables/exp1_resolved_outcomes.tex"
)
DEFAULT_REPETITIONS = 20_000
DEFAULT_SEED = 20_260_824
DEFAULT_TIMEOUT = 60.0
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"

_DATE_SUFFIX_RE = re.compile(r"_\d{4}-\d{2}-\d{2}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(
    path: Path, *, skip_malformed: bool = False
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if skip_malformed:
                    continue
                raise ValueError(f"Malformed JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    return []


def _normalise_probability(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value.endswith("%"):
            value = float(value[:-1]) / 100.0
        else:
            value = float(value)
    else:
        value = float(value)
    if value > 1.0:
        value /= 100.0
    if not 0.0 <= value <= 1.0:
        return None
    return value


def _terminal_outcome(market: dict[str, Any]) -> int | None:
    """Return a strict binary outcome, or None when settlement is not usable."""
    if market.get("closed") is not True:
        return None
    outcomes = [str(value).lower() for value in _parse_list(market.get("outcomes"))]
    if outcomes != ["yes", "no"]:
        return None
    try:
        prices = [float(value) for value in _parse_list(market.get("outcomePrices"))]
    except (TypeError, ValueError):
        return None
    if prices == [1.0, 0.0]:
        return 1
    if prices == [0.0, 1.0]:
        return 0
    return None


def _lean_market(row: dict[str, Any]) -> dict[str, Any]:
    """Retain only fields needed to audit strict outcome eligibility."""
    return {
        "id": str(row.get("id") or ""),
        "question": row.get("question"),
        "conditionId": row.get("conditionId"),
        "endDate": row.get("endDate"),
        "closed": row.get("closed"),
        "closedTime": row.get("closedTime"),
        "outcomes": row.get("outcomes"),
        "outcomePrices": row.get("outcomePrices"),
        "umaResolutionStatus": row.get("umaResolutionStatus"),
        "umaEndDate": row.get("umaEndDate"),
        "updatedAt": row.get("updatedAt"),
    }


def _request_markets(
    market_ids: list[str], *, closed: bool, timeout: float
) -> list[dict[str, Any]]:
    params: list[tuple[str, str]] = [
        ("closed", str(closed).lower()),
        ("limit", str(len(market_ids))),
    ]
    params.extend(("id", market_id) for market_id in market_ids)
    request = Request(
        f"{GAMMA_MARKETS_URL}?{urlencode(params)}",
        headers={
            "User-Agent": "exp1-resolved-outcome-audit/1.0",
            "Connection": "close",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Gamma markets endpoint did not return a list")
    return [row for row in payload if isinstance(row, dict)]


def fetch_snapshot(roster_path: Path, output_path: Path, timeout: float) -> dict:
    roster = _read_jsonl(roster_path)
    market_ids = [str(row["market_id"]) for row in roster]
    if len(market_ids) != 100 or len(set(market_ids)) != 100:
        raise ValueError("The frozen roster must contain exactly 100 unique markets")

    fetched: dict[str, dict[str, Any]] = {}
    for closed in (True, False):
        for row in _request_markets(market_ids, closed=closed, timeout=timeout):
            market_id = str(row.get("id") or "")
            if market_id in market_ids:
                fetched[market_id] = _lean_market(row)
    missing = sorted(set(market_ids) - set(fetched))
    if missing:
        raise ValueError(f"Gamma API omitted frozen market IDs: {missing}")

    queried_at = datetime.now(timezone.utc).isoformat()
    snapshot = {
        "schema_version": 1,
        "queried_at": queried_at,
        "source": GAMMA_MARKETS_URL,
        "query": (
            "Two documented list-market queries over the exact roster IDs, one "
            "with closed=true and one with closed=false."
        ),
        "roster_path": str(roster_path.relative_to(EXPERIMENT_ROOT)),
        "roster_sha256": _sha256(roster_path),
        "n_roster": len(market_ids),
        "n_api_records": len(fetched),
        "markets": [fetched[market_id] for market_id in market_ids],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return snapshot


def _model_name(path: Path) -> str | None:
    manifest = path.with_name(path.stem + ".manifest.json")
    if manifest.exists():
        try:
            value = json.loads(manifest.read_text(encoding="utf-8")).get("model")
            if value:
                return str(value)
        except (json.JSONDecodeError, OSError):
            pass
    match = re.match(r"forecasts_(.+)_\d{4}-\d{2}-\d{2}$", path.stem)
    return match.group(1) if match else None


def load_forecasts(
    forecast_dir: Path,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, list[str]]]:
    """Mirror the maintained newest-file/last-row/median-run forecast loader."""
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    source_files: dict[str, list[str]] = {}

    for path in sorted(forecast_dir.glob("forecasts_*.jsonl"), reverse=True):
        model = _model_name(path)
        if not model:
            continue
        by_model.setdefault(model, {})
        latest_in_file: dict[str, dict[str, Any]] = {}
        for row in _read_jsonl(path, skip_malformed=True):
            task_id = str(row.get("task_id") or "")
            if task_id:
                latest_in_file[_DATE_SUFFIX_RE.sub("", task_id)] = row

        used = False
        for task_id, row in latest_in_file.items():
            if task_id in by_model[model]:
                continue
            runs = [
                value
                for value in (
                    _normalise_probability(raw)
                    for raw in (row.get("yes_prob_runs") or [])
                )
                if value is not None
            ]
            probability = (
                float(statistics.median(runs))
                if runs
                else _normalise_probability(row.get("yes_prob"))
            )
            market_id = str(row.get("market_id") or "")
            by_model[model][task_id] = {
                "task_id": task_id,
                "market_id": market_id,
                "question": row.get("question"),
                "yes_prob": probability,
                "forecast_at": row.get("forecast_at"),
                "n_runs": len(runs) if runs else (1 if probability is not None else 0),
                "source_file": str(path.relative_to(EXPERIMENT_ROOT)),
            }
            used = True
        if used:
            source_files.setdefault(model, []).append(
                str(path.relative_to(EXPERIMENT_ROOT))
            )

    return by_model, source_files


def _log_loss(probabilities: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return -(outcomes * np.log(clipped) + (1.0 - outcomes) * np.log(1.0 - clipped))


def _interval(estimate: float, draws: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "estimate": float(estimate),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def score_probabilities(
    probabilities: np.ndarray,
    market_probabilities: np.ndarray,
    outcomes: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if not (len(probabilities) == len(market_probabilities) == len(outcomes)):
        raise ValueError("Forecast, market, and outcome arrays must be aligned")
    if len(probabilities) == 0:
        raise ValueError("Cannot score an empty forecast array")

    model_correct = (probabilities > 0.5) == outcomes
    market_correct = (market_probabilities > 0.5) == outcomes
    model_brier_rows = (probabilities - outcomes) ** 2
    market_brier_rows = (market_probabilities - outcomes) ** 2
    model_log_rows = _log_loss(probabilities, outcomes)
    market_log_rows = _log_loss(market_probabilities, outcomes)

    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(outcomes), size=(repetitions, len(outcomes)))
    accuracy_diff_rows = model_correct.astype(float) - market_correct.astype(float)
    brier_diff_rows = model_brier_rows - market_brier_rows
    log_diff_rows = model_log_rows - market_log_rows

    accuracy_diff_draws = accuracy_diff_rows[sampled].mean(axis=1)
    brier_diff_draws = brier_diff_rows[sampled].mean(axis=1)
    log_diff_draws = log_diff_rows[sampled].mean(axis=1)

    return {
        "n": int(len(outcomes)),
        "accuracy": {
            "model_correct": int(model_correct.sum()),
            "model_rate": float(model_correct.mean()),
            "market_correct": int(market_correct.sum()),
            "market_rate": float(market_correct.mean()),
            "difference_model_minus_market": _interval(
                float(accuracy_diff_rows.mean()), accuracy_diff_draws
            ),
        },
        "disagreements": {
            "same_side": int(((probabilities > 0.5) == (market_probabilities > 0.5)).sum()),
            "model_only_correct": int((model_correct & ~market_correct).sum()),
            "market_only_correct": int((~model_correct & market_correct).sum()),
            "both_wrong": int((~model_correct & ~market_correct).sum()),
        },
        "brier": {
            "model": float(model_brier_rows.mean()),
            "market": float(market_brier_rows.mean()),
            "difference_model_minus_market": _interval(
                float(brier_diff_rows.mean()), brier_diff_draws
            ),
        },
        "log_loss": {
            "model": float(model_log_rows.mean()),
            "market": float(market_log_rows.mean()),
            "difference_model_minus_market": _interval(
                float(log_diff_rows.mean()), log_diff_draws
            ),
        },
    }


def build_report(
    roster_path: Path,
    snapshot_path: Path,
    forecast_dir: Path,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    roster_rows = _read_jsonl(roster_path)
    roster = {str(row["market_id"]): row for row in roster_rows}
    if len(roster_rows) != 100 or len(roster) != 100:
        raise ValueError("The frozen roster must contain exactly 100 unique markets")

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot.get("roster_sha256") != _sha256(roster_path):
        raise ValueError("Outcome snapshot does not match the frozen roster checksum")
    api_markets = {
        str(row.get("id") or ""): row for row in snapshot.get("markets", [])
    }
    if set(api_markets) != set(roster):
        raise ValueError("Outcome snapshot IDs do not exactly match the frozen roster")

    strict_outcomes = {
        market_id: outcome
        for market_id, row in api_markets.items()
        if (outcome := _terminal_outcome(row)) is not None
    }
    yes_count = sum(strict_outcomes.values())
    invalid_closed = [
        market_id
        for market_id, row in api_markets.items()
        if row.get("closed") is True and _terminal_outcome(row) is None
    ]
    forecasts, source_files = load_forecasts(forecast_dir)

    per_market: list[dict[str, Any]] = []
    for market_id in sorted(strict_outcomes, key=int):
        market = roster[market_id]
        per_market.append(
            {
                "market_id": market_id,
                "task_id": _DATE_SUFFIX_RE.sub("", str(market.get("task_id") or "")),
                "question": market.get("question"),
                "outcome": strict_outcomes[market_id],
                "market_yes_probability": float(market["yes_price"]),
                "end_time": market.get("end_time"),
                "closed_time": api_markets[market_id].get("closedTime"),
            }
        )
    market_by_id = {row["market_id"]: row for row in per_market}

    per_model: dict[str, Any] = {}
    for model, records in forecasts.items():
        records_by_market = {
            row["market_id"]: row
            for row in records.values()
            if row.get("market_id") and row.get("yes_prob") is not None
        }
        shared_ids = sorted(set(strict_outcomes) & set(records_by_market), key=int)
        if not shared_ids:
            continue
        forecast_times = sorted(
            str(records_by_market[market_id]["forecast_at"])
            for market_id in shared_ids
            if records_by_market[market_id].get("forecast_at")
        )
        probabilities = np.asarray(
            [records_by_market[market_id]["yes_prob"] for market_id in shared_ids],
            dtype=float,
        )
        market_probabilities = np.asarray(
            [market_by_id[market_id]["market_yes_probability"] for market_id in shared_ids],
            dtype=float,
        )
        outcomes = np.asarray(
            [strict_outcomes[market_id] for market_id in shared_ids], dtype=float
        )
        scores = score_probabilities(
            probabilities,
            market_probabilities,
            outcomes,
            repetitions=repetitions,
            seed=seed,
        )
        scores.update(
            {
                "n_strict_outcomes_available": len(strict_outcomes),
                "n_missing_forecasts": len(strict_outcomes) - len(shared_ids),
                "missing_market_ids": sorted(
                    set(strict_outcomes) - set(shared_ids), key=int
                ),
                "forecast_time_min": forecast_times[0] if forecast_times else None,
                "forecast_time_max": forecast_times[-1] if forecast_times else None,
                "source_files": source_files.get(model, []),
                "per_market": [
                    {
                        "market_id": market_id,
                        "outcome": strict_outcomes[market_id],
                        "model_yes_probability": float(
                            records_by_market[market_id]["yes_prob"]
                        ),
                        "market_yes_probability": market_by_id[market_id][
                            "market_yes_probability"
                        ],
                        "model_correct": bool(
                            (records_by_market[market_id]["yes_prob"] > 0.5)
                            == strict_outcomes[market_id]
                        ),
                        "market_correct": bool(
                            (market_by_id[market_id]["market_yes_probability"] > 0.5)
                            == strict_outcomes[market_id]
                        ),
                    }
                    for market_id in shared_ids
                ],
            }
        )
        per_model[model] = scores

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "design": {
            "roster": str(roster_path.relative_to(EXPERIMENT_ROOT)),
            "roster_sha256": _sha256(roster_path),
            "outcome_snapshot": str(snapshot_path.relative_to(EXPERIMENT_ROOT)),
            "outcome_snapshot_sha256": _sha256(snapshot_path),
            "outcome_snapshot_queried_at": snapshot.get("queried_at"),
            "outcome_rule": (
                "closed=true and exact terminal YES/NO prices [1,0] or [0,1]"
            ),
            "forecast_estimand": (
                "Median of parseable repeated initial YES forecasts per agent-market, "
                "matching the maintained Experiment 1 viewer."
            ),
            "market_benchmark": (
                "YES probability stored in the frozen June 9 selection record."
            ),
            "classification_rule": "Predict YES iff probability > 0.5.",
            "proper_scores": (
                "Mean Brier score and natural-log loss; lower is better."
            ),
        },
        "bootstrap": {
            "unit": "market",
            "repetitions": repetitions,
            "seed": seed,
            "rng": "numpy.random.Generator(PCG64)",
            "interval": "percentile 95% (2.5th, 97.5th percentiles)",
            "contrast": "paired model-minus-market difference",
        },
        "outcomes": {
            "n_roster": len(roster),
            "n_strict_terminal": len(strict_outcomes),
            "n_yes": int(yes_count),
            "n_no": int(len(strict_outcomes) - yes_count),
            "n_closed_nonterminal": len(invalid_closed),
            "closed_nonterminal_market_ids": sorted(invalid_closed, key=int),
            "n_not_strict_terminal": len(roster) - len(strict_outcomes),
        },
        "per_market": per_market,
        "per_model": per_model,
    }


# Hosted systems first; open-weight rows in descending parameter count.
_MODEL_DISPLAY = {
    "claude-opus-4-8": "Claude Opus~4.8",
    "gpt-5.4": "GPT-5.4",
    "DeepSeek-V4-Pro": "DeepSeek V4-Pro",
    "qwen2.5:72b": "Qwen2.5-72B",
    "llama3.3:70b": "Llama-3.3-70B",
    "llama3.1:70b": "Llama-3.1-70B",
    "qwen2.5:32b": "Qwen2.5-32B",
    "qwen2.5:14b": "Qwen2.5-14B",
    "llama3.1:8b": "Llama-3.1-8B",
    "qwen2.5:7b": "Qwen2.5-7B",
}
_MODEL_ORDER = tuple(_MODEL_DISPLAY)


def _tex_number(value: float) -> str:
    text = f"{value:.3f}"
    if text.startswith("-0."):
        return "-." + text[3:]
    if text.startswith("0."):
        return "." + text[2:]
    return text


def _tex_contrast(summary: dict[str, float]) -> str:
    return (
        "$"
        + _tex_number(summary["estimate"])
        + " ["
        + _tex_number(summary["ci_low"])
        + ","
        + _tex_number(summary["ci_high"])
        + "]$"
    )


def render_latex_table(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "% Generated by exp1_prospective/agent/evaluate_resolved_outcomes.py",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{$n$} & \textbf{Correct A/M} & "
        r"\textbf{Brier A/M} & \textbf{$\Delta$Brier [95\% CI]} & "
        r"\textbf{Log A/M} & \textbf{$\Delta$log [95\% CI]} \\",
        r"\midrule",
    ]
    for model in _MODEL_ORDER:
        row = report["per_model"][model]
        accuracy = row["accuracy"]
        brier = row["brier"]
        log_loss = row["log_loss"]
        lines.append(
            f"{_MODEL_DISPLAY[model]} & {row['n']} & "
            f"{accuracy['model_correct']}/{accuracy['market_correct']} & "
            + "$"
            + f"{_tex_number(brier['model'])}/{_tex_number(brier['market'])}"
            + "$ & "
            + f"{_tex_contrast(brier['difference_model_minus_market'])} & "
            + "$"
            + f"{_tex_number(log_loss['model'])}/{_tex_number(log_loss['market'])}"
            + "$ & "
            + f"{_tex_contrast(log_loss['difference_model_minus_market'])} \\\\"
        )
        if model == "DeepSeek-V4-Pro":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--latex-output", type=Path, default=DEFAULT_LATEX_OUTPUT)
    parser.add_argument(
        "--forecast-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "data/initial_forecasts",
    )
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.fetch:
        snapshot = fetch_snapshot(args.roster, args.snapshot, args.timeout)
        print(
            f"Fetched {snapshot['n_api_records']} markets -> {args.snapshot}"
        )
    if not args.snapshot.exists():
        parser.error(f"Outcome snapshot does not exist: {args.snapshot} (use --fetch)")

    report = build_report(
        args.roster,
        args.snapshot,
        args.forecast_dir,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    render_latex_table(report, args.latex_output)
    print(
        f"Scored {report['outcomes']['n_strict_terminal']} strict outcomes "
        f"across {len(report['per_model'])} models -> {args.output}"
    )


if __name__ == "__main__":
    main()
