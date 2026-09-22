#!/usr/bin/env python3
"""Descriptive analysis of the development-only paired context-reversal pilot.

The compiled design, never the observed responses, defines all denominators.
Whole families are bootstrapped, retaining their contexts, repeats, and arms.
Missing probabilities remain missing; primary binary outcomes count them as
failures without substituting a forecast value.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

CONTEXTS = ("positive", "negative", "broken", "masked")
CONDITIONS = ("baseline", "new_news", "no_news", "repeated_news")
STATUSES = {"ok", "parse_error", "blocked_baseline", "generation_error"}
MATERIAL_STATUS = "authored_development_unvalidated"
DEFAULT_SEED = 20260921
DEFAULT_BOOTSTRAP_DRAWS = 2000
EQUIVALENCE_MARGIN_PP = 2.0


@dataclass(frozen=True)
class Sample:
    family: str
    value: float | None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            records.append(record)
    return records


def validate_design(design: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not design:
        raise ValueError("The planned design is empty")
    units: dict[str, dict[str, Any]] = {}
    cells: set[tuple[str, int, str]] = set()
    family_repeats: dict[tuple[str, int], set[str]] = defaultdict(set)
    domains: dict[str, str] = {}
    for row in design:
        for field in ("trial_id", "family_id", "domain"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"Design requires nonempty {field}")
        trial, family = row["trial_id"], row["family_id"]
        context, repeat = row.get("context_id"), row.get("repeat")
        if context not in CONTEXTS:
            raise ValueError(f"Unknown design context: {context!r}")
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
            raise ValueError("Design repeat must be a nonnegative integer")
        if row.get("material_status") != MATERIAL_STATUS:
            raise ValueError("This analyzer requires authored_development_unvalidated materials")
        if trial in units or (family, repeat, context) in cells:
            raise ValueError("Duplicate planned trial or family/repeat/context")
        if family in domains and domains[family] != row["domain"]:
            raise ValueError(f"Inconsistent domain for family {family}")
        domains[family] = row["domain"]
        units[trial] = row
        cells.add((family, repeat, context))
        family_repeats[family, repeat].add(context)
    if any(contexts != set(CONTEXTS) for contexts in family_repeats.values()):
        raise ValueError("Each planned family/repeat must contain all four contexts")
    return units


def index_responses(
    responses: list[dict[str, Any]], units: dict[str, dict[str, Any]], model_key: str,
) -> tuple[dict[tuple[str, str], dict[str, Any]], int]:
    indexed = {}
    seen = set()
    other_models = 0
    for row in responses:
        if not isinstance(row.get("model_key"), str) or not row["model_key"]:
            raise ValueError("Response is missing model_key")
        if not isinstance(row.get("trial_id"), str) or not isinstance(row.get("condition"), str):
            raise ValueError("Response requires string trial_id and condition")
        key = (row["model_key"], row["trial_id"], row["condition"])
        if key in seen:
            raise ValueError(f"Duplicate response key: {key}")
        seen.add(key)
        if row["model_key"] != model_key:
            other_models += 1
            continue
        trial, condition = row.get("trial_id"), row.get("condition")
        if trial not in units:
            raise ValueError(f"Response trial is absent from the planned design: {trial}")
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown response condition: {condition}")
        expected_stage = "baseline" if condition == "baseline" else "update"
        if row.get("stage") != expected_stage:
            raise ValueError(f"Incorrect stage for {trial}/{condition}")
        if row.get("status") not in STATUSES:
            raise ValueError(f"Unknown response status for {trial}/{condition}")
        if isinstance(row.get("repeat"), bool) or not isinstance(row.get("repeat"), int):
            raise ValueError(f"Response repeat must be an integer: {trial}")
        if "material_status" in row and row["material_status"] != MATERIAL_STATUS:
            raise ValueError(f"Response material status mismatch: {trial}")
        for field in ("family_id", "domain", "context_id", "repeat"):
            if row.get(field) != units[trial][field]:
                raise ValueError(f"Response/design mismatch for {trial}: {field}")
        indexed[trial, condition] = row
    return indexed, other_models


def probability(row: dict[str, Any] | None) -> tuple[float | None, str]:
    if row is None:
        return None, "missing_record"
    if row["status"] != "ok":
        return None, row["status"]
    value = row.get("probability")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, "invalid_probability"
    if not math.isfinite(value) or not 0 <= value <= 1:
        return None, "invalid_probability"
    return float(value), "ok"


class FamilyBootstrap:
    def __init__(self, families: list[str], draws: int, seed: int):
        if draws < 1:
            raise ValueError("bootstrap draws must be positive")
        self.families = sorted(set(families))
        if not self.families:
            raise ValueError("No families supplied")
        self.lookup = {family: i for i, family in enumerate(self.families)}
        self.draws = draws
        rng = np.random.default_rng(seed)
        self.indices = rng.integers(0, len(self.families), (draws, len(self.families)))

    def summarize(self, samples: list[Sample], *, binary: bool = False) -> dict[str, Any]:
        sums = np.zeros(len(self.families))
        counts = np.zeros(len(self.families))
        observed = sum(sample.value is not None for sample in samples)
        observed_families = {sample.family for sample in samples if sample.value is not None}
        for sample in samples:
            index = self.lookup[sample.family]
            if sample.value is not None:
                sums[index] += sample.value
            if binary or sample.value is not None:
                counts[index] += 1
        denominator = int(counts.sum())
        estimate = float(sums.sum() / denominator) if denominator else None
        result = {
            "estimate": estimate,
            "n_planned": len(samples),
            "n_observed": observed,
            "n_missing": len(samples) - observed,
            "n_families_observed": len(observed_families),
            "denominator": denominator,
            "missing_policy": "counts_as_failure" if binary else "observed_only_reported_explicitly",
            "ci95": None,
            "ci90": None,
            "bootstrap_draws_requested": self.draws,
            "bootstrap_draws_defined": 0,
            "interval_status": "insufficient_observed_families",
        }
        if binary:
            result["n_success"] = int(sums.sum())
            result["n_failure_including_missing"] = len(samples) - int(sums.sum())
        # A single observed family cannot establish between-family uncertainty.
        if len(observed_families) < 2:
            return result
        numerators = sums[self.indices].sum(axis=1)
        denominators = counts[self.indices].sum(axis=1)
        defined = denominators > 0
        result["bootstrap_draws_defined"] = int(defined.sum())
        # Do not silently discard resamples having no observed measurements.
        if not defined.all():
            result["interval_status"] = "undefined_resamples_due_to_missingness"
            return result
        estimates = numerators / denominators
        result["ci95"] = [float(x) for x in np.quantile(estimates, (0.025, 0.975))]
        result["ci90"] = [float(x) for x in np.quantile(estimates, (0.05, 0.95))]
        result["interval_status"] = "descriptive_family_bootstrap"
        return result


def stability_diagnostic(signed: dict[str, Any], absolute: dict[str, Any]) -> dict[str, Any]:
    result = {
        "margin_pp": EQUIVALENCE_MARGIN_PP,
        "status": "indeterminate",
        "criterion": "90% signed-mean CI strictly inside +/-2 pp AND upper 95% mean-absolute CI below 2 pp",
        "signed_mean_equivalent": None,
        "absolute_movement_bounded": None,
        "interpretation": "Exploratory development diagnostic; nonsignificance is not equivalence.",
    }
    if signed["n_missing"] or absolute["n_missing"]:
        result["reason"] = "missing_planned_observations"
    elif signed["ci90"] is None or absolute["ci95"] is None:
        result["reason"] = "insufficient_family_uncertainty"
    else:
        low, high = signed["ci90"]
        result["signed_mean_equivalent"] = -EQUIVALENCE_MARGIN_PP < low and high < EQUIVALENCE_MARGIN_PP
        result["absolute_movement_bounded"] = absolute["ci95"][1] < EQUIVALENCE_MARGIN_PP
        result["status"] = "criterion_met" if result["signed_mean_equivalent"] and result["absolute_movement_bounded"] else "criterion_not_met"
        result["reason"] = "complete_planned_observations"
    return result


def _movement_summary(samples: list[Sample], bootstrap: FamilyBootstrap) -> dict[str, Any]:
    return {
        "signed_mean_pp": bootstrap.summarize(samples),
        "mean_absolute_pp": bootstrap.summarize([
            Sample(item.family, None if item.value is None else abs(item.value)) for item in samples
        ]),
    }


def analyze(
    design: list[dict[str, Any]], responses: list[dict[str, Any]], model_key: str,
    *, bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS, seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if not model_key:
        raise ValueError("model_key must be nonempty")
    units = validate_design(design)
    indexed, other_models = index_responses(responses, units, model_key)
    families = sorted({row["family_id"] for row in design})
    bootstrap = FamilyBootstrap(families, bootstrap_draws, seed)
    coverage = {context: {condition: Counter() for condition in CONDITIONS} for context in CONTEXTS}
    baseline = {context: [] for context in CONTEXTS}
    raw = {context: {arm: [] for arm in CONDITIONS[1:]} for context in CONTEXTS}
    adjusted = {context: [] for context in CONTEXTS}
    repeated_minus_null = {context: [] for context in CONTEXTS}
    paired: dict[tuple[str, int], dict[str, tuple[float | None, float | None]]] = defaultdict(dict)
    for row in design:
        family, context, trial = row["family_id"], row["context_id"], row["trial_id"]
        values = {}
        for condition in CONDITIONS:
            record = indexed.get((trial, condition))
            value, reason = probability(record)
            values[condition] = value
            counter = coverage[context][condition]
            counter["expected"] += 1
            counter["present"] += int(record is not None)
            counter["valid"] += int(value is not None)
            counter["missing_record"] += int(record is None)
            counter["invalid_present"] += int(record is not None and value is None)
            counter[f"reason:{reason}"] += 1
        initial = values["baseline"]
        baseline[context].append(Sample(family, None if initial is None else 100 * initial))
        deltas = {}
        for condition in CONDITIONS[1:]:
            value = values[condition]
            delta = None if initial is None or value is None else 100 * (value - initial)
            raw[context][condition].append(Sample(family, delta))
            deltas[condition] = delta
        # Baseline is required for a valid paired trial even though it cancels.
        drift_adjusted = None if initial is None or values["new_news"] is None or values["no_news"] is None else 100 * (values["new_news"] - values["no_news"])
        repeat_adjusted = None if initial is None or values["repeated_news"] is None or values["no_news"] is None else 100 * (values["repeated_news"] - values["no_news"])
        adjusted[context].append(Sample(family, drift_adjusted))
        repeated_minus_null[context].append(Sample(family, repeat_adjusted))
        paired[family, row["repeat"]][context] = (deltas["new_news"], drift_adjusted)

    def direction_summary(samples_by_context: dict[str, list[Sample]]) -> dict[str, Any]:
        result = {}
        pooled = []
        for context, sign in (("positive", 1), ("negative", -1)):
            samples = [Sample(item.family, None if item.value is None else float(sign * item.value > 0)) for item in samples_by_context[context]]
            result[context] = bootstrap.summarize(samples, binary=True)
            pooled.extend(samples)
        result["pooled"] = bootstrap.summarize(pooled, binary=True)
        return result

    def pair_summary(index: int) -> dict[str, Any]:
        success, contrast = [], []
        for (family, _repeat), contexts in paired.items():
            positive, negative = contexts["positive"][index], contexts["negative"][index]
            complete = positive is not None and negative is not None
            success.append(Sample(family, float(positive > 0 and negative < 0) if complete else None))
            contrast.append(Sample(family, positive - negative if complete else None))
        return {
            "both_directions_correct": bootstrap.summarize(success, binary=True),
            "positive_minus_negative_pp": bootstrap.summarize(contrast),
        }

    raw_summaries = {context: {arm: _movement_summary(samples, bootstrap) for arm, samples in arms.items()} for context, arms in raw.items()}
    adjusted_summaries = {context: _movement_summary(samples, bootstrap) for context, samples in adjusted.items()}
    coverage_dict = {context: {arm: dict(counter) for arm, counter in arms.items()} for context, arms in coverage.items()}
    totals = Counter()
    for arms in coverage.values():
        for counter in arms.values():
            totals.update(counter)
    return {
        "schema_version": "context_reversal_analysis_v1",
        "material_status": MATERIAL_STATUS,
        "analysis_status": "development_unvalidated_descriptive_only",
        "model_key": model_key,
        "n_families": len(families),
        "n_planned_units": len(design),
        "n_planned_response_records": 4 * len(design),
        "other_model_records_ignored": other_models,
        "bootstrap": {"unit": "family_id", "draws": bootstrap_draws, "seed": seed, "preserves": ["contexts", "repeats", "update_arms", "baseline_pairs"]},
        "interpretation": [
            "Unvalidated authored development materials; no confirmatory or population claims.",
            "Primary outcomes use raw new-news minus baseline updates; zero and missing updates fail direction scoring.",
            "Both signs must be correct for paired reversal; all planned family/repeat pairs remain in the denominator.",
            "Magnitude summaries use observed complete measurements with missing counts displayed; missing forecasts are never zero-imputed.",
            "Intervals resample whole families and retain all repeats, contexts, and arms; they are descriptive.",
            "Masked contexts have no prespecified null or expected direction and are reported separately.",
            "Development diagnostics must not be used as confirmatory comparisons or to select a model.",
        ],
        "coverage": {"totals": dict(totals), "by_context_condition": coverage_dict},
        "baseline_probability_percent": {context: bootstrap.summarize(samples) for context, samples in baseline.items()},
        "primary": {
            "direction_correct": direction_summary({context: arms["new_news"] for context, arms in raw.items()}),
            "paired_reversal": pair_summary(0),
        },
        "raw_updates": raw_summaries,
        "drift_adjusted": {
            "definition": "100 * (new_news_probability - no_news_probability)",
            "direction_correct": direction_summary(adjusted),
            "paired_reversal": pair_summary(1),
            "by_context": adjusted_summaries,
        },
        "repeated_minus_no_news": {context: _movement_summary(samples, bootstrap) for context, samples in repeated_minus_null.items()},
        "broken_stability": {
            "raw_new_news": stability_diagnostic(**{key: raw_summaries["broken"]["new_news"][value] for key, value in (("signed", "signed_mean_pp"), ("absolute", "mean_absolute_pp"))}),
            "drift_adjusted": stability_diagnostic(**{key: adjusted_summaries["broken"][value] for key, value in (("signed", "signed_mean_pp"), ("absolute", "mean_absolute_pp"))}),
        },
    }


def _format_metric(metric: dict[str, Any], percent: bool = False) -> str:
    if metric["estimate"] is None:
        return "unavailable"
    scale = 100 if percent else 1
    text = f"{scale * metric['estimate']:.2f}" + ("%" if percent else "")
    interval = metric["ci95"]
    if interval is not None:
        text += f" [{scale * interval[0]:.2f}, {scale * interval[1]:.2f}]"
    else:
        text += " [interval unavailable]"
    return text


def markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        f"# Context-reversal development pilot: {report['model_key']}", "",
        "Authored, unvalidated materials. These are exploratory descriptive diagnostics, not confirmatory evidence or population estimates.", "",
        f"Planned: {report['n_families']} families, {report['n_planned_units']} context/repeat units, {report['n_planned_response_records']} response records.", "",
        "## Coverage", "",
        "| Context | Condition | Planned | Present | Valid probability | Missing record | Invalid present |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for context, arms in report["coverage"]["by_context_condition"].items():
        for arm, item in arms.items():
            lines.append(f"| {context} | {arm} | {item['expected']} | {item['present']} | {item['valid']} | {item['missing_record']} | {item['invalid_present']} |")
    lines.extend(["", "## Planned-denominator outcomes", "", "Zeros and invalid or missing trials count as failures. Paired reversal requires both signs to be correct.", "", "| Outcome | Estimate [95% family interval] | Success / planned | Missing measurements |", "|---|---|---:|---:|"])
    for prefix, source in (("Primary raw", report["primary"]), ("Drift adjusted", report["drift_adjusted"])):
        for name in ("positive", "negative", "pooled"):
            metric = source["direction_correct"][name]
            lines.append(f"| {prefix}: {name} direction | {_format_metric(metric, True)} | {metric['n_success']} / {metric['n_planned']} | {metric['n_missing']} |")
        metric = source["paired_reversal"]["both_directions_correct"]
        lines.append(f"| {prefix}: paired reversal | {_format_metric(metric, True)} | {metric['n_success']} / {metric['n_planned']} | {metric['n_missing']} |")
    lines.extend(["", "## Baselines and movement", "", "All values below are percentage points (baseline values are probabilities in percent). Magnitudes are observed-only, with missingness shown; no forecasts are zero-imputed. Masked has no expected direction or null target.", "", "| Context | Measurement | Signed mean [95% family interval] | Mean absolute [95% family interval] | Observed / planned |", "|---|---|---|---|---:|"])
    for context in CONTEXTS:
        initial = report["baseline_probability_percent"][context]
        lines.append(f"| {context} | baseline probability | {_format_metric(initial)} | — | {initial['n_observed']} / {initial['n_planned']} |")
        measurements = list(report["raw_updates"][context].items()) + [("new_news minus no_news", report["drift_adjusted"]["by_context"][context]), ("repeated_news minus no_news", report["repeated_minus_no_news"][context])]
        for name, summary in measurements:
            signed, absolute = summary["signed_mean_pp"], summary["mean_absolute_pp"]
            lines.append(f"| {context} | {name} | {_format_metric(signed)} | {_format_metric(absolute)} | {signed['n_observed']} / {signed['n_planned']} |")
    lines.extend(["", "## Broken-link stability", "", "Criterion: complete planned measurements, a 90% signed-mean interval strictly inside ±2 pp, and an upper 95% bound for mean absolute movement below 2 pp. Nonsignificance does not establish equivalence; opposing large changes do not establish stability.", ""])
    for name, diagnostic in report["broken_stability"].items():
        movement = report["raw_updates"]["broken"]["new_news"] if name == "raw_new_news" else report["drift_adjusted"]["by_context"]["broken"]
        interval = movement["signed_mean_pp"]["ci90"]
        absolute_interval = movement["mean_absolute_pp"]["ci95"]
        interval_text = "unavailable" if interval is None else f"[{interval[0]:.2f}, {interval[1]:.2f}] pp"
        upper_text = "unavailable" if absolute_interval is None else f"{absolute_interval[1]:.2f} pp"
        lines.append(f"- {name}: **{diagnostic['status']}** ({diagnostic['reason']}); signed 90% interval {interval_text}, upper 95% absolute-movement bound {upper_text}.")
    lines.extend(["", f"Uncertainty: {report['bootstrap']['draws']:,} whole-family bootstrap draws, seed {report['bootstrap']['seed']}; contexts, repetitions, and arms remain paired. Missing-only resamples make magnitude intervals unavailable rather than being dropped.", "", "Use these diagnostics to inspect the development task, not to make confirmatory model comparisons or select a model.", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--responses", required=True, type=Path, nargs="+")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output", required=True, type=Path, help="Directory for summary.json and summary.md")
    parser.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    try:
        responses = [row for path in args.responses for row in read_jsonl(path)]
        design_sha256 = hashlib.sha256(args.design.read_bytes()).hexdigest()
        for row in responses:
            if row.get("model_key") == args.model_key and "input_sha256" in row and row["input_sha256"] != design_sha256:
                raise ValueError("Response input_sha256 does not match the supplied design")
        report = analyze(read_jsonl(args.design), responses, args.model_key, bootstrap_draws=args.bootstrap_draws, seed=args.seed)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    report["provenance"] = {
        "design": {"path": str(args.design), "sha256": design_sha256},
        "responses": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in args.responses],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output / "summary.md").write_text(markdown_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
