#!/usr/bin/env python3
"""Held-out deterministic audit of preserved Coin City rationales."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data" / "coin_city_stable_relationship_claude_n250_v4"
DESIGN = RUN / "design"
COMPARISON = (
    ROOT
    / "local_results"
    / "symbol_context_model_comparison_20260824"
    / "exploratory_results.json"
)
OUTPUT = RUN / "analysis" / "rationale_audit_v1.json"
PROTOCOL_VERSION = "coin_city_robustness_v1"
TASK = re.compile(r"_(\d{4})_c([0-4])$")
CITY = re.compile(r"\bcity\s*[- ]?([ab])\b", re.I)
LINK_PATTERNS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"city\s*c.{0,140}?\b(?:same|shares?|similar|like|matches?|matching|aligns?|comparable)\b.{0,120}?city\s*[- ]?([ab])\b",
        r"city\s*c.{0,140}?city\s*[- ]?([ab])\b.{0,100}?\b(?:same|shares?|similar|like|matches?|matching|aligns?|comparable)\b",
        r"\b(?:same|shares?|similar|like|matches?|matching|aligns?|comparable)\b.{0,100}?city\s*[- ]?([ab])\b",
        r"\b(?:using|from|follows?|apply|applying|inferred from|based on)\b.{0,100}?city\s*[- ]?([ab])\b",
    )
)
SELECTION = re.compile(
    r"\b(?:same|shares?|similar|like|matches?|matching|aligns?|comparable|"
    r"inferred from|based on|apply|using)\b",
    re.I,
)
SYMBOL = re.compile(r"\b(?:KIV|ZOR)\b", re.I)
EVIDENCE = re.compile(
    r"\b(?:responsiveness|cases?|data|matched|average|observed|pattern|pairs?|"
    r"per\s+(?:unit|point)|slope|response)\b",
    re.I,
)
NATIONAL = re.compile(r"\bnational(?:ly)?(?:[- ]news|\s+coverage)?\b", re.I)
LOCAL = re.compile(r"\blocal(?:ly)?(?:[- ]news|\s+coverage)?\b", re.I)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def development_episodes() -> set[int]:
    return set(
        sorted(
            range(250),
            key=lambda episode: hashlib.sha256(
                f"rationale-coder-v1:{episode}".encode()
            ).hexdigest(),
        )[:50]
    )


def classify_reference(text: str) -> tuple[str | None, str]:
    linked = {
        match.group(1).upper()
        for pattern in LINK_PATTERNS
        for match in pattern.finditer(text)
    }
    if len(linked) == 1:
        return next(iter(linked)), "explicit_link"
    if len(linked) > 1:
        return None, "ambiguous_link"
    mentioned = {match.group(1).upper() for match in CITY.finditer(text)}
    if len(mentioned) == 1:
        return next(iter(mentioned)), "unique_mention"
    return None, "ambiguous_or_absent"


def semantic_regime(text: str) -> bool | None:
    national = bool(NATIONAL.search(text))
    local = bool(LOCAL.search(text))
    if national == local:
        return None
    return national


def expected_reference(episode: dict, strong: bool) -> str:
    strong_city = episode["strong_reference_city"]
    if strong:
        return strong_city
    return "B" if strong_city == "A" else "A"


def rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def aggregate_symbol(records: list[dict], episodes: dict[int, dict]) -> dict:
    counts = Counter()
    for row in records:
        counts["records"] += 1
        rationale = row.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            continue
        text = rationale.strip()
        counts["rationales"] += 1
        counts["mentions_symbol"] += bool(SYMBOL.search(text))
        counts["selection_language"] += bool(SELECTION.search(text))
        counts["symbol_evidence_language"] += bool(
            SYMBOL.search(text) and EVIDENCE.search(text)
        )
        reference, method = classify_reference(text)
        counts[f"reference_method:{method}"] += 1
        if reference is None:
            continue
        counts["reference_codable"] += 1
        episode = episodes[row["episode"]]
        correct = expected_reference(episode, bool(episode["target_strong"]))
        counts["names_matching_reference"] += reference == correct
        prediction = row.get("predicted_poll")
        if isinstance(prediction, (int, float)):
            query_start = float(episode["query_starting_poll"])
            query_news = float(episode["query_net_news"])
            implied = (float(prediction) - query_start) / query_news
            fitted = {
                city: fit_slope(episode[f"reference_{city.lower()}"])
                for city in ("A", "B")
            }
            other = "B" if reference == "A" else "A"
            counts["forecast_codable"] += 1
            counts["forecast_closer_to_named_reference"] += abs(
                implied - fitted[reference]
            ) <= abs(implied - fitted[other])
    return {
        "counts": dict(counts),
        "rationale_presence": rate(counts["rationales"], counts["records"]),
        "symbol_mention": rate(counts["mentions_symbol"], counts["rationales"]),
        "selection_language": rate(counts["selection_language"], counts["rationales"]),
        "symbol_evidence_language": rate(
            counts["symbol_evidence_language"], counts["rationales"]
        ),
        "reference_codable": rate(counts["reference_codable"], counts["rationales"]),
        "matching_reference_all_rationales": rate(
            counts["names_matching_reference"], counts["rationales"]
        ),
        "matching_reference_when_codable": rate(
            counts["names_matching_reference"], counts["reference_codable"]
        ),
        "forecast_closer_to_named_reference": rate(
            counts["forecast_closer_to_named_reference"], counts["forecast_codable"]
        ),
    }


def aggregate_wrong(records: list[dict], episodes: dict[int, dict]) -> dict:
    counts = Counter()
    for row in records:
        counts["records"] += 1
        rationale = row.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            continue
        text = rationale.strip()
        counts["rationales"] += 1
        episode = episodes[row["episode"]]
        displayed_strong = not bool(episode["target_strong"])
        displayed_ref = expected_reference(episode, displayed_strong)
        true_ref = expected_reference(episode, not displayed_strong)
        reference, method = classify_reference(text)
        counts[f"reference_method:{method}"] += 1
        if reference is not None:
            counts["reference_codable"] += 1
            counts["names_false_cue_reference"] += reference == displayed_ref
            counts["names_true_reference"] += reference == true_ref
        regime = semantic_regime(text)
        if regime is not None:
            counts["semantic_codable"] += 1
            counts["names_false_cue_regime"] += regime == displayed_strong
            counts["names_true_regime"] += regime == bool(episode["target_strong"])
    return {
        "counts": dict(counts),
        "rationale_presence": rate(counts["rationales"], counts["records"]),
        "reference_codable": rate(counts["reference_codable"], counts["rationales"]),
        "false_cue_reference_when_codable": rate(
            counts["names_false_cue_reference"], counts["reference_codable"]
        ),
        "true_reference_when_codable": rate(
            counts["names_true_reference"], counts["reference_codable"]
        ),
        "semantic_codable": rate(counts["semantic_codable"], counts["rationales"]),
        "false_cue_regime_when_codable": rate(
            counts["names_false_cue_regime"], counts["semantic_codable"]
        ),
        "true_regime_when_codable": rate(
            counts["names_true_regime"], counts["semantic_codable"]
        ),
    }


def fit_slope(rows: list[dict]) -> float:
    x = np.asarray([row["net_news"] for row in rows], dtype=float)
    y = np.asarray(
        [row["ending_poll"] - row["starting_poll"] for row in rows], dtype=float
    )
    return float(np.sum(x * y) / np.sum(x * x))


def task_metadata(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        match = TASK.search(row.get("task_id", ""))
        if match is None:
            continue
        enriched = dict(row)
        enriched["episode"] = int(match.group(1))
        enriched["c_cases"] = int(match.group(2))
        output.append(enriched)
    return output


def main() -> None:
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    episodes = {
        int(row["episode"]): row for row in read_jsonl(DESIGN / "episodes.jsonl")
    }
    dev = development_episodes()
    audit = set(range(250)) - dev

    presence = {}
    symbol_results = {}
    for model, result in comparison["models"].items():
        model_presence = {}
        for arm in ("abc_no_context", "abc_context", "abc_symbol_context"):
            path = Path(comparison["response_paths"][f"{model}:{arm}"])
            rows = read_jsonl(path)
            model_presence[arm] = {
                "records": len(rows),
                "rationales": sum(
                    isinstance(r.get("rationale"), str) and bool(r["rationale"].strip())
                    for r in rows
                ),
            }
        presence[model] = model_presence
        symbol_path = Path(comparison["response_paths"][f"{model}:abc_symbol_context"])
        symbol = [
            row for row in task_metadata(read_jsonl(symbol_path)) if row["c_cases"] == 0
        ]
        audit_rows = [row for row in symbol if row["episode"] in audit]
        dev_rows = [row for row in symbol if row["episode"] in dev]
        curve = result["curves"]["0"]
        symbol_results[model] = {
            "audit": aggregate_symbol(audit_rows, episodes),
            "development": aggregate_symbol(dev_rows, episodes),
            "forecast_mapping": {
                "symbol_rho": curve["arms"]["abc_symbol_context"]["rho"],
                "no_context_rho": curve["arms"]["abc_no_context"]["rho"],
                "symbol_minus_no_context_rho": curve["paired_discrimination"][
                    "symbol_minus_no_context_rho"
                ],
                "symbol_regime_accuracy": curve["arms"]["abc_symbol_context"][
                    "regime_accuracy"
                ],
            },
        }

    wrong_results = {}
    for path in sorted((RUN / "responses").glob("responses_*_abc_wrong_context.jsonl")):
        rows = [row for row in task_metadata(read_jsonl(path)) if row["c_cases"] == 0]
        if not rows:
            continue
        model = str(
            rows[0].get("model")
            or path.name.removeprefix("responses_").removesuffix(
                "_abc_wrong_context.jsonl"
            )
        )
        wrong_results[model] = {
            "audit": aggregate_wrong(
                [row for row in rows if row["episode"] in audit], episodes
            ),
            "development": aggregate_wrong(
                [row for row in rows if row["episode"] in dev], episodes
            ),
            "source": str(path.resolve()),
        }

    total_records = sum(
        cell["records"] for model in presence.values() for cell in model.values()
    )
    total_rationales = sum(
        cell["rationales"] for model in presence.values() for cell in model.values()
    )
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "classification": "post_hoc_heldout_rationale_audit",
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "development_episodes": sorted(dev),
        "audit_episodes": sorted(audit),
        "coding": {
            "reference": "explicit City-C linkage patterns, then unique City-A/B mention fallback; ambiguous statements uncoded",
            "semantic_regime": "exclusive national-versus-local lexical mention; ambiguous statements uncoded",
            "symbol_evidence": "KIV/ZOR plus prespecified evidence-language dictionary",
            "interpretation": "behavioral self-report, not privileged evidence of computation",
        },
        "presence": {
            "total_records": total_records,
            "total_rationales": total_rationales,
            "coverage": total_rationales / total_records,
            "by_model_arm": presence,
        },
        "symbol_k0": symbol_results,
        "inverted_semantic_k0": wrong_results,
        "sources": {
            "comparison": str(COMPARISON.resolve()),
            "episodes": str((DESIGN / "episodes.jsonl").resolve()),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "records": total_records,
                "rationales": total_rationales,
                "coverage": total_rationales / total_records,
                "symbol_models": len(symbol_results),
                "wrong_context_models": len(wrong_results),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
