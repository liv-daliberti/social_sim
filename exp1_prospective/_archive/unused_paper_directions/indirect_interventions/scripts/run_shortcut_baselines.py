#!/usr/bin/env python3
"""Grouped bridge-masked baselines on the locked human-retained families."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, make_pipeline


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CANDIDATES = ROOT / "data/candidates/core_candidates.jsonl"
DEFAULT_VALIDATION = ROOT / "reports/core_human_validation.json"
DEFAULT_OUTPUT = ROOT / "reports/shortcut_baselines.json"
LABELS = ("increase", "decrease", "no_effect")
ROLE_LABEL = {
    "positive": "increase",
    "negative": "decrease",
    "broken": "no_effect",
}
POSITIVE = {
    "gain", "gains", "growth", "rise", "rises", "strong", "support", "surge",
    "record", "success", "improve", "improves", "increase", "increases",
}
NEGATIVE = {
    "decline", "declines", "fall", "falls", "weak", "failure", "crisis",
    "loss", "losses", "drop", "drops", "decrease", "decreases",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sentiment_prediction(text: str) -> str:
    words = set(re.findall(r"[a-z]+", text.lower()))
    score = len(words & POSITIVE) - len(words & NEGATIVE)
    return "increase" if score > 0 else ("decrease" if score < 0 else "no_effect")


def evaluate(y: list[str], prediction: list[str], groups: list[str]) -> dict[str, Any]:
    per_market = {}
    for group in sorted(set(groups)):
        positions = [index for index, value in enumerate(groups) if value == group]
        per_market[group] = float(np.mean([
            prediction[index] == y[index] for index in positions
        ]))
    return {
        "macro_f1": float(f1_score(y, prediction, labels=LABELS, average="macro")),
        "accuracy": float(accuracy_score(y, prediction)),
        "market_macro_accuracy": float(np.mean(list(per_market.values()))),
        "triplet_accuracy_with_fixed_tau_shift": float(accuracy_score(y, prediction)),
        "complete_triplet_success": 0.0,
    }


def expand_family_predictions(
    candidates: list[dict[str, Any]],
    mapping: dict[str, str],
) -> list[str]:
    result = []
    for candidate in candidates:
        prediction = mapping[candidate["candidate_id"]]
        result.extend([prediction, prediction, prediction])
    return result


def load_prediction_file(
    path: Path | None,
    candidates: list[dict[str, Any]],
) -> list[str] | None:
    if path is None:
        return None
    rows = read_jsonl(path)
    mapping = {row["candidate_id"]: row["prediction"] for row in rows}
    expected = {row["candidate_id"] for row in candidates}
    if set(mapping) != expected or not set(mapping.values()) <= set(LABELS):
        raise SystemExit(f"prediction file is incomplete or invalid: {path}")
    return expand_family_predictions(candidates, mapping)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--embedding-predictions", type=Path)
    parser.add_argument("--evidence-instruction-predictions", type=Path)
    parser.add_argument("--question-evidence-predictions", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    if not validation.get("ready_to_freeze"):
        raise SystemExit("refusing baseline audit before human retention gates pass")
    retained = set(validation["retention"]["retained_candidate_ids"])
    candidates = sorted(
        (row for row in read_jsonl(args.candidates) if row["candidate_id"] in retained),
        key=lambda row: row["candidate_id"],
    )
    if len(candidates) < 240:
        raise SystemExit("retained candidate pool is below the registered minimum")

    evidence_text, question_text, y, groups = [], [], [], []
    for candidate in candidates:
        evidence = candidate["evidence"]["headline"] + " " + candidate["evidence"]["text"]
        question_evidence = candidate["market"]["question"] + " " + evidence
        for role in ("positive", "negative", "broken"):
            evidence_text.append(evidence)
            question_text.append(question_evidence)
            y.append(ROLE_LABEL[role])
            groups.append(candidate["market"]["task_id"])
    splitter = GroupKFold(n_splits=args.folds)
    models: dict[str, dict[str, Any]] = {}
    sentiment = [sentiment_prediction(text) for text in evidence_text]
    models["headline_sentiment"] = evaluate(y, sentiment, groups)
    majority = ["no_effect"] * len(y)
    models["majority_class"] = evaluate(y, majority, groups)

    tfidf = make_pipeline(
        FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30_000)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30_000)),
        ]),
        LogisticRegression(max_iter=3000, random_state=args.seed),
    )
    tfidf_prediction = cross_val_predict(
        tfidf, question_text, y, groups=groups, cv=splitter, method="predict"
    ).tolist()
    models["tfidf_logistic_question_evidence"] = evaluate(y, tfidf_prediction, groups)

    hashing = make_pipeline(
        HashingVectorizer(n_features=2**14, alternate_sign=True, ngram_range=(1, 2)),
        LogisticRegression(max_iter=3000, random_state=args.seed),
    )
    hashing_prediction = cross_val_predict(
        hashing, evidence_text, y, groups=groups, cv=splitter, method="predict"
    ).tolist()
    models["frozen_hashing_linear_evidence"] = evaluate(y, hashing_prediction, groups)

    optional = {
        "frozen_sentence_embedding_linear": load_prediction_file(
            args.embedding_predictions, candidates
        ),
        "evidence_only_instruction_classifier": load_prediction_file(
            args.evidence_instruction_predictions, candidates
        ),
        "question_evidence_instruction_classifier": load_prediction_file(
            args.question_evidence_predictions, candidates
        ),
    }
    missing = []
    for name, prediction in optional.items():
        if prediction is None:
            missing.append(name)
        else:
            models[name] = evaluate(y, prediction, groups)
    threshold_pass = all(row["macro_f1"] <= 0.40 for row in models.values())
    complete_required = not missing
    report = {
        "protocol_version": "indirect-core-v1",
        "grouping": "market",
        "folds": args.folds,
        "candidate_count": len(candidates),
        "record_count": len(y),
        "labels": list(LABELS),
        "models": models,
        "missing_required_baselines": missing,
        "bridge_masked_macro_f1_maximum": 0.40,
        "threshold_pass": threshold_pass,
        "complete_required_baselines": complete_required,
        "ready_for_target_freeze": threshold_pass and complete_required,
        "note": (
            "Every evidence item is repeated under positive, negative, and broken "
            "labels; grouped splits keep all copies and their market in one fold."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "threshold_pass": threshold_pass,
        "complete_required_baselines": complete_required,
        "missing": missing,
        "macro_f1": {name: row["macro_f1"] for name, row in models.items()},
    }, indent=2))
    if not report["ready_for_target_freeze"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
