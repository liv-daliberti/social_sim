#!/usr/bin/env python3
"""Audit lexical discrimination on frozen development materials, not model outputs.

Only the compiled visible prompts enter features. Intended condition labels are
supervised targets, never feature text. Optional mechanism-class metadata is used
only for cross-validation grouping, never for feature text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
DEFAULT_PLAN = ROOT / "runs/development_v1/development_plan.jsonl"
DEFAULT_OUTPUT = ROOT / "results/lexical_audit"
FEATURES = ("news_only", "question_and_news", "full_visible_context_and_news")
CLASSIFIERS = ("multinomial_nb", "logistic_regression")
CONTEXTS = ("positive", "negative", "broken")
SEED = 20260921


def between(text: str, start: str, end: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"Visible prompt lacks unique expected delimiters: {start!r}, {end!r}")
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def extract_examples(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    examples = {}
    for row in rows:
        context = row["context_id"]
        if context == "masked":
            continue
        if context not in CONTEXTS:
            raise ValueError(f"Unexpected context {context!r}")
        if row.get("material_status") != "authored_development_unvalidated":
            raise ValueError("Audit requires unvalidated development materials")
        baseline, update = row["baseline_prompt"], row["update_templates"]["new_news"]
        question = between(baseline, "\n\nQUESTION\n", "\n\nRESOLUTION RULE\n")
        news = between(update, "\n\nMESSAGE\n", "\n\nRevise your previous probability")
        example = {
            "family_id": row["family_id"],
            "context_id": context,
            "news_only": news,
            "question_and_news": question + "\n\n" + news,
            "full_visible_context_and_news": baseline + "\n\nMESSAGE\n" + news,
        }
        key = (row["family_id"], context)
        if key in examples and examples[key] != example:
            raise ValueError("Repeat prompts differ; cannot deduplicate the frozen design")
        examples[key] = example
    families = sorted({family for family, _ in examples})
    if len(families) < 2:
        raise ValueError("Family-held-out audit needs at least two families")
    for family in families:
        if {context for candidate, context in examples if candidate == family} != set(CONTEXTS):
            raise ValueError(f"Incomplete context triplet for family {family}")
        for feature in FEATURES[:2]:
            if len({examples[family, context][feature] for context in CONTEXTS}) != 1:
                raise ValueError(f"Expected matched {feature} within family {family}")
    return [examples[family, context] for family in families for context in CONTEXTS]


def make_classifier(name: str):
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    if name == "multinomial_nb":
        estimator = MultinomialNB(alpha=1.0)
    elif name == "logistic_regression":
        estimator = LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", max_iter=2000, random_state=SEED)
    else:
        raise ValueError(f"Unknown fixed classifier {name}")
    return make_pipeline(vectorizer, estimator)


def classification_metrics(target, predicted, groups) -> dict[str, Any]:
    families = sorted(set(groups))
    return {
        "accuracy": float(accuracy_score(target, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "confusion_matrix_labels_0_1": confusion_matrix(target, predicted, labels=[0, 1]).tolist(),
        "family_all_contexts_correct_rate": float(np.mean([np.all(predicted[groups == family] == target[groups == family]) for family in families])),
    }


def evaluate_tasks(examples: list[dict[str, str]], mechanism_classes: dict[str, str] | None = None) -> dict[str, Any]:
    tasks = {}
    for task in ("relevance", "sign"):
        selected = [row for row in examples if task == "relevance" or row["context_id"] != "broken"]
        groups = np.asarray([row["family_id"] for row in selected])
        target = np.asarray([int(row["context_id"] != "broken") if task == "relevance" else int(row["context_id"] == "positive") for row in selected])
        split_groups = np.asarray([mechanism_classes[family] for family in groups]) if mechanism_classes else groups
        splits = list(LeaveOneGroupOut().split(selected, target, split_groups))
        majority = np.empty(len(selected), dtype=int)
        for train, test in splits:
            assert not set(groups[train]) & set(groups[test])
            assert not set(split_groups[train]) & set(split_groups[test])
            counts = np.bincount(target[train], minlength=2)
            # Deterministic label-0 tie break for the balanced sign task.
            majority[test] = int(np.argmax(counts))
        results = []
        for feature in FEATURES:
            text = np.asarray([row[feature] for row in selected], dtype=object)
            for classifier in CLASSIFIERS:
                predicted = np.empty(len(selected), dtype=int)
                for train, test in splits:
                    pipeline = make_classifier(classifier)
                    pipeline.fit(text[train].tolist(), target[train])
                    predicted[test] = pipeline.predict(text[test].tolist())
                results.append({
                    "feature_set": feature,
                    "classifier": classifier,
                    **classification_metrics(target, predicted, groups),
                    "out_of_family_predictions": [
                        {"family_id": row["family_id"], "context_id": row["context_id"], "target": int(truth), "prediction": int(prediction)}
                        for row, truth, prediction in zip(selected, target, predicted)
                    ],
                })
        tasks[task] = {
            "target_definition": {"0": "broken", "1": "positive_or_negative"} if task == "relevance" else {"0": "negative", "1": "positive"},
            "n_examples": len(selected),
            "n_families": len(set(groups)),
            "n_folds": len(splits),
            "class_counts_0_1": np.bincount(target, minlength=2).tolist(),
            "training_majority_baseline": classification_metrics(target, majority, groups),
            "results": results,
        }
    return tasks


def audit(examples: list[dict[str, str]], mechanism_classes: dict[str, str] | None = None) -> dict[str, Any]:
    tasks = evaluate_tasks(examples)
    mechanism_audit = None
    if mechanism_classes is not None:
        families = {row["family_id"] for row in examples}
        if set(mechanism_classes) != families:
            raise ValueError("Mechanism-class map does not match the planned families")
        counts = {group: sum(mechanism_classes[family] == group for family in families) for group in sorted(set(mechanism_classes.values()))}
        if len(counts) < 2:
            raise ValueError("Mechanism-class CV requires at least two classes")
        mechanism_audit = {
            "cv": "leave_one_mechanism_class_out; every family sharing a class is held out together",
            "class_family_counts": counts,
            "n_mechanism_classes": len(counts),
            "tasks": evaluate_tasks(examples, mechanism_classes),
        }
    return {
        "schema_version": "context_reversal_lexical_audit_v1",
        "material_status": "authored_development_unvalidated",
        "audit_status": "diagnostic_not_confirmatory",
        "feature_sources": "Only baseline_prompt and new_news update_template visible text from the compiled frozen plan; intended context IDs are targets only.",
        "private_metadata_used_as_features": False,
        "mechanism_class_metadata_used_only_for_cv_grouping": mechanism_classes is not None,
        "model_responses_read": False,
        "masked_excluded": True,
        "repeats_deduplicated": True,
        "cv": "leave_one_family_out; vectorizer and classifier fit only on training families",
        "mechanism_class_cv": mechanism_audit,
        "settings": {
            "tfidf": {"analyzer": "word", "ngram_range": [1, 2], "lowercase": True, "min_df": 1},
            "multinomial_nb": {"alpha": 1.0},
            "logistic_regression": {"C": 1.0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000, "random_state": SEED},
            "hyperparameter_search": False,
            "sklearn_version": sklearn.__version__,
        },
        "interpretation": [
            "Identical news and question-plus-news within each family guarantee no within-family discrimination for these feature sets.",
            "The full visible prompt can expose lexical relational cues; classifier success is a shortcut diagnostic, not evidence of relational reasoning.",
            "Classifier failure does not establish that language-model success reflects reasoning or rule out richer lexical shortcuts.",
            "The 20 authored families reuse five mechanism classes and are not 20 independent mechanisms; class-held-out CV probes shared-template generalization.",
            "Small, authored, unvalidated development materials and intended labels do not support confirmatory or population claims.",
            "The fixed audit does not modify materials, model inputs, or the prescribed model roster.",
        ],
        "tasks": tasks,
    }


def markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Frozen-material lexical audit", "",
        "Exploratory diagnostic on authored, unvalidated development materials. No model responses entered this audit. Features use visible prompts only; mechanism-class metadata, when supplied, is used exclusively to form held-out groups.", "",
        "All repetitions are deduplicated. Leave-one-family-out cross-validation keeps every context from a family out of training together; TF-IDF is fit inside each fold. Hyperparameters are fixed, with no search. Masked cases are excluded.", "",
        "| Task | Features | Classifier | Accuracy | Balanced accuracy | Whole-family correctness |",
        "|---|---|---|---:|---:|---:|",
    ]
    for task, results in report["tasks"].items():
        majority = results["training_majority_baseline"]
        lines.append(f"| {task} | none | training majority | {majority['accuracy']:.3f} | {majority['balanced_accuracy']:.3f} | {majority['family_all_contexts_correct_rate']:.3f} |")
        for row in results["results"]:
            lines.append(f"| {task} | {row['feature_set']} | {row['classifier']} | {row['accuracy']:.3f} | {row['balanced_accuracy']:.3f} | {row['family_all_contexts_correct_rate']:.3f} |")
    mechanism = report["mechanism_class_cv"]
    if mechanism is not None:
        lines.extend(["", "## Leave-one-mechanism-class-out check", "", f"The authored families reuse {mechanism['n_mechanism_classes']} mechanism classes: " + ", ".join(f"{name} ({count} families)" for name, count in mechanism["class_family_counts"].items()) + ". They are not 20 independent mechanisms. This check holds every family from one class out together; the class labels never enter the classifier features.", "", "| Task | Features | Classifier | Accuracy | Balanced accuracy | Whole-family correctness |", "|---|---|---|---:|---:|---:|"])
        for task, results in mechanism["tasks"].items():
            majority = results["training_majority_baseline"]
            lines.append(f"| {task} | none | training majority | {majority['accuracy']:.3f} | {majority['balanced_accuracy']:.3f} | {majority['family_all_contexts_correct_rate']:.3f} |")
            for row in results["results"]:
                lines.append(f"| {task} | {row['feature_set']} | {row['classifier']} | {row['accuracy']:.3f} | {row['balanced_accuracy']:.3f} | {row['family_all_contexts_correct_rate']:.3f} |")
    lines.extend(["", "Relevance labels are positive/negative versus broken (2:1 class balance); sign uses positive versus negative (1:1). Whole-family correctness requires all three relevance labels, or both sign labels, to be correct. Labels are prediction targets and never feature text.", "", "Identical news and question-plus-news guarantee no within-family discrimination. Full prompts may expose lexical relational cues. Success identifies an available lexical route; failure of these limited classifiers does not prove relational reasoning or eliminate more capable shortcuts.", "", "Fixed settings: word unigram/bigram TF-IDF; MultinomialNB alpha=1; logistic regression C=1, balanced class weights, liblinear, maximum 2,000 iterations. Results are diagnostic, not confirmatory; materials and model inputs remain unchanged.", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--families", type=Path, default=DEFAULT_PLAN.with_name("development_families.jsonl"), help="Frozen family metadata; only mechanism_class is used, solely for CV grouping")
    args = parser.parse_args(argv)
    raw = args.plan.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    family_raw = args.families.read_bytes()
    mechanism_classes = {}
    for line in family_raw.splitlines():
        if not line.strip():
            continue
        family = json.loads(line)
        family_id = family["family_id"]
        group = family["private_metadata"]["mechanism_class"]
        if family_id in mechanism_classes or not isinstance(group, str) or not group:
            raise ValueError("Invalid or duplicate mechanism-class mapping")
        mechanism_classes[family_id] = group
    report = audit(extract_examples(rows), mechanism_classes)
    report["provenance"] = {"compiled_plan": str(args.plan), "sha256": hashlib.sha256(raw).hexdigest(), "grouping_metadata": str(args.families), "grouping_metadata_sha256": hashlib.sha256(family_raw).hexdigest()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output / "summary.md").write_text(markdown_summary(report))
    for cv_name, task_reports in (("leave_family_out", report["tasks"]), ("leave_mechanism_class_out", report["mechanism_class_cv"]["tasks"])):
        print(cv_name)
        print(json.dumps({task: [{key: row[key] for key in ("feature_set", "classifier", "accuracy", "balanced_accuracy", "family_all_contexts_correct_rate")} for row in data["results"] if row["feature_set"] == "full_visible_context_and_news"] for task, data in task_reports.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
