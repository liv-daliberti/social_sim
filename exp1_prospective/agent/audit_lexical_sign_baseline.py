#!/usr/bin/env python3
"""Lexical baselines for Experiment 1 packet direction.

The construct-validity discussion raises the possibility that a shallow sign
classifier could recover a packet's direction from its wording alone, without
representing the mechanism the packet describes. This script tests that directly
by fitting text-only classifiers on the 900 frozen packets and scoring them
against the registered direction key.

Three tasks, each evaluated with grouped cross-validation so that no market
contributes packets to both the training and the test side of a fold:

  3-class     pro_H1 vs anti_H1 vs orthogonal        (chance 1/3)
  probative   directional vs orthogonal              (majority 2/3)
  sign        pro_H1 vs anti_H1                      (chance 1/2)

`sign` is the one that matters. It is the task the agents perform when they move
a forecast the right way, and their unconditional directional correctness on it
is 97.3-98.4% for the hosted systems. Two feature settings are reported: the
snippet alone, which is the bare classifier the objection describes, and the
snippet with its market question prepended, which is the information the agents
also had.

Models are multinomial naive Bayes on token counts and L2-regularised logistic
regression on TF-IDF, both implemented here so the check has no dependency
beyond numpy.

Outputs:
  data/results/lexical_sign_baseline.json
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CF_FILE = ROOT / "data" / "counterfactuals" / "counterfactuals_2026-06-10.jsonl"
OUT_JSON = ROOT / "data" / "results" / "lexical_sign_baseline.json"

N_FOLDS = 5
SEED = 0
TOKEN_RE = re.compile(r"[a-z][a-z']+")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    # Bucket digit runs so specific figures cannot act as market fingerprints.
    text = re.sub(r"\d[\d,\.]*", " <num> ", text)
    toks = TOKEN_RE.findall(text) + ["<num>"] * text.count("<num>")
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


def build_vocab(docs: list[list[str]], min_df: int = 2) -> dict[str, int]:
    df = Counter()
    for d in docs:
        df.update(set(d))
    kept = sorted(t for t, c in df.items() if c >= min_df)
    return {t: i for i, t in enumerate(kept)}


def count_matrix(docs, vocab) -> np.ndarray:
    X = np.zeros((len(docs), len(vocab)), dtype=np.float64)
    for i, d in enumerate(docs):
        for t, c in Counter(d).items():
            j = vocab.get(t)
            if j is not None:
                X[i, j] = c
    return X


def tfidf(train_counts: np.ndarray, *mats: np.ndarray):
    df = (train_counts > 0).sum(axis=0)
    idf = np.log((1 + train_counts.shape[0]) / (1 + df)) + 1.0
    out = []
    for m in mats:
        w = m * idf
        norm = np.linalg.norm(w, axis=1, keepdims=True)
        out.append(w / np.maximum(norm, 1e-12))
    return out


def fit_nb(X: np.ndarray, y: np.ndarray, n_classes: int, alpha: float = 1.0):
    logprior, loglik = [], []
    for c in range(n_classes):
        Xc = X[y == c]
        logprior.append(math.log(max(len(Xc), 1) / len(X)))
        counts = Xc.sum(axis=0) + alpha
        loglik.append(np.log(counts / counts.sum()))
    return np.array(logprior), np.vstack(loglik)


def predict_nb(X, logprior, loglik):
    return np.argmax(X @ loglik.T + logprior, axis=1)


def fit_logreg(X, y, n_classes, l2=1.0, epochs=300, lr=0.5):
    n, d = X.shape
    W = np.zeros((n_classes, d))
    b = np.zeros(n_classes)
    Y = np.zeros((n, n_classes))
    Y[np.arange(n), y] = 1.0
    for _ in range(epochs):
        z = X @ W.T + b
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        g = (p - Y) / n
        W -= lr * (g.T @ X + l2 * W / n)
        b -= lr * g.sum(axis=0)
    return W, b


def predict_logreg(X, W, b):
    return np.argmax(X @ W.T + b, axis=1)


def grouped_folds(groups: list[str], n_folds: int) -> list[np.ndarray]:
    uniq = sorted(set(groups))
    rng = np.random.RandomState(SEED)
    rng.shuffle(uniq)
    assign = {g: i % n_folds for i, g in enumerate(uniq)}
    idx = np.array([assign[g] for g in groups])
    return [np.where(idx == f)[0] for f in range(n_folds)]


def run_task(texts, labels, groups, n_classes, name):
    docs = [tokenize(t) for t in texts]
    y = np.array(labels)
    folds = grouped_folds(groups, N_FOLDS)
    acc = {"naive_bayes": [], "logistic_regression": []}
    for f in range(N_FOLDS):
        te = folds[f]
        tr = np.concatenate([folds[g] for g in range(N_FOLDS) if g != f])
        vocab = build_vocab([docs[i] for i in tr])
        Xtr = count_matrix([docs[i] for i in tr], vocab)
        Xte = count_matrix([docs[i] for i in te], vocab)
        lp, ll = fit_nb(Xtr, y[tr], n_classes)
        acc["naive_bayes"].append(float((predict_nb(Xte, lp, ll) == y[te]).mean()))
        Ttr, Tte = tfidf(Xtr, Xtr, Xte)
        W, b = fit_logreg(Ttr, y[tr], n_classes)
        acc["logistic_regression"].append(float((predict_logreg(Tte, W, b) == y[te]).mean()))
    majority = max(Counter(labels).values()) / len(labels)
    return {
        "task": name,
        "n": len(labels),
        "classes": n_classes,
        "majority_class_rate": round(majority, 3),
        "chance_rate": round(1 / n_classes, 3),
        "naive_bayes_accuracy": round(float(np.mean(acc["naive_bayes"])), 3),
        "naive_bayes_by_fold": [round(a, 3) for a in acc["naive_bayes"]],
        "logistic_regression_accuracy": round(float(np.mean(acc["logistic_regression"])), 3),
        "logistic_regression_by_fold": [round(a, 3) for a in acc["logistic_regression"]],
    }


def main() -> None:
    packets = [json.loads(l) for l in CF_FILE.read_text().splitlines() if l.strip()]
    groups = [p["task_id"] for p in packets]
    snippet = [p.get("evidence_text") or "" for p in packets]
    with_q = [f"{p.get('question','')} {p.get('evidence_text') or ''}" for p in packets]
    direction = [p["direction"] for p in packets]

    results = []
    three = {"pro_H1": 0, "anti_H1": 1, "orthogonal": 2}
    results.append(run_task(snippet, [three[d] for d in direction], groups, 3,
                            "3-class direction (snippet only)"))
    results.append(run_task(snippet, [0 if d == "orthogonal" else 1 for d in direction],
                            groups, 2, "probative vs orthogonal (snippet only)"))

    sign_idx = [i for i, d in enumerate(direction) if d != "orthogonal"]
    sy = [0 if direction[i] == "pro_H1" else 1 for i in sign_idx]
    sg = [groups[i] for i in sign_idx]
    results.append(run_task([snippet[i] for i in sign_idx], sy, sg, 2,
                            "SIGN: pro vs anti (snippet only)"))
    results.append(run_task([with_q[i] for i in sign_idx], sy, sg, 2,
                            "SIGN: pro vs anti (snippet + market question)"))

    report = {
        "analysis": "exp1_lexical_sign_baseline",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "counterfactual_file": CF_FILE.name,
        "n_packets": len(packets),
        "cross_validation": f"{N_FOLDS}-fold, grouped by market (no market spans folds)",
        "agent_reference": {
            "hosted_unconditional_directional_correctness": "97.3-98.4%",
            "human_panel_exact_direction": "76.2%",
        },
        "results": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")

    print(f"{'task':46s} {'chance':>7s} {'major':>7s} {'NB':>7s} {'LogReg':>7s}")
    for r in results:
        print(f"{r['task']:46s} {r['chance_rate']:7.3f} {r['majority_class_rate']:7.3f} "
              f"{r['naive_bayes_accuracy']:7.3f} {r['logistic_regression_accuracy']:7.3f}")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
