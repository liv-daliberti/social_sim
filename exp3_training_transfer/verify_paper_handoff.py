#!/usr/bin/env python3
"""Prove that every registered campaign entered the canonical paper PDF."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PAPER_ROOTS = (REPO / "paper",)
FAILURES = {
    "BOOT_FAIL", "CANCELLED", "DEADLINE", "FAILED", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
}
CURRENT_TABLES = (
    "exp3c_structural_ood_greedy.tex",
    "exp3c_structural_ood_stochastic.tex",
    "exp3c_structural_ood_overall.tex",
    "exp3c_structural_ood_secondary.tex",
    "exp3b_model_roster_results.tex",
)
COIN_TABLES = (
    "exp3_coin_structural_primary.tex",
    "exp3_coin_structural_cues.tex",
    "exp3_coin_structural_diagnostics.tex",
    "exp3_coin_structural_summary.tex",
)
COIN_MODELS = {"qwen3_4b", "qwen3_8b", "llama3_1_8b"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_state(value: str) -> str:
    return value.upper().split()[0].split("+", 1)[0] if value else "UNKNOWN"


def scheduler_states(job_ids: list[str]) -> dict[str, str]:
    account = subprocess.run([
        "sacct", "-X", "-n", "-P", "-j", ",".join(job_ids),
        "-o", "JobIDRaw,State",
    ], text=True, capture_output=True, check=False)
    if account.returncode:
        raise AssertionError(f"sacct failed: {account.stderr.strip()}")
    states = {}
    for line in account.stdout.splitlines():
        if "|" not in line:
            continue
        job_id, state = line.split("|", 1)
        if "." not in job_id:
            states[job_id] = normalize_state(state)
    queue = subprocess.run([
        "squeue", "-h", "-j", ",".join(job_ids), "-o", "%i|%T",
    ], text=True, capture_output=True, check=False)
    if queue.returncode:
        raise AssertionError(f"squeue failed: {queue.stderr.strip()}")
    for line in queue.stdout.splitlines():
        if "|" in line:
            job_id, state = line.split("|", 1)
            states[job_id] = normalize_state(state)
    return {job_id: states.get(job_id, "UNKNOWN") for job_id in job_ids}


def numeric_ids(rows: list[dict], expected: int, label: str) -> list[str]:
    ids = [str(row.get("job_id", "")) for row in rows]
    if len(ids) != expected or len(set(ids)) != expected or any(not value.isdigit() for value in ids):
        raise AssertionError(f"{label}: expected {expected} distinct numeric IDs")
    return ids


def registered_jobs(mechanism: dict, poly: dict, coin: dict, handoff: dict) -> dict[str, list[str]]:
    current = numeric_ids(mechanism.get("submissions", []), 48, "mechanism roster")
    current += numeric_ids(poly.get("submissions", []), 6, "Polymarket training roster")
    current += numeric_ids(poly.get("locked_evaluations", []), 2, "Polymarket locked tests")
    coin_jobs = numeric_ids(coin.get("canaries", []), 6, "Coin City canaries")
    gate = coin.get("canary_gate", {})
    coin_jobs += numeric_ids([gate], 1, "Coin City gate")
    coin_jobs += numeric_ids(coin.get("full_training", []), 21, "Coin City training roster")
    coin_jobs += numeric_ids(coin.get("base_evaluations", []), 3, "Coin City base roster")
    finalizers = handoff.get("replacement_finalizers") or {}
    paper = [str(finalizers.get("current", "")), str(finalizers.get("coin_city", ""))]
    if len(set(paper)) != 2 or any(not value.isdigit() for value in paper):
        raise AssertionError("handoff must contain two distinct numeric finalizer IDs")
    all_ids = current + coin_jobs + paper
    if len(all_ids) != 89 or len(set(all_ids)) != 89:
        raise AssertionError("campaign handoff must contain exactly 89 distinct jobs")
    return {"current_scientific": current, "coin_city_dag": coin_jobs, "paper": paper}


def verify_hash_map(mapping: dict[str, str], expected: int, label: str) -> None:
    if len(mapping) != expected:
        raise AssertionError(f"{label}: expected {expected} hashed files, found {len(mapping)}")
    for relative, expected_hash in mapping.items():
        path = REPO / relative
        if not path.is_file() or sha256(path) != expected_hash:
            raise AssertionError(f"{label}: missing or hash-mismatched file {relative}")


def verify_coin_estimate_shapes(coin: dict) -> None:
    estimates = coin.get("registered_estimates") or {}
    primary = estimates.get("primary") or []
    transfer = estimates.get("transfer_cells") or []
    cue_by_k = estimates.get("cue_by_k") or []
    cue_overall = estimates.get("cue_overall") or []
    override = estimates.get("evidence_override") or []
    if len(primary) != 6 or {
        (row.get("model"), row.get("decode")) for row in primary
    } != {(model, decode) for model in COIN_MODELS
         for decode in ("greedy", "stochastic")}:
        raise AssertionError("Coin City result lacks the exact six primary estimates")
    if len(transfer) != 12 or {
        (row.get("model"), row.get("domain"), row.get("target_structure"))
        for row in transfer
    } != {(model, domain, structure) for model in COIN_MODELS
         for domain in ("coin_city", "coin_harbor")
         for structure in ("direct_a", "mediated_b")}:
        raise AssertionError("Coin City result lacks the exact 12 transfer-cell estimates")
    if len(cue_by_k) != 12 or {
        (row.get("model"), row.get("k")) for row in cue_by_k
    } != {(model, k) for model in COIN_MODELS for k in (0, 2, 4, 8)}:
        raise AssertionError("Coin City result lacks the exact 12 cue-by-k estimates")
    for label, rows in (("cue-overall", cue_overall), ("override", override)):
        if len(rows) != 3 or {row.get("model") for row in rows} != COIN_MODELS:
            raise AssertionError(f"Coin City result lacks the exact three {label} estimates")

    def numeric_leaves(value):
        if isinstance(value, dict):
            for nested in value.values():
                yield from numeric_leaves(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from numeric_leaves(nested)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield float(value)

    leaves = list(numeric_leaves(estimates))
    if not leaves or any(not math.isfinite(value) for value in leaves):
        raise AssertionError("Coin City registered estimates contain nonfinite values")


def verify_result_manifests(mechanism_ledger: Path, poly_ledger: Path,
                            coin_ledger: Path, handoff: dict) -> None:
    mechanism_path = ROOT / "mechanism_family" / "reports" / "registered_results.json"
    poly_path = ROOT / "polymarket" / "reports" / "exp3b_model_roster_summary.json"
    coin_path = ROOT / "coin_city_structural" / "reports" / "registered_results.json"
    for path in (mechanism_path, poly_path, coin_path):
        if not path.is_file():
            raise AssertionError(f"missing final result manifest {path.relative_to(REPO)}")

    mechanism = load(mechanism_path)
    if mechanism.get("ledger_sha256") != sha256(mechanism_ledger):
        raise AssertionError("mechanism result does not match the effective ledger")
    if int(mechanism.get("row_draws", -1)) != 48 * (720 + 3600):
        raise AssertionError("mechanism result has the wrong row-draw count")
    verify_hash_map(mechanism.get("score_file_sha256", {}), 96, "mechanism scores")

    poly = load(poly_path)
    if poly.get("protocol_version") != "exp3b_three_model_roster_v1":
        raise AssertionError("Polymarket roster result has the wrong protocol")
    if int(poly.get("test_tasks", 0)) != 1024 or len(poly.get("rows", [])) != 3:
        raise AssertionError("Polymarket roster result is not the exact three-model test")
    extension = (poly.get("run_ledgers") or {}).get("extension", {})
    if extension.get("sha256") != sha256(poly_ledger):
        raise AssertionError("Polymarket result does not match the effective extension ledger")
    for item in (poly.get("inputs") or {}).values():
        for key in ("summary", "raw"):
            path = REPO / item[f"{key}_path"]
            if not path.is_file() or sha256(path) != item[f"{key}_sha256"]:
                raise AssertionError(f"Polymarket {key} artifact drifted: {path}")

    coin = load(coin_path)
    if coin.get("protocol") != "coin_city_structural_transfer_v1":
        raise AssertionError("Coin City result has the wrong protocol")
    if coin.get("ledger_sha256") != sha256(coin_ledger):
        raise AssertionError("Coin City result does not match the effective ledger")
    if (int(coin.get("validated_jobs", -1)), int(coin.get("validated_tasks_per_job", -1)),
            int(coin.get("row_draws", -1))) != (24, 1440, 24 * (1440 + 7200)):
        raise AssertionError("Coin City result has the wrong exact-roster dimensions")
    verify_coin_estimate_shapes(coin)
    verify_hash_map(coin.get("score_file_sha256", {}), 48, "Coin City scores")
    analysis = (handoff.get("coin_analysis_extension")
                or handoff.get("coin_analysis_repair") or {})
    renderer = REPO / str(analysis.get("renderer", ""))
    if not renderer.is_file() or sha256(renderer) != analysis.get("renderer_sha256"):
        raise AssertionError("Coin City renderer no longer matches the pinned repair hash")
    if coin.get("analysis_code_sha256") != analysis.get("renderer_sha256"):
        raise AssertionError("Coin City result was not generated by the pinned renderer")


def verify_tables_and_pdfs() -> None:
    tables = CURRENT_TABLES + COIN_TABLES
    newest_table = 0.0
    for name in tables:
        paths = [root / "tables" / name for root in PAPER_ROOTS]
        if any(not path.is_file() or path.stat().st_size == 0 for path in paths):
            raise AssertionError(f"missing or empty generated table {name}")
        if paths[0].read_bytes() != paths[1].read_bytes():
            raise AssertionError(f"paper copies disagree for generated table {name}")
        newest_table = max(newest_table, *(path.stat().st_mtime for path in paths))

    markers = (
        "Coin City structural/domain transfer endpoint",
        "Complete expanded structural-OOD factorial",
        "Matched Experiment 4 model-roster extension",
        "Registered endpoint result",
    )
    for root in PAPER_ROOTS:
        pdf = root / "main.pdf"
        if not pdf.is_file() or pdf.stat().st_mtime < newest_table:
            raise AssertionError(f"paper PDF is absent or older than its tables: {pdf}")
        text = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
        normalized = " ".join(text.split())
        for marker in markers:
            if marker not in normalized:
                raise AssertionError(f"{pdf}: rendered marker missing: {marker}")


def verify_finalizer_logs(paper_ids: list[str]) -> None:
    paths = (
        ROOT / "logs" / f"finalize_current_{paper_ids[0]}.out",
        ROOT / "coin_city_structural" / "logs" / f"finalize_{paper_ids[1]}.out",
    )
    markers = (
        "Current structural-OOD and Polymarket-roster tables rendered; ICLR PDF compiled",
        "Coin City tables rendered and the ICLR paper compiled",
    )
    for path, marker in zip(paths, markers):
        if not path.is_file() or marker not in path.read_text(encoding="utf-8", errors="replace"):
            raise AssertionError(f"finalizer success marker missing: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-ledger", type=Path, required=True)
    parser.add_argument("--polymarket-ledger", type=Path, required=True)
    parser.add_argument("--coin-ledger", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    paths = [path.resolve() for path in (
        args.mechanism_ledger, args.polymarket_ledger, args.coin_ledger, args.handoff
    )]
    mechanism, poly, coin, handoff = map(load, paths)
    jobs = registered_jobs(mechanism, poly, coin, handoff)
    all_ids = jobs["current_scientific"] + jobs["coin_city_dag"] + jobs["paper"]
    states = scheduler_states(all_ids)
    counts = Counter(states.values())
    failed = {job_id: state for job_id, state in states.items() if state in FAILURES}
    if failed:
        raise SystemExit(f"campaign contains failed jobs: {failed}")
    incomplete = {job_id: state for job_id, state in states.items() if state != "COMPLETED"}
    if incomplete:
        print("scheduler state: " + ", ".join(
            f"{count} {state.lower()}" for state, count in sorted(counts.items())
        ))
        if args.allow_incomplete:
            print(f"handoff is healthy but incomplete: {len(incomplete)}/89 jobs remain")
            return
        raise SystemExit(f"campaign is incomplete: {incomplete}")

    verify_result_manifests(paths[0], paths[1], paths[2], handoff)
    verify_tables_and_pdfs()
    verify_finalizer_logs(jobs["paper"])
    print("PASS: 87 scientific/gate jobs and two finalizers completed; all result hashes, "
          "nine synchronized tables, and both rendered PDFs verified")


if __name__ == "__main__":
    main()
