"""Render both fixed local-diagnostic tables offline, retaining invalid outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIRECTION = (("qwen2_5_72b_instruct", "Qwen2.5-72B"), ("llama3_1_70b_instruct", "Llama-3.1-70B"), ("qwen3_32b", "Qwen3-32B"))
PROBABILITY = (("qwen3_32b_unconstrained_nonthinking", "Unconstrained, thinking disabled"), ("qwen3_32b_thinking", "Unconstrained, thinking enabled"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(source: dict) -> Path:
    path = Path(source["path"])
    if digest(path) != source["sha256"]:
        raise ValueError(f"Stored source hash differs: {path}")
    return path


def count(metric: dict, denominator: int) -> str:
    success = metric["n_success"]
    if (metric["n_planned"] != denominator or metric["denominator"] != denominator
            or type(success) is not int or not 0 <= success <= denominator):
        raise ValueError("Unexpected planned scoring denominator or success count")
    return f"{success}/{denominator}"


def coverage(summary: dict, expected: int) -> str:
    counts = summary["coverage"]["totals"]
    if (summary["n_families"] != 20 or summary["n_planned_units"] != 80
            or summary["n_planned_response_records"] != expected or counts["expected"] != expected
            or counts["present"] != expected or counts["missing_record"] != 0
            or counts["valid"] + counts["invalid_present"] != expected):
        raise ValueError("Expected every planned record from all 20 families and four contexts")
    return f"{counts['valid']}/{expected}"


def table(header: str, rows: list[list[str]]) -> str:
    lines = [r"\begin{tabular}{@{}l" + "c" * (len(rows[0]) - 1) + "@{}}", r"\toprule", header + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""])


def render(completion: dict, reference: dict) -> tuple[str, str]:
    expected_arms = {f"direction:{key}" for key, _ in DIRECTION} | {f"probability:{key}" for key, _ in PROBABILITY}
    if (not completion.get("analysis_complete") or not completion.get("record_complete")
            or completion.get("expected_records") != 1360 or set(completion["arms"]) != expected_arms):
        raise ValueError("All five prescribed arms must be analyzed and record-complete")
    for name, arm in completion["arms"].items():
        if name != f"{arm['task']}:{arm['model_key']}" or arm["metrics"]["model_key"] != arm["model_key"]:
            raise ValueError("Arm identity and analysis model differ")
        if not arm["analysis_succeeded"] or not arm["record_complete"]:
            raise ValueError("Every prescribed arm must be analyzed and record-complete")
    if reference["selection"]["repeat"] != 0 or reference["n_families"] != 20:
        raise ValueError("Original reference must retain all 20 families at repeat zero")
    if completion["designs"]["probability"]["sha256"] != reference["provenance"]["frontier_plan"]["sha256"]:
        raise ValueError("Original and new probability designs differ")
    direction_rows = []
    for key, label in DIRECTION:
        summary = completion["arms"][f"direction:{key}"]["metrics"]
        metrics = summary["companion"]
        direction_rows.append([label, count(metrics["direction_correct"]["pooled"], 40),
                               count(metrics["paired_reversal"]["both_directions_correct"], 20),
                               count(metrics["broken_new_news_unchanged"], 20),
                               count(metrics["controls_unchanged"], 160), coverage(summary, 240)])
    original = reference["models"]["qwen3_32b"]["summary_report"]
    if original["model_key"] != "qwen3_32b":
        raise ValueError("Original reference model must be Qwen3-32B")
    probability_rows = []
    for label, summary in [("Original constrained, thinking disabled", original)] + [
            (label, completion["arms"][f"probability:{key}"]["metrics"]) for key, label in PROBABILITY]:
        metrics = summary["primary"]
        broken = summary["raw_updates"]["broken"]["new_news"]["mean_absolute_pp"]
        if broken["n_planned"] != 20 or broken["n_observed"] + broken["n_missing"] != 20:
            raise ValueError("Broken-link magnitude must retain its planned 20 trials")
        estimate = broken["estimate"]
        if estimate is not None and (not math.isfinite(estimate) or estimate < 0):
            raise ValueError("Invalid broken-link mean absolute movement")
        movement = "---" if estimate is None else f"{estimate:.2f}"
        if broken["n_missing"]:
            movement += f" ({broken['n_observed']}/20 observed)"
        probability_rows.append([label, count(metrics["direction_correct"]["pooled"], 40),
                                 count(metrics["paired_reversal"]["both_directions_correct"], 20), movement, coverage(summary, 320)])
    return (table(r"Model & Correct sign & Paired reversal & Broken unchanged & Controls unchanged & Valid/planned", direction_rows),
            table(r"Qwen3-32B deployment & Correct sign & Paired reversal & Broken $|\Delta p|$ (pp) & Valid/planned", probability_rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", type=Path, default=HERE / "runs/local_diagnostics_v1/completion.json")
    parser.add_argument("--reference", type=Path, default=HERE / "results/frontier_gpt56_pilot_v1/comparison.json")
    parser.add_argument("--output-dir", type=Path, default=HERE.parents[1] / "paper/tables")
    parser.add_argument("--dry-run", action="store_true", help="Validate/render without writing tables")
    args = parser.parse_args()
    if args.dry_run and not args.completion.exists():
        print(f"Waiting for {args.completion}; no tables written.")
        return 0
    try:
        completion, reference = json.loads(args.completion.read_text()), json.loads(args.reference.read_text())
        tables = render(completion, reference)
        for arm in completion["arms"].values():
            path = verify_source({"path": arm["summary_path"], "sha256": arm["summary_sha256"]})
            if json.loads(path.read_text()) != arm["metrics"]:
                raise ValueError("Stored arm metrics differ from their source summary")
        verify_source(reference["provenance"]["frontier_plan"])
        audit = json.loads(verify_source(reference["provenance"]["local_repeat0_audit"]).read_text())
        for field in ("responses", "response_manifest", "original_full_plan"):
            verify_source(audit["models"]["qwen3_32b"]["provenance"][field])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    header = "% Generated by make_diagnostics_paper_tables.py; exploratory diagnostics, all planned failures retained.\n"
    header += f"% completion SHA256 {digest(args.completion)}; reference SHA256 {digest(args.reference)}\n"
    for name, content in zip(("exp1_context_direction.tex", "exp1_context_reasoning.tex"), tables):
        if args.dry_run:
            print(f"% {name}\n{header}{content}")
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            path = args.output_dir / name
            path.write_text(header + content)
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
