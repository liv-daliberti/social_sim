#!/usr/bin/env python3
"""Verify and summarize the completed frontier/local repeat-0 development comparison.

Offline only: reads frozen plans, manifests, responses, and the frontier summary.
Original response records and input hashes are never changed. The original local
three-repeat aggregate is independent of these matched-subset artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOCAL_RUN = HERE / "runs/development_v1"
FRONTIER_RUN = HERE / "runs/frontier_gpt56_pilot_v1"
RESULTS = HERE / "results/frontier_gpt56_pilot_v1"
LOCAL_MODELS = (
    ("qwen2_5_72b_instruct", "Qwen2.5-72B"),
    ("llama3_1_70b_instruct", "Llama-3.1-70B"),
    ("qwen3_32b", "Qwen3-32B"),
)
FRONTIER = ("gpt_5_6_frontier", "GPT-5.6")
MODELS = (*LOCAL_MODELS, FRONTIER)
CONTEXTS = ("positive", "negative", "broken", "masked")
CONDITIONS = ("baseline", "new_news", "no_news", "repeated_news")
CAVEAT = (
    "Exploratory deployment comparison on authored, independently unvalidated development materials. "
    "All 20 original families and all four contexts are retained, using repeat 0 only. "
    "The frontier extension was added after local outcomes. Generation settings differ: local "
    "temperature 0.7 with constrained decoding and 128 output tokens; GPT-5.6 uses low reasoning "
    "with 2048 output tokens and temperature/seed omitted. Qwen3 uses non-thinking mode. "
    "These descriptive results do not isolate model capability, establish a confirmatory ranking, "
    "or support population claims or model selection."
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def source(path: Path) -> dict:
    return {"path": str(path), "sha256": sha(path.read_bytes())}


def recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_analyzer(submission: dict, frontier_manifest: dict):
    path = LOCAL_RUN / "code/analyze.py"
    digest = sha(path.read_bytes())
    require(digest == submission["files_sha256"]["analyze.py"] == frontier_manifest["code_sha256"]["analyze.py"], "Frozen analyzer provenance differs")
    require(path.read_bytes() == (HERE / "analyze.py").read_bytes() == (FRONTIER_RUN / "code/analyze.py").read_bytes(), "Analyzer snapshots differ")
    spec = importlib.util.spec_from_file_location("context_reversal_matched_analyzer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, source(path)


def verify_responses(path: Path, manifest: dict, plan: list[dict], model: str, *, local: bool) -> tuple[list[dict], dict]:
    """Check complete source data against its original plan before any subsetting."""
    config = manifest["config"]
    input_path = recorded_path(config["input_path"])
    input_hash = sha(input_path.read_bytes())
    signature = sha(canonical(config).encode())
    require(read_rows(input_path) == plan, f"{model}: unexpected input plan")
    require(input_hash == config["input_sha256"], f"{model}: original input hash differs")
    require(signature == manifest["run_signature"] and config["model_key"] == model, f"{model}: manifest signature/model differs")
    require(manifest["status"] == "complete", f"{model}: run is incomplete")
    require(config["unit_count"] == len(plan), f"{model}: unit count differs")
    expected = {(unit["trial_id"], condition) for unit in plan for condition in CONDITIONS}
    units = {unit["trial_id"]: unit for unit in plan}
    require(len(units) == len(plan), f"{model}: duplicate planned units")
    rows = read_rows(path)
    indexed = {(row["trial_id"], row["condition"]): row for row in rows}
    require(len(rows) == len(indexed) == manifest["records"] == manifest["expected_records"] == len(expected), f"{model}: response count/uniqueness differs")
    require(set(indexed) == expected, f"{model}: response keys differ from original plan")
    for row in rows:
        unit, condition = units[row["trial_id"]], row["condition"]
        require(row["input_sha256"] == input_hash and row["run_signature"] == signature and row["model_key"] == model, f"{model}: raw response provenance differs")
        require(all(row[key] == unit[key] for key in ("trial_id", "family_id", "domain", "context_id", "repeat", "material_status")), f"{model}: response metadata differs")
        require(row["stage"] == ("baseline" if condition == "baseline" else "update"), f"{model}: invalid response stage")
        require(row["status"] == "ok", f"{model}: nonvalid response")
        value = row["probability"]
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1, f"{model}: invalid probability")
        parsed = json.loads(row["raw"])
        require(type(parsed) is dict and set(parsed) == {"probability"} and type(parsed["probability"]) in (int, float) and parsed["probability"] == value, f"{model}: raw/parsed probability differs")
        baseline = indexed[row["trial_id"], "baseline"]["probability"]
        prior = None if condition == "baseline" else baseline
        prompt = unit["baseline_prompt"] if condition == "baseline" else unit["update_templates"][condition].replace("__PRIOR_PROBABILITY__", json.dumps(baseline, allow_nan=False))
        require(row["prior_probability"] == prior, f"{model}: update prior differs from own baseline")
        require(row["prompt"] == prompt and row["prompt_sha256"] == sha(prompt.encode()), f"{model}: prompt or prompt hash differs")
        if local:
            require(re.fullmatch(config["decode"]["guided_regex"], row["raw"]) is not None, f"{model}: constrained output differs")
            seed_key = canonical([config["decode"]["seed"], row["trial_id"], row["stage"], condition])
            require(row["seed"] == int(sha(seed_key.encode())[:16], 16) % (2**31 - 1), f"{model}: request seed differs")
        else:
            require(row["requested_model"] == row["returned_model"] == config["model"], f"{model}: returned model differs")
            require(row["response_received"] and row["request_attempted"] and row["response_status"] == "completed", f"{model}: response was not completed")
    return rows, {"status": "passed", "original_plan_input_hash_checked_records": len(rows), "run_signature_checked_records": len(rows), "metadata_prompt_own_baseline_raw_parse_checked_records": len(rows), "input_sha256_retained": input_hash, "run_signature": signature}


def manual_metrics(plan: list[dict], rows: list[dict], report: dict) -> dict:
    """Independent arithmetic check, plus pooled control drift for concise reporting."""
    indexed = {(row["trial_id"], row["condition"]): row["probability"] for row in rows}
    samples = {context: {condition: [] for condition in CONDITIONS[1:]} for context in CONTEXTS}
    pairs = {}
    for unit in plan:
        trial, context = unit["trial_id"], unit["context_id"]
        baseline = indexed[trial, "baseline"]
        for condition in CONDITIONS[1:]:
            samples[context][condition].append(100 * (indexed[trial, condition] - baseline))
        pairs.setdefault(unit["family_id"], {})[context] = indexed[trial, "new_news"] - baseline
    sign = sum(x > 0 for x in samples["positive"]["new_news"]) + sum(x < 0 for x in samples["negative"]["new_news"])
    pair = sum(item["positive"] > 0 and item["negative"] < 0 for item in pairs.values())
    require(sign == report["primary"]["direction_correct"]["pooled"]["n_success"], "Independent direction tally differs")
    require(pair == report["primary"]["paired_reversal"]["both_directions_correct"]["n_success"], "Independent paired tally differs")
    for context in CONTEXTS:
        for condition in CONDITIONS[1:]:
            values = samples[context][condition]
            expected = report["raw_updates"][context][condition]
            for name, value in (("signed_mean_pp", sum(values) / len(values)), ("mean_absolute_pp", sum(abs(x) for x in values) / len(values))):
                require(math.isclose(value, expected[name]["estimate"], rel_tol=1e-12, abs_tol=1e-12), "Independent movement tally differs")
    controls = {}
    for condition in ("no_news", "repeated_news"):
        values = [x for context in CONTEXTS for x in samples[context][condition]]
        controls[condition] = {"n_planned": len(values), "n_observed": len(values), "signed_mean_pp": sum(values) / len(values), "mean_absolute_pp": sum(abs(x) for x in values) / len(values), "n_exact_zero": sum(x == 0 for x in values)}
    return {"direction_success": sign, "paired_success": pair, "control_drift_pooled_all_contexts": controls}


def validate_report(report: dict) -> None:
    require(report["material_status"] == "authored_development_unvalidated" and report["analysis_status"] == "development_unvalidated_descriptive_only", "Unexpected analysis status")
    require((report["n_families"], report["n_planned_units"], report["n_planned_response_records"]) == (20, 80, 320), "Unexpected matched cohort size")
    coverage = report["coverage"]["totals"]
    require(coverage["expected"] == coverage["present"] == coverage["valid"] == 320 and coverage["missing_record"] == coverage["invalid_present"] == 0, "Expected 320 valid records per model")
    metrics = (report["primary"]["direction_correct"]["pooled"], report["primary"]["paired_reversal"]["both_directions_correct"], report["raw_updates"]["broken"]["new_news"]["mean_absolute_pp"])
    for metric, denominator in zip(metrics, (40, 20, 20)):
        require(metric["n_planned"] == metric["n_observed"] == denominator and metric["n_missing"] == 0 and metric["ci95"] is not None, "Unexpected scoring denominator or uncertainty")


def collect() -> tuple[dict, dict]:
    submission_path = LOCAL_RUN / "submission.json"
    submission = read_json(submission_path)
    plan_path, plan_manifest_path = FRONTIER_RUN / "plan.jsonl", FRONTIER_RUN / "plan.manifest.json"
    plan, plan_manifest = read_rows(plan_path), read_json(plan_manifest_path)
    plan_hash = sha(plan_path.read_bytes())
    require(plan_hash == plan_manifest["design_sha256"], "Frontier plan hash differs")
    require(len(plan) == 80 and {row["repeat"] for row in plan} == {0}, "Expected 80 repeat-0 units")
    require(sorted({row["family_id"] for row in plan}) == [f"dev_{i:02}" for i in range(1, 21)], "Expected every original family")
    require([job["model_key"] for job in submission["jobs"]] == [key for key, _ in LOCAL_MODELS], "Local prescribed roster differs")
    analyzer, analyzer_source = load_analyzer(submission, plan_manifest)
    analyzer.validate_design(plan)
    selection = {"rule": "All original families and contexts, repeat == 0; no outcome-based selection", "repeat": 0, "family_ids": sorted({row["family_id"] for row in plan}), "contexts": list(CONTEXTS), "trial_ids_in_plan_order": [row["trial_id"] for row in plan], "n_units": 80, "n_records_per_model": 320, "n_direction_trials_per_model": 40, "n_pairs_per_model": 20, "n_broken_trials_per_model": 20}
    audit = {"schema_version": "context_reversal_matched_local_repeat0_v1", "analysis_status": "development_unvalidated_descriptive_only", "caveat": CAVEAT, "provenance": {"submission": source(submission_path), "frontier_matched_plan": source(plan_path), "frontier_plan_manifest": source(plan_manifest_path), "analyzer": analyzer_source, "generator": source(Path(__file__))}, "selection": selection, "models": {}}
    model_data = {}
    for job, (key, label) in zip(submission["jobs"], LOCAL_MODELS):
        path = recorded_path(job["response_path"])
        manifest_path = Path(str(path) + ".manifest.json")
        manifest = read_json(manifest_path)
        config = manifest["config"]
        full_plan_path = recorded_path(config["input_path"])
        full_plan = read_rows(full_plan_path)
        require(len(full_plan) == 240 and {row["repeat"] for row in full_plan} == {0, 1, 2}, f"{key}: expected original three-repeat plan")
        require(sha(full_plan_path.read_bytes()) == submission["design_sha256"] == plan_manifest["source_design_sha256"], f"{key}: full-plan provenance differs")
        require([row for row in full_plan if row["repeat"] == 0] == plan, f"{key}: original repeat-0 rows differ from frontier plan")
        require(config["runner_sha256"] == sha((LOCAL_RUN / "code/run_local.py").read_bytes()) == submission["files_sha256"]["run_local.py"], f"{key}: frozen runner differs")
        require(config["decode"]["temperature"] == 0.7 and config["decode"]["max_tokens"] == 128, f"{key}: documented local settings differ")
        require(key != "qwen3_32b" or config["decode"]["chat_template_mode"] == "qwen-no-thinking", "Qwen3 thinking setting differs")
        rows, verification = verify_responses(path, manifest, full_plan, key, local=True)
        selected = [row for row in rows if row["repeat"] == 0]
        require(len(selected) == 320, f"{key}: selected count differs")
        before = sha(canonical(selected).encode())
        report = analyzer.analyze(plan, selected, key)
        validate_report(report)
        require(sha(canonical(selected).encode()) == before, f"{key}: selected responses modified")
        manual = manual_metrics(plan, selected, report)
        raw_lines = path.read_bytes().splitlines(keepends=True)
        selected_lines = [i for i, line in enumerate(raw_lines, 1) if line.strip() and json.loads(line)["repeat"] == 0]
        verification.update({"exact_matching_plan_rows": 80, "selected_response_records_unchanged": True, "independent_tallies_match": True})
        audit["models"][key] = {"display_name": label, "provenance": {"responses": source(path), "response_manifest": source(manifest_path), "original_full_plan": source(full_plan_path), "original_full_plan_units": 240, "original_response_records": 960, "input_sha256_values_retained": sorted({row["input_sha256"] for row in selected}), "decode": config["decode"], "local_model": config["local_model"]}, "selection": {"source_line_numbers_1_based": selected_lines, "selected_raw_lines_sha256": sha(b"".join(raw_lines[i-1] for i in selected_lines)), "selected_records_canonical_sha256": before, "n_selected_records": len(selected)}, "verification": verification, "compact_metrics": manual, "summary_report": report}
        model_data[key] = {"label": label, "deployment": {"backend": "local_vllm", "decode": config["decode"], "snapshot_revision": config["local_model"]["snapshot_revision"]}, "compact_metrics": manual, "summary_report": report}

    summary_path = RESULTS / "summary.json"
    frontier_report = read_json(summary_path)
    provenance = frontier_report["provenance"]
    require(recorded_path(provenance["design"]["path"]).resolve() == plan_path.resolve() and provenance["design"]["sha256"] == plan_hash, "Frontier summary plan provenance differs")
    require(len(provenance["responses"]) == 1, "Expected one frontier response source")
    response_source = provenance["responses"][0]
    response_path = recorded_path(response_source["path"])
    require(response_path.resolve() == (HERE / "responses/frontier_gpt56_pilot_v1/gpt_5_6_frontier.jsonl").resolve(), "Unexpected frontier response source")
    require(sha(response_path.read_bytes()) == response_source["sha256"], "Frontier summary response hash differs")
    manifest_path = Path(str(response_path) + ".manifest.json")
    manifest = read_json(manifest_path)
    config = manifest["config"]
    require(config["model"] == plan_manifest["resolved_model"] == "gpt-5.6-sol", "Frontier resolved model differs")
    require(config["decode"]["reasoning_effort"] == "low" and config["decode"]["max_output_tokens"] == 2048 and config["decode"]["temperature"] == config["decode"]["seed"] == "omitted", "Documented frontier settings differ")
    for filename, expected_hash in config["code_sha256"].items():
        require(sha((FRONTIER_RUN / "code" / filename).read_bytes()) == expected_hash == plan_manifest["code_sha256"][filename], f"Frontier runner hash differs: {filename}")
    rows, verification = verify_responses(response_path, manifest, plan, FRONTIER[0], local=False)
    require(frontier_report["model_key"] == FRONTIER[0], "Frontier summary model differs")
    bootstrap = frontier_report["bootstrap"]
    require(bootstrap == next(iter(model_data.values()))["summary_report"]["bootstrap"], "Bootstrap settings differ across models")
    recomputed = analyzer.analyze(plan, rows, FRONTIER[0], bootstrap_draws=bootstrap["draws"], seed=bootstrap["seed"])
    require(recomputed == {key: value for key, value in frontier_report.items() if key != "provenance"}, "Saved frontier summary differs from recomputation")
    validate_report(frontier_report)
    manual = manual_metrics(plan, rows, frontier_report)
    verification["saved_summary_equals_recomputed_analysis"] = True
    model_data[FRONTIER[0]] = {"label": FRONTIER[1], "deployment": {"backend": "openai_responses_api", "resolved_model": config["model"], "decode": config["decode"]}, "verification": verification, "compact_metrics": manual, "summary_report": frontier_report}
    comparison = {"schema_version": "context_reversal_frontier_comparison_v1", "material_status": "authored_development_unvalidated", "analysis_status": "development_unvalidated_descriptive_only", "caveat": CAVEAT, "generation_settings_matched": False, "model_order": [key for key, _ in MODELS], "selection": selection, "n_families": 20, "n_planned_units_per_model": 80, "n_planned_response_records_per_model": 320, "bootstrap": bootstrap, "provenance": {"frontier_plan": source(plan_path), "frontier_plan_manifest": source(plan_manifest_path), "frontier_summary": source(summary_path), "frontier_responses": source(response_path), "frontier_response_manifest": source(manifest_path), "analyzer": analyzer_source, "generator": source(Path(__file__))}, "models": model_data}
    return audit, comparison


def interval(metric: dict, scale: float, *, latex: bool = False) -> str:
    low, high = metric["ci95"]
    estimate = metric["estimate"] * scale
    return (f"${estimate:.1f}$ $[{low * scale:.1f},\\,{high * scale:.1f}]$" if latex else f"{estimate:.1f} [{low * scale:.1f}, {high * scale:.1f}]")


def metrics(report: dict) -> tuple[dict, dict, dict]:
    return report["primary"]["direction_correct"]["pooled"], report["primary"]["paired_reversal"]["both_directions_correct"], report["raw_updates"]["broken"]["new_news"]["mean_absolute_pp"]


def render_markdown(comparison: dict) -> str:
    lines = ["# Matched repeat-0 frontier development comparison", "", CAVEAT, "", "Each model has 320/320 valid responses: 20 families × four contexts × one baseline and three update arms. Correct sign uses 40 positive/negative trials; paired reversal uses 20 family pairs; broken-link movement uses 20 trials. Each update uses its own model's context-specific baseline. Zero updates fail directional scoring.", "", "Brackets are descriptive 95% intervals from 2,000 whole-family bootstrap draws (seed 20260921), preserving contexts and update arms. No paired between-model significance test is claimed.", "", "| Model | Correct sign (%) [95% interval] | Paired reversal (%) [95% interval] | Broken mean absolute update (pp) [95% interval] |", "|---|---:|---:|---:|"]
    for key, label in MODELS:
        report = comparison["models"][key]["summary_report"]
        sign, pair, broken = metrics(report)
        lines.append(f"| {label} | {interval(sign, 100)}; {sign['n_success']}/40 | {interval(pair, 100)}; {pair['n_success']}/20 | {interval(broken, 1)}; 20/20 |")
    lines.extend(["", "| Model | Broken stability criterion | No-news mean / mean absolute (pp) | Repeated-news mean / mean absolute (pp) |", "|---|---|---:|---:|"])
    for key, label in MODELS:
        model = comparison["models"][key]
        controls = model["compact_metrics"]["control_drift_pooled_all_contexts"]
        no_news, repeat = controls["no_news"], controls["repeated_news"]
        status = model["summary_report"]["broken_stability"]["raw_new_news"]["status"]
        lines.append(f"| {label} | {status} | {no_news['signed_mean_pp']:.4f} / {no_news['mean_absolute_pp']:.4f} | {repeat['signed_mean_pp']:.4f} / {repeat['mean_absolute_pp']:.4f} |")
    lines.extend(["", "Control drift pools all 80 context units per model. The broken-link criterion requires complete observations, a signed-mean 90% interval strictly within ±2 pp, and the upper 95% mean-absolute bound below 2 pp. Masked contexts have no prespecified direction or null target; context-specific controls and drift-adjusted results remain in comparison.json.", "", "Provenance: local_repeat0_audit.json records exact equality of the original repeat-0 plan rows, source response line numbers and hashes, checks against each model's own baseline, and unchanged original full-plan input hashes. The frontier summary's recorded file hashes are verified and its entire analysis is recomputed before this report is generated. The original three-repeat local aggregate is preserved.", "", "Regenerate offline from the repository root: `python exp1_prospective/context_reversal/summarize_frontier.py`.", ""])
    return "\n".join(lines)


def render_table(comparison: dict, comparison_hash: str) -> str:
    lines = ["% Generated by exp1_prospective/context_reversal/summarize_frontier.py", "% Exploratory unvalidated deployment comparison; generation settings differ.", "% Repeat0: 320 valid records/model; sign n=40, paired n=20, broken n=20; descriptive 95% family intervals.", "% Source comparison.json SHA256 " + comparison_hash, r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Model & Correct sign (\%) & Paired reversal (\%) & Broken $|\Delta p|$ (pp) \\", r"\midrule"]
    for key, label in MODELS:
        report = comparison["models"][key]["summary_report"]
        validate_report(report)
        sign, pair, broken = metrics(report)
        lines.append(" & ".join([label, interval(sign, 100, latex=True), interval(pair, 100, latex=True), interval(broken, 1, latex=True)]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--table-output", type=Path, default=REPO / "paper/tables/exp1_context_reversal_frontier.tex")
    args = parser.parse_args(argv)
    try:
        audit, comparison = collect()
        audit_path = args.output_dir / "local_repeat0_audit.json"
        audit_text = json_text(audit)
        comparison["provenance"]["local_repeat0_audit"] = {"path": str(audit_path.resolve()), "sha256": sha(audit_text.encode())}
        comparison_text = json_text(comparison)
        markdown = render_markdown(comparison)
        table = render_table(comparison, sha(comparison_text.encode()))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    outputs = ((audit_path, audit_text), (args.output_dir / "comparison.json", comparison_text), (args.output_dir / "comparison.md", markdown), (args.table_output, table))
    for path, text in outputs:
        path.write_text(text)
        print(path)
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
