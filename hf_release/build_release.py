#!/usr/bin/env python3
"""Build the anonymous Hugging Face dataset repository used by the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_OUTPUT = HERE / "build"
PARQUET_COMPRESSION = "zstd"
DATASET_PLACEHOLDER = "ANONYMOUS_NAMESPACE/ANONYMOUS_DATASET"
REPO_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def sanitize(value: Any) -> Any:
    """Remove machine-local path prefixes while preserving relative provenance."""
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        value = value.replace(str(ROOT), "${REPOSITORY_ROOT}")
        value = re.sub(
            r"/n/fs/similarity/social_sim(?=/|$)",
            "${REPOSITORY_ROOT}",
            value,
        )
        return value
    return value


def json_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create a stable union schema and JSON-encode irregular nested values."""
    materialized = [sanitize(dict(row)) for row in rows]
    if not materialized:
        raise ValueError("Refusing to write an empty Parquet table")

    keys = sorted({key for row in materialized for key in row})
    kinds: dict[str, set[type]] = {
        key: {type(row[key]) for row in materialized if row.get(key) is not None}
        for key in keys
    }
    scalar_list_columns = {
        key
        for key in keys
        if kinds[key] == {list}
        and all(
            not isinstance(member, (dict, list))
            for candidate in materialized
            for member in (candidate.get(key) or [])
        )
    }

    normalized: list[dict[str, Any]] = []
    for row in materialized:
        output: dict[str, Any] = {}
        for key in keys:
            item = row.get(key)
            observed = kinds[key]
            if item is None:
                output[key] = None
            elif dict in observed:
                output[key] = json_text(item)
            elif list in observed:
                if key in scalar_list_columns:
                    output[key] = item
                else:
                    output[key] = json_text(item)
            elif observed <= {int, float}:
                output[key] = float(item) if float in observed else int(item)
            elif observed == {bool}:
                output[key] = bool(item)
            elif len(observed) > 1:
                output[key] = str(item)
            else:
                output[key] = item
        normalized.append(output)
    return normalized


def write_parquet(rows: Iterable[dict[str, Any]], destination: Path) -> int:
    normalized = _normalize_rows(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(normalized)
    pq.write_table(
        table,
        destination,
        compression=PARQUET_COMPRESSION,
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
    )
    return table.num_rows


def read_arrow(path: Path) -> pa.Table:
    with pa.memory_map(str(path), "r") as source:
        try:
            reader = pa.ipc.open_stream(source)
        except pa.ArrowInvalid:
            reader = pa.ipc.open_file(source)
        return reader.read_all()


def convert_arrow(source: Path, destination: Path) -> int:
    table = read_arrow(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        destination,
        compression=PARQUET_COMPRESSION,
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
    )
    return table.num_rows


def copy_json_sanitized(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = sanitize(read_json(source))
    destination.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_json(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def copy_text(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding="utf-8").replace(str(ROOT), "${REPOSITORY_ROOT}")
    destination.write_text(text, encoding="utf-8")


def build_exp1(output: Path, counts: dict[str, int]) -> None:
    base = ROOT / "exp1_prospective"
    report_path = base / "data/results/consistency_report_2026-06-17.json"
    report = read_json(report_path)

    rows: list[dict[str, Any]] = []
    for model, model_result in report["per_model"].items():
        for market in model_result["per_market"]:
            common = {
                "model": model,
                "task_id": market.get("task_id"),
                "question": market.get("question"),
                "category": market.get("category"),
                "market_price": market.get("market_price"),
                "market_initial_yes_prob": market.get("yes_prob"),
            }
            for packet in market.get("cf_results", []):
                packet_fields = {
                    "cf_id": packet.get("cf_id"),
                    "cf_index": packet.get("cf_index"),
                    "direction": packet.get("direction"),
                    "evidence_headline": packet.get("evidence_headline"),
                    "mechanism_targeted": packet.get("mechanism_targeted"),
                    "slot_type": packet.get("slot_type"),
                }
                for run in packet.get("runs", []):
                    rows.append({**common, **packet_fields, **run})

    counts["exp1_updates"] = write_parquet(rows, output / "data/exp1/updates.parquet")

    material_path = base / "stage3_materials_annotation/generated_v5/public_items.jsonl"
    counts["exp1_review_materials"] = write_parquet(
        read_jsonl(material_path), output / "data/exp1/review_materials.parquet"
    )

    artifacts = output / "artifacts/exp1"
    copy_json_sanitized(report_path, artifacts / report_path.name)
    copy_json_sanitized(
        base / "data/results/threshold_robustness.json",
        artifacts / "threshold_robustness.json",
    )
    agreement_path = (
        base / "stage3_materials_annotation/data/exports/agreement_current.json"
    )
    if agreement_path.exists():
        agreement = read_json(agreement_path)
        human_key = dict(agreement.get("human_key", {}))
        human_key.pop("per_reviewer", None)
        public_agreement = {
            key: agreement[key]
            for key in (
                "n_items",
                "n_reviewers",
                "threshold_pp",
                "human_human",
                "registered_gate",
                "models_full_coverage",
            )
            if key in agreement
        }
        if human_key:
            public_agreement["human_key"] = human_key
        write_json(
            sanitize(public_agreement),
            artifacts / "human_review/agreement_aggregate.json",
        )
    for name in ("PROTOCOL.md", "PROTOCOL_DEVIATION_2026-08-21.md"):
        source = base / "stage3_materials_annotation" / name
        if source.exists():
            copy_text(source, artifacts / "human_review" / name)


def build_exp2(output: Path, counts: dict[str, int]) -> None:
    root = (
        ROOT / "exp2_v2/biased_news/data/coin_city_stable_relationship_claude_n250_v4"
    )
    design = root / "design"
    arm_files = {
        "baseline": design / "tasks_baseline.jsonl",
        "abc_no_context": design / "tasks_abc_no_context.jsonl",
        "abc_context": design / "tasks_abc_context.jsonl",
        "abc_wrong_context": design / "tasks_abc_wrong_context.jsonl",
        "abc_symbol_context": design / "tasks_abc_symbol_context.jsonl",
    }
    answer = {row["task_id"]: row for row in read_jsonl(design / "answer_key.jsonl")}

    task_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    task_rows: list[dict[str, Any]] = []
    for arm, path in arm_files.items():
        for task in read_jsonl(path):
            key = (arm, task["task_id"])
            task_lookup[key] = task
            gold = answer.get(task["task_id"], {})
            task_rows.append(
                {
                    "arm": arm,
                    "task_id": task.get("task_id"),
                    "prompt_sha256": task.get("prompt_sha256"),
                    "prompt": task.get("prompt"),
                    "episode": gold.get("episode"),
                    "c_cases": gold.get("c_cases"),
                    "strong_reference_city": gold.get("strong_reference_city"),
                    "target_strong": gold.get("target_strong"),
                    "cue_correct": gold.get("cue_correct"),
                    "cue_strong": gold.get("cue_strong"),
                    "gold_expected_poll": gold.get("gold_expected_poll"),
                    "baselines_json": json_text(gold.get("baselines")),
                }
            )
    counts["exp2_coin_city_tasks"] = write_parquet(
        task_rows, output / "data/exp2/tasks.parquet"
    )

    response_files = [
        path
        for path in sorted((root / "responses").glob("responses_*.jsonl"))
        if "corrupt" not in path.name
    ]
    matched_dir = root / "responses/symbol_control_matched_20260813"
    response_files.extend(sorted(matched_dir.glob("responses_*.jsonl")))

    response_rows: list[dict[str, Any]] = []
    for path in response_files:
        analysis_set = (
            "symbol_control_matched" if path.parent == matched_dir else "production"
        )
        for response in read_jsonl(path):
            arm = response.get("arm")
            task_id = response.get("task_id")
            task = task_lookup.get((arm, task_id), {})
            gold = answer.get(task_id, {})
            predicted = response.get("predicted_poll")
            target = gold.get("gold_expected_poll")
            response_rows.append(
                {
                    "analysis_set": analysis_set,
                    "source_file": path.name,
                    "model": response.get("model"),
                    "arm": arm,
                    "task_id": task_id,
                    "prompt_sha256": response.get("prompt_sha256"),
                    "prompt": task.get("prompt"),
                    "episode": gold.get("episode"),
                    "c_cases": gold.get("c_cases"),
                    "strong_reference_city": gold.get("strong_reference_city"),
                    "target_strong": gold.get("target_strong"),
                    "cue_correct": gold.get("cue_correct"),
                    "cue_strong": gold.get("cue_strong"),
                    "gold_expected_poll": target,
                    "predicted_poll": predicted,
                    "absolute_error": (
                        abs(float(predicted) - float(target))
                        if predicted is not None and target is not None
                        else None
                    ),
                    "rationale": response.get("rationale"),
                    "raw_response": response.get("raw"),
                    "response_received": response.get("response_received"),
                    "attempts": response.get("attempts"),
                    "error": response.get("error"),
                }
            )
    counts["exp2_coin_city_responses"] = write_parquet(
        response_rows, output / "data/exp2/responses.parquet"
    )

    artifacts = output / "artifacts/exp2"
    for path in (
        design / "manifest.json",
        design / "validation.json",
        root / "analysis/multimodel_results.json",
        root / "analysis/symbol_context_results.json",
    ):
        copy_json_sanitized(path, artifacts / path.name)


def build_arrow_family(
    output: Path,
    counts: dict[str, int],
    count_prefix: str,
    destination_dir: str,
    sources: dict[str, Path],
) -> None:
    for split_name, source in sources.items():
        destination = output / destination_dir / f"{split_name}.parquet"
        counts[f"{count_prefix}_{split_name}"] = convert_arrow(source, destination)


def _score_rows(
    paths: Iterable[Path], relative_to: Path = ROOT
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            source_name = str(path.relative_to(relative_to))
        except ValueError:
            source_name = path.name
        for row in read_jsonl(path):
            row["source_file"] = source_name
            rows.append(row)
    return rows


def _endpoint_score_file(run_dir: Path, stochastic: bool) -> Path:
    if stochastic:
        return run_dir / "stochastic_n5.scores.jsonl"
    candidates = sorted(run_dir.glob("debug_*/eval_results/301.scores.jsonl"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one greedy endpoint score file in {run_dir}")
    return candidates[0]


def build_exp3_coin_city(output: Path, counts: dict[str, int]) -> None:
    root = ROOT / "exp3_training_transfer/coin_city_structural"
    data = root / "data"
    arrow_sources = {
        "train_causal": data / "causal/train/train/data-00000-of-00001.arrow",
        "train_population_prior": data
        / "population_prior/train/train/data-00000-of-00001.arrow",
        "train_structureless": data
        / "structureless/train/train/data-00000-of-00001.arrow",
        "test": data / "causal/heldout/train/data-00000-of-00001.arrow",
    }
    build_arrow_family(
        output,
        counts,
        "exp3_coin_city",
        "data/exp3_coin_city",
        arrow_sources,
    )

    run_names = [
        "causal_qwen3_4b_s42_20260820_231218_j30723979",
        "causal_qwen3_4b_s43_20260820_231219_j30723980",
        "causal_qwen3_4b_s44_20260820_231658_j30723981",
        "population_prior_qwen3_4b_s42_20260821_000228_j30723982",
        "population_prior_qwen3_4b_s43_20260821_000919_j30723983",
        "population_prior_qwen3_4b_s44_20260821_020625_j30723984",
        "structureless_qwen3_4b_s42_20260821_021618_j30723998",
        "structureless_qwen3_4b_s43_20260821_022934_j30723999",
        "structureless_qwen3_4b_s44_20260821_031007_j30724000",
    ]
    runs = [root / "reports" / name for name in run_names]
    score_files = [root / "reports/base_qwen3_4b_j30724001/greedy.scores.jsonl"]
    score_files += [_endpoint_score_file(run, False) for run in runs]
    score_files += [root / "reports/base_qwen3_4b_j30724001/stochastic_n5.scores.jsonl"]
    score_files += [_endpoint_score_file(run, True) for run in runs]
    counts["exp3_coin_city_scores"] = write_parquet(
        _score_rows(score_files), output / "data/exp3_coin_city/scores.parquet"
    )

    artifacts = output / "artifacts/exp3_coin_city"
    for path in (
        data / "build_manifest.json",
        root / "protocol/coin_city_structural_manifest.json",
        root / "reports/qwen3_4b_step300_greedy.json",
        root / "reports/qwen3_4b_step300_stochastic.json",
    ):
        copy_json_sanitized(path, artifacts / path.name)


def _exp4_task_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for row in rows:
        row["price_history_json"] = json_text(row.pop("price_history", None))
    return rows


def build_exp4(output: Path, counts: dict[str, int]) -> None:
    root = ROOT / "exp3_training_transfer/polymarket"
    registered = root / "data/exp3b_registered"
    split_sources = {
        "train": registered / "train.tasks.jsonl",
        "validation": registered / "dev.tasks.jsonl",
        "test": registered / "test.tasks.jsonl",
    }
    for split, source in split_sources.items():
        counts[f"exp4_{split}"] = write_parquet(
            _exp4_task_rows(source), output / f"data/exp4/{split}.parquet"
        )

    locked = root / "reports/exp3b_locked_test_j30505540.jsonl"
    counts["exp4_locked_test_outputs"] = write_parquet(
        read_jsonl(locked), output / "data/exp4/locked_test_outputs.parquet"
    )

    update_root = root / "reports/exp3b_exp1_reapplication_j30535058"
    update_files = [
        update_root / f"{name}.jsonl"
        for name in ("base", "seed_42", "seed_43", "seed_44")
    ]
    counts["exp4_evidence_updates"] = write_parquet(
        _score_rows(update_files), output / "data/exp4/evidence_updates.parquet"
    )

    artifacts = output / "artifacts/exp4"
    for path in (
        registered / "manifest.json",
        root / "protocol/exp3b_preflight.json",
        root / "reports/exp3b_baselines.json",
        root / "reports/exp3b_locked_test_j30505540.summary.json",
        update_root / "summary.json",
    ):
        copy_json_sanitized(path, artifacts / path.name)


def build_mechanism_study(output: Path, counts: dict[str, int]) -> None:
    training = ROOT / "exp3_training_transfer"
    mechanism = training / "mechanism_family"
    mechanism_arms = [
        f"{disclosure}_{arm}"
        for disclosure in ("disclosed", "undisclosed")
        for arm in ("causal_family", "population_prior", "structureless")
    ]
    mechanism_sources: dict[str, Path] = {}
    for arm in mechanism_arms:
        mechanism_sources[f"train_{arm}"] = (
            mechanism / f"data/{arm}/train/train/data-00000-of-00001.arrow"
        )
        mechanism_sources[f"test_{arm}"] = (
            mechanism / f"data/{arm}/heldout/train/data-00000-of-00001.arrow"
        )
    build_arrow_family(
        output,
        counts,
        "appendix_mechanism",
        "data/appendix_mechanism",
        mechanism_sources,
    )

    result_path = mechanism / "reports/c3_completed_qwen3_4b_factorial.json"
    result = read_json(result_path)
    score_paths = [ROOT / relative for relative in result["score_files"]]
    counts["appendix_mechanism_scores"] = write_parquet(
        _score_rows(score_paths), output / "data/appendix_mechanism/scores.parquet"
    )
    for path in (
        mechanism / "data/build_manifest.json",
        mechanism / "protocol/c3_mechanism_manifest.json",
        result_path,
    ):
        copy_json_sanitized(path, output / "artifacts/appendix_mechanism" / path.name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_release_metadata(output: Path, counts: dict[str, int]) -> None:
    exclusions = [
        "model checkpoints and optimizer state",
        "debug logs, caches, PID files, and failed or corrupt pilots",
        "mutable post-freeze daily market snapshots",
        "the upstream 21 GB Polymarket scrape",
        "individual human-review ratings, reviewer IDs, timestamps, and durations",
    ]
    manifest = {
        "release": "anonymous_inductive_forecasting_data_v1",
        "row_counts": dict(sorted(counts.items())),
        "exclusions": exclusions,
        "notes": {
            "paths": "Absolute machine-local paths were replaced or omitted.",
            "third_party": "See LICENSE.md for component-specific source terms.",
        },
    }
    (output / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    checksum_path = output / "SHA256SUMS"
    files = sorted(
        path for path in output.rglob("*") if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in files),
        encoding="utf-8",
    )


def prepare_output(output: Path, force: bool) -> None:
    resolved = output.resolve()
    if (
        resolved == Path("/")
        or resolved == ROOT.resolve()
        or resolved == HERE.resolve()
    ):
        raise ValueError(f"Unsafe output directory: {resolved}")
    if output.exists():
        if not force:
            raise FileExistsError(f"{output} exists; pass --force to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--repo-id",
        help="Neutral namespace/repository-name to embed in the dataset card",
    )
    args = parser.parse_args()
    if args.repo_id and not REPO_ID_PATTERN.fullmatch(args.repo_id):
        parser.error("--repo-id must be namespace/repository-name")

    output = args.output
    prepare_output(output, args.force)
    card = (HERE / "DATASET_CARD.md").read_text(encoding="utf-8")
    if args.repo_id:
        card = card.replace(DATASET_PLACEHOLDER, args.repo_id)
    (output / "README.md").write_text(card, encoding="utf-8")
    shutil.copy2(HERE / "LICENSE.md", output / "LICENSE.md")

    counts: dict[str, int] = {}
    build_exp1(output, counts)
    build_exp2(output, counts)
    build_exp3_coin_city(output, counts)
    build_exp4(output, counts)
    build_mechanism_study(output, counts)
    write_release_metadata(output, counts)

    total_bytes = sum(
        path.stat().st_size for path in output.rglob("*") if path.is_file()
    )
    print(f"Built {output} ({total_bytes / 1024 / 1024:.1f} MiB)")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value:,} rows")
    print(f"Next: {sys.executable} {HERE / 'validate_release.py'} {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
