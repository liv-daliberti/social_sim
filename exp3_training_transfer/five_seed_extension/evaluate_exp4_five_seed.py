#!/usr/bin/env python3
"""Five-seed wrapper around the frozen Exp4 318-market evaluator."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
POLY_SCRIPTS = REPO / "exp3_training_transfer" / "polymarket" / "scripts"
sys.path.insert(0, str(POLY_SCRIPTS))

import evaluate_scale_holdout as frozen  # noqa: E402


SEEDS = (42, 43, 44, 45, 46)
PROTOCOL_VERSION = "exp3_exp4_five_seed_extension_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_job_adapter(_model_key: str, job_id: str) -> Path:
    reports = REPO / "exp3_training_transfer" / "polymarket" / "reports"
    roots = []
    for path in reports.glob(f"*_j{job_id}"):
        if not path.is_dir():
            continue
        adapters = sorted(path.glob("debug_*/saved_models/step_*"))
        if adapters:
            roots.append((path, adapters[-1]))
    if len(roots) != 1:
        raise AssertionError(
            f"expected one Polymarket training report for job {job_id}, got {roots}"
        )
    adapter = roots[0][1]
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / name).is_file():
            raise AssertionError(f"job {job_id} adapter is missing {name}")
    return adapter.resolve()


def output_prefix(argv: list[str]) -> Path:
    try:
        index = argv.index("--output-prefix")
        return Path(argv[index + 1])
    except (ValueError, IndexError) as error:
        raise SystemExit("--output-prefix is required") from error


def install_tensor_parallelism() -> None:
    size = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
    if size == 1:
        return
    import vllm

    original = vllm.LLM

    def tensor_parallel_llm(*args: Any, **kwargs: Any) -> Any:
        if "tensor_parallel_size" in kwargs:
            raise AssertionError("tensor parallelism specified twice")
        kwargs["tensor_parallel_size"] = size
        return original(*args, **kwargs)

    vllm.LLM = tensor_parallel_llm


def main() -> None:
    prefix = output_prefix(sys.argv)
    frozen.SEEDS = SEEDS
    frozen.resolve_job_adapter = resolve_job_adapter
    install_tensor_parallelism()
    frozen.main()

    summary_path = prefix.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["parent_protocol_version"] = summary["protocol_version"]
    summary["protocol_version"] = PROTOCOL_VERSION
    summary["training_seeds"] = list(SEEDS)
    summary["parent_evaluator_sha256"] = sha256(POLY_SCRIPTS / "evaluate_scale_holdout.py")
    summary["five_seed_wrapper_sha256"] = sha256(Path(__file__).resolve())
    summary["seed_extension_status"] = "post_parent_result_prospective_additional_seeds"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
