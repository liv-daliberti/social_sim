#!/usr/bin/env python3
"""Run the frozen scale evaluator with two-way TP for dense Qwen3-32B."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import evaluate_scale_holdout as frozen


MODEL = "Qwen/Qwen3-32B"
MODEL_KEY = "qwen3_32b"
REPORTS = Path(__file__).resolve().parents[1] / "reports"


def require_locked_identity(argv: list[str]) -> None:
    pairs = dict(zip(argv[1::2], argv[2::2]))
    if pairs.get("--model") != MODEL or pairs.get("--model-key") != MODEL_KEY:
        raise SystemExit("Qwen3-32B evaluator received an unregistered model identity")
    if pairs.get("--template") != "auto_no_think":
        raise SystemExit("Qwen3-32B evaluator requires non-thinking decoding")


def resolve_job_adapter(_model_key: str, job_id: str) -> Path:
    roots = sorted(REPORTS.glob(f"qwen32_train_{MODEL_KEY}_market_*_j{job_id}"))
    if len(roots) != 1:
        raise AssertionError(
            f"expected one Qwen3-32B training report for job {job_id}, got {roots}"
        )
    adapters = sorted(roots[0].glob("debug_*/saved_models/step_*"))
    if not adapters:
        raise AssertionError(f"no final Qwen3-32B adapter for job {job_id}")
    adapter = adapters[-1]
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / name).is_file():
            raise AssertionError(f"job {job_id} adapter is missing {name}")
    return adapter.resolve()


def main() -> None:
    require_locked_identity(sys.argv)
    frozen.resolve_job_adapter = resolve_job_adapter
    import vllm

    original_llm = vllm.LLM

    def tensor_parallel_llm(*args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", args[0] if args else None)
        if model != MODEL:
            raise AssertionError(f"unexpected evaluation model: {model}")
        if "tensor_parallel_size" in kwargs:
            raise AssertionError("tensor parallelism was already specified")
        kwargs["tensor_parallel_size"] = 2
        return original_llm(*args, **kwargs)

    vllm.LLM = tensor_parallel_llm
    frozen.main()


if __name__ == "__main__":
    main()
