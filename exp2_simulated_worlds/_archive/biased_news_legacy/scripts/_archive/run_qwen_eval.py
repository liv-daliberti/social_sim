#!/usr/bin/env python3
"""Biased-news eval using a locally-served Qwen model (vLLM or Ollama).

Serves the same task as the cloud eval wrappers but routes to a local endpoint.
The `extra_body={"enable_thinking": False}` is automatically applied for Qwen
models by run_grpo.py.

Default model: Qwen/Qwen3-4B-Instruct-2507 (the GRPO training model).
Override with --model for other Qwen sizes or checkpoints.

vLLM (default, port 8000):
    vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000
    python scripts/run_qwen_eval.py dryrun --max-tasks 4

Ollama (port 11434):
    ollama pull qwen3:4b
    python scripts/run_qwen_eval.py dryrun --api-base http://localhost:11434/v1 \\
        --model qwen3:4b --max-tasks 4

Custom model or checkpoint:
    python scripts/run_qwen_eval.py eval --model Qwen/Qwen3-8B-Instruct \\
        --output-dir reports/eval_qwen3_8b

All other flags are forwarded to run_grpo.py.
"""

import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

DEFAULT_MODEL    = os.getenv("QWEN_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
DEFAULT_API_BASE = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_API_KEY  = os.getenv("OPENAI_API_KEY", "local-no-key-required")
DEFAULT_OUT      = "reports/eval_qwen"


def main() -> None:
    argv  = sys.argv[1:]
    flags = " ".join(argv)

    extra: list[str] = []
    if "--model"      not in flags: extra += ["--model",      DEFAULT_MODEL]
    if "--api-base"   not in flags: extra += ["--api-base",   DEFAULT_API_BASE]
    if "--api-key"    not in flags: extra += ["--api-key",    DEFAULT_API_KEY]
    if "--output-dir" not in flags: extra += ["--output-dir", DEFAULT_OUT]
    # no --azure-auth: local endpoints use standard Bearer (or no auth)

    cmd = [sys.executable, str(_HERE / "run_grpo.py")] + extra + argv
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
