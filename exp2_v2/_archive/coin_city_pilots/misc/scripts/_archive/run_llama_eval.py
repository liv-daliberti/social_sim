#!/usr/bin/env python3
"""Biased-news eval using a locally-served Llama model (vLLM or Ollama).

Serves the same task as the cloud eval wrappers but routes to a local endpoint.

Default model: meta-llama/Llama-3.1-8B-Instruct
Override with --model for other sizes (70B, 3.2, etc.).

vLLM (default, port 8000):
    vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
    python scripts/run_llama_eval.py dryrun --max-tasks 4

Ollama (port 11434):
    ollama pull llama3.1:8b
    python scripts/run_llama_eval.py dryrun --api-base http://localhost:11434/v1 \\
        --model llama3.1:8b --max-tasks 4

Custom model:
    python scripts/run_llama_eval.py eval --model meta-llama/Llama-3.3-70B-Instruct \\
        --output-dir reports/eval_llama_70b

All other flags are forwarded to run_grpo.py.
"""

import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

DEFAULT_MODEL    = os.getenv("LLAMA_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
DEFAULT_API_BASE = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_API_KEY  = os.getenv("OPENAI_API_KEY", "local-no-key-required")
DEFAULT_OUT      = "reports/eval_llama"


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
