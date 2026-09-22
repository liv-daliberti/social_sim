"""Run a frozen group of local diagnostic arms inside one GPU allocation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--group", required=True)
    args = parser.parse_args()
    submission = json.loads(args.submission.read_text())
    root = Path(__file__).resolve().parent
    jobs = [job for job in submission["jobs"] if job["group"] == args.group]
    if not jobs:
        raise ValueError("No registered arms in the requested group")
    failed = []
    for job in jobs:
        for name, expected in submission["code_sha256"].items():
            path = root / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Frozen code changed: {name}")
        for field in ("direction_design", "probability_design"):
            path = Path(submission[field])
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != submission[field + "_sha256"]:
                raise ValueError(f"Frozen input changed: {field}")
        print(f"Starting {job['task']}: {job['model_key']}", flush=True)
        result = subprocess.run(job["inference_command"], check=False)
        if result.returncode:
            failed.append({"model_key": job["model_key"], "stage": "inference",
                           "returncode": result.returncode})
        if Path(job["response_path"]).is_file():
            analysis = subprocess.run(job["analysis_command"], check=False)
            if analysis.returncode:
                failed.append({"model_key": job["model_key"], "stage": "analysis",
                               "returncode": analysis.returncode})
    print(json.dumps({"group": args.group, "process_failures": failed}), flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
