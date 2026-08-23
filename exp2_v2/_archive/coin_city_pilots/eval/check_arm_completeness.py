#!/usr/bin/env python3
"""Report exactly which prompts an arm is missing, and optionally re-run them.

Scoring used to restrict silently to the prompts every model happened to have
answered. That hides two failure modes: a model that died partway through looks
like a smaller sample, and a model that produced *nothing* disappears from the
comparison entirely without a word. Both were live possibilities in this project.

This tool makes the gap explicit and, with --resubmit, fills it. The runners are
resumable and key on task ID, so re-submitting the same job completes the
missing work rather than duplicating what is already there.

Exit status is 0 only when every requested model has answered every requested
prompt, so this can gate an analysis step in a shell pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_TASK_ID = re.compile(r"^(c2v\d+)_(\d{4})_v([0-3])_k(\d)$")
_DEFAULT_MODELS = ("gpt-5.4", "DeepSeek-V4-Pro", "claude-opus-4-8")


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def _selected_task_ids(tasks_path: Path, episodes, prefixes, variants) -> set:
    episodes, prefixes, variants = set(episodes), set(prefixes), set(variants)
    selected = set()
    for line in tasks_path.read_text().splitlines():
        if not line.strip():
            continue
        task_id = json.loads(line)["task_id"]
        match = _TASK_ID.fullmatch(task_id)
        if match is None:
            raise ValueError(f"unrecognised task ID: {task_id!r}")
        _, episode, variant, prefix = match.groups()
        if (
            int(episode) in episodes
            and int(variant) in variants
            and int(prefix) in prefixes
        ):
            selected.add(task_id)
    return selected


def _answered(responses_dir: Path, model: str) -> set:
    path = responses_dir / f"responses_{_slug(model)}.jsonl"
    if not path.exists():
        return set()
    answered = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("predicted_poll") is not None:
            answered.add(record["task_id"])
    return answered


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(_DEFAULT_MODELS))
    parser.add_argument("--episodes", type=int, nargs="+", default=list(range(48)))
    parser.add_argument(
        "--prefixes", type=int, nargs="+", default=[0, 1, 2, 3, 4, 6, 8]
    )
    parser.add_argument("--variants", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--resubmit",
        action="store_true",
        help="sbatch the missing work instead of only reporting it",
    )
    parser.add_argument("--slurm", type=Path, default=None)
    parser.add_argument(
        "--env",
        nargs="*",
        default=[],
        help="extra KEY=VALUE pairs to export to the resubmitted job",
    )
    args = parser.parse_args()

    if args.resubmit and args.slurm is None:
        parser.error("--resubmit requires --slurm")

    selected = _selected_task_ids(
        args.tasks, args.episodes, args.prefixes, args.variants
    )
    print(f"arm: {args.responses}")
    print(f"requested per model: {len(selected)} prompts")
    print()
    print(f"  {'model':20s} {'answered':>9s} {'missing':>8s}  status")
    incomplete = []
    for model in args.models:
        answered = _answered(args.responses, model) & selected
        missing = selected - answered
        if not missing:
            status = "complete"
        elif not answered:
            # The case that used to vanish silently.
            status = "NO RESPONSES AT ALL"
        else:
            status = "partial"
        if missing:
            incomplete.append((model, len(missing)))
        print(
            f"  {model:20s} {len(answered):9d} {len(missing):8d}  {status}"
        )

    if not incomplete:
        print("\nevery requested prompt is answered by every model.")
        return

    print(
        f"\n{len(incomplete)} model(s) incomplete. Scoring this arm now would "
        "drop those prompts from every model, silently shrinking the sample."
    )
    if not args.resubmit:
        print("re-run the missing work with --resubmit --slurm <script>, or")
        print("pass --allow-partial to the analysis to score anyway.")
        sys.exit(1)

    for model, count in incomplete:
        command = ["sbatch"]
        exports = [f"MODEL={model}"] + list(args.env)
        command += ["--export=ALL," + ",".join(exports), str(args.slurm)]
        print(f"\nresubmitting {model} ({count} missing): {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode != 0:
            sys.exit(result.returncode)
    print("\nresubmitted; the runners skip already-answered task IDs.")
    sys.exit(1)


if __name__ == "__main__":
    main()
