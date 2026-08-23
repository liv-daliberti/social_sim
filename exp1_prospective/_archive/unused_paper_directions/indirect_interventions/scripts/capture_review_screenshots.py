#!/usr/bin/env python3
"""Capture reproducible appendix screenshots from the live review templates."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

# Permit direct execution without requiring an installed repository package.
SCRIPT = Path(__file__).resolve()
REPOSITORY = SCRIPT.parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from flask import render_template, session
from werkzeug.serving import make_server

from exp1_prospective.indirect_interventions.review_app.app import (
    DIRECTIONS,
    create_app,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
DEFAULT_OUTPUT = REPO / "paper/figures"
DEFAULT_ICLR_OUTPUT = REPO / "paper/ICLR/figures"


def capture(firefox: str, url: str, output: Path, width: int, height: int) -> None:
    with tempfile.TemporaryDirectory(prefix="review-firefox-") as profile:
        subprocess.run(
            [
                firefox,
                "--headless",
                "--no-remote",
                "--profile",
                profile,
                "--window-size",
                f"{width},{height}",
                "--screenshot",
                str(output),
                url,
            ],
            check=True,
            timeout=60,
        )
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"Firefox did not create {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--firefox", default=shutil.which("firefox") or "firefox")
    parser.add_argument("--port", type=int, default=5062)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iclr-output-dir", type=Path, default=DEFAULT_ICLR_OUTPUT)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="review-capture-") as temporary:
        app = create_app(
            {
                "TESTING": False,
                "SECRET_KEY": "appendix-screenshot-session",
                "DATABASE": Path(temporary) / "screenshots.sqlite3",
                "ADMIN_TOKEN": "",
            }
        )
        reviewers = app.extensions["reviewers"]
        tasks = app.extensions["review_tasks"]
        reviewer = next(
            row for row in reviewers.values() if row["panel"] == "full_context"
        )
        task_id = reviewer["task_ids"][0]
        task = tasks["full_context"][task_id]

        def render_review_preview() -> str:
            session["reviewer_id"] = reviewer["reviewer_id"]
            return render_template(
                "review.html",
                reviewer=reviewer,
                task=task,
                saved={},
                directions=DIRECTIONS,
                index=0,
                total=len(reviewer["task_ids"]),
                completed=0,
                previous_id=None,
                next_id=reviewer["task_ids"][1],
            )

        @app.get("/_appendix_preview/full-context")
        def full_context_preview():
            return render_review_preview()


        server = make_server("127.0.0.1", args.port, app)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            login = args.output_dir / "exp1_review_site_login.png"
            full = args.output_dir / "exp1_review_site_full_context.png"
            capture(args.firefox, f"http://127.0.0.1:{args.port}/", login, 1400, 760)
            capture(
                args.firefox,
                f"http://127.0.0.1:{args.port}/_appendix_preview/full-context",
                full,
                1400,
                1750,
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)

    args.iclr_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(login, args.iclr_output_dir / login.name)
    shutil.copy2(full, args.iclr_output_dir / full.name)
    print(f"wrote {login}")
    print(f"wrote {full}")
    print(f"copied both screenshots to {args.iclr_output_dir}")


if __name__ == "__main__":
    main()
