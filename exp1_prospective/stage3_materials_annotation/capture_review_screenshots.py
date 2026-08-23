#!/usr/bin/env python3
"""Capture reproducible appendix screenshots from the final v5 review site."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

SCRIPT = Path(__file__).resolve()
REPOSITORY = SCRIPT.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from flask import render_template, session
from werkzeug.serving import make_server

from exp1_prospective.stage3_materials_annotation.review_site.app import (
    DIRECTIONS,
    TERNARY,
    create_app,
)


HERE = SCRIPT.parent
DEFAULT_OUTPUT = REPOSITORY / "paper/figures"
DEFAULT_ICLR_OUTPUT = REPOSITORY / "paper/ICLR/figures"


def capture(firefox: str, url: str, output: Path, width: int, height: int) -> None:
    with tempfile.TemporaryDirectory(prefix="stage3-review-firefox-") as profile:
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
    parser.add_argument("--port", type=int, default=5063)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iclr-output-dir", type=Path, default=DEFAULT_ICLR_OUTPUT)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="stage3-review-capture-") as temporary:
        app = create_app(
            {
                "TESTING": False,
                "SECRET_KEY": "appendix-screenshot-session",
                "DATABASE": Path(temporary) / "screenshots.sqlite3",
                "DATABASE_URL": "",
                "ADMIN_TOKEN": "",
                "ENABLE_PRACTICE": True,
                "PRACTICE_ONLY": False,
                "SESSION_COOKIE_SECURE": False,
            }
        )
        reviewers = app.extensions["annotation_reviewers"]
        items = app.extensions["annotation_items"]
        reviewer = reviewers["annotator_01"]
        item = next(
            row
            for row in items.values()
            if "China GDP growth in Q2 2026" in row["question"]
        )
        item_id = str(item["item_id"])
        index = reviewer["item_ids"].index(item_id)

        @app.get("/_appendix_preview/consent")
        def consent_preview():
            session["reviewer_id"] = reviewer["reviewer_id"]
            return render_template("consent.html", reviewer=reviewer)

        @app.get("/_appendix_preview/item")
        def item_preview():
            session["reviewer_id"] = reviewer["reviewer_id"]
            return render_template(
                "review.html",
                reviewer=reviewer,
                item=item,
                saved={},
                directions=DIRECTIONS,
                ternary=TERNARY,
                index=index,
                total=len(reviewer["item_ids"]),
                completed=0,
                previous_id=reviewer["item_ids"][index - 1] if index > 0 else None,
                next_id=(
                    reviewer["item_ids"][index + 1]
                    if index + 1 < len(reviewer["item_ids"])
                    else None
                ),
            )

        server = make_server("127.0.0.1", args.port, app)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            paths = {
                "login": args.output_dir / "exp1_stage3_review_login.png",
                "consent": args.output_dir / "exp1_stage3_review_consent.png",
                "item": args.output_dir / "exp1_stage3_review_item.png",
            }
            capture(
                args.firefox,
                f"http://127.0.0.1:{args.port}/",
                paths["login"],
                1200,
                760,
            )
            capture(
                args.firefox,
                f"http://127.0.0.1:{args.port}/_appendix_preview/consent",
                paths["consent"],
                1200,
                1180,
            )
            capture(
                args.firefox,
                f"http://127.0.0.1:{args.port}/_appendix_preview/item",
                paths["item"],
                1200,
                2400,
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)

    args.iclr_output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        shutil.copy2(path, args.iclr_output_dir / path.name)
        print(f"wrote {path}")
    print(f"copied screenshots to {args.iclr_output_dir}")


if __name__ == "__main__":
    main()
