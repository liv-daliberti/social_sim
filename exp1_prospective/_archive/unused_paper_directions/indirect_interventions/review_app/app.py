#!/usr/bin/env python3
"""Blinded human-review app for the indirect-intervention pilot."""

from __future__ import annotations

import argparse
import csv
import hmac
import io
import json
import os
import secrets
import sqlite3
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    Response,
    abort,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_PACKET = ROOT / "data/development/pilot_v2_review_packet.json"
DEFAULT_REVIEWERS = ROOT / "data/development/pilot_v2_reviewers.json"
DEFAULT_DB = ROOT / "data/annotations/pilot_v2_reviews.sqlite3"

DIRECTIONS = (
    ("increase", "Makes YES more likely"),
    ("decrease", "Makes YES less likely"),
    ("no_effect", "Does not affect YES"),
    ("ambiguous", "The connection is unclear"),
)
EXPORT_FIELDS = (
    "reviewer_id",
    "panel",
    "task_id",
    "direction",
    "confidence",
    "started_at",
    "submitted_at",
    "duration_seconds",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
        PACKET_PATH=Path(os.environ.get("INDIRECT_REVIEW_PACKET", DEFAULT_PACKET)),
        REVIEWERS_PATH=Path(os.environ.get("INDIRECT_REVIEWERS", DEFAULT_REVIEWERS)),
        DATABASE=Path(os.environ.get("INDIRECT_REVIEW_DB", DEFAULT_DB)),
        ADMIN_TOKEN=os.environ.get("REVIEW_ADMIN_TOKEN", ""),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    if test_config:
        app.config.update(test_config)

    packet = load_json(Path(app.config["PACKET_PATH"]))
    reviewer_file = load_json(Path(app.config["REVIEWERS_PATH"]))
    tasks = {
        panel: {task["task_id"]: task for task in panel_tasks}
        for panel, panel_tasks in packet["panels"].items()
    }
    reviewers = {row["reviewer_id"]: row for row in reviewer_file["reviewers"]}
    access_codes = {row["access_code"]: row["reviewer_id"] for row in reviewer_file["reviewers"]}
    app.extensions["review_packet"] = packet
    app.extensions["review_tasks"] = tasks
    app.extensions["reviewers"] = reviewers

    def get_db() -> sqlite3.Connection:
        if "db" not in g:
            db_path = Path(app.config["DATABASE"])
            db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(db_path, timeout=30)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=30000")
            g.db = connection
        return g.db

    def init_db() -> None:
        database = Path(app.config["DATABASE"])
        database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database)
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'annotations'"
        ).fetchone()
        if table_exists:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(annotations)")
            }
            if columns != set(EXPORT_FIELDS):
                # Preserve the two requested judgments and timing data when opening
                # a database created by the earlier, longer review form.
                connection.executescript(
                    """
                    ALTER TABLE annotations RENAME TO annotations_long_form;
                    DROP INDEX IF EXISTS idx_annotations_task;
                    CREATE TABLE annotations (
                        reviewer_id TEXT NOT NULL,
                        panel TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        confidence INTEGER NOT NULL,
                        started_at TEXT NOT NULL,
                        submitted_at TEXT NOT NULL,
                        duration_seconds REAL NOT NULL,
                        PRIMARY KEY (reviewer_id, task_id)
                    );
                    INSERT INTO annotations (
                        reviewer_id, panel, task_id, direction, confidence,
                        started_at, submitted_at, duration_seconds
                    )
                    SELECT reviewer_id, panel, task_id, direction, confidence,
                           started_at, submitted_at, duration_seconds
                    FROM annotations_long_form;
                    DROP TABLE annotations_long_form;
                    """
                )
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                reviewer_id TEXT NOT NULL,
                panel TEXT NOT NULL,
                task_id TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                PRIMARY KEY (reviewer_id, task_id)
            );
            CREATE INDEX IF NOT EXISTS idx_annotations_task
                ON annotations (panel, task_id);
            """
        )
        connection.close()

    init_db()

    @app.teardown_appcontext
    def close_db(_error: BaseException | None) -> None:
        connection = g.pop("db", None)
        if connection is not None:
            connection.close()

    def csrf_token() -> str:
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_urlsafe(24)
        return session["csrf_token"]

    app.jinja_env.globals["csrf_token"] = csrf_token

    def valid_csrf() -> bool:
        supplied = request.form.get("csrf_token", "")
        expected = session.get("csrf_token", "")
        return bool(supplied and expected and hmac.compare_digest(supplied, expected))

    def login_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if session.get("reviewer_id") not in reviewers:
                return redirect(url_for("login"))
            return view(*args, **kwargs)

        return wrapped

    def current_reviewer() -> dict[str, Any]:
        return reviewers[session["reviewer_id"]]

    def completed_ids(reviewer_id: str) -> set[str]:
        rows = get_db().execute(
            "SELECT task_id FROM annotations WHERE reviewer_id = ?", (reviewer_id,)
        ).fetchall()
        return {row["task_id"] for row in rows}

    def annotation(reviewer_id: str, task_id: str) -> dict[str, Any]:
        row = get_db().execute(
            "SELECT * FROM annotations WHERE reviewer_id = ? AND task_id = ?",
            (reviewer_id, task_id),
        ).fetchone()
        return dict(row) if row else {}

    def integer_field(name: str, low: int, high: int, errors: list[str]) -> int | None:
        raw = request.form.get(name, "").strip()
        try:
            value = int(raw)
        except ValueError:
            errors.append(f"{name.replace('_', ' ').capitalize()} is required.")
            return None
        if not low <= value <= high:
            errors.append(f"{name.replace('_', ' ').capitalize()} must be {low}–{high}.")
            return None
        return value

    def validate_form() -> tuple[dict[str, Any], list[str]]:
        errors: list[str] = []
        direction = request.form.get("direction", "")
        if direction not in {value for value, _ in DIRECTIONS}:
            errors.append("Select an update direction.")
        confidence = integer_field("confidence", 1, 5, errors)
        values: dict[str, Any] = {
            "direction": direction,
            "confidence": confidence,
        }
        return values, errors

    @app.route("/", methods=("GET", "POST"))
    def login():
        if request.method == "POST":
            if not valid_csrf():
                abort(400, "Invalid form token. Reload and try again.")
            code = request.form.get("access_code", "").strip()
            reviewer_id = next(
                (rid for saved_code, rid in access_codes.items() if hmac.compare_digest(code, saved_code)),
                None,
            )
            if reviewer_id is None:
                flash("That reviewer code was not recognized.", "error")
            else:
                session.clear()
                session["reviewer_id"] = reviewer_id
                session["csrf_token"] = secrets.token_urlsafe(24)
                return redirect(url_for("review_index"))
        elif session.get("reviewer_id") in reviewers:
            return redirect(url_for("review_index"))
        return render_template("login.html")

    @app.get("/review")
    @login_required
    def review_index():
        reviewer = current_reviewer()
        done = completed_ids(reviewer["reviewer_id"])
        next_id = next((task_id for task_id in reviewer["task_ids"] if task_id not in done), None)
        if next_id is None:
            return render_template("done.html", reviewer=reviewer, complete=len(done))
        return redirect(url_for("review_task", task_id=next_id))

    @app.route("/review/<task_id>", methods=("GET", "POST"))
    @login_required
    def review_task(task_id: str):
        reviewer = current_reviewer()
        assigned = reviewer["task_ids"]
        if task_id not in assigned:
            abort(404)
        task = tasks[reviewer["panel"]].get(task_id)
        if task is None:
            abort(404)
        started_key = f"started:{task_id}"
        if started_key not in session:
            session[started_key] = {"iso": utc_now(), "clock": time.time()}
        saved = annotation(reviewer["reviewer_id"], task_id)

        if request.method == "POST":
            if not valid_csrf():
                abort(400, "Invalid form token. Reload and try again.")
            values, errors = validate_form()
            if not errors:
                timing = session.get(started_key, {})
                started_at = saved.get("started_at") or timing.get("iso") or utc_now()
                duration = max(0.0, time.time() - float(timing.get("clock", time.time())))
                if saved:
                    duration += float(saved.get("duration_seconds") or 0)
                columns = (
                    "reviewer_id", "panel", "task_id", *values.keys(),
                    "started_at", "submitted_at", "duration_seconds",
                )
                payload = (
                    reviewer["reviewer_id"], reviewer["panel"], task_id, *values.values(),
                    started_at, utc_now(), round(duration, 3),
                )
                assignments = ", ".join(f"{column}=excluded.{column}" for column in columns[3:])
                placeholders = ", ".join("?" for _ in columns)
                get_db().execute(
                    f"INSERT INTO annotations ({', '.join(columns)}) VALUES ({placeholders}) "
                    f"ON CONFLICT(reviewer_id, task_id) DO UPDATE SET {assignments}",
                    payload,
                )
                get_db().commit()
                session.pop(started_key, None)
                flash("Rating saved.", "success")
                done = completed_ids(reviewer["reviewer_id"])
                next_id = next((item for item in assigned if item not in done), None)
                if next_id is None:
                    return redirect(url_for("review_index"))
                return redirect(url_for("review_task", task_id=next_id))
            for error in errors:
                flash(error, "error")
            saved = {**saved, **values}

        done = completed_ids(reviewer["reviewer_id"])
        index = assigned.index(task_id)
        return render_template(
            "review.html", reviewer=reviewer, task=task, saved=saved,
            directions=DIRECTIONS, index=index,
            total=len(assigned), completed=len(done),
            previous_id=assigned[index - 1] if index else None,
            next_id=assigned[index + 1] if index + 1 < len(assigned) else None,
        )

    @app.post("/logout")
    def logout():
        if not valid_csrf():
            abort(400)
        session.clear()
        return redirect(url_for("login"))

    def require_admin() -> None:
        expected = app.config.get("ADMIN_TOKEN", "")
        supplied = request.args.get("token", "")
        if not expected or not supplied or not hmac.compare_digest(str(expected), supplied):
            abort(404)

    def all_annotations() -> list[dict[str, Any]]:
        rows = get_db().execute(
            "SELECT * FROM annotations ORDER BY reviewer_id, submitted_at, task_id"
        ).fetchall()
        return [{field: row[field] for field in EXPORT_FIELDS} for row in rows]

    @app.get("/admin")
    def admin():
        require_admin()
        counts = {
            row["reviewer_id"]: row["count"]
            for row in get_db().execute(
                "SELECT reviewer_id, COUNT(*) AS count FROM annotations GROUP BY reviewer_id"
            ).fetchall()
        }
        progress = [
            {"reviewer_id": row["reviewer_id"], "panel": row["panel"],
             "completed": counts.get(row["reviewer_id"], 0), "total": len(row["task_ids"])}
            for row in reviewers.values()
        ]
        return render_template("admin.html", progress=progress, token=request.args["token"])

    @app.get("/admin/export.json")
    def export_json():
        require_admin()
        body = {"protocol_version": packet["protocol_version"], "exported_at": utc_now(),
                "annotations": all_annotations()}
        return Response(
            json.dumps(body, indent=2, sort_keys=True) + "\n", mimetype="application/json",
            headers={"Content-Disposition": "attachment; filename=indirect_annotations.json"},
        )

    @app.get("/admin/export.csv")
    def export_csv():
        require_admin()
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=EXPORT_FIELDS)
        writer.writeheader()
        writer.writerows(all_annotations())
        return Response(
            output.getvalue(), mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=indirect_annotations.csv"},
        )

    @app.get("/healthz")
    def health():
        return {"status": "ok", "protocol_version": packet["protocol_version"]}

    return app


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5060)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
