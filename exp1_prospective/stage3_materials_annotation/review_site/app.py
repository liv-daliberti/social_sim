#!/usr/bin/env python3
"""Blinded browser interface for Stage 3 materials annotation."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GENERATED = ROOT / "generated_v6"
DEFAULT_PUBLIC = GENERATED / "public_items.jsonl"
DEFAULT_ASSIGNMENTS = GENERATED / "assignments.json"
DEFAULT_DB = ROOT / "data/reviews_v6.sqlite3"
PRACTICE_ID = "self_review"
PRACTICE_CODE = "self"
EXPORT_FIELDS = (
    "reviewer_id",
    "cohort",
    "item_id",
    "consent_confirmed",
    "conditional_direction",
    "direction_confidence",
    "clarity",
    "plausibility",
    "usable_premise",
    "started_at",
    "submitted_at",
    "duration_seconds",
)
DIRECTIONS = (
    ("more_likely", "YES becomes more likely"),
    ("less_likely", "YES becomes less likely"),
    ("no_material_effect", "No material effect on YES"),
    ("ambiguous", "Ambiguous or cannot be determined"),
)
TERNARY = (("yes", "Yes"), ("unclear", "Unclear"), ("no", "No"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def practice_order(item_ids: list[str]) -> list[str]:
    return sorted(
        item_ids,
        key=lambda item_id: hashlib.sha256(
            f"stage3-materials-self-review|{item_id}".encode("utf-8")
        ).hexdigest(),
    )


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("STAGE3_ANNOTATION_SECRET_KEY")
        or secrets.token_hex(32),
        PUBLIC_ITEMS=Path(os.environ.get("STAGE3_ANNOTATION_ITEMS", DEFAULT_PUBLIC)),
        ASSIGNMENTS=Path(
            os.environ.get("STAGE3_ANNOTATION_ASSIGNMENTS", DEFAULT_ASSIGNMENTS)
        ),
        DATABASE=Path(os.environ.get("STAGE3_ANNOTATION_DB", DEFAULT_DB)),
        DATABASE_URL=os.environ.get("DATABASE_URL", ""),
        ADMIN_TOKEN=os.environ.get("STAGE3_ANNOTATION_ADMIN_TOKEN", ""),
        REVIEWER_CODES_JSON=os.environ.get(
            "STAGE3_ANNOTATION_REVIEWER_CODES", ""
        ),
        PRODUCTION=os.environ.get("STAGE3_ANNOTATION_PRODUCTION", "0") == "1",
        ENABLE_PRACTICE=os.environ.get(
            "STAGE3_ANNOTATION_ENABLE_PRACTICE", "1"
        )
        == "1",
        PRACTICE_ONLY=os.environ.get("STAGE3_ANNOTATION_PRACTICE_ONLY", "0") == "1",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get(
            "STAGE3_ANNOTATION_SECURE_COOKIE", "0"
        )
        == "1",
        MAX_CONTENT_LENGTH=16 * 1024,
    )
    if test_config:
        app.config.update(test_config)

    public_rows = read_jsonl(Path(app.config["PUBLIC_ITEMS"]))
    items = {str(row["item_id"]): row for row in public_rows}
    if len(items) != 18:
        raise RuntimeError(f"expected 18 public items, found {len(items)}")
    assignments = load_json(Path(app.config["ASSIGNMENTS"]))
    registered_ids = {
        str(row["reviewer_id"]) for row in assignments["assignments"]
    }
    try:
        configured_codes = json.loads(app.config["REVIEWER_CODES_JSON"] or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("STAGE3_ANNOTATION_REVIEWER_CODES must be valid JSON") from exc
    if not isinstance(configured_codes, dict):
        raise RuntimeError("STAGE3_ANNOTATION_REVIEWER_CODES must be a JSON object")
    configured_codes = {
        str(reviewer_id): str(code).strip()
        for reviewer_id, code in configured_codes.items()
    }
    if configured_codes and set(configured_codes) != registered_ids:
        raise RuntimeError(
            "STAGE3_ANNOTATION_REVIEWER_CODES must contain exactly the frozen "
            f"reviewer IDs: {', '.join(sorted(registered_ids))}"
        )
    if configured_codes and (
        any(len(code) < 12 for code in configured_codes.values())
        or len(set(configured_codes.values())) != len(configured_codes)
        or PRACTICE_CODE in configured_codes.values()
    ):
        raise RuntimeError("reviewer codes must be unique, at least 12 characters, and not 'self'")
    if app.config["PRODUCTION"]:
        if not configured_codes:
            raise RuntimeError("production deployment requires private reviewer codes")
        if not app.config["ADMIN_TOKEN"]:
            raise RuntimeError("production deployment requires an admin token")
        if app.config["ENABLE_PRACTICE"] or app.config["PRACTICE_ONLY"]:
            raise RuntimeError("practice access must be disabled in production")

    reviewers: dict[str, dict[str, Any]] = {}
    for row in assignments["assignments"]:
        reviewer_id = str(row["reviewer_id"])
        reviewers[reviewer_id] = {
            "reviewer_id": reviewer_id,
            "access_code": configured_codes.get(reviewer_id, reviewer_id),
            "cohort": "registered",
            "item_ids": list(row["item_ids_in_order"]),
        }
    if app.config["ENABLE_PRACTICE"]:
        reviewers[PRACTICE_ID] = {
            "reviewer_id": PRACTICE_ID,
            "access_code": PRACTICE_CODE,
            "cohort": "practice",
            "item_ids": practice_order(list(items)),
        }
    expected_reviewer_count = len(registered_ids) + int(app.config["ENABLE_PRACTICE"])
    if len(reviewers) != expected_reviewer_count:
        raise RuntimeError(f"expected {expected_reviewer_count} configured reviewers")
    access_codes = {
        str(row["access_code"]): reviewer_id
        for reviewer_id, row in reviewers.items()
        if not app.config["PRACTICE_ONLY"] or row["cohort"] == "practice"
    }
    app.extensions["annotation_items"] = items
    app.extensions["annotation_reviewers"] = reviewers

    def db_sql(statement: str) -> str:
        if app.config["DATABASE_URL"]:
            return statement.replace("?", "%s")
        return statement

    def connect_db() -> Any:
        database_url = str(app.config["DATABASE_URL"] or "")
        if database_url:
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as exc:
                raise RuntimeError(
                    "psycopg is required when DATABASE_URL is configured"
                ) from exc
            return psycopg.connect(database_url, row_factory=dict_row)
        path = Path(app.config["DATABASE"])
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def get_db() -> Any:
        if "db" not in g:
            g.db = connect_db()
        return g.db

    def execute(statement: str, parameters: tuple[Any, ...] = ()) -> Any:
        return get_db().execute(db_sql(statement), parameters)

    def init_db() -> None:
        statements = (
            """CREATE TABLE IF NOT EXISTS consents (
                reviewer_id TEXT PRIMARY KEY,
                cohort TEXT NOT NULL,
                consented_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS annotations (
                reviewer_id TEXT NOT NULL,
                cohort TEXT NOT NULL,
                item_id TEXT NOT NULL,
                conditional_direction TEXT NOT NULL,
                direction_confidence INTEGER NOT NULL,
                clarity INTEGER NOT NULL,
                plausibility INTEGER NOT NULL,
                usable_premise TEXT NOT NULL,
                started_at TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                PRIMARY KEY (reviewer_id, item_id)
            )""",
            """CREATE INDEX IF NOT EXISTS idx_stage3_annotation_item
                ON annotations (cohort, item_id)""",
        )
        connection = connect_db()
        try:
            for statement in statements:
                connection.execute(statement)
            connection.commit()
        finally:
            connection.close()

    init_db()

    @app.after_request
    def security_headers(response: Response) -> Response:
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; form-action 'self'; frame-ancestors 'none'"
        )
        return response

    @app.teardown_appcontext
    def close_db(_error: BaseException | None) -> None:
        connection = g.pop("db", None)
        if connection is not None:
            connection.close()

    def csrf_token() -> str:
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_urlsafe(24)
        return str(session["csrf_token"])

    app.jinja_env.globals["csrf_token"] = csrf_token

    def valid_csrf() -> bool:
        supplied = request.form.get("csrf_token", "")
        expected = session.get("csrf_token", "")
        return bool(
            supplied
            and expected
            and hmac.compare_digest(str(supplied), str(expected))
        )

    def login_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if session.get("reviewer_id") not in reviewers:
                return redirect(url_for("login"))
            return view(*args, **kwargs)

        return wrapped

    def current_reviewer() -> dict[str, Any]:
        return reviewers[str(session["reviewer_id"])]

    def has_consent(reviewer_id: str) -> bool:
        row = execute(
            "SELECT 1 FROM consents WHERE reviewer_id = ?", (reviewer_id,)
        ).fetchone()
        return row is not None

    def completed_ids(reviewer_id: str) -> set[str]:
        rows = execute(
            "SELECT item_id FROM annotations WHERE reviewer_id = ?", (reviewer_id,)
        ).fetchall()
        return {str(row["item_id"]) for row in rows}

    def saved_annotation(reviewer_id: str, item_id: str) -> dict[str, Any]:
        row = execute(
            "SELECT * FROM annotations WHERE reviewer_id = ? AND item_id = ?",
            (reviewer_id, item_id),
        ).fetchone()
        return dict(row) if row else {}

    def integer_field(name: str, errors: list[str]) -> int | None:
        raw = request.form.get(name, "").strip()
        try:
            value = int(raw)
        except ValueError:
            errors.append(f"{name.replace('_', ' ').capitalize()} is required.")
            return None
        if value not in range(1, 6):
            errors.append(f"{name.replace('_', ' ').capitalize()} must be 1–5.")
            return None
        return value

    def validate_form() -> tuple[dict[str, Any], list[str]]:
        errors: list[str] = []
        direction = request.form.get("conditional_direction", "")
        usable = request.form.get("usable_premise", "")
        if direction not in {value for value, _label in DIRECTIONS}:
            errors.append("Select a conditional direction.")
        if usable not in {value for value, _label in TERNARY}:
            errors.append("Select whether the report is a usable premise.")
        return (
            {
                "conditional_direction": direction,
                "direction_confidence": integer_field(
                    "direction_confidence", errors
                ),
                "clarity": integer_field("clarity", errors),
                "plausibility": integer_field("plausibility", errors),
                "usable_premise": usable,
            },
            errors,
        )

    @app.route("/", methods=("GET", "POST"))
    def login():
        if request.method == "POST":
            if not valid_csrf():
                abort(400, "Invalid form token. Reload and try again.")
            supplied = request.form.get("access_code", "").strip()
            reviewer_id = next(
                (
                    reviewer_id
                    for code, reviewer_id in access_codes.items()
                    if hmac.compare_digest(supplied, code)
                ),
                None,
            )
            if reviewer_id is None:
                flash("That reviewer code was not recognized.", "error")
            else:
                session.clear()
                session["reviewer_id"] = reviewer_id
                session["csrf_token"] = secrets.token_urlsafe(24)
                return redirect(url_for("consent"))
        elif session.get("reviewer_id") in reviewers:
            reviewer = current_reviewer()
            if has_consent(reviewer["reviewer_id"]):
                return redirect(url_for("review_index"))
            return redirect(url_for("consent"))
        return render_template("login.html")

    @app.route("/consent", methods=("GET", "POST"))
    @login_required
    def consent():
        reviewer = current_reviewer()
        if has_consent(reviewer["reviewer_id"]):
            return redirect(url_for("review_index"))
        if request.method == "POST":
            if not valid_csrf():
                abort(400, "Invalid form token. Reload and try again.")
            if request.form.get("adult_consent") != "yes":
                flash("Consent and confirmation that you are at least 18 are required.", "error")
            else:
                execute(
                    "INSERT INTO consents (reviewer_id, cohort, consented_at) "
                    "VALUES (?, ?, ?) ON CONFLICT(reviewer_id) DO UPDATE SET "
                    "cohort=excluded.cohort, consented_at=excluded.consented_at",
                    (reviewer["reviewer_id"], reviewer["cohort"], utc_now()),
                )
                get_db().commit()
                return redirect(url_for("review_index"))
        return render_template("consent.html", reviewer=reviewer)

    @app.get("/review")
    @login_required
    def review_index():
        reviewer = current_reviewer()
        if not has_consent(reviewer["reviewer_id"]):
            return redirect(url_for("consent"))
        done = completed_ids(reviewer["reviewer_id"])
        next_id = next(
            (item_id for item_id in reviewer["item_ids"] if item_id not in done),
            None,
        )
        if next_id is None:
            return render_template(
                "done.html", reviewer=reviewer, item_ids=reviewer["item_ids"]
            )
        return redirect(url_for("review_item", item_id=next_id))

    @app.route("/review/<item_id>", methods=("GET", "POST"))
    @login_required
    def review_item(item_id: str):
        reviewer = current_reviewer()
        if not has_consent(reviewer["reviewer_id"]):
            return redirect(url_for("consent"))
        assigned = reviewer["item_ids"]
        if item_id not in assigned or item_id not in items:
            abort(404)
        index = assigned.index(item_id)
        item = items[item_id]
        started_key = f"started:{item_id}"
        if started_key not in session:
            session[started_key] = {"iso": utc_now(), "clock": time.time()}
        saved = saved_annotation(reviewer["reviewer_id"], item_id)

        if request.method == "POST":
            if not valid_csrf():
                abort(400, "Invalid form token. Reload and try again.")
            values, errors = validate_form()
            if not errors:
                timing = session.get(started_key, {})
                started_at = saved.get("started_at") or timing.get("iso") or utc_now()
                duration = max(
                    0.0, time.time() - float(timing.get("clock", time.time()))
                )
                if saved:
                    duration += float(saved.get("duration_seconds") or 0)
                columns = (
                    "reviewer_id",
                    "cohort",
                    "item_id",
                    *values.keys(),
                    "started_at",
                    "submitted_at",
                    "duration_seconds",
                )
                payload = (
                    reviewer["reviewer_id"],
                    reviewer["cohort"],
                    item_id,
                    *values.values(),
                    started_at,
                    utc_now(),
                    round(duration, 3),
                )
                placeholders = ", ".join("?" for _ in columns)
                updates = ", ".join(
                    f"{column}=excluded.{column}" for column in columns[3:]
                )
                execute(
                    f"INSERT INTO annotations ({', '.join(columns)}) "
                    f"VALUES ({placeholders}) "
                    f"ON CONFLICT(reviewer_id, item_id) DO UPDATE SET {updates}",
                    payload,
                )
                get_db().commit()
                session.pop(started_key, None)
                next_id = next(
                    (
                        candidate
                        for candidate in assigned[index + 1 :] + assigned[: index + 1]
                        if candidate not in completed_ids(reviewer["reviewer_id"])
                    ),
                    None,
                )
                if next_id is None:
                    return redirect(url_for("review_index"))
                return redirect(url_for("review_item", item_id=next_id))
            for error in errors:
                flash(error, "error")
            saved = {**saved, **values}

        done = completed_ids(reviewer["reviewer_id"])
        return render_template(
            "review.html",
            reviewer=reviewer,
            item=item,
            saved=saved,
            directions=DIRECTIONS,
            ternary=TERNARY,
            index=index,
            total=len(assigned),
            completed=len(done),
            previous_id=assigned[index - 1] if index > 0 else None,
            next_id=assigned[index + 1] if index + 1 < len(assigned) else None,
        )

    @app.post("/logout")
    def logout():
        if not valid_csrf():
            abort(400, "Invalid form token.")
        session.clear()
        return redirect(url_for("login"))

    def authorized_admin() -> bool:
        expected = str(app.config.get("ADMIN_TOKEN") or "")
        supplied = request.args.get("token", "")
        return bool(expected and supplied and hmac.compare_digest(expected, supplied))

    def export_csv(cohort: str) -> Response:
        if not authorized_admin():
            abort(403)
        rows = execute(
            "SELECT * FROM annotations WHERE cohort = ? ORDER BY reviewer_id, item_id",
            (cohort,),
        ).fetchall()
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=EXPORT_FIELDS)
        writer.writeheader()
        for raw in rows:
            row = dict(raw)
            writer.writerow(
                {
                    "reviewer_id": row["reviewer_id"],
                    "cohort": row["cohort"],
                    "item_id": row["item_id"],
                    "consent_confirmed": "yes",
                    "conditional_direction": row["conditional_direction"],
                    "direction_confidence": row["direction_confidence"],
                    "clarity": row["clarity"],
                    "plausibility": row["plausibility"],
                    "usable_premise": row["usable_premise"],
                    "started_at": row["started_at"],
                    "submitted_at": row["submitted_at"],
                    "duration_seconds": row["duration_seconds"],
                }
            )
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=stage3_{cohort}_annotations.csv"
            },
        )

    @app.get("/export/registered.csv")
    def export_registered():
        return export_csv("registered")

    @app.get("/export/practice.csv")
    def export_practice():
        return export_csv("practice")

    @app.get("/healthz")
    def health() -> Response:
        execute("SELECT 1").fetchone()
        return jsonify({"status": "ok"})

    @app.get("/admin/status")
    def admin_status() -> Response:
        if not authorized_admin():
            abort(403)
        ratings = {
            row["reviewer_id"]: int(row["count"])
            for row in execute(
                "SELECT reviewer_id, COUNT(*) AS count FROM annotations "
                "WHERE cohort = ? GROUP BY reviewer_id ORDER BY reviewer_id",
                ("registered",),
            ).fetchall()
        }
        consented = [
            str(row["reviewer_id"])
            for row in execute(
                "SELECT reviewer_id FROM consents WHERE cohort = ? ORDER BY reviewer_id",
                ("registered",),
            ).fetchall()
        ]
        return jsonify(
            {
                "status": "ok",
                "consented_reviewers": consented,
                "ratings_by_reviewer": ratings,
                "completed_reviewers": sorted(
                    reviewer_id
                    for reviewer_id, count in ratings.items()
                    if count == 18
                ),
                "expected_reviewers": len(registered_ids),
                "expected_ratings": len(registered_ids) * len(items),
            }
        )

    return app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    app = create_app({"DATABASE": args.database})
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
