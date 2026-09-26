#!/usr/bin/env python3
"""Export review annotations from SQLite to stable JSON and CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DB = ROOT / "data/annotations/pilot_v2_reviews.sqlite3"
DEFAULT_JSON = ROOT / "data/annotations/pilot_v2_annotations.json"
DEFAULT_CSV = ROOT / "data/annotations/pilot_v2_annotations.csv"
FIELDS = (
    "reviewer_id", "panel", "task_id", "direction", "confidence", "started_at",
    "submitted_at", "duration_seconds",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    if not args.db.exists():
        raise SystemExit(f"database not found: {args.db}")

    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT * FROM annotations ORDER BY reviewer_id, submitted_at, task_id"
    ).fetchall()
    connection.close()
    annotations = [{field: row[field] for field in FIELDS} for row in rows]
    payload = {
        "protocol_version": "indirect-pilot-v2",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "annotation_count": len(annotations),
        "annotations": annotations,
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(annotations)
    print(f"exported {len(annotations)} annotations to {args.json} and {args.csv}")


if __name__ == "__main__":
    main()
