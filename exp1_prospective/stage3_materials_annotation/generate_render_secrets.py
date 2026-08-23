#!/usr/bin/env python3
"""Print fresh Render secrets without saving them to the repository."""

from __future__ import annotations

import json
import secrets


def main() -> int:
    codes = {
        f"annotator_{index:02d}": f"s3-{secrets.token_urlsafe(12)}"
        for index in range(1, 8)
    }
    print("STAGE3_ANNOTATION_ADMIN_TOKEN")
    print(secrets.token_urlsafe(32))
    print("\nSTAGE3_ANNOTATION_REVIEWER_CODES")
    print(json.dumps(codes, separators=(",", ":")))
    print("\nREVIEWER HANDOUT")
    for reviewer_id, code in codes.items():
        print(f"{reviewer_id}: {code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
