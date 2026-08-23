#!/usr/bin/env python3
"""Create and upload the validated anonymous dataset repository."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from huggingface_hub import HfApi

from validate_release import validate


DATASET_PLACEHOLDER = "ANONYMOUS_NAMESPACE/ANONYMOUS_DATASET"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Neutral namespace/repository-name")
    parser.add_argument("--release", type=Path, default=Path(__file__).parent / "build")
    parser.add_argument("--private", action="store_true", help="Create a private repo")
    parser.add_argument(
        "--allow-personal-namespace",
        action="store_true",
        help="Override the anonymous-review safeguard (not recommended)",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", args.repo_id):
        parser.error("repo_id must be namespace/repository-name")

    errors = validate(
        args.release,
        allowed_repo_id=args.repo_id if args.allow_personal_namespace else None,
    )
    if errors:
        print("Refusing to upload an invalid release:")
        for error in errors:
            print(f"- {error}")
        return 1

    readme = (args.release / "README.md").read_text(encoding="utf-8")
    if DATASET_PLACEHOLDER in readme or f'"{args.repo_id}"' not in readme:
        print(
            "Refusing to upload a dataset card with a placeholder or mismatched "
            "repository ID. Rebuild with: python hf_release/build_release.py "
            f"--force --repo-id {args.repo_id}"
        )
        return 2

    api = HfApi()
    identity = api.whoami()
    username = identity.get("name") or identity.get("fullname")
    namespace = args.repo_id.split("/", 1)[0]
    if username and namespace.casefold() == str(username).casefold() and not args.allow_personal_namespace:
        print(
            "Refusing to publish an anonymous-review artifact under the logged-in "
            "personal namespace. Use a neutral organization, or pass "
            "--allow-personal-namespace only after review anonymity is no longer needed."
        )
        return 3

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(args.release),
        commit_message="Publish anonymous study data release",
    )
    info = api.dataset_info(args.repo_id)
    print(f"Published and verified: https://huggingface.co/datasets/{info.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
