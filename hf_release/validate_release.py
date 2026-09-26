#!/usr/bin/env python3
"""Validate row counts, checksums, anonymity, and secrets in an HF release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq


EXPECTED_MINIMUMS = {
    "exp1_updates": 38_740,
    "exp1_review_materials": 18,
    "exp2_coin_city_tasks": 6_250,
    "exp2_coin_city_responses": 33_750,
    "exp3_coin_city_train_causal": 4_800,
    "exp3_coin_city_train_population_prior": 4_800,
    "exp3_coin_city_train_structureless": 4_800,
    "exp3_coin_city_test": 1_440,
    "exp4_train": 1_736,
    "exp4_validation": 512,
    "exp4_test": 1_024,
    "exp4_locked_test_outputs": 1_024,
}

TEXT_SUFFIXES = {".md", ".json", ".jsonl", ".csv", ".txt", ".yaml", ".yml"}
FORBIDDEN_PATTERNS = {
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Bearer token": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{16,}", re.I),
    "machine-local path": re.compile(
        r"(?<![A-Za-z0-9._~:/?#\[\]@!$&()*+,;=%-])"
        r"(?:/n/fs/|/home/|/Users/)[^\s\"']+"
    ),
    "research account": re.compile(
        r"\bod2961\b|od2961@princeton\.edu|\bliv-daliberti\b", re.I
    ),
    "author email": re.compile(r"manoel@cs\.princeton\.edu", re.I),
    "author name": re.compile(
        r"Liv\s+d['’]Aliberti|Lillio\s+Mok|Kate\s+Sieck|Manoel\s+Horta\s+Ribeiro",
        re.I,
    ),
    "identifying endpoint": re.compile(r"liv\.services\.ai\.azure\.com", re.I),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_text(label: str, text: str, errors: list[str]) -> None:
    for description, pattern in FORBIDDEN_PATTERNS.items():
        match = pattern.search(text)
        if match:
            errors.append(f"{label}: found {description}: {match.group(0)[:80]!r}")


def validate(root: Path, *, allowed_repo_id: str | None = None) -> list[str]:
    errors: list[str] = []
    for required in ("README.md", "LICENSE.md", "release_manifest.json", "SHA256SUMS"):
        if not (root / required).is_file():
            errors.append(f"Missing {required}")

    manifest_path = root / "release_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        counts = manifest.get("row_counts", {})
        for key, minimum in EXPECTED_MINIMUMS.items():
            actual = counts.get(key)
            if actual is None or actual < minimum:
                errors.append(
                    f"{key}: expected at least {minimum:,} rows, got {actual!r}"
                )

    sums_path = root / "SHA256SUMS"
    if sums_path.is_file():
        for line_number, line in enumerate(sums_path.read_text().splitlines(), 1):
            expected, separator, relative = line.partition("  ")
            if not separator:
                errors.append(f"SHA256SUMS:{line_number}: invalid format")
                continue
            path = root / relative
            if not path.is_file():
                errors.append(f"SHA256SUMS:{line_number}: missing {relative}")
            elif sha256(path) != expected:
                errors.append(
                    f"SHA256SUMS:{line_number}: checksum mismatch for {relative}"
                )

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {
            "README.md",
            "LICENSE.md",
            "SHA256SUMS",
        }:
            text = path.read_text(encoding="utf-8", errors="replace")
            # An explicitly authorized personal-namespace upload must place its
            # exact repository ID in the dataset card so the loading examples
            # work. Mask only that full ID, and only in README.md; identity
            # strings anywhere else (including the released data) still fail.
            if relative == "README.md" and allowed_repo_id:
                text = text.replace(
                    allowed_repo_id, "ALLOWED_NAMESPACE/ALLOWED_DATASET"
                )
            scan_text(relative, text, errors)
        elif path.suffix == ".parquet":
            table = pq.read_table(path)
            if table.num_rows == 0:
                errors.append(f"{relative}: empty Parquet table")
            for column_name in table.column_names:
                column = table[column_name]
                if not column.type.equals(column.type):  # pragma: no cover; defensive
                    continue
                if str(column.type) in {"string", "large_string"}:
                    for chunk in column.chunks:
                        for index, value in enumerate(chunk.to_pylist()):
                            if value:
                                scan_text(
                                    f"{relative}:{column_name}:{index}", value, errors
                                )
                                if errors:
                                    return errors

    readme = (
        (root / "README.md").read_text(encoding="utf-8")
        if (root / "README.md").exists()
        else ""
    )
    if "configs:" not in readme or "license: other" not in readme:
        errors.append("README.md is missing dataset configs or license metadata")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument(
        "--allow-repo-id",
        help="Permit only this exact repository ID in README.md",
    )
    args = parser.parse_args()
    errors = validate(args.release, allowed_repo_id=args.allow_repo_id)
    if errors:
        print("Release validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    files = sum(1 for path in args.release.rglob("*") if path.is_file())
    size = sum(
        path.stat().st_size for path in args.release.rglob("*") if path.is_file()
    )
    print(f"Release validation passed: {files} files, {size / 1024 / 1024:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
