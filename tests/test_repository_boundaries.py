"""Keep retired research code outside every maintained execution graph."""

from __future__ import annotations

import ast
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_GIT_ROW_PAYLOADS = {
    # These small files predate the publication boundary on origin/main.
    "exp1_prospective/data/daily_tracking/snapshot_2026-06-05.jsonl",
    "exp1_prospective/data/daily_tracking/snapshot_2026-06-07.jsonl",
    "exp1_prospective/data/daily_tracking/snapshot_2026-06-08.jsonl",
    "exp1_prospective/data/daily_tracking/history.jsonl",
    "exp1_prospective/data/initial_forecasts/forecasts_gpt-5.4_2026-06-11.jsonl",
    "exp1_prospective/data/selected_markets/diverse_2026-06-05.jsonl",
    "exp1_prospective/data/selected_markets/diverse_2026-06-07.jsonl",
    "exp1_prospective/data/updated_forecasts/updated_2026-06-07.jsonl",
    "exp1_prospective/data/updated_forecasts/updated_2026-06-08.jsonl",
    "exp1_prospective/data/updated_forecasts/updated_claude-opus-4-8_2026-06-07.jsonl",
    # This is the one selection consumed by the frozen paper workflow.
    "exp1_prospective/data/selected_markets/diverse_2026-06-09.jsonl",
}
PRUNED_DIRS = {
    ".git",
    ".runtime",
    ".venv",
    "_archive",
    "build",
    "data",
    "logs",
    "reports",
    "results",
    "runs",
}


def active_files(suffix: str):
    for directory, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in PRUNED_DIRS and not name.startswith(".venv")
        ]
        base = Path(directory)
        for name in filenames:
            if name.endswith(suffix):
                yield base / name


def test_active_python_never_imports_archive_modules():
    violations = []
    for path in active_files(".py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any("_archive" in name.split(".") for name in names):
                violations.append(str(path.relative_to(ROOT)))
    assert not violations, f"active Python imports archived modules: {violations}"


def test_operational_entrypoints_never_execute_archive_paths():
    entrypoints = [
        ROOT / "render.yaml",
        ROOT / "render-stage3.yaml",
        ROOT / "scripts/test_active.sh",
        ROOT / "start_viewer.sh",
    ]
    violations = [
        str(path.relative_to(ROOT))
        for path in entrypoints
        if "_archive" in path.read_text(encoding="utf-8")
    ]
    assert not violations, f"operational entrypoints reference archives: {violations}"


def test_manuscript_has_one_physical_source_tree():
    paper = ROOT / "paper"
    compatibility_alias = paper / "ICLR"

    assert (
        compatibility_alias.is_symlink()
    ), "paper/ICLR may only be a temporary compatibility symlink, not a mirror"
    assert compatibility_alias.resolve() == paper.resolve()

    main = (paper / "main.tex").read_text(encoding="utf-8")
    assert r"\author{Anonymous authors" in main
    # The manuscript is two hand-edited sources: the body and the appendix.
    assert r"\input{appendix}" in main
    assert (paper / "appendix.tex").is_file()


def test_active_workflows_write_only_to_the_canonical_paper_tree():
    legacy_path = "paper" + "/ICLR"
    legacy_path_constructor = re.compile(r"/\s*['\"]ICLR['\"]")
    allowed = {
        ROOT / "README.md",
        ROOT / "REPRODUCIBILITY.md",
        ROOT / "paper/README.md",
        ROOT / "paper/tests/test_layout.py",
        ROOT / "scripts/lint_active.sh",
        Path(__file__).resolve(),
    }
    suffixes = (".py", ".sh", ".sbatch", ".yaml", ".yml", ".md")
    violations = [
        str(path.relative_to(ROOT))
        for suffix in suffixes
        for path in active_files(suffix)
        if path not in allowed
        and (
            legacy_path in path.read_text(encoding="utf-8", errors="replace")
            or (
                suffix in (".py", ".sh", ".sbatch", ".yaml", ".yml")
                and legacy_path_constructor.search(
                    path.read_text(encoding="utf-8", errors="replace")
                )
            )
        )
    ]
    assert (
        not violations
    ), f"active workflows still target the legacy paper mirror: {violations}"


def test_every_maintained_test_directory_is_in_the_review_suite():
    runner = (ROOT / "scripts/test_active.sh").read_text(encoding="utf-8")
    compact_runner = " ".join(runner.split())
    missing = []
    for path in active_files(".py"):
        if not path.name.startswith("test_"):
            continue
        relative_directory = path.parent.relative_to(ROOT).as_posix()
        suite_directory = path.parent.parent.relative_to(ROOT).as_posix()
        runs_from_suite_directory = f'"{suite_directory}" tests' in compact_runner
        if relative_directory not in runner and not runs_from_suite_directory:
            missing.append(relative_directory)

    assert not missing, (
        "maintained test directories omitted from test_active.sh: "
        f"{sorted(set(missing))}"
    )


def test_manuscripts_never_include_archive_paths():
    include = re.compile(r"\\(?:input|include|includegraphics)\b[^\n]*_archive")
    violations = []
    for path in active_files(".tex"):
        if include.search(path.read_text(encoding="utf-8")):
            violations.append(str(path.relative_to(ROOT)))
    assert not violations, f"manuscripts include archived files: {violations}"


def test_git_candidate_set_excludes_large_research_payloads():
    """Keep local corpora and generated datasets out of a normal Git add."""
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    candidates = [Path(raw.decode()) for raw in result.stdout.split(b"\0") if raw]

    prohibited = []
    oversized = []
    for relative in candidates:
        path = ROOT / relative
        if not path.is_file():
            # Tracked deletions are part of the commit diff, not its payload.
            continue
        parts = relative.parts
        text = relative.as_posix()
        if text not in ALLOWED_GIT_ROW_PAYLOADS and (
            text.startswith("exp3_training_transfer/polymarket/raw/")
            or text.startswith("exp1_prospective/data/_recovery_audit/")
            or relative.suffix == ".arrow"
            or ("_archive" in parts and "data" in parts[parts.index("_archive") + 1 :])
            or (
                text.startswith("exp1_prospective/data/")
                and relative.suffix == ".jsonl"
                and any(
                    name in parts
                    for name in (
                        "daily_tracking",
                        "initial_forecasts",
                        "raw_markets",
                        "selected_markets",
                        "updated_forecasts",
                    )
                )
            )
        ):
            prohibited.append(text)

        if path.stat().st_size >= 95_000_000:
            oversized.append((text, path.stat().st_size))

    assert (
        not prohibited
    ), f"large local research payloads are Git-eligible: {prohibited[:10]}"
    assert not oversized, f"Git candidates approach GitHub's 100 MB limit: {oversized}"
