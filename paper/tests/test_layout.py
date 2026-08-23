from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


PAPER_DIR = Path(__file__).resolve().parents[1]
ICLR_DIR = PAPER_DIR / "ICLR"

MIRRORED_FIGURES = (
    "exp1_direction_selective_updating.pdf",
    "exp1_direction_selective_updating.png",
    "exp1_ehc_sensitivity_combined.pdf",
    "exp2_coin_city_prompt_arms.pdf",
    "exp2_coin_city_prompt_arms.png",
    "exp2_coin_city_results.pdf",
    "exp2_coin_city_results.png",
)


def test_transfer_grid_is_a_readable_right_wrapped_figure() -> None:
    figure_source = (ICLR_DIR / "fig_exp3_transfer_grid.tex").read_text()
    match = re.search(
        r"\\begin\{wrapfigure\}\{r\}\{([0-9.]+)\\textwidth\}",
        figure_source,
    )
    assert match, "the transfer grid must remain a right-side wrapfigure"
    width = float(match.group(1))
    assert 0.38 <= width <= 0.41, (
        "the transfer grid must remain readable without returning to the oversized layout"
    )

    methods_source = (ICLR_DIR / "methods_section.tex").read_text()
    assert re.search(
        r"\\newpage\s*\\subsubsection\{Coin City: Training",
        methods_source,
    ), "reserve a fresh page before starting the wrapped Figure 2 block"


def test_shared_manuscript_sources_and_generated_figures_stay_synchronized() -> None:
    root_tex = {
        path.relative_to(PAPER_DIR)
        for path in PAPER_DIR.rglob("*.tex")
        if "ICLR" not in path.relative_to(PAPER_DIR).parts
        and "_archive" not in path.relative_to(PAPER_DIR).parts
    }
    iclr_tex = {
        path.relative_to(ICLR_DIR)
        for path in ICLR_DIR.rglob("*.tex")
        if "_archive" not in path.relative_to(ICLR_DIR).parts
    }
    shared_tex = (root_tex & iclr_tex) - {Path("main.tex")}
    drifted = [
        str(relative)
        for relative in sorted(shared_tex)
        if (PAPER_DIR / relative).read_bytes() != (ICLR_DIR / relative).read_bytes()
    ]
    assert not drifted, f"shared manuscript sources drifted: {drifted}"

    figure_drift = [
        name
        for name in MIRRORED_FIGURES
        if (PAPER_DIR / "figures" / name).read_bytes()
        != (ICLR_DIR / "figures" / name).read_bytes()
    ]
    assert not figure_drift, f"generated manuscript figures drifted: {figure_drift}"


@pytest.fixture(scope="session")
def iclr_pages() -> list[str]:
    for executable in ("latexmk", "pdftotext"):
        if shutil.which(executable) is None:
            pytest.fail(f"{executable} is required for paper layout tests")

    build = subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "main.tex",
        ],
        cwd=ICLR_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert build.returncode == 0, build.stdout[-8000:]

    extracted = subprocess.run(
        ["pdftotext", "-layout", "main.pdf", "-"],
        cwd=ICLR_DIR,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    return [" ".join(page.split()) for page in extracted.split("\f")]


def _unique_page(pages: list[str], phrase: str) -> int:
    matches = [index for index, page in enumerate(pages) if phrase in page]
    assert len(matches) == 1, (
        f"expected one page containing {phrase!r}, found {matches}"
    )
    return matches[0]


def test_transfer_grid_is_visible_beside_its_methods_subsection(
    iclr_pages: list[str],
) -> None:
    methods_page = _unique_page(
        iclr_pages, "Training uses the direct, one-period relationship"
    )
    caption_page = _unique_page(iclr_pages, "Transfer grid. Training uses")

    assert caption_page == methods_page, (
        "the transfer grid and its wrapped Experiment 3 methods text must share a page"
    )
    assert "mediator m and persists." in iclr_pages[caption_page], (
        "the transfer grid's complete caption must remain inside the rendered page"
    )
