from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest


PAPER_DIR = Path(__file__).resolve().parents[1]


def test_figure_five_a_is_modestly_larger_without_stacking() -> None:
    source = (PAPER_DIR / "experiment1_section.tex").read_text()
    assert r"\begin{subfigure}[c]{0.525\textwidth}" in source
    assert r"\end{subfigure}\hspace{0.01\textwidth}" in source
    assert r"\begin{subfigure}[c]{0.455\textwidth}" in source


def test_main_result_figures_keep_local_post_caption_clearance() -> None:
    experiment2 = (PAPER_DIR / "experiment2_section.tex").read_text()
    experiment3_figure = (PAPER_DIR / "main.tex").read_text()
    assert "\\label{fig:coin-city-results}\n\\vspace{10pt}\n\\end{figure}" in experiment2
    assert "\\begin{wrapfigure}{r}{0.35\\textwidth}\n\\vspace{0pt}" in experiment2
    assert "\\label{fig:exp3-transfer-results}\n\\vspace{10pt}\n\\end{figure}" in experiment3_figure


def test_coin_city_prompt_figure_follows_methods_experiment_one() -> None:
    source = (PAPER_DIR / "methods_section.tex").read_text()
    exp1 = source.index(r"\subsection{Experiment 1: Evidence-Selective Updating}")
    figure = source.index(r"\label{fig:coin-city-prompts}")
    exp2 = source.index(
        r"\subsection{Experiment 2: Context-Guided Relationship Selection}"
    )
    assert exp1 < figure < exp2


def test_canonical_manuscript_is_the_anonymous_iclr_submission() -> None:
    source = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")

    assert r"\usepackage{iclr2027_conference,times}" in source
    assert r"\author{Anonymous authors" in source
    assert r"\input{appendix}" in source
    assert r"\subsection*{AI use statement}" in (
        PAPER_DIR / "main.tex"
    ).read_text(encoding="utf-8")
    assert r"\bibliography{references}" in source
    assert "@princeton.edu" not in source
    assert "ICLR/" not in source

    compatibility_alias = PAPER_DIR / "ICLR"
    assert compatibility_alias.is_symlink()
    assert compatibility_alias.resolve() == PAPER_DIR.resolve()


def test_submission_copy_is_final_form() -> None:
    main = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")
    frontmatter = main
    methods = (PAPER_DIR / "methods_section.tex").read_text(encoding="utf-8")

    title = re.search(r"\\title\{(.*?)\}", main, flags=re.DOTALL)
    assert title
    assert title.group(1).count(r"\\") == 1, "the title must remain on two lines"

    abstract = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        frontmatter,
        flags=re.DOTALL,
    )
    assert abstract
    abstract_without_footnote = re.sub(
        r"\\footnote\{.*?^\}\.",
        " ",
        abstract.group(1),
        flags=re.DOTALL | re.MULTILINE,
    )
    abstract_words = re.findall(
        r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", abstract_without_footnote
    )
    assert len(abstract_words) <= 155, f"abstract has {len(abstract_words)} words"

    assert (
        r"\includegraphics[width=\linewidth]{figures/exp1_protocol_four_panels.pdf}"
        in methods
    ), "Figure 4 must use the physically cropped four-panel asset"
    assert r"\subsubsection" not in methods
    assert "Stage 1:" not in methods
    assert "Stage 2:" not in methods
    assert "Stage 3:" not in methods
    for heading in (
        r"\subsection{Experiment 1: Evidence-Selective Updating}",
        r"\subsection{Experiment 2: Context-Guided Relationship Selection}",
        r"\subsection{Experiment 3: Controlled Relationship Transfer}",
        r"\subsection{Experiment 4: Historical-Market Training and Sealed Evaluation}",
    ):
        assert heading in methods
    figure_text = subprocess.run(
        [
            "pdftotext",
            "-raw",
            str(PAPER_DIR / "figures/exp1_protocol_four_panels.pdf"),
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "Three packets" not in figure_text
    assert all(
        figure_text.count(heading) == 1
        for heading in (
            "The question",
            "Agent event model",
            "Manufacture evidence",
            "Coherent update",
        )
    ), "Figure 4 headings must appear exactly once in the raw text stream"

    figure_overlay = (
        PAPER_DIR / "figures/exp1_protocol_four_panels_overlay.tex"
    ).read_text(encoding="utf-8")
    assert "(176.16,-83.53) rectangle (328.90,66.67)" in figure_overlay
    assert "(176.16,-83.53) rectangle (337.80,66.67)" not in figure_overlay

    active_source = "\n".join(
        path.read_text(encoding="utf-8") for path in PAPER_DIR.glob("*.tex")
    )
    provisional = re.compile(
        r"\b(?:interim|provisional|pending|still running|still completing|"
        r"incomplete at the analysis freeze|results? to come|live campaign)\b",
        flags=re.IGNORECASE,
    )
    assert not provisional.search(
        active_source
    ), "submission copy contains provisional status language"

    banned_tics = re.compile(
        r"\b(?:simple|simply|merely|basic|sharper|ladder|load-bearing)\b",
        flags=re.IGNORECASE,
    )
    assert not banned_tics.search(
        active_source
    ), "submission copy contains disallowed vague or AI-styled wording"

    misnamed_training = re.compile(
        r"\bcausal (?:training|reward|arm|policy|supervision)\b",
        flags=re.IGNORECASE,
    )
    assert not misnamed_training.search(
        active_source
    ), "correct-label supervision must be called episode-matched, not causal"


def test_every_declared_latex_input_exists() -> None:
    include = re.compile(r"\\(?:input|include)\{([^}]+)\}")
    missing = []
    seen = set()
    pending = ["main.tex"]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        source = PAPER_DIR / name
        # Comments mention \input paths without declaring them.
        body = re.sub(
            r"(?<!\\\\)%.*", "", source.read_text(encoding="utf-8")
        )
        for relative in include.findall(body):
            target = PAPER_DIR / relative
            if not target.suffix:
                target = target.with_suffix(".tex")
            if not target.is_file():
                missing.append((name, relative))
            else:
                pending.append(str(target.relative_to(PAPER_DIR)))

    assert not missing, f"missing LaTeX inputs: {missing}"
    # The manuscript is exactly two hand-edited sources plus generated tables.
    assert seen == {"main.tex", "appendix.tex"} | {
        name for name in seen if name.startswith("tables/")
    }


def test_main_experiments_link_to_their_appendices() -> None:
    methods = (PAPER_DIR / "methods_section.tex").read_text()
    assert r"Appendix~\ref{app:coin-city}" in methods

    expected = {
        "experiment1_section.tex": "app:exp1",
        "experiment2_section.tex": "app:coin-city",
        "experiment3_section.tex": "app:exp3",
    }
    for filename, label in expected.items():
        source = (PAPER_DIR / filename).read_text()
        assert rf"Appendix~\ref{{{label}}}" in source
    assert (
        r"Appendix~\ref{app:exp4}"
        in (PAPER_DIR / "experiment3_section.tex").read_text()
    )


def test_complete_coin_city_roster_is_reported() -> None:
    appendix = (PAPER_DIR / "appendix.tex").read_text()

    main = (PAPER_DIR / "experiment3_section.tex").read_text()
    table = (PAPER_DIR / "tables/exp3_coin_structural_primary.tex").read_text()

    assert "for 18 runs" in appendix
    assert r"\input{tables/exp3_coin_structural_primary}" in appendix
    assert "Qwen3-8B at round 300 under five-draw stochastic" in main
    assert "The Llama comparison" in main
    assert "spans Llama-3.1-8B" in main
    assert "Llama-3.1-70B scale check" in main
    assert "same task at sufficient scale" in main
    assert "Qwen3-4B" in table
    assert "Llama-3.1-8B" in table
    assert "Qwen3-8B" not in table
    assert "Greedy" not in table
    assert "greedy decoding" not in appendix.lower()
    assert "Llama nevertheless uses the qualitative" in appendix
    assert "Exploratory Llama-3.1-70B scale check" in appendix
    assert "roughly one tenth the direct-response error of 8B" in appendix
    assert "episode-conditioned structural" in appendix
    assert "transfer remains unresolved" in appendix
    assert "registered factorial estimates" in appendix
    assert "conclusive evidence" not in appendix
    assert "sensitivity to the supplied" in appendix
    assert "mechanism cue" in appendix
    assert "0.42 [-0.12,0.96]" in table


def test_polymarket_reporting_hierarchy_uses_main_scale_figure() -> None:
    main = (PAPER_DIR / "experiment3_section.tex").read_text()
    appendix = (PAPER_DIR / "appendix.tex").read_text()
    original_table = (
        PAPER_DIR / "tables/exp3b_qwen3_8b_endpoint_results.tex"
    ).read_text()
    architecture_table = (
        PAPER_DIR / "tables/exp3b_model_roster_results.tex"
    ).read_text()
    scale_table = (PAPER_DIR / "tables/exp4_qwen_scale_results.tex").read_text()
    normalized_main = " ".join(main.split())

    assert (
        r"\includegraphics[width=\linewidth]{figures/exp4_qwen_scale_results.pdf}"
        in main
    )
    assert r"\input{tables/exp3b_qwen3_8b_endpoint_results}" not in main
    for checkpoint in (
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3-14B",
        "Qwen3-32B",
    ):
        assert checkpoint in scale_table
    for checkpoint in ("Llama-3.2-3B", "Llama-3.1-8B"):
        assert checkpoint in scale_table
    assert "Qwen3 at 1.7B, 4B, 8B, 14B, and 32B" in normalized_main
    assert "Llama 3B/8B replication shows the same qualitative" in normalized_main
    assert "on a logarithmic parameter axis" in normalized_main
    assert "categorical" in main
    assert r"\input{tables/exp3b_qwen3_8b_endpoint_results}" in appendix
    assert "Untrained Qwen3-8B" in original_table
    assert r"\input{tables/exp3b_model_roster_results}" in appendix
    assert "Qwen3-4B" in architecture_table
    assert "Llama-3.1-8B" in architecture_table
    assert "Qwen3-8B" not in architecture_table
    assert r"\input{tables/exp4_qwen_scale_results}" in appendix
    assert (
        r"\includegraphics[width=\linewidth]{figures/exp4_llama_scale_results.pdf}"
        in appendix
    )
    assert r"\label{fig:exp4-llama-scale-results}" in appendix
    assert "five-draw non-thinking stochastic decoding" in " ".join(main.split())
    assert "contemporaneous Polymarket" in normalized_main
    assert "contemporaneous Polymarket users" not in normalized_main
    assert "train-only Platt" not in main
    assert "train-only Platt" not in appendix


def test_iclr_2027_disclosures_remain_complete_and_anonymous() -> None:
    statement = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")
    normalized_statement = " ".join(statement.split())
    required_ai_disclosures = (
        r"\subsection*{AI use statement}",
        "produced the forecasts and experimental responses",
        "assisted with software and manuscript preparation",
        "verified numerical claims",
        "checked citations against original sources",
        "take responsibility",
    )
    missing = [
        phrase
        for phrase in required_ai_disclosures
        if phrase not in normalized_statement
    ]
    assert not missing, f"AI-use statement is missing required disclosures: {missing}"
    assert "research ideation" not in statement

    acknowledgments = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")
    assert r"\subsubsection*{Acknowledgments}" in acknowledgments
    assert "redacted for double-blind review" in acknowledgments

    for roster_entry in (
        "Qwen3-8B, Qwen3-4B, and Llama-3.1-8B",
        "common stochastic decoder",
        "family-disjoint historical markets",
    ):
        assert roster_entry in normalized_statement


def test_submission_archive_is_derived_from_compiled_inputs(tmp_path: Path) -> None:
    output = tmp_path / "submission.zip"
    build = subprocess.run(
        ["./make_submission_zip.sh", str(output)],
        cwd=PAPER_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert build.returncode == 0, build.stdout[-8000:]

    manifest = {
        line.strip()
        for line in (PAPER_DIR / ".archive_manifest").read_text().splitlines()
        if line.strip()
    }
    with zipfile.ZipFile(output) as archive:
        members = set(archive.namelist())

    assert members == manifest
    assert {
        "main.tex",
        "main.bbl",
        "references.bib",
        "appendix.tex",
        "tables/exp2_symbol_context_results.tex",
        "tables/exp2_symbol_context_scaling.tex",
        "tables/exp3_coin_qwen3_8b_stochastic_data.tex",
        "tables/exp3_coin_structural_primary.tex",
        "tables/exp3b_qwen3_8b_endpoint_results.tex",
        "tables/exp3b_model_roster_results.tex",
    } <= members
    assert not any("_archive" in Path(name).parts for name in members)
    assert not any(
        name in members for name in ("main.aux", "main.log", "main.fls", "main.pdf")
    )


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
        cwd=PAPER_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert build.returncode == 0, build.stdout[-8000:]

    extracted = subprocess.run(
        ["pdftotext", "-layout", "main.pdf", "-"],
        cwd=PAPER_DIR,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    return [" ".join(page.split()) for page in extracted.split("\f")]


def _unique_page(pages: list[str], phrase: str) -> int:
    matches = [index for index, page in enumerate(pages) if phrase in page]
    assert (
        len(matches) == 1
    ), f"expected one page containing {phrase!r}, found {matches}"
    return matches[0]


def test_artifact_links_remain_visible_on_the_first_page(
    iclr_pages: list[str],
) -> None:
    frontmatter = (PAPER_DIR / "main.tex").read_text()
    assert "https://huggingface.co/datasets/anonymous-review/" in frontmatter
    assert "https://github.com/anonymous-review/" in frontmatter
    assert "Study dataset and artifacts" in iclr_pages[0]
    assert "Code repository" in iclr_pages[0]
    assert "Links anonymized for review" in iclr_pages[0]
    for required in (
        r"\micon{huggingface.png}",
        r"\micon{gh_logo.png}",
        "https://huggingface.co/datasets/anonymous-review/forecasting-study-data",
        "https://github.com/anonymous-review/forecasting-study-code",
    ):
        assert required in frontmatter
    assert all("Code repository" not in page for page in iclr_pages[1:])


def test_transfer_grid_is_visible_beside_its_methods_subsection(
    iclr_pages: list[str],
) -> None:
    # The grid wraps into the opening Experiment 3 methods paragraph.
    methods_page = _unique_page(
        iclr_pages, "Why matching matters."
    )
    caption_page = _unique_page(iclr_pages, "Transfer grid. Training uses")

    assert (
        caption_page == methods_page
    ), "the transfer grid and its wrapped Experiment 3 methods text must share a page"
    assert (
        "four are evaluated." in iclr_pages[caption_page]
    ), "the transfer grid's complete caption must remain inside the rendered page"
    assert "Columns vary domain" in iclr_pages[caption_page]
    assert "response mechanism." in iclr_pages[caption_page]


def test_main_paper_fits_the_nine_page_limit(iclr_pages: list[str]) -> None:
    conclusion_page = _unique_page(
        iclr_pages, "presuming a general world model"
    )
    references_page = _unique_page(
        iclr_pages, "R EFERENCES"
    )

    assert conclusion_page <= 8, "the conclusion must end by numbered page 9"
    assert (
        references_page > conclusion_page
    ), "references must begin after the main text"


def test_top_level_tex_group_delimiters_are_balanced() -> None:
    for path in sorted(PAPER_DIR.glob("*.tex")):
        source = path.read_text(encoding="utf-8")
        begin_count = source.count(r"\begingroup")
        end_count = source.count(r"\endgroup")
        assert (
            begin_count == end_count
        ), f"{path.name} has {begin_count} begingroup and {end_count} endgroup"
