"""Regression checks for the nine-page SocialAgent workshop manuscript."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
WORKSHOP = PAPER / "socialagent2026"
BODY_FILES = (
    "abstract.tex",
    "introduction.tex",
    "related_work.tex",
    "evidence_ladder.tex",
    "transfer_and_grounding.tex",
    "discussion.tex",
)


def read(name: str) -> str:
    return (WORKSHOP / name).read_text(encoding="utf-8")


def flat(text: str) -> str:
    """Collapse the line breaks pdftotext inserts, so phrases match across them."""
    return re.sub(r"\s+", " ", text)


def pdf_page(number: int) -> str:
    return subprocess.run(
        [
            "pdftotext",
            "-f",
            str(number),
            "-l",
            str(number),
            "-layout",
            str(WORKSHOP / "main.pdf"),
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_official_neurips_workshop_mode_and_anonymity() -> None:
    main = read("main.tex")
    assert r"\usepackage[dblblindworkshop]{neurips_2026}" in main
    assert r"\workshoptitle{SocialAgent:" in main
    assert r"\author{Anonymous Author(s)}" in main
    assert "Can language models forecast inductively?" in main
    assert "@princeton.edu" not in main

    first_page = pdf_page(1)
    assert "Anonymous Author(s)" in first_page
    assert "Can language models forecast inductively?" in first_page


def test_page_one_keeps_anonymous_artifact_links() -> None:
    abstract = read("abstract.tex")
    first_page = pdf_page(1)
    assert r"\footnotemark" in abstract
    assert r"\footnotetext{" in abstract
    assert r"{\centering\small" not in abstract
    for required in (
        r"\micon{huggingface.png}",
        r"\micon{gh_logo.png}",
        "https://huggingface.co/datasets/anonymous-review/forecasting-study-data",
        "https://github.com/anonymous-review/forecasting-study-code",
    ):
        assert required in abstract
    assert "Study dataset and artifacts" in first_page
    assert "Code repository" in first_page
    assert "Links anonymized for review" in first_page


def test_exp3_moves_the_cross_scale_qwen_synthesis_to_the_appendix() -> None:
    transfer = flat(read("transfer_and_grounding.tex"))
    appendix = flat(read("experiment3_appendix.tex"))
    for required in (
        "Qwen3-family summary",
        "$.319$ $[.134,.489]$",
        "$.251$ $[.058,.470]$",
        "$.120$ $[-.027,.261]$",
        "$.230$ $[.128,.333]$",
        "equal-weight synthesis across the Qwen3 family",
        "eight of nine size--seed estimates",
        "11 of 12 size--cell estimates",
        "not scale invariance",
    ):
        assert required in appendix
    assert "Full cell estimates, intervals" in transfer
    assert "5.72" not in transfer
    assert "equal-weight synthesis" not in transfer
    assert "post-hoc" not in transfer.lower()
    assert "post hoc" not in transfer.lower()


def test_abstract_is_high_level_and_compact() -> None:
    source = read("abstract.tex")
    body = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}", source, re.DOTALL
    )
    assert body
    words = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", body.group(1))
    assert len(words) <= 180
    assert "30,094" not in source
    assert "Qwen" not in source
    assert "Across a battery of experiments" in source
    assert "update their beliefs coherently" in source
    assert "Training on families of predictive" in source
    assert "forecast inductively" in source
    assert "crowds are inconclusive" not in source


def test_inductive_forecasting_story_and_plain_language_are_preserved() -> None:
    introduction = read("introduction.tex")
    assert r"\textbf{Present work.}" in introduction
    assert r"\noindent\textbf{Results.}" in introduction
    assert "a demanding form of reasoning central to decisions" in flat(introduction)
    assert "fundamentally inductive" in introduction
    assert "This distinction is particularly consequential" in introduction
    for experiment in range(1, 5):
        assert rf"\hyperref[sec:exp{experiment}]{{Exp.~{experiment}}}" in introduction
    assert "topically related but orthogonal information" in introduction
    assert "holding\nnumerical evidence fixed" in introduction
    assert "unseen synthetic structures and held-out historical" in introduction
    assert r"\begin{wrapfigure}{r}{0.44\textwidth}" in introduction
    assert r"\setlength{\columnsep}{6pt}" in introduction
    assert r"\vspace{-4.2em}" in introduction
    assert r"\vspace{-1.0cm}" in introduction
    assert r"\caption{\textbf{Motivation.}" in introduction
    assert "Judgmental forecasts combine observations with" in introduction
    assert r"$P(H) = 0.68$" in introduction
    assert r"App.~\ref{app:carnival-coin-probe}" in introduction
    assert "for details" not in introduction
    assert r"$1.9$--$7.5\times$" in introduction
    assert r"$r=.52$--$.70$" in introduction
    assert r"MAE $4.46$ vs.~$4.71$" in flat(introduction)
    assert "at all three tested Qwen3 sizes" in flat(introduction)
    assert "All nine systems update" in introduction
    assert "forecasts track context-selected relationships" in introduction
    assert "answer-matched shuffled control" in flat(introduction)
    assert "several trained checkpoints reach crowd performance" in flat(introduction)
    assert "Qwen gains narrow with scale" not in introduction
    assert "care not only about what will" in introduction
    results = introduction.split(r"\noindent\textbf{Results.}", 1)[1]
    assert len(re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", results)) <= 130

    body = "\n".join(read(name) for name in BODY_FILES)
    forbidden_phrases = (
        "intervention ladder",
        "simply",
        "bounded view",
        "perhaps",
        "arguably",
        "family and size are confounded",
        "no evidence of transfer",
        "does not reproduce the effect",
        "fails to replicate",
        "qwen-specific",
    )
    for phrase in forbidden_phrases:
        assert phrase not in body.lower()
    assert "For social agents in particular" in body
    assert "We designed the experiments around alternatives left open" in body
    assert r"\section{Do forecasts change for the right reason?}" in body
    assert r"\section{Related work: why accuracy is not enough}" in body
    assert (
        r"\section{Limitations: what these studies do and do not establish}"
        in body
    )
    assert r"\section{Discussion: why inductive reasoning matters}" in body
    assert "Forecasting is our testbed for inductive reasoning" in body
    assert "not the boundary of the claim" in flat(body)
    assert r"\noindent\textbf{Judgmental forecasting.}" in body
    assert r"\noindent\textbf{Induction and relational reasoning.}" in body
    assert r"\noindent\textbf{Social agents and conditional validity.}" in body
    assert r"\paragraph{" not in body
    for heading in (
        r"\subsection{Exp.~1: can the model separate evidence from topical chatter?}",
        r"\subsection{Exp.~2: let context select a relationship, then challenge it}",
        r"\subsection{Exp.~3: transfer under domain and mechanism shift}",
        r"\subsection{Exp.~4: external grounding in unseen historical markets}",
    ):
        assert heading in body
    for experiment in range(1, 5):
        assert rf"\textbf{{\emph{{What Exp.~{experiment} establishes.}}}}" in body


def test_all_ten_figure_assemblies_remain_in_the_main_body() -> None:
    introduction = read("introduction.tex")
    evidence = read("evidence_ladder.tex")
    transfer = read("transfer_and_grounding.tex")

    assert r"\begin{wrapfigure}{r}{0.44\textwidth}" in introduction
    assert "teaser_revised.pdf" in introduction

    assert r"\begin{subfigure}[c]{0.525\textwidth}" in evidence
    assert (
        r"\includegraphics[width=.92\linewidth]"
        r"{exp1_direction_selective_updating.pdf}"
    ) in evidence
    assert r"\begin{subfigure}[c]{0.455\textwidth}" in evidence
    assert r"\textbf{Abst.}" in evidence
    assert r"\textbf{Sens.}" in evidence
    assert r"\textbf{EHC$_{\geq3}$}" in evidence

    assert "exp2_coin_city_results.pdf" in evidence
    assert r"\begin{minipage}{\linewidth}" in evidence
    assert r"\multicolumn{4}{c}{Forecast MAE [95\% CI]}" in evidence
    assert r"\rho(\hat g,g_C)" in evidence

    assert r"\begin{wrapfigure}[23]{r}{0.43\textwidth}" in evidence
    assert "exp2_symbol_decoding.pdf" in evidence
    assert "exp2_causal_patch.pdf" in evidence
    # The transfer grid stays a compact half-page wrap in Section 4. It is
    # declared after the subsection heading so wrapfig cannot absorb the heading.
    assert r"\input{fig_exp3_transfer_grid}" not in evidence
    assert r"\input{fig_exp3_transfer_grid}" in transfer
    subsection = transfer.index(
        r"\subsection{Exp.~3: transfer under domain and mechanism shift}"
    )
    transfer_grid_input = transfer.index(r"\input{fig_exp3_transfer_grid}")
    opening = transfer.index(r"\textbf{Why matching matters.}")
    assert subsection < transfer_grid_input < opening
    assert r"\newcommand{\transfergridlines}{15}" in transfer
    assert r"\renewenvironment{wrapfigure}[3][]" not in transfer
    transfer_grid = (WORKSHOP / "fig_exp3_transfer_grid.tex").read_text(
        encoding="utf-8"
    )
    # The count is a parameter now: the ICLR build keeps 13 via providecommand.
    assert r"\providecommand{\transfergridlines}{13}" in transfer_grid
    assert (
        r"\begin{wrapfigure}[\transfergridlines]{r}{0.48\textwidth}" in transfer_grid
    )
    assert r"\vspace{-0.8em}" in transfer_grid
    assert r"\caption{\textbf{Transfer grid.}" in transfer_grid
    assert r"\footnotesize\textbf{Transfer grid.}" not in transfer_grid
    assert r"\vspace{0.1em}" in transfer_grid

    page_six = pdf_page(6)
    page_seven = pdf_page(7)
    assert "Figure 7:" in page_six
    assert page_six.index("Figure 7:") < page_six.index("4.1")
    assert page_six.index("4.1") < page_six.index("Figure 8:")
    assert "Figure 9:" in page_seven

    pdf_text = "\n".join(pdf_page(page) for page in range(1, 10))
    figures = {int(value) for value in re.findall(r"Figure\s+(\d+):", pdf_text)}
    assert figures == set(range(1, 11))


def test_activation_patch_figure_reports_the_effects() -> None:
    source = read("evidence_ladder.tex")
    figure = source.index(r"\label{fig:coin-city-causal-patch}")
    figure_start = source.rfind(r"\begin{figure}", 0, figure)
    assert source[figure_start:].startswith(r"\begin{figure}[!t]")
    assert "exp2_causal_patch.pdf" in source[:figure]
    assert r"\(+.333\)" in source
    assert r"\(-.035\)" in source
    assert "Changing the induced convention changes the forecast before" in source

    results_path = (
        PAPER.parent
        / "exp2_v2"
        / "biased_news"
        / "data"
        / "coin_city_stable_relationship_claude_n250_v4"
        / "mechanistic_probe"
        / "qwen3_14b_symbol_relational_v2"
        / "relational_probe_results.json"
    )
    results = json.loads(results_path.read_text(encoding="utf-8"))
    primary = results["patching"]["conditions"]["cross_selected_window"]
    assert round(primary["k0"]["mean"], 3) == 0.333
    assert round(primary["k4"]["mean"], 3) == -0.035
    assert round(primary["k0"]["null_q025"], 3) == -0.177
    assert round(primary["k0"]["null_q975"], 3) == 0.178
    assert round(primary["k4"]["null_q025"], 3) == -0.073
    assert round(primary["k4"]["null_q975"], 3) == 0.073

    page_five = pdf_page(5)
    page_six = pdf_page(6)
    assert "Figure 7:" not in page_five
    assert "Figure 7:" in page_six
    assert "random-swap range" in flat(page_five.lower())
    example = patch_figure_example()
    for text in (
        "DONOR RUN",
        "RECIPIENT RUN",
        "label-token states",
        "KIV",
        "ZOR",
        f"{example['unpatched_poll']:.1f}",
        f"{example['patched_poll']:.1f}",
    ):
        assert text in page_six
    # The figure states the intervention, not just its outcome. These are
    # separate text runs in the PDF, so each is checked on its own line.
    for text in (
        "write the donor's vectors",
        "over the recipient's",
        "all other layers and positions run unchanged",
    ):
        assert text in page_six


def patch_figure_example() -> dict:
    """The episode the causal-patch figure selects, loaded from the frozen run."""
    spec = importlib.util.spec_from_file_location(
        "coin_city_causal_patch_figure",
        PAPER / "generate_coin_city_causal_patch_figure.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    conditions = module.load()
    return module.load_example(conditions["k0"]["mean"])


def test_activation_patch_caption_matches_the_displayed_episode() -> None:
    """The caption quotes the example, so it must track the selection rule."""
    example = patch_figure_example()
    caption = read("evidence_ladder.tex")
    start = caption.index(
        "Changing the induced convention changes the forecast before"
    )
    caption = caption[start : caption.index(r"\label{fig:coin-city-causal-patch}")]
    assert r"Table~\ref{tab:coin-city-causal-patch}" in caption
    assert f"from {example['unpatched_poll']:.1f} to" in caption
    assert f"{example['patched_poll']:.1f}" in caption
    # The patched recipient lands on the donor's own unpatched forecast.
    assert example["patched_poll"] == example["donor_poll"]
    # Selected to be representative, not the largest effect available.
    assert abs(example["effect"] - 0.333) < 0.05


def test_every_numbered_figure_has_a_body_callout() -> None:
    body = "\n".join(read(name) for name in BODY_FILES)
    labels = (
        "fig:world-knowledge-evidence",
        "fig:exp1-pipeline",
        "fig:exp1-updating",
        "fig:coin-city-prompt",
        "fig:coin-city-results",
        "fig:coin-city-symbol-decoding",
        "fig:coin-city-causal-patch",
        "fig:exp3-transfer-grid",
        "fig:exp3-transfer-results",
        "fig:sealed-market-grounding",
    )
    for label in labels:
        assert rf"\ref{{{label}}}" in body


def test_figures_two_and_three_follow_requested_reading_order() -> None:
    source = read("evidence_ladder.tex")
    figure_two = source.index(r"\label{fig:exp1-pipeline}")
    design = source.index(r"\noindent\textbf{Design.}")
    finding = source.index(r"\noindent\textbf{Finding: models first decide")
    figure_three = source.index(r"\label{fig:exp1-updating}")
    experiment_two = source.index(
        r"\subsection{Exp.~2: let context select a relationship"
    )

    figure_two_start = source.rfind(r"\begin{figure}", 0, figure_two)
    figure_three_start = source.rfind(r"\begin{figure}", 0, figure_three)

    # Figure 2 is declared while page two is still being built so it can anchor
    # the top of page three. Figure 3 remains after the finding as a bottom float.
    assert source[figure_two_start:].startswith(r"\begin{figure}[!t]")
    assert source[figure_three_start:].startswith(r"\begin{figure}[!b]")
    assert figure_two < design < finding < figure_three < experiment_two

    page_three = pdf_page(3)
    page_four = pdf_page(4)
    assert "Figure 2:" in page_three
    assert "Figure 3:" in page_three
    assert "Finding: models first decide" in page_three
    assert page_three.index("Figure 2:") < page_three.index("Design.")
    assert page_three.index("Design.") < page_three.index("Finding:")
    assert page_three.index("Finding:") < page_three.index("Figure 3:")
    assert "Figure 2:" not in page_four
    assert "Figure 3:" not in page_four
    assert "A minimal requirement is clear" not in source
    assert "Forecasting is not only about assigning a probability" in source


def test_figure_four_follows_design_and_precedes_finding() -> None:
    source = read("evidence_ladder.tex")
    figure_four = (
        r"\includegraphics[" + "\n"
        r"  width=.90\linewidth," + "\n"
        r"  trim=11pt 14.5pt 3.5pt 7pt," + "\n"
        r"  clip" + "\n"
        r"]{exp2_coin_city_prompt_arms.pdf}"
    )
    assert figure_four in source
    design = source.index("We compare five versions of the same problem")
    figure = source.index(r"\label{fig:coin-city-prompt}")
    figure_start = source.rfind(r"\begin{figure}", 0, figure)
    finding = source.index(
        r"\noindent\textbf{Finding: the forecast follows the selected analogue.}"
    )
    figure_five = source.index(r"\label{fig:coin-city-results}")
    assert source[figure_start:].startswith(r"\begin{figure}[!t]")
    assert figure < design < finding < figure_five

    page_four = pdf_page(4)
    page_five = pdf_page(5)
    assert "Figure 4:" in page_four
    assert page_four.index("Figure 4:") < page_four.index("Finding:")
    assert "Figure 5:" not in page_four
    assert "Figure 5:" in page_five


def test_figure_five_keeps_its_table_below_the_plot() -> None:
    source = read("evidence_ladder.tex")
    plot = source.index("exp2_coin_city_results.pdf")
    table = source.index(r"\multicolumn{4}{c}{Forecast MAE [95\% CI]}")
    caption = source.index(
        r"\caption{\textbf{Context improves forecasts before target cases"
    )
    assert plot < table < caption

    page_five = pdf_page(5)
    assert "Forecast MAE [95% CI]" in page_five
    assert "Figure 5:" in page_five


def test_figure_six_wraps_the_arbitrary_convention_result_on_page_five() -> None:
    source = read("evidence_ladder.tex")
    figure = source.index(r"\begin{wrapfigure}[23]{r}{0.43\textwidth}")
    finding = source.index(
        r"\noindent\textbf{Finding: the forecast follows the selected analogue.}"
    )
    convention_subpart = source.index(
        r"\subsubsection{Infer a convention with no stable semantics}"
    )
    setup = source.index("This behavior shows that models can use")
    decoding = source.index("Figure~\\ref{fig:coin-city-symbol-decoding} asks")
    causal = source.index("Decoding is still correlational.")
    takeaway = source.index(r"\noindent\textbf{\emph{What Exp.~2 establishes.}}")
    assert finding < convention_subpart < figure < setup < decoding < causal < takeaway
    assert r"\vspace{-1.0em}" in source[figure:setup]
    assert r"\vspace{-1.8em}" in source[figure:]

    page_five = pdf_page(5)
    page_six = pdf_page(6)
    page_four = pdf_page(4)
    assert "Finding: the forecast follows the selected analogue." in page_four
    assert "Infer a convention with no stable semantics" in page_four
    assert "Figure 6:" in page_five
    assert "What Exp. 2 establishes." in page_five
    assert "Figure 7:" not in page_five
    assert "Figure 6:" not in page_six
    assert "Figure 7:" in page_six
    assert page_six.index("Figure 7:") < page_six.index("Section 4 asks")


def test_figure_ten_is_at_the_bottom_of_page_seven() -> None:
    source = read("transfer_and_grounding.tex")
    figure = source.index("exp4_qwen_scale_results.pdf")
    assert r"\begin{figure}[!b]" in source[:figure]
    # Declare the float before Exp. 4 so it can occupy the bottom of page seven.
    assert figure < source.index(r"\subsection{Exp.~4: external grounding")
    assert figure < source.index("Figure~\\ref{fig:sealed-market-grounding}")

    page_seven = pdf_page(7)
    page_eight = pdf_page(8)
    assert "Figure 10:" in page_seven
    assert "Figure 10:" not in page_eight
    assert page_seven.index("Exp. 4:") < page_seven.index("Figure 10:")


def test_qwen32_and_separate_llama_scale_results_flow_to_workshop() -> None:
    transfer = flat(read("transfer_and_grounding.tex"))
    appendix = read("appendix.tex")
    shared_appendix = (WORKSHOP / "experiment3_appendix.tex").read_text(
        encoding="utf-8"
    )

    assert "Qwen3 checkpoints from 1.7B to 32B" in transfer
    assert "trained mean Brier is \\(.1265\\)" in transfer
    assert "versus \\(.1266\\) for the base" in transfer
    assert "smallest observed gain" in transfer
    assert "base and trained Brier are both" not in transfer
    assert "Llama 3B/8B replication shows the same qualitative" in transfer
    assert "log parameter axis" in transfer
    assert r"\input{experiment3_appendix}" in appendix
    assert "figures/exp4_llama_scale_results.pdf" in shared_appendix
    assert r"\label{fig:exp4-llama-scale-results}" in shared_appendix


def test_coin_harbor_is_defined_against_coin_city() -> None:
    transfer = " ".join(read("transfer_and_grounding.tex").split())
    assert "Coin City campaign news and candidate support become Coin Harbor" in transfer
    assert "shipping bulletins" in transfer
    assert "cargo-flow index over tide cycles" in transfer
    assert "populations, noise, units, and time scale" in transfer
    assert "persistent two-step response replaces the direct" in transfer


def test_llama_70b_capacity_result_flows_to_workshop() -> None:
    transfer = " ".join(read("transfer_and_grounding.tex").split())
    appendix = read("appendix.tex")
    shared_appendix = (WORKSHOP / "experiment3_appendix.tex").read_text(
        encoding="utf-8"
    )

    assert "the Llama comparisons are in" in transfer
    assert "spans Llama-3.1-8B and Llama-3.1-70B" not in transfer
    assert "larger model also learns the task" not in transfer
    assert "post-registration" not in transfer
    assert r"\input{experiment3_appendix}" in appendix
    assert "Additional Llama-3.1-70B scale check" in shared_appendix
    assert "post-registration" not in shared_appendix
    assert r"\label{fig:exp4-scale-results}" in transfer


def test_coin_city_is_defined_at_first_body_reference() -> None:
    introduction = flat(read("introduction.tex"))
    first = introduction.index("In Coin City")
    definition = "In Coin City, our controlled task"
    assert introduction.index(definition) == first


def test_main_content_is_exactly_nine_pages_before_references() -> None:
    page_nine = pdf_page(9)
    page_ten = pdf_page(10)
    assert "Links anonymized for review." not in page_nine
    heading = r"^\s*(?:\d+\s+)?References\s*$"
    assert not re.search(heading, page_nine, re.MULTILINE)
    assert re.search(heading, page_ten, re.MULTILINE)


def test_appendix_opens_with_the_iclr_organization_guide() -> None:
    appendix = read("appendix.tex")
    guide = appendix.index(r"\input{appendix_guide}")
    page_break = appendix.index(r"\clearpage", guide)
    first_appendix = appendix.index(r"\input{teaser_probe_appendix}")
    assert guide < page_break < first_appendix

    guide_source = (WORKSHOP / "appendix_guide.tex").read_text(encoding="utf-8")
    assert r"\section*{Appendices}" in guide_source
    assert r"\appendixguidechapter{app:carnival-coin-probe}" in guide_source
    assert r"\appendixguidechapter{app:exp1}" in guide_source
    assert r"\appendixguidechapter{app:coin-city}" in guide_source
    assert r"\appendixguidechapter{app:exp3}" in guide_source
    assert r"\appendixguidechapter{app:exp4}" in guide_source



def test_both_manuscripts_use_one_shared_bibliography() -> None:
    workshop_main = read("main.tex")
    iclr_main = (PAPER / "main.tex").read_text(encoding="utf-8")
    references = (PAPER / "references.bib").read_text(encoding="utf-8")

    assert r"\bibliography{../references}" in workshop_main
    assert r"\bibliography{references}" in iclr_main
    assert "workshop_references" not in workshop_main
    assert not (WORKSHOP / "workshop_references.bib").exists()
    for key in (
        "park2023generative",
        "aher2023simulate",
        "argyle2023out",
        "bisbee2024synthetic",
    ):
        assert references.count("{" + key + ",") == 1


def test_workshop_build_has_no_serious_latex_warnings() -> None:
    log = read("main.log")
    forbidden = (
        "LaTeX Error",
        "There were undefined references",
        "Citation `",
        "Reference `",
        "Overfull \\hbox",
        "Overfull \\vbox",
        "Package wrapfig Warning",
        "multiply defined",
    )
    assert not any(message in log for message in forbidden)
