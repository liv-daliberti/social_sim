#!/usr/bin/env python3
"""Experiment 2 — SKILL table (LaTeX), the primary Exp 2 result.

Compact wraptable matching Figure~\\ref{fig:exp2-skill}: per forecaster and per k it pairs the two
ERROR metrics the figure plots, both lower-is-better:

    eps_k = mean |g_hat - g|            (latent-recovery error; Fig panel A)
    pi_k  = mean_s |p_hat(s) - mu_k(s)| (next-poll error, poll points; Fig panel B)

The fair statistical bar is the naive frequentist (unregularized through-origin fit) at EVERY k:
with a single poll it cannot identify g (no baseline of its own) and reads the news at face value,
so its k=1 error is its worst (eps~.46). Cells are shaded green where a model beats the frequentist
on that metric, red where it falls short. The Bayes oracle is the ceiling; always-guess-the-prior is
a secondary no-information reference (calibrated on eps but blind); the fixed shortcuts (ignore g=0,
face value g=1) sit far above and live in the caption. The informativeness axis rho is demoted to the
appendix scatter (Fig~\\ref{fig:exp2-informativeness}).

Reuses generate_exp2_skill.py (recovery + poll readouts), so table and figure are consistent.
Run from repo root:  python paper/generate_exp2_skill_table.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp2_skill", _HERE / "generate_exp2_skill.py")
_s = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s)
_m = _s._m

K_COLS = [1, 2, 3, 5]
PRIOR = 0.55
C_WIN = "green!22"
C_LOSE = "red!18"

FRONTIER = [
    ("claude-opus-4-8",  "Claude Opus~4.8"),
    ("gpt-5.4",          "GPT-5.4"),
    ("DeepSeek-V4-Pro",  "DeepSeek V4-Pro"),
]
OPENWEIGHT = [   # size-ordered, largest -> smallest
    ("qwen2.5-72b",       "Qwen~2.5-72B"),
    ("llama3.3-70b",      "Llama~3.3-70B"),
    ("llama3.1-70b",      "Llama~3.1-70B"),
    ("qwen2.5-32b",       "Qwen~2.5-32B"),
    ("mistral-small-24b", "Mistral-Small-24B"),
    ("qwen2.5-14b",       "Qwen~2.5-14B"),
    ("mistral-nemo-12b",  "Mistral-Nemo-12B"),
    ("llama3.1-8b",       "Llama~3.1-8B"),
    ("qwen2.5-7b",        "Qwen~2.5-7B"),
]


def _fmt_e(v):                                  # recovery error: <1, leading-dot style (.20)
    if v is None or (isinstance(v, float) and v != v):
        return "---"
    return f"{v:.2f}".replace("0.", ".").replace("-0.", "-.")


def _fmt_p(v):                                  # next-poll error: poll points, 1 decimal (2.4)
    if v is None or (isinstance(v, float) and v != v):
        return "---"
    return f"{v:.1f}"


# ── metric readouts ────────────────────────────────────────────────────────────
def _eps(recs, k):
    return _s._mae(recs, k, _s.g_model)


def _pi(recs, k):
    return _s._pmae(recs, k, _s.p_model)


def _eps_bar(ref, k):                           # frequentist recovery-error bar (defined at all k)
    return _s._mae(ref, k, _s.g_freq_unclamped)


def _pi_bar(ref, k):                            # frequentist next-poll-error bar
    return _s._pmae(ref, k, _s.p_freq)


def _cell(val, bar, fmt):
    s = fmt(val)
    if val is None or bar is None:
        return s
    if val < bar - 1e-9:                         # lower error than the bar = beats it
        return f"\\cellcolor{{{C_WIN}}}{s}"
    if val > bar + 1e-9:
        return f"\\cellcolor{{{C_LOSE}}}{s}"
    return s


def _row(label, recs, ebar, pbar, tag=""):
    cells = []
    for k in K_COLS:
        cells.append(_cell(_eps(recs, k), ebar[k], _fmt_e))
        cells.append(_cell(_pi(recs, k), pbar[k], _fmt_p))
    return f"{label}{tag} & " + " & ".join(cells) + " \\\\"


def _ref_row(label, efn, pfn):
    cells = []
    for k in K_COLS:
        cells.append(_fmt_e(efn(k)))
        cells.append(_fmt_p(pfn(k)))
    return f"{label} & " + " & ".join(cells) + " \\\\"


def _caption(fr_n, ow_n):
    def _nlab(ns):
        ns = sorted(set(ns))
        return str(ns[0]) if len(ns) == 1 else f"{ns[0]}\\text{{--}}{ns[-1]}"
    return (
        "Forecast skill from a sparse history.  Each $k$ pairs the two errors of "
        "Fig.~\\ref{fig:exp2-skill} (both lower is better): the \\emph{latent-recovery} error "
        "$\\varepsilon_k{=}\\overline{|\\hat g-g|}$ and the \\emph{next-poll} error "
        "$\\pi_k{=}\\overline{|\\hat p_{t+1}-p_{t+1}|}$ (poll points).  The \\emph{Bayes oracle} is the "
        "ceiling.  The fair statistical bar is the \\emph{naive frequentist} (unregularized fit); with "
        "one poll it cannot identify $g$ and reads the news at face value, so its $k{=}1$ error is its "
        "worst.  Cells are \\colorbox{" + C_WIN + "}{green} where a model beats the frequentist on that "
        "metric, \\colorbox{" + C_LOSE + "}{red} where it falls short.  \\emph{Always-prior} (guess the "
        "mean) is a secondary no-information reference---calibrated on $\\varepsilon$ but blind; the "
        "fixed shortcuts (ignore $g{=}0$, face value $g{=}1$) sit far above ($\\varepsilon{\\approx}.46$--"
        "$.54$, $\\pi{\\approx}3.9$--$4.3$).  The frontier agents beat the frequentist on both metrics "
        "from a single week and stay near the oracle.  "
        f"Frontier $n{{=}}{_nlab(fr_n)}$; open-weight $n{{=}}{_nlab(ow_n)}$ (partial rows noisier).  "
        "\\textsuperscript{$\\dagger$}Frontier probe.")


def main():
    by = _m.load_fair_records()
    if not by:
        print("No parity results found in", _m._REC); return
    ref = max(by.values(), key=len)
    ebar = {k: _eps_bar(ref, k) for k in K_COLS}
    pbar = {k: _pi_bar(ref, k) for k in K_COLS}

    frontier = [(s, l) for s, l in FRONTIER if s in by]
    openw = [(s, l) for s, l in OPENWEIGHT if s in by]
    fr_n = [len(by[s]) for s, _ in frontier]
    ow_n = [len(by[s]) for s, _ in openw]

    spec = "l@{\\hskip 5pt}" + " ".join("cc" for _ in K_COLS)
    groups = " & ".join(f"\\multicolumn{{2}}{{c}}{{$k{{=}}{k}$}}" for k in K_COLS)
    cmids = "".join(f"\\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}" for i in range(len(K_COLS)))
    subhdr = " & ".join(r"$\varepsilon{\downarrow}$ & $\pi{\downarrow}$" for _ in K_COLS)

    L = []
    L.append("% AUTO-GENERATED by paper/generate_exp2_skill_table.py — do not edit by hand.")
    L.append("% Requires \\usepackage{wrapfig}, booktabs, and \\usepackage[table]{xcolor}.")
    L.append("\\begin{wraptable}{r}{0.56\\textwidth}")
    L.append("\\centering")
    L.append("\\vspace{-1.2em}")
    L.append("\\setlength{\\tabcolsep}{2pt}")
    L.append(f"\\caption{{{_caption(fr_n, ow_n)}}}")
    L.append("\\label{tab:exp2-skill}")
    L.append("\\footnotesize")
    L.append(f"\\begin{{tabular}}{{{spec}}}")
    L.append("\\toprule")
    L.append(f" & {groups} \\\\")
    L.append(cmids)
    L.append(f"\\textbf{{Forecaster}} & {subhdr} \\\\")
    L.append("\\midrule")
    L.append(_ref_row("Bayes oracle", lambda k: _s._mae(ref, k, _s.g_bayes),
                      lambda k: _s._pmae(ref, k, _s.p_oracle)))
    L.append(_ref_row("Naive freq.", lambda k: _eps_bar(ref, k), lambda k: _pi_bar(ref, k)))
    L.append(_ref_row("Always-prior", lambda k: _s._mae(ref, k, lambda r, k: PRIOR),
                      lambda k: _s._pmae(ref, k, _s.p_prior)))
    L.append("\\midrule")
    for s, l in frontier:
        L.append(_row(l, by[s], ebar, pbar, tag="\\textsuperscript{$\\dagger$}"))
    L.append("\\midrule")
    for s, l in openw:
        L.append(_row(l, by[s], ebar, pbar))
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{wraptable}")
    tex = "\n".join(L)

    out = _HERE / "tables" / "exp2_skill_table.tex"
    out.parent.mkdir(exist_ok=True)
    out.write_text(tex + "\n")
    print(tex)
    print(f"\n  Saved {out}")
    print("  eps bar (freq): " + "  ".join(f"k{k}={_fmt_e(ebar[k])}" for k in K_COLS))
    print("  pi  bar (freq): " + "  ".join(f"k{k}={_fmt_p(pbar[k])}" for k in K_COLS))


if __name__ == "__main__":
    main()
