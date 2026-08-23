#!/usr/bin/env python3
"""Experiment 2 — NEXT-POLL forecast-error table (LaTeX), the second headline metric.

Companion to the latent-recovery table (generate_exp2_skill_table.py). Per forecaster and per k
it reports the next-poll forecast error

    p_err_k = mean_s |p_hat(s) - mu_k(s)|   (poll points; LOWER is better)

where mu_k(s) = clip(50 + phi*(o_k-50) + g*s) is the ground-truth expected poll under held-out
shock s (see generate_exp2_skill.py). The fair statistical bar is the naive frequentist
(unregularized through-origin slope, predict last_poll + g_hat*s), UNDEFINED at k=1, so the k=1 bar
is instead always-guess-the-prior (last_poll + 0.55*s). p_err cells are green where a model beats
that per-k bar and red where it falls short. The Bayes oracle is the ceiling; the fixed shortcuts
(face value g=1, ignore news g=0) are the floor and get their own reference rows here (unlike the
compact main-text table) since this appendix table has the width.

Reuses generate_exp2_skill.py's poll readouts, so table and figure Panel B are guaranteed
consistent. Run from repo root:  python paper/generate_exp2_poll_table.py
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
C_WIN = "green!22"
C_LOSE = "red!18"

FRONTIER = [
    ("claude-opus-4-8",  "Claude Opus~4.8"),
    ("gpt-5.4",          "GPT-5.4"),
    ("DeepSeek-V4-Pro",  "DeepSeek V4-Pro"),
]
OPENWEIGHT = [
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


def _fmt(v, places=2):
    if v is None or (isinstance(v, float) and v != v):
        return "---"
    return f"{v:.{places}f}"


def _perr(recs, k):
    return _s._pmae(recs, k, _s.p_model)


def _floor(ref, k):
    """Per-k fair bar on poll error: naive frequentist where identified (k>=2), else always-prior."""
    fq = _s._pmae(ref, k, _s.p_freq)
    return fq if fq is not None else _s._pmae(ref, k, _s.p_prior)


def _cell(val, floor):
    s = _fmt(val)
    if val is None or floor is None:
        return s
    if val < floor - 1e-9:
        return f"\\cellcolor{{{C_WIN}}}{s}"
    if val > floor + 1e-9:
        return f"\\cellcolor{{{C_LOSE}}}{s}"
    return s


def _row(label, recs, floors, tag=""):
    cells = [_cell(_perr(recs, k), floors[k]) for k in K_COLS]
    return f"{label}{tag} & " + " & ".join(cells) + " \\\\"


def _ref_row(label, fn):
    return f"{label} & " + " & ".join(_fmt(fn(k)) for k in K_COLS) + " \\\\"


def _caption(fr_n, ow_n):
    def _nlab(ns):
        ns = sorted(set(ns))
        return str(ns[0]) if len(ns) == 1 else f"{ns[0]}\\text{{--}}{ns[-1]}"
    return (
        "\\textbf{Next-poll forecast error} $\\overline{|\\hat p_{t+1}-p_{t+1}|}$ (poll points, lower is "
        "better), the second headline metric: does recovering $g$ translate into good forecasts? For "
        "each held-out shock $s$ the target is the ground-truth expected poll "
        "$\\mu_k(s){=}\\mathrm{clip}(50+\\phi(o_k{-}50)+g s)$; the counterfactual shocks never realize, so "
        "this is the DGP's conditional mean, the Bayes-optimal point forecast (even the oracle, which "
        "must filter $o_k$ and $g$, sits above $0$). The \\emph{Bayes oracle} is the ceiling and the "
        "fixed shortcuts (face value $g{=}1$, ignore news $g{=}0$) the floor. The fair statistical bar "
        "is the \\emph{naive frequentist} (unregularized slope, predict $p_T{+}\\hat g s$); undefined at "
        "$k{=}1$, so there the bar is \\emph{always-guess-the-prior}. Cells are \\colorbox{" + C_WIN +
        "}{green} where a model beats that per-$k$ bar, \\colorbox{" + C_LOSE + "}{red} where it falls "
        "short. The frontier agents forecast to $\\approx2$ poll points---near the oracle, far below "
        "every shortcut, and beating the overshooting frequentist through the sparse regime. "
        f"Frontier $n{{=}}{_nlab(fr_n)}$; open-weight $n{{=}}{_nlab(ow_n)}$.  "
        "\\textsuperscript{$\\dagger$}Frontier probe.")


def main():
    by = _m.load_fair_records()
    if not by:
        print("No parity results found in", _m._REC); return
    ref = max(by.values(), key=len)
    floors = {k: _floor(ref, k) for k in K_COLS}

    frontier = [(s, l) for s, l in FRONTIER if s in by]
    openw = [(s, l) for s, l in OPENWEIGHT if s in by]
    fr_n = [len(by[s]) for s, _ in frontier]
    ow_n = [len(by[s]) for s, _ in openw]

    spec = "l@{\\hskip 6pt}" + "c" * len(K_COLS)
    hdr = " & ".join(f"$k{{=}}{k}$" for k in K_COLS)

    L = []
    L.append("% AUTO-GENERATED by paper/generate_exp2_poll_table.py — do not edit by hand.")
    L.append("% Requires booktabs and \\usepackage[table]{xcolor}.")
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append(f"\\caption{{{_caption(fr_n, ow_n)}}}")
    L.append("\\label{tab:exp2-poll}")
    L.append("\\footnotesize")
    L.append(f"\\begin{{tabular}}{{{spec}}}")
    L.append("\\toprule")
    L.append(f"\\textbf{{Forecaster}} & {hdr} \\\\")
    L.append("\\midrule")
    L.append(_ref_row("Bayes oracle", lambda k: _s._pmae(ref, k, _s.p_oracle)))
    L.append(_ref_row("Naive freq.", lambda k: _s._pmae(ref, k, _s.p_freq)))
    L.append(_ref_row("Always-prior", lambda k: _s._pmae(ref, k, _s.p_prior)))
    L.append(_ref_row("Face value ($g{=}1$)", lambda k: _s._pmae(ref, k, lambda r, k: _s.p_fixed(r, k, 1.0))))
    L.append(_ref_row("Ignore news ($g{=}0$)", lambda k: _s._pmae(ref, k, lambda r, k: _s.p_fixed(r, k, 0.0))))
    L.append("\\midrule")
    for s, l in frontier:
        L.append(_row(l, by[s], floors, tag="\\textsuperscript{$\\dagger$}"))
    L.append("\\midrule")
    for s, l in openw:
        L.append(_row(l, by[s], floors))
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{table}")
    tex = "\n".join(L)

    out = _HERE / "tables" / "exp2_poll_table.tex"
    out.parent.mkdir(exist_ok=True)
    out.write_text(tex + "\n")
    print(tex)
    print(f"\n  Saved {out}")
    print("  poll-error bar (floor): " + "  ".join(f"k{k}={_fmt(floors[k])}" for k in K_COLS))


if __name__ == "__main__":
    main()
