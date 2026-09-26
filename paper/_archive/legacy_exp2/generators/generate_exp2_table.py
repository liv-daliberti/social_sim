#!/usr/bin/env python3
"""Experiment 2 — information-parity recovery table (LaTeX), sparse-data framing.

Emits a COMPACT wraptable (matching the section's figure/table layout) led by the
sparse-evidence contrast: after a single week the matched same-information forecaster
is provably unidentified (rho=0 — one poll, no baseline, g not estimable), the oracle
(handed the baseline) is already ~0.66, and the agents land between. A second rho column
at the horizon (k=10) shows the forecaster overtaking once data accumulates.

Columns (narrow wraptable, so kept to five): Forecaster | rho_{k=1} | rho_{k=10} |
beta (k=10) | stated-rate rho (k=10). The mean g_hat and the bootstrap SE on rho_{k=1}
are summarised in the caption rather than given their own columns.

Reuses the estimator definitions from generate_exp2_curve_parity.py so the table and the
curve are guaranteed consistent. Reads
exp2_simulated_worlds/biased_news/data/recovery_curve_parity/results_*.jsonl.

Run from repo root:  python paper/generate_exp2_table.py
"""
from __future__ import annotations

import importlib.util
import random
import statistics
from pathlib import Path

_HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "exp2_parity", _HERE / "generate_exp2_curve_parity.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

K_COLS = [1, 3, 5]           # recovery columns shown (sparse -> mid -> horizon)
K_EARLY = 1
K_HORIZON = 5                # horizon for beta / mean / stated-rho (probe caps at k=5)
N_BOOT = 2000
random.seed(0)

# heatmap shades (conditional formatting): green where a model beats the
# same-information forecaster at that k, red where it falls short.
C_WIN = "green!22"
C_LOSE = "red!18"

# rows ordered by model SIZE (descending): largest at the top, smallest at the bottom.
FRONTIER = [   # frontier tier, descending by (estimated) size
    ("claude-opus-4-8",  "Claude Opus~4.8"),
    ("gpt-5.4",          "GPT-5.4"),
    ("DeepSeek-V4-Pro",  "DeepSeek V4-Pro"),
]
OPENWEIGHT = [  # open-weight tier, descending by parameter count
    ("qwen2.5-72b",  "Qwen~2.5-72B"),
    ("llama3.3-70b", "Llama~3.3-70B"),
    ("llama3.1-70b", "Llama~3.1-70B"),
    ("qwen2.5-32b",  "Qwen~2.5-32B"),
    ("qwen2.5-14b",  "Qwen~2.5-14B"),
    ("llama3.1-8b",  "Llama~3.1-8B"),
    ("qwen2.5-7b",   "Qwen~2.5-7B"),
]


def _load():
    """Per model, its newest NON-EMPTY results file only (see generate_exp2_curve_parity.
    load_fair_records): the fair 2026-06-29 parity rerun supersedes the pre-fair 06-22/24 runs
    instead of being backfilled by them, so the table and the figure read identical city sets."""
    return _m.load_fair_records()


def _at(rec, k, key):
    return next((c.get(key) for c in rec["curve"] if c["k"] == k), None)


def _rho(recs, k, key):
    return _m.corr([r["g"] for r in recs], [_at(r, k, key) for r in recs])


def _beta(recs, k, key):
    pairs = [(r["g"], _at(r, k, key)) for r in recs if _at(r, k, key) is not None]
    if len(pairs) < 3:
        return None
    g = [a for a, _ in pairs]
    mg = statistics.mean(g)
    vg = sum((x - mg) ** 2 for x in g)
    if vg < 1e-9:
        return None
    mgh = statistics.mean([b for _, b in pairs])
    return sum((a - mg) * (b - mgh) for a, b in pairs) / vg


def _mean(recs, k, key):
    vals = [_at(r, k, key) for r in recs if _at(r, k, key) is not None]
    return statistics.mean(vals) if vals else None


def _boot_se_rho(recs, k, key):
    pairs = [(r["g"], _at(r, k, key)) for r in recs if _at(r, k, key) is not None]
    if len(pairs) < 3:
        return None
    out = []
    n = len(pairs)
    for _ in range(N_BOOT):
        s = [pairs[random.randrange(n)] for _ in range(n)]
        c = _m.corr([a for a, _ in s], [b for _, b in s])
        if c is not None:
            out.append(c)
    return statistics.pstdev(out) if out else None


def _fmt(v, places=2):
    if v is None:
        return "---"
    s = f"{v:.{places}f}"
    return s.replace("0.", ".").replace("-0.", "-.")


def _model_row(slug, label, recs, frontier):
    return {
        "label": label, "frontier": frontier, "n": len(recs),
        "rk": {k: _rho(recs, k, "lm_g") for k in K_COLS},     # recovery per k
        "bk": {k: _beta(recs, k, "lm_g") for k in K_COLS},    # tracking slope per k
        "se1": _boot_se_rho(recs, K_EARLY, "lm_g"),
    }


def _references(ref):
    """Reference band: the two oracles (ceiling) and the two same-information forecasters
    (the fair bar). Green/red shading keys off the forecaster bar only (see _thresholds),
    so the oracle rows are context, not the threshold."""
    g = [r["g"] for r in ref]

    def col(role, fn=None, key=None):
        gh = (lambda r, k: _at(r, k, key)) if key else fn
        rk, bk = {}, {}
        for k in K_COLS:
            rk[k] = _m.corr(g, [gh(r, k) for r in ref])
            pairs = [(r["g"], gh(r, k)) for r in ref if gh(r, k) is not None]
            if len(pairs) >= 3:
                gs = [a for a, _ in pairs]
                mg = statistics.mean(gs)
                vg = sum((x - mg) ** 2 for x in gs)
                mgh = statistics.mean([b for _, b in pairs])
                bk[k] = sum((a - mg) * (b - mgh) for a, b in pairs) / vg if vg > 1e-9 else None
            else:
                bk[k] = None
        return dict(rk=rk, bk=bk, se1=None, frontier=False, n=len(ref), role=role)

    return [
        {"label": "Bayes oracle", **col("oracle", key="bayes_g")},
        {"label": "OLS oracle",   **col("oracle", fn=_m.ols_oracle_ghat)},
        {"label": "Bayes fcst.",  **col("forecaster", fn=lambda r, k: _m.forecast_ghat(r, k, _m.naive_bayes_ghat))},
        {"label": "OLS fcst.",    **col("forecaster", fn=lambda r, k: _m.forecast_ghat(r, k, _m.naive_ghat))},
    ]


def _thresholds(ref_rows):
    """Per-k bar = the harder (max) of the same-information FORECASTERS only (not the
    oracles); a model cell is green above it, red below."""
    fcst = [r for r in ref_rows if r.get("role") == "forecaster"]
    thr = {}
    for k in K_COLS:
        vals = [r["rk"][k] for r in fcst if r["rk"].get(k) is not None]
        thr[k] = max(vals) if vals else None
    return thr


def _cell(val, thr=None, color=False, bold=False):
    s = _fmt(val)
    if bold:
        s = f"\\textbf{{{s}}}"
    if color and val is not None and thr is not None:
        if val > thr + 1e-9:
            return f"\\cellcolor{{{C_WIN}}}{s}"
        if val < thr - 1e-9:
            return f"\\cellcolor{{{C_LOSE}}}{s}"
    return s


def _row_tex(row, thr, color):
    tag = "\\textsuperscript{$\\dagger$}" if row["frontier"] else ""
    cells = []
    for k in K_COLS:                                  # paired (rho colored, beta plain) per k
        cells.append(_cell(row["rk"].get(k), thr.get(k), color=color))
        cells.append(_fmt(row["bk"].get(k)))
    return f"{row['label']}{tag} & " + " & ".join(cells) + " \\\\"


def _caption(rows):
    fr_se = [r["se1"] for r in rows if r["frontier"] and r["se1"] is not None]
    ow_se = [r["se1"] for r in rows if (not r["frontier"]) and r["se1"] is not None]
    fr_n = sorted({r["n"] for r in rows if r["frontier"]})
    ow_n = sorted({r["n"] for r in rows if not r["frontier"]})

    def _nlab(ns):     # compact n label: a single value, else min--max range
        return str(ns[0]) if len(ns) == 1 else f"{ns[0]}\\text{{--}}{ns[-1]}"

    se_clause = ""
    if ow_se:
        ow_lab = _nlab(ow_n) if ow_n else "250"
        se_clause += f"  Bootstrap SE on $\\rho_{{1}}$ is $\\le{_fmt(max(ow_se))}$ (open-weight, $n{{=}}{ow_lab}$)"
        if fr_se:
            nlab = _nlab(fr_n) if fr_n else "25"
            se_clause += f" and ${_fmt(min(fr_se))}$--${_fmt(max(fr_se))}$ (frontier, $n{{=}}{nlab}$)"
        se_clause += "."
    return (
        "Latent recovery (information parity), led by the \\emph{sparse} regime "
        "(true mean $g{=}0.55$).  $\\rho_k$ is recovery after $k$ weeks, paired with its tracking "
        "slope $\\beta$ ($\\beta{\\to}1$ tracks the latent, $\\beta{\\to}0$ ignores it).  The "
        "\\emph{oracles} are the ceiling---all that \\emph{could} be recovered if the full generative "
        "structure (the $o_0{=}50$ baseline and mean-reversion $\\phi$) were known; the "
        "\\emph{forecasters} are the fair bar, given only what the agent is given.  $\\rho$ cells are "
        "shaded \\colorbox{" + C_WIN + "}{green} where a model beats the same-information forecaster "
        "at that $k$ and \\colorbox{" + C_LOSE + "}{red} where it falls short.  At $k{=}1$ the "
        "forecaster is \\emph{unidentified} ($\\rho{=}.00$---one poll, no baseline; the "
        f"baseline-knowing oracle is already at $.66$), so agents land above it; by $k{{=}}{K_COLS[-1]}$ it "
        "has caught and overtaken them." + se_clause +
        "  \\textsuperscript{$\\dagger$}Frontier probe.")


def _latex(ref_rows, frontier_rows, open_rows):
    thr = _thresholds(ref_rows)
    all_rows = ref_rows + frontier_rows + open_rows
    spec = "l@{\\hskip 5pt}" + " ".join("cc" for _ in K_COLS)      # label + (rho,beta) per k
    groups = " & ".join(f"\\multicolumn{{2}}{{c}}{{$k{{=}}{k}$}}" for k in K_COLS)
    cmids = "".join(f"\\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}" for i in range(len(K_COLS)))
    subhdr = " & ".join("$\\rho{\\uparrow}$ & $\\beta{\\uparrow}$" for _ in K_COLS)
    L = []
    L.append("% AUTO-GENERATED by paper/generate_exp2_table.py — do not edit by hand.")
    L.append("% Requires \\usepackage{wrapfig}, booktabs, and \\usepackage[table]{xcolor}.")
    L.append("\\begin{wraptable}{r}{0.56\\textwidth}")
    L.append("\\centering")
    L.append("\\vspace{-1.2em}")
    L.append("\\setlength{\\tabcolsep}{3pt}")
    L.append(f"\\caption{{{_caption(all_rows)}}}")
    L.append("\\label{tab:exp2-recovery}")
    L.append("\\footnotesize")
    L.append(f"\\begin{{tabular}}{{{spec}}}")
    L.append("\\toprule")
    L.append(f" & {groups} \\\\")
    L.append(cmids)
    L.append(f"\\textbf{{Forecaster}} & {subhdr} \\\\")
    L.append("\\midrule")
    for r in ref_rows:                       # the bar itself — uncoloured
        L.append(_row_tex(r, thr, color=False))
    L.append("\\midrule")
    for r in frontier_rows:
        L.append(_row_tex(r, thr, color=True))
    L.append("\\midrule")
    for r in open_rows:
        L.append(_row_tex(r, thr, color=True))
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    L.append("\\end{wraptable}")
    return "\n".join(L)


def main():
    by = _load()
    if not by:
        print("No parity results found in", _m._REC)
        return
    ref = max(by.values(), key=len)
    ref_rows = _references(ref)
    frontier_rows = [_model_row(s, lbl, by[s], True) for s, lbl in FRONTIER if s in by]
    open_rows = [_model_row(s, lbl, by[s], False) for s, lbl in OPENWEIGHT if s in by]
    # rows kept in OPENWEIGHT/FRONTIER list order = ascending model size

    tex = _latex(ref_rows, frontier_rows, open_rows)
    out = _HERE / "tables" / "exp2_recovery_table.tex"
    out.parent.mkdir(exist_ok=True)
    out.write_text(tex + "\n")
    print(tex)
    print(f"\n  Saved {out}")
    thr = _thresholds(ref_rows)
    print("  forecaster bar:  " + "  ".join(f"k{k}={_fmt(thr[k])}" for k in K_COLS))
    for r in frontier_rows:
        rs = "  ".join(f"k{k}={_fmt(r['rk'][k])}" for k in K_COLS)
        print(f"    {r['label']:<18} {rs}  (SE_k1={_fmt(r['se1'])}, n={r['n']})")


if __name__ == "__main__":
    main()
