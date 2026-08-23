#!/usr/bin/env python3
"""Generate the Experiment 1 direction-selective-updating figure.

exp1_direction_selective_updating.pdf — single panel:
  Mean SIGNED Δp̂ by CF direction × model — causal selectivity
  (pro-H1 pushes ↑, anti-H1 ↓, orthogonal ≈ 0).
Run from repo root:
    python paper/generate_figures.py
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.colors as mcolors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent / "exp1_prospective"
_OUT  = Path(__file__).resolve().parent / "figures"
_ICLR_OUT = Path(__file__).resolve().parent / "ICLR" / "figures"
_LOGOS = _OUT / "logos"
_OUT.mkdir(exist_ok=True)
_ICLR_OUT.mkdir(exist_ok=True)

# ── brand logos (per-model markers) ─────────────────────────────────────────────
# MODEL_BRAND (defined below) maps each model key → logo filename.
_logo_cache: dict[str, np.ndarray] = {}

# vendor PDFs whose stem differs from the brand filename we use
_PDF_ALIAS = {"llamma": "llama"}

def _refresh_logos_from_pdf() -> None:
    """Rasterise any logos/<name>.pdf into a transparent black silhouette PNG
    (so dropping in the exact vendor PDFs and rerunning replaces any recreated
    PNGs automatically). The logos are black-on-white art, so we key white→alpha."""
    import shutil, subprocess, tempfile, os
    if not shutil.which("pdftoppm"):
        return
    for pdf in _LOGOS.glob("*.pdf"):
        stem = _PDF_ALIAS.get(pdf.stem, pdf.stem)
        png  = _LOGOS / f"{stem}.png"
        if png.is_file() and png.stat().st_mtime >= pdf.stat().st_mtime:
            continue
        try:
            with tempfile.TemporaryDirectory() as td:
                base = os.path.join(td, "raw")
                subprocess.run(["pdftoppm", "-png", "-r", "600", "-singlefile",
                                str(pdf), base], check=True, capture_output=True)
                im   = plt.imread(base + ".png")[..., :3]          # RGB in [0,1]
                lum  = im.mean(axis=2)
                alpha = np.clip((1.0 - lum) * 255, 0, 255).astype(np.uint8)
                rgba = np.zeros((*alpha.shape, 4), np.uint8)        # pure black ink
                rgba[..., 3] = alpha
                from PIL import Image
                out = Image.fromarray(rgba)
                if out.getbbox():                                  # trim + square pad
                    out = out.crop(out.getbbox())
                s = max(out.size)
                canv = Image.new("RGBA", (s, s), (0, 0, 0, 0))
                canv.paste(out, ((s - out.width) // 2, (s - out.height) // 2), out)
                canv.resize((256, 256), Image.LANCZOS).save(png)
            print(f"  Logo: {pdf.name} → {png.name}")
        except Exception as e:  # pragma: no cover
            print(f"  WARN could not convert {pdf.name}: {e}")

def _logo(key: str) -> np.ndarray | None:
    """Return the RGBA logo for a model key (via MODEL_BRAND), or None."""
    if key not in _logo_cache:
        fname = MODEL_BRAND.get(key, "")
        path  = _LOGOS / fname
        _logo_cache[key] = plt.imread(path) if path.is_file() else None
    return _logo_cache[key]

def _oimage(key: str, target_px: float, tint: str | None = None) -> OffsetImage | None:
    """OffsetImage scaled so every brand renders at the same display height
    (logos have different native resolutions, so a fixed zoom would make the
    256px qwen/llama marks twice as big as the frontier logos).  If `tint` is
    given, the (black) source art is recoloured to that colour, alpha preserved."""
    img = _logo(key)
    if img is None:
        return None
    if tint is not None and img.ndim == 3 and img.shape[2] == 4:
        img = np.array(img, dtype=float)
        img[..., 0], img[..., 1], img[..., 2] = mcolors.to_rgb(tint)
    return OffsetImage(img, zoom=target_px / img.shape[0])

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Nimbus Sans", "DejaVu Sans", "Helvetica", "Arial"],
    "font.size":         13,
    "axes.titlesize":    14,
    "axes.labelsize":    13,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e5e7eb",
    "grid.linewidth":    0.6,
    "axes.linewidth":    0.8,
    "legend.fontsize":   11,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#d1d5db",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

# ── colours ───────────────────────────────────────────────────────────────────
C_PRO   = "#159244"   # green (nudged slightly darker)
C_ANTI  = "#cb2121"   # red   (nudged slightly darker)
C_ORTHO = "#7c3aed"
C_FRONT = "#ea580c"
C_QWEN  = "#615CED"   # Qwen brand purple   (matches generate_paper_figures.py)
C_LLAMA = "#1877F2"   # Meta/Llama brand blue

def _family(key: str) -> str:
    """Model family from its brand logo: 'qwen', 'llama', or 'frontier'."""
    brand = MODEL_BRAND.get(key, "")
    return "qwen" if brand == "qwen.png" else "llama" if brand == "llama.png" else "frontier"

FAMILY_COLOR = {"qwen": C_QWEN, "llama": C_LLAMA, "frontier": C_FRONT}

DIR_COLOR = {"pro_H1": C_PRO, "anti_H1": C_ANTI, "orthogonal": C_ORTHO}
DIR_LABEL = {"pro_H1": "pro-$H_1$", "anti_H1": "anti-$H_1$", "orthogonal": "orthogonal"}
DIRECTIONS = ["pro_H1", "anti_H1", "orthogonal"]

# ordered model list — strictly ascending parameter count, frontier last
MODELS = [
    # key                 short label        size    is_frontier  brand logo
    ("qwen2.5:7b",       "Qwen 2.5-7B",     "7B",   False, "qwen.png"),
    ("llama3.1:8b",      "Llama 3.1-8B",    "8B",   False, "llama.png"),
    ("qwen2.5:14b",      "Qwen 2.5-14B",    "14B",  False, "qwen.png"),
    ("qwen2.5:32b",      "Qwen 2.5-32B",    "32B",  False, "qwen.png"),
    ("llama3.3:70b",     "Llama 3.3-70B",   "70B",  False, "llama.png"),
    ("llama3.1:70b",     "Llama 3.1-70B",   "70B",  False, "llama.png"),
    ("qwen2.5:72b",      "Qwen 2.5-72B",    "72B",  False, "qwen.png"),
    ("DeepSeek-V4-Pro",  "DeepSeek V4",     "DS",   True,  "deepseek.png"),
    ("gpt-5.4",          "GPT-5.4",         "GPT",  True,  "openai.png"),
    ("claude-opus-4-8",  "Claude Opus 4.8", "Claude",True, "claude.png"),
]
MODEL_BRAND    = {m[0]: m[4] for m in MODELS}

# The main comparison includes only deployments whose archived revisions use a
# consistent probability scale. Qwen 2.5-7B remains documented in the appendix.
MAIN_MODELS = [m for m in MODELS if m[0] != "qwen2.5:7b"]
MAIN_MODEL_KEYS = [m[0] for m in MAIN_MODELS]

# short names printed under the frontier bars (local bars get the param size)
FRONTIER_LABEL = {
    "DeepSeek-V4-Pro": "Pro-V4",
    "gpt-5.4":         "5.4",
    "claude-opus-4-8": "Opus-4.8",
}

# per-model logo size multiplier (the Claude mark has more internal padding, so
# it reads smaller than the others at a uniform display height)
LOGO_SCALE = {
    "claude-opus-4-8": 1.45,
}

# ── data loader ───────────────────────────────────────────────────────────────
def _load_update_records() -> list[dict]:
    """Load all update records, deduplicating by (model, update_id)."""
    uf_dir = _ROOT / "data" / "updated_forecasts"
    best: dict[tuple, dict] = {}
    for path in sorted(uf_dir.glob("*.jsonl"), reverse=True):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid   = r.get("update_id", "")
            model = r.get("forecast_model", "")
            if not uid or not model:
                continue
            key = (model, uid)
            prev = best.get(key)
            if prev is None:
                best[key] = r
            elif prev.get("delta_yes_prob") is None and r.get("delta_yes_prob") is not None:
                best[key] = r
    recs = list(best.values())
    print(f"  Update records: {len(recs):,}")
    return recs

def _normalise(r: dict) -> tuple[float | None, float | None, float | None]:
    """Return (initial_yes_prob, updated_yes_prob, delta_yes_prob) in [0,1]."""
    iyp = r.get("initial_yes_prob")
    uyp = r.get("updated_yes_prob")
    dyp = r.get("delta_yes_prob")
    if iyp is None or dyp is None:
        return None, None, None
    if abs(iyp) > 1.5:
        iyp = iyp / 100
        if uyp is not None:
            uyp = uyp / 100
        dyp = dyp / 100
    iyp = max(0.0, min(1.0, iyp))
    return iyp, uyp, dyp


# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — Mean |Δp̂| by direction × model (causal selectivity)
# ══════════════════════════════════════════════════════════════════════════════

def _panel_A(ax: plt.Axes, records: list[dict]) -> None:
    # collect mean SIGNED Δp̂ per (model, direction).  Showing the sign (not |Δ|)
    # makes the direction selectivity visible as an up/down split about zero:
    # pro-H1 evidence pushes the forecast up (+), anti-H1 down (−), orthogonal ≈ 0.
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in records:
        d   = r.get("direction", "")
        mod = r.get("forecast_model", "")
        if d not in DIRECTIONS or mod not in MAIN_MODEL_KEYS:
            continue
        iyp, _, dyp = _normalise(r)
        if dyp is not None:
            buckets[(mod, d)].append(dyp)

    # bar positions: group by direction, models ordered small→large within group
    n_models  = len(MAIN_MODEL_KEYS)
    group_gap = 0.14          # tight spacing between pro / anti / orthogonal groups
    bar_w     = 0.085

    dir_positions = {
        "pro_H1":    0.0,
        "anti_H1":   n_models * bar_w + group_gap,
        "orthogonal":2 * (n_models * bar_w + group_gap),
    }

    # x in data coords, y in axes fraction — for per-bar logos / labels below 0
    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    max_abs = 0.0    # track extent (pp) for symmetric y-limits
    for m_idx, (key, _, size, frontier, _) in enumerate(MAIN_MODELS):
        fam       = _family(key)
        col       = FAMILY_COLOR[fam]          # qwen=purple, llama=blue, frontier=orange
        alpha_mod = 1.0 if frontier else 0.9
        tint      = col if fam in ("qwen", "llama") else None  # tint family marks only
        for direction in DIRECTIONS:
            vals = buckets.get((key, direction), [])
            if not vals:
                continue
            mean = np.mean(vals)
            se   = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            xc   = dir_positions[direction] + m_idx * bar_w + bar_w * 0.44
            ax.bar(xc, mean * 100, bar_w * 0.88,
                   color=col, alpha=alpha_mod, zorder=3)
            if se > 0:
                ax.errorbar(xc, mean * 100, yerr=se * 100,
                            fmt="none", color="black", capsize=1.5,
                            linewidth=0.7, zorder=4)
            max_abs = max(max_abs, abs(mean * 100) + se * 100)

        # logo (uniform display size) + size label beneath each bar.
        # Direction labels sit just under the axis (see below), so the per-bar
        # marks are pushed down to leave room for them.
        for direction in DIRECTIONS:
            xc = dir_positions[direction] + m_idx * bar_w + bar_w * 0.44
            oi = _oimage(key, target_px=9.0 * LOGO_SCALE.get(key, 1.0), tint=tint)
            if oi is not None:
                ab = AnnotationBbox(oi, (xc, -0.18),
                                    xycoords=blend, frameon=False,
                                    box_alignment=(0.5, 1.0),
                                    clip_on=False, zorder=5)
                ax.add_artist(ab)
            # text label beneath each bar: param size for local models,
            # model name for the frontier agents (DeepSeek / GPT / Claude)
            lbl = size if not frontier else FRONTIER_LABEL.get(key, size)
            lcol = "#6b7280" if not frontier else FAMILY_COLOR["frontier"]
            ax.text(xc, -0.27, lbl, transform=blend,
                    ha="center", va="top", fontsize=13, rotation=90,
                    color=lcol, clip_on=False,
                    fontweight="bold" if frontier else "normal")

    # direction group labels — placed just below the x-axis (closest to it)
    for direction in DIRECTIONS:
        cx = dir_positions[direction] + (n_models - 1) * bar_w / 2 + bar_w * 0.44
        ax.text(cx, -0.05, DIR_LABEL[direction], transform=blend,
                ha="center", va="top", fontsize=22,
                color=DIR_COLOR[direction], fontweight="bold",
                clip_on=False)

    # x-axis label (below the per-bar marks)
    x_mid = dir_positions["anti_H1"] + (n_models - 1) * bar_w / 2 + bar_w * 0.44
    ax.text(x_mid, -0.55,
            "Counterfactual evidence direction  (models ordered 8B → frontier)",
            transform=blend, ha="center", va="top", fontsize=19,
            color="#374151", clip_on=False)

    # symmetric limits about zero so the pro-up / anti-down split reads as a mirror
    lim = max(max_abs * 1.16, 1.0)
    ax.set_ylim(-lim, lim)
    # faint half-plane cue: above 0 the forecast was pushed toward YES, below toward NO
    ax.axhspan(0,   lim, color=C_PRO,  alpha=0.045, zorder=0, linewidth=0)
    ax.axhspan(-lim, 0,  color=C_ANTI, alpha=0.045, zorder=0, linewidth=0)
    ax.axhline(0, color="#4b5563", linewidth=1.2, zorder=2)
    ax.set_ylabel("Mean signed revision  $\\Delta\\hat{p}$  (pp)", fontsize=19)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlim(-bar_w * 0.5,
                dir_positions["orthogonal"] + n_models * bar_w + bar_w * 0.5)
    ax.set_xticks([])

    # sign cue (placed in the empty lower-right, below the near-zero orthogonal bars)
    ax.text(0.995, 0.02,
            "$\\blacktriangle$ above 0: forecast pushed toward YES\n"
            "$\\blacktriangledown$ below 0: forecast pushed toward NO",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
            color="#4b5563", linespacing=1.4,
            bbox=dict(fc="white", ec="#e5e7eb", lw=0.6, pad=4, alpha=0.9))

    # legend: model family (matches the Qwen/Llama colours in the other figures)
    leg = [
        mpatches.Patch(color=C_QWEN,  alpha=0.9, label="Qwen 2.5 (local)"),
        mpatches.Patch(color=C_LLAMA, alpha=0.9, label="Llama 3.x (local)"),
        mpatches.Patch(color=C_FRONT,            label="Frontier"),
    ]
    ax.legend(handles=leg, loc="upper right", bbox_to_anchor=(0.995, 1.0),
              handlelength=0.8, fontsize=15)

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Loading data …")
    _refresh_logos_from_pdf()
    records = _load_update_records()

    def _save(fig, stem):
        for ext in ("pdf", "png"):
            path = _OUT / f"{stem}.{ext}"
            mirror = _ICLR_OUT / path.name
            # PDF text/lines are vector (crisp at any zoom); render the PNG at
            # very high dpi so its rasterised text is sharp too. Save once and
            # copy so both maintained manuscript trees are byte-identical.
            fig.savefig(path, bbox_inches="tight", dpi=900 if ext == "png" else 300)
            shutil.copyfile(path, mirror)
            print(f"  Saved {path}")
            print(f"  Mirrored {mirror}")
        plt.close(fig)

    print("Composing exp1_direction_selective_updating (signed, single panel) …")
    fig = plt.figure(figsize=(9.4, 5.4))
    ax  = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.95, bottom=0.24)
    _panel_A(ax, records)
    _save(fig, "exp1_direction_selective_updating")
    print("Done.")


if __name__ == "__main__":
    main()
