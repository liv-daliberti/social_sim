#!/usr/bin/env python3
"""Generate the paper's combined Experiment 1 scale figure.

Produces (in paper/figures/):
  exp1_ehc_sensitivity_combined.pdf — EHC and sensitivity side by side

Run from repo root:
    python paper/generate_paper_figures.py
"""

from __future__ import annotations

import json
import math
import shutil
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import matplotlib.colors as mcolors
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.image import BboxImage
from matplotlib.legend_handler import HandlerBase
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
_ROOT  = Path(__file__).resolve().parent.parent / "exp1_prospective"
_OUT   = Path(__file__).resolve().parent / "figures"
_ICLR_OUT = Path(__file__).resolve().parent / "ICLR" / "figures"
_LOGOS = _OUT / "logos"
_OUT.mkdir(exist_ok=True)
_ICLR_OUT.mkdir(exist_ok=True)

# ── font / style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    # Embed real TrueType outlines (Type 42) instead of matplotlib's default
    # Type 3 fonts, which many PDF viewers render soft/blurry. Type 42 keeps
    # text crisp at any zoom and selectable/searchable in the PDF.
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
    # DejaVu Sans first: it embeds as pure TrueType (FontFile2) with no
    # font-type mismatch, so text renders crisply in every PDF viewer. Nimbus
    # Sans is an OpenType/CFF font that, under pdf.fonttype=42, embeds with a
    # type mismatch that some viewers render soft.
    "font.family":           "sans-serif",
    "font.sans-serif":       ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size":             9,
    "axes.titlesize":        10,
    "axes.labelsize":        9,
    "xtick.labelsize":       8.5,
    "ytick.labelsize":       8.5,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.grid":             True,
    "grid.color":            "#e5e7eb",
    "grid.linewidth":        0.6,
    "axes.linewidth":        0.8,
    "xtick.major.width":     0.8,
    "ytick.major.width":     0.8,
    "legend.fontsize":       8,
    "legend.framealpha":     0.9,
    "legend.edgecolor":      "#d1d5db",
    "figure.dpi":            150,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
})

# ── colour constants ───────────────────────────────────────────────────────────
C_PRO    = "#16a34a"   # green
C_ANTI   = "#dc2626"   # red
C_ORTHO  = "#6b7280"   # gray
C_LOCAL  = "#2563eb"   # blue (fallback)
C_QWEN   = "#615CED"   # Qwen brand purple
C_LLAMA  = "#1877F2"   # Facebook/Meta brand blue — Llama family
C_FRONT  = "#ea580c"   # orange

_MODEL_FAMILY = {
    "qwen2.5:7b":   "qwen",
    "qwen2.5:14b":  "qwen",
    "qwen2.5:32b":  "qwen",
    "qwen2.5:72b":  "qwen",
    "llama3.1:8b":  "llama",
    "llama3.1:70b": "llama",
    "llama3.3:70b": "llama",
}

def _local_color(key: str) -> str:
    fam = _MODEL_FAMILY.get(key, "local")
    return C_QWEN if fam == "qwen" else C_LLAMA if fam == "llama" else C_LOCAL

# ── model catalogue (ordered for x-axis) ──────────────────────────────────────
#   key         label             param_class   frontier?
MODELS = [
    ("llama3.1:8b",       "Llama 3.1-8B",    "8B",     False),
    ("qwen2.5:7b",        "Qwen 2.5-7B",     "7B",     False),
    ("qwen2.5:14b",       "Qwen 2.5-14B",    "14B",    False),
    ("qwen2.5:32b",       "Qwen 2.5-32B",    "32B",    False),
    ("qwen2.5:72b",       "Qwen 2.5-72B",    "72B",    False),
    ("llama3.1:70b",      "Llama 3.1-70B",   "70B",    False),
    ("llama3.3:70b",      "Llama 3.3-70B",   "70B",    False),
    ("DeepSeek-V4-Pro",   "DeepSeek V4-Pro", "DS",     True),
    ("gpt-5.4",           "GPT-5.4",         "GPT",    True),
    ("claude-opus-4-8",   "Claude Opus 4.8", "Claude", True),
]

# Logos for frontier models — PNG files in paper/figures/logos/
FRONTIER_LOGOS = {
    "DeepSeek-V4-Pro":  "deepseek",
    "gpt-5.4":          "openai",
    "claude-opus-4-8":  "claude",   # Claude starburst mark (not Anthropic "AI\" lettermark)
}
_LOGO_CACHE: dict[str, np.ndarray] = {}

# Per-logo zoom multipliers — Claude starburst is visually lighter than the
# DeepSeek whale / OpenAI swirl so it needs a larger zoom to match visual weight.
_LOGO_ZOOM_SCALE = {
    "deepseek": 1.0,
    "openai":   1.0,
    "claude":   1.45,
}

def _get_logo(name: str) -> np.ndarray | None:
    if name in _LOGO_CACHE:
        return _LOGO_CACHE[name]
    path = _LOGOS / f"{name}.png"
    if not path.exists():
        return None
    img = plt.imread(str(path))
    _LOGO_CACHE[name] = img
    return img


# Local-model family marks — the Qwen hexagram and Llama silhouette, drawn in
# place of plain dots.  Source art is black-on-transparent; we tint each to its
# family colour so the blue/red Qwen/Llama coding is preserved.
LOCAL_LOGOS      = {"qwen": "qwen", "llama": "llama"}
_LOCAL_LOGO_TINT = {"qwen": C_QWEN, "llama": C_LLAMA}
# Per-family display zoom (tuned so both marks read at a similar visual size to
# the frontier logos despite different source-pixel dimensions / ink coverage).
_LOCAL_LOGO_ZOOM = {"qwen": 0.018, "llama": 0.0108}
_LOCAL_LOGO_CACHE: dict[tuple[str, str], np.ndarray] = {}

def _get_local_logo(family: str) -> np.ndarray | None:
    """Return the family logo recoloured to its family tint (alpha preserved)."""
    name = LOCAL_LOGOS.get(family)
    tint = _LOCAL_LOGO_TINT.get(family)
    if name is None or tint is None:
        return None
    ck = (name, tint)
    if ck in _LOCAL_LOGO_CACHE:
        return _LOCAL_LOGO_CACHE[ck]
    img = _get_logo(name)
    if img is None:
        return None
    out = np.array(img, dtype=float)
    if out.ndim == 3 and out.shape[2] == 4:
        r, g, b = mcolors.to_rgb(tint)
        out[..., 0], out[..., 1], out[..., 2] = r, g, b   # keep original alpha
    _LOCAL_LOGO_CACHE[ck] = out
    return out


class _HandlerLogo(HandlerBase):
    """Legend handler that draws a logo image (aspect-preserved) as the marker."""
    def __init__(self, img: np.ndarray):
        self._img = img
        super().__init__()

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        h_px, w_px = self._img.shape[0], self._img.shape[1]
        asp = w_px / h_px
        box_h = height
        box_w = box_h * asp
        x0 = xdescent + (width - box_w) / 2
        bb  = mtransforms.Bbox.from_bounds(x0, ydescent, box_w, box_h)
        tbb = mtransforms.TransformedBbox(bb, trans)
        image = BboxImage(tbb, interpolation="antialiased")
        image.set_data(self._img)
        return [image]


# ── load consistency report ────────────────────────────────────────────────────
def _load_report() -> dict:
    reports = sorted((_ROOT / "data" / "results").glob("consistency_report_*.json"),
                     reverse=True)
    if not reports:
        raise FileNotFoundError("No consistency report found")
    return json.loads(reports[0].read_text())



# ══════════════════════════════════════════════════════════════════════════════
# Shared layout for scale-axis figures
# ══════════════════════════════════════════════════════════════════════════════

# x positions: local models on log₂(param_count) scale so that proportional
# distances reflect actual size differences (7B→14B = ×2 = same log gap as
# 14B→28B, etc.).  Frontier models placed after a fixed gap.
_LOG2_MIN   = math.log2(7)
_LOG2_MAX   = math.log2(72)
_LOG_SPAN   = _LOG2_MAX - _LOG2_MIN  # ≈ 3.363

def _lx(b: float, total: float = 5.4) -> float:
    """Log₂-scale x position for a model with b billion parameters."""
    return (math.log2(b) - _LOG2_MIN) / _LOG_SPAN * total

_XPOS = {
    # local — log₂-scaled; 7B and 8B are nearly coincident (correctly so)
    "qwen2.5:7b":       _lx(7),           # 0.00
    "llama3.1:8b":      _lx(8),           # 0.31
    "qwen2.5:14b":      _lx(14),          # 1.61
    "qwen2.5:32b":      _lx(32),          # 3.52
    "llama3.1:70b":     _lx(70) - 0.56,  # 4.78 (jitter left — needs 0.65+ unit gap for rotated labels)
    "llama3.3:70b":     _lx(70) + 0.08,  # 5.42 (jitter right)
    "qwen2.5:72b":      _lx(72) + 0.55,  # 5.95
    # frontier — placed after separator; spacing is visual not parametric.
    # Spaced 1.3 units apart so the wide multi-line rotated tick labels
    # ("GPT 5.4 …" / "Claude Opus 4.8 …") don't collide in the narrower
    # combined-figure panels.
    "DeepSeek-V4-Pro":  7.2,
    "gpt-5.4":          8.5,
    "claude-opus-4-8":  9.8,
}
_XLABELS = {
    "llama3.1:8b":      "8B",
    "qwen2.5:7b":       "7B",
    "qwen2.5:14b":      "14B",
    "qwen2.5:32b":      "32B",
    "llama3.1:70b":     "L3.1\n70B",
    "llama3.3:70b":     "L3.3\n70B",
    "qwen2.5:72b":      "72B",
    "DeepSeek-V4-Pro":  "DS\nV4-Pro\n(1.6T MoE)",
    "gpt-5.4":          "GPT\n5.4\n(est. ~3T)",
    "claude-opus-4-8":  "Claude\nOpus 4.8\n(~5.3T MoE)",
}

# pts per x-data-unit for a standalone 7-inch-wide figure
_FIG_W_IN  = 7.0
_XLIM_SPAN = 10.7  # 10.4 - (-0.3)
_PTS_PER_X = _FIG_W_IN * 72 / _XLIM_SPAN   # ≈ 49.9 pts/unit


# ── core drawing helper ────────────────────────────────────────────────────────

def _draw_scale_panel(
    ax: plt.Axes,
    values: dict[str, float],
    ymin: float,
    ymax: float,
    yformat: str,                             # "pct" or "x"
    label_offsets: dict[str, tuple[float, float]],  # model_key → (dx_pts, flip_above)
    ylabel: str,
    title: str = "",
    yticks: list[float] | None = None,
    xlabel: str | None = None,
    show_zone_labels: bool = True,
    logo_zoom: float = 0.10,
    pts_per_x: float = _PTS_PER_X,
    zone_fs: float = 8.0,
    val_fs: float = 7.0,
    tick_fs: float = 7.5,
    model_keys: tuple[str, ...] | None = None,
) -> None:
    """Draw a single scale-axis panel onto an existing Axes object.

    Extracted so that both standalone figures and the combined side-by-side
    figure can reuse the same rendering logic.

    ``zone_fs`` / ``val_fs`` / ``tick_fs`` control the in-panel zone-label,
    value-annotation, and x-tick-label font sizes so the combined figure can
    use larger text (it is downscaled less aggressively in the paper).
    """
    sep_x = 6.55
    ax.axvline(sep_x, color="#94a3b8", linewidth=1.0, linestyle="--", zorder=1)

    if show_zone_labels:
        # Blended transform: x in data coords, y in axes fraction (avoids
        # phantom-text artifacts from placing text near ymax in data coords)
        _blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(sep_x - 0.18, 0.97, "Local models",
                transform=_blended, ha="right", va="top", fontsize=zone_fs, color="#64748b")
        ax.text(sep_x + 0.18, 0.97, "Frontier systems",
                transform=_blended, ha="left",  va="top", fontsize=zone_fs, color="#64748b")

    # Value-label offset: ~3.5% of y range above/below the data point.
    # Using data coordinates rather than textcoords="offset points" prevents
    # the tight-bbox renderer from capturing annotation bboxes above the axes
    # and producing phantom text at the top of the saved PDF.
    y_off = (ymax - ymin) * 0.035

    shown_models = [row for row in MODELS if model_keys is None or row[0] in model_keys]

    for key, _, _, frontier in shown_models:
        y = values.get(key)
        if y is None:
            continue
        x = _XPOS[key]

        logo_name = FRONTIER_LOGOS.get(key)
        if logo_name is not None:
            logo_img       = _get_logo(logo_name)
            effective_zoom = logo_zoom * _LOGO_ZOOM_SCALE.get(logo_name, 1.0)
        else:
            fam            = _MODEL_FAMILY.get(key, "local")
            logo_img       = _get_local_logo(fam)
            effective_zoom = _LOCAL_LOGO_ZOOM.get(fam, 0.02)

        if logo_img is not None:
            imagebox = OffsetImage(logo_img, zoom=effective_zoom)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, zorder=4,
                                box_alignment=(0.5, 0.5))
            ax.add_artist(ab)
        else:
            col = _local_color(key)
            ax.scatter(x, y, color=col, marker="o", s=90, zorder=4,
                       edgecolors="white", linewidths=0.8)

        # Value annotation.  Round half-up via Decimal so e.g. 6.35 -> "6.4×"
        # matches the manuscript tables (plain %.1f would float-round to 6.3).
        dx_pts, flip = label_offsets.get(key, (0, False))
        if yformat == "pct":
            val_str = f"{y:.1%}"
        else:
            _r = float(Decimal(str(y)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
            val_str = f"{_r:.1f}×"
        x_text  = x + dx_pts / pts_per_x
        y_text  = y + y_off if flip else y - y_off
        va      = "bottom" if flip else "top"
        ax.text(x_text, y_text, val_str,
                ha="center", va=va, fontsize=val_fs, color="#475569", zorder=5)

    # x-axis tick labels — rotated so closely spaced small-model labels don't clash
    ax.set_xticks([_XPOS[k] for k, _, _, _ in shown_models])
    ax.set_xticklabels([_XLABELS[k] for k, _, _, _ in shown_models], fontsize=tick_fs,
                       linespacing=1.2, rotation=38, ha="right")
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if yticks is not None:
        ax.set_yticks(yticks)
    if yformat == "pct":
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    else:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}×" if v > 0 else "0")

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.3, 10.4)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9.5, pad=7)

    # Legend: Qwen and Llama families, shown with their tinted logo marks
    # (frontier systems are identified by their own logos in-plot).
    qwen_logo  = _get_local_logo("qwen")
    llama_logo = _get_local_logo("llama")
    if qwen_logo is not None and llama_logo is not None:
        q_proxy, l_proxy = mpatches.Patch(), mpatches.Patch()
        ax.legend(
            [q_proxy, l_proxy], ["Qwen", "Llama"],
            handler_map={q_proxy: _HandlerLogo(qwen_logo),
                         l_proxy: _HandlerLogo(llama_logo)},
            loc="upper left", handletextpad=0.6, labelspacing=0.7,
        )
    else:
        leg_handles = [
            mlines.Line2D([], [], color=C_QWEN,  marker="o", linestyle="none",
                          markersize=7, label="Qwen"),
            mlines.Line2D([], [], color=C_LLAMA, marker="o", linestyle="none",
                          markersize=7, label="Llama"),
        ]
        ax.legend(handles=leg_handles, loc="upper left", handletextpad=0.5)




# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Sensitivity ratio by model scale
# ══════════════════════════════════════════════════════════════════════════════

_SENS_LABEL_OFFSETS = {
    "llama3.1:8b":      ( 0,  True),   # above — 8B/7B are close in x
    "qwen2.5:7b":       ( 0,  False),
    "qwen2.5:14b":      ( 0,  False),
    "qwen2.5:32b":      ( 0,  False),
    "llama3.1:70b":     ( 0,  True),   # above — L3.1 and L3.3 are close
    "llama3.3:70b":     ( 0,  False),
    "qwen2.5:72b":      ( 0,  False),
    "DeepSeek-V4-Pro":  (-18, False),
    "gpt-5.4":          (  0, False),
    "claude-opus-4-8":  ( 18, False),
}

# Qwen 2.5-7B is intentionally excluded from sensitivity panels: its archived
# absolute revisions mix 0--1 and 0--100 scales, so the derived ratio is not a
# comparable measurement. It remains in EHC, which does not use magnitudes.
_SENSITIVITY_MODEL_KEYS = tuple(
    key for key, _, _, _ in MODELS if key != "qwen2.5:7b"
)

# ══════════════════════════════════════════════════════════════════════════════
# EHC label placement
# ══════════════════════════════════════════════════════════════════════════════

_EHC_LABEL_OFFSETS = {
    "llama3.1:8b":      (  0,  True),   # above — 8B/7B share nearly same x
    "qwen2.5:7b":       (  0,  False),
    "qwen2.5:14b":      (  0,  False),
    "qwen2.5:32b":      (  0,  False),
    # 70B cluster: L3.1 above, L3.3 below, Q72B below at distinct x
    "llama3.1:70b":     (  0,  True),
    "llama3.3:70b":     (  0,  False),
    "qwen2.5:72b":      (  0,  False),
    "DeepSeek-V4-Pro":  (-18,  False),
    "gpt-5.4":          (  0,  False),
    "claude-opus-4-8":  ( 18,  False),
}

# ══════════════════════════════════════════════════════════════════════════════
# EHC + sensitivity combined side-by-side
# ══════════════════════════════════════════════════════════════════════════════

def fig_combined_ehc_sensitivity(report: dict) -> None:
    """Two-panel figure: EHC (left) and sensitivity ratio (right) vs model scale."""

    ehc_values  = {}
    sens_values = {}
    for key, _, _, _ in MODELS:
        pm = report["per_model"].get(key, {})
        ehc  = pm.get("consistency", {}).get("EHC_rate")
        sens = pm.get("anchoring", {}).get("sensitivity_ratio")
        if ehc  is not None: ehc_values[key]  = ehc
        if sens is not None and key in _SENSITIVITY_MODEL_KEYS:
            sens_values[key] = sens

    # Layout: a less-extreme aspect ratio than before (was 14×4.4) so the paper
    # downscales it far less aggressively — the previous 3.2:1 figure shrank all
    # text to near-illegible sizes at text width.  Tight margins + small wspace
    # keep the two panels as large as possible.
    _fig_w, _fig_h = 13.5, 4.9
    _left, _right, _wspace = 0.06, 0.98, 0.16

    # Each subplot panel is narrower than the standalone 7-inch figure, so
    # recalculate pts_per_x to keep dx_pts label offsets visually consistent.
    # For two equal panels: panel_frac = (right-left) / (2 + wspace).
    _panel_w_in     = _fig_w * (_right - _left) / (2 + _wspace)
    _pts_per_x_comb = _panel_w_in * 72 / _XLIM_SPAN

    # Larger in-panel text: the combined figure is downscaled less in the paper,
    # and these sizes restore legibility at print scale.
    _panel_kw = dict(
        show_zone_labels=True,
        logo_zoom=0.10,
        pts_per_x=_pts_per_x_comb,
        zone_fs=10.0,
        val_fs=9.5,
        tick_fs=9.5,
    )

    rc = {
        "savefig.bbox":    None,
        "axes.labelsize":  12,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.5,
    }
    with plt.rc_context(rc):
        fig, (ax_ehc, ax_sens) = plt.subplots(1, 2, figsize=(_fig_w, _fig_h))

        _draw_scale_panel(
            ax_ehc, ehc_values,
            ymin=0.84, ymax=1.005,
            yformat="pct",
            label_offsets=_EHC_LABEL_OFFSETS,
            ylabel="Evidence–Hypothesis Consistency (EHC)",
            xlabel="Model scale / system class",
            yticks=[0.85, 0.90, 0.95, 1.00],
            **_panel_kw,
        )

        _draw_scale_panel(
            ax_sens, sens_values,
            ymin=0, ymax=30,
            yformat="x",
            label_offsets=_SENS_LABEL_OFFSETS,
            ylabel="Sensitivity ratio\n(mean directional / mean orthogonal $|\\Delta\\hat{p}|$)",
            xlabel="Model scale / system class",
            yticks=[0, 5, 10, 15, 20, 25, 30],
            model_keys=_SENSITIVITY_MODEL_KEYS,
            **_panel_kw,
        )

        fig.subplots_adjust(
            left=_left, right=_right,
            bottom=0.26, top=0.97,
            wspace=_wspace,
        )
        out = _OUT / "exp1_ehc_sensitivity_combined.pdf"
        mirror = _ICLR_OUT / out.name
        fig.savefig(out, dpi=600)
        shutil.copyfile(out, mirror)
        print(f"  Saved {out}")
        print(f"  Mirrored {mirror}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Loading data …")
    report = _load_report()
    print(f"  Report: {report['date']}")

    print("Generating combined figure …")
    fig_combined_ehc_sensitivity(report)
    print("Done.")


if __name__ == "__main__":
    main()
