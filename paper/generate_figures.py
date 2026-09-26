#!/usr/bin/env python3
"""Generate the Experiment 1 direction-selective-updating figure.

exp1_direction_selective_updating.pdf — single panel:
  Mean SIGNED Δp̂ by CF direction × model — causal selectivity
  (pro-H1 pushes right, anti-H1 left, orthogonal ≈ 0), one row per model.
Run from repo root:
    python paper/generate_figures.py
"""
from __future__ import annotations

import json
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
_OUT = Path(__file__).resolve().parent / "figures"
_LOGOS = _OUT / "logos"
_OUT.mkdir(exist_ok=True)

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
        png = _LOGOS / f"{stem}.png"
        if png.is_file() and png.stat().st_mtime >= pdf.stat().st_mtime:
            continue
        try:
            with tempfile.TemporaryDirectory() as td:
                base = os.path.join(td, "raw")
                subprocess.run(
                    ["pdftoppm", "-png", "-r", "600", "-singlefile", str(pdf), base],
                    check=True,
                    capture_output=True,
                )
                im = plt.imread(base + ".png")[..., :3]  # RGB in [0,1]
                lum = im.mean(axis=2)
                alpha = np.clip((1.0 - lum) * 255, 0, 255).astype(np.uint8)
                rgba = np.zeros((*alpha.shape, 4), np.uint8)  # pure black ink
                rgba[..., 3] = alpha
                from PIL import Image

                out = Image.fromarray(rgba)
                if out.getbbox():  # trim + square pad
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
        path = _LOGOS / fname
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
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e5e7eb",
        "grid.linewidth": 0.6,
        "axes.linewidth": 0.8,
        "legend.fontsize": 11,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#d1d5db",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# ── colours ───────────────────────────────────────────────────────────────────
C_PRO = "#159244"  # green (nudged slightly darker)
C_ANTI = "#cb2121"  # red   (nudged slightly darker)
C_ORTHO = "#7c3aed"
C_FRONT = "#ea580c"
C_QWEN = "#615CED"  # Qwen brand purple   (matches generate_paper_figures.py)
C_LLAMA = "#1877F2"  # Meta/Llama brand blue


def _family(key: str) -> str:
    """Model family from its brand logo: 'qwen', 'llama', or 'frontier'."""
    brand = MODEL_BRAND.get(key, "")
    return (
        "qwen"
        if brand == "qwen.png"
        else "llama"
        if brand == "llama.png"
        else "frontier"
    )


FAMILY_COLOR = {"qwen": C_QWEN, "llama": C_LLAMA, "frontier": C_FRONT}

DIR_COLOR = {"pro_H1": C_PRO, "anti_H1": C_ANTI, "orthogonal": C_ORTHO}
DIR_LABEL = {"pro_H1": "pro-$H_1$", "anti_H1": "anti-$H_1$", "orthogonal": "orthogonal"}
DIRECTIONS = ["pro_H1", "anti_H1", "orthogonal"]

# ordered model list — strictly ascending parameter count, frontier last
MODELS = [
    # key                 short label        size    is_frontier  brand logo
    ("qwen2.5:7b", "Qwen2.5-7B", "7B", False, "qwen.png"),
    ("llama3.1:8b", "Llama-3.1-8B", "8B", False, "llama.png"),
    ("qwen2.5:14b", "Qwen2.5-14B", "14B", False, "qwen.png"),
    ("qwen2.5:32b", "Qwen2.5-32B", "32B", False, "qwen.png"),
    ("llama3.3:70b", "Llama-3.3-70B", "70B", False, "llama.png"),
    ("llama3.1:70b", "Llama-3.1-70B", "70B", False, "llama.png"),
    ("qwen2.5:72b", "Qwen2.5-72B", "72B", False, "qwen.png"),
    ("DeepSeek-V4-Pro", "DeepSeek V4", "DS", True, "deepseek.png"),
    ("gpt-5.4", "GPT-5.4", "GPT", True, "openai.png"),
    ("claude-opus-4-8", "Claude Opus 4.8", "Claude", True, "claude.png"),
]
MODEL_BRAND = {m[0]: m[4] for m in MODELS}

# The main comparison includes only deployments whose archived revisions use a
# consistent probability scale. Qwen2.5-7B remains documented in the appendix.
MAIN_MODELS = [m for m in MODELS if m[0] != "qwen2.5:7b"]
MAIN_MODEL_KEYS = [m[0] for m in MAIN_MODELS]

# Row labels, in the order and spelling of the table beside the figure (Fig. 3b):
# hosted systems first, then open-weight models by descending parameter count.
ROW_ORDER = [
    ("claude-opus-4-8", "Opus 4.8"),
    ("gpt-5.4", "GPT-5.4"),
    ("DeepSeek-V4-Pro", "V4-Pro"),
    ("qwen2.5:72b", "Qwen2.5-72B"),
    ("llama3.3:70b", "Llama-3.3-70B"),
    ("llama3.1:70b", "Llama-3.1-70B"),
    ("qwen2.5:32b", "Qwen2.5-32B"),
    ("qwen2.5:14b", "Qwen2.5-14B"),
    ("llama3.1:8b", "Llama-3.1-8B"),
]
N_HOSTED = 3

# ── frozen clustered-uncertainty loader ──────────────────────────────────────
_CLUSTERED_PATH = _ROOT / "data" / "results" / "clustered_movement_uncertainty.json"


def _load_clustered_uncertainty() -> dict:
    """Load the frozen market-clustered movement estimates and intervals."""
    if not _CLUSTERED_PATH.exists():
        raise FileNotFoundError(
            f"Missing {_CLUSTERED_PATH}; run "
            "exp1_prospective/agent/evaluate_clustered_uncertainty.py"
        )
    report = json.loads(_CLUSTERED_PATH.read_text())
    bootstrap = report.get("bootstrap", {})
    if bootstrap.get("cluster_unit") != "market task_id":
        raise ValueError("Unexpected uncertainty artifact or cluster unit")
    if int(bootstrap.get("repetitions", 0)) < 1_000:
        raise ValueError("Clustered uncertainty artifact has too few repetitions")
    return report


# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — Mean |Δp̂| by direction × model (causal selectivity)
# ══════════════════════════════════════════════════════════════════════════════


def _panel_A(ax: plt.Axes, clustered: dict) -> None:
    # Mean signed revision and its 95% interval come from the frozen
    # market-clustered bootstrap. Showing the sign makes selectivity visible as
    # a left/right split: pro-H1 evidence pushes right, anti-H1 pushes left, and
    # orthogonal evidence remains near zero. One row per model keeps the labels
    # horizontal and aligned with the table beside the figure.
    uncertainty = clustered["per_model"]
    bar_h = 0.26
    offsets = {"pro_H1": -bar_h, "anti_H1": 0.0, "orthogonal": bar_h}
    hosted_gap = 0.45  # extra space separating hosted from open-weight rows

    y_rows = [i + (hosted_gap if i >= N_HOSTED else 0.0) for i in range(len(ROW_ORDER))]
    max_abs = 0.0
    for y, (key, _) in zip(y_rows, ROW_ORDER):
        for direction in DIRECTIONS:
            row = uncertainty[key]["signed_revision"][direction]
            mean = float(row["estimate"]) * 100
            ci_low = float(row["ci_low"]) * 100
            ci_high = float(row["ci_high"]) * 100
            yc = y + offsets[direction]
            ax.barh(yc, mean, bar_h * 0.9, color=DIR_COLOR[direction], zorder=3)
            ax.errorbar(
                mean,
                yc,
                xerr=np.asarray([[mean - ci_low], [ci_high - mean]]),
                fmt="none",
                color="black",
                capsize=1.5,
                linewidth=0.8,
                zorder=4,
            )
            max_abs = max(max_abs, abs(ci_low), abs(ci_high))

    lim = max(max_abs * 1.08, 1.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(y_rows[-1] + 0.6, y_rows[0] - 0.6)  # first model at the top
    ax.axvline(0, color="#4b5563", linewidth=1.2, zorder=2)
    ax.axhline((y_rows[N_HOSTED - 1] + y_rows[N_HOSTED]) / 2, color="#9ca3af",
               linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.set_yticks(y_rows)
    ax.set_yticklabels([label for _, label in ROW_ORDER], fontsize=16)
    for tick, (key, _) in zip(ax.get_yticklabels(), ROW_ORDER):
        tick.set_color(FAMILY_COLOR[_family(key)])
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean signed revision $\\Delta\\hat{p}$ (pp)", fontsize=16)

    leg = [mpatches.Patch(color=DIR_COLOR[d], label=DIR_LABEL[d]) for d in DIRECTIONS]
    ax.legend(
        handles=leg,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        handlelength=0.9,
        columnspacing=1.0,
        fontsize=16,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    print("Loading data …")
    _refresh_logos_from_pdf()
    clustered = _load_clustered_uncertainty()
    print(
        "  Cluster bootstrap: "
        f"{clustered['bootstrap']['repetitions']:,} market resamples"
    )

    def _save(fig, stem):
        for ext in ("pdf", "png"):
            path = _OUT / f"{stem}.{ext}"
            # PDF text/lines are vector (crisp at any zoom); render the PNG at
            # very high dpi so its rasterised text is sharp too.
            fig.savefig(path, bbox_inches="tight", dpi=900 if ext == "png" else 300)
            print(f"  Saved {path}")
        plt.close(fig)

    print("Composing exp1_direction_selective_updating (signed, single panel) …")
    # Fig. 3 places this asset at about half text width (~2.6 in), so the
    # 5.2 in source canvas prints at half scale: 16 pt labels print near 8 pt.
    fig = plt.figure(figsize=(5.2, 4.6))
    ax = fig.add_subplot(1, 1, 1)
    _panel_A(ax, clustered)
    _save(fig, "exp1_direction_selective_updating")
    print("Done.")


if __name__ == "__main__":
    main()
