#!/usr/bin/env python3
"""Draw the Coin City causal activation-patching result.

The figure walks one held-out episode through the intervention itself: two
prompts that differ only in the City C label tokens, the forward pass in which
the donor's hidden states at those tokens are written over the recipient's, and
the forecast that comes out, placed on the responsiveness scale the episode's
two reference cities define. The across-episode averages live in the text and in
the appendix table, so they are not repeated here.

The displayed episode is selected by a fixed, documented rule rather than by
hand: among sealed episodes whose effect is positive in both patch directions,
the script takes the one whose symmetric effect is closest to the reported mean.
The script reads the frozen powered run and fails closed if its provenance or
stability gate does not match the paper.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT
    / "exp2_v2"
    / "biased_news"
    / "data"
    / "coin_city_stable_relationship_claude_n250_v4"
    / "mechanistic_probe"
    / "qwen3_14b_symbol_relational_v2"
)
RESULTS = RUN_DIR / "relational_probe_results.json"
MANIFEST = RUN_DIR / "task_manifest.json"
PROTOCOL = RUN_DIR / "relational_protocol_manifest.json"
PREFLIGHT = RUN_DIR / "tokenization_preflight.json"
TASKS = RUN_DIR / "tasks.jsonl"
GENERATIONS = RUN_DIR / "activation_patch_generations.jsonl"
OUTPUT_PDF = ROOT / "paper" / "figures" / "exp2_causal_patch.pdf"
OUTPUT_PNG = ROOT / "paper" / "figures" / "exp2_causal_patch.png"
EXPECTED_STUDY = "qwen3_14b_symbol_relational_v2"
EXPECTED_SEALED_EPISODES = 88
CONDITION = "cross_selected_window"

# The donor regime is blue, the recipient's own regime green.
DONOR = "#0072B2"
DONOR_FILL = "#DCEAF6"
RECIPIENT = "#1B7837"
RECIPIENT_FILL = "#DFEFE0"
INK = "#222222"
NEUTRAL = "#6A6A6A"
RULE = "#9A9A9A"

# Each stage of the walkthrough gets a pastel panel; the last one gets none.
STAGES = {
    1: {"fill": "#FCF3DA", "badge": "#B8871B"},
    2: {"fill": "#FBE8D8", "badge": "#BC5E28"},
    3: {"fill": "#EFEAF7", "badge": "#69499C"},
    4: {"fill": None, "badge": "#5F5F5F"},
}


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 7.4,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load() -> dict[str, dict[str, float]]:
    results = json.loads(RESULTS.read_text())
    manifest = json.loads(MANIFEST.read_text())
    if results.get("status") != "complete":
        raise SystemExit(f"status is {results.get('status')!r}")
    if results.get("study") != EXPECTED_STUDY:
        raise SystemExit(f"study is {results.get('study')!r}")
    if manifest.get("episode_counts", {}).get("test") != EXPECTED_SEALED_EPISODES:
        raise SystemExit("the sealed test does not contain 88 episodes")
    if not results.get("stability_gate", {}).get("passed"):
        raise SystemExit("the frozen stability gate did not pass")
    conditions = results.get("patching", {}).get("conditions", {}).get(CONDITION)
    if not conditions or any(depth not in conditions for depth in ("k0", "k4")):
        raise SystemExit(f"missing primary patch condition {CONDITION!r}")
    return conditions


def episode_effects(
    tasks: list[dict], generations: list[dict]
) -> dict[int, dict[str, float]]:
    """Reproduce the per-episode k=0 effect the analysis averages."""
    by_key = {
        (r["episode"], r["c_cases"], r["recipient_arm"], r["condition"]): r
        for r in generations
    }
    donor_task = {r["episode"]: r for r in tasks if r["c_cases"] == 0 and r["arm"] == "abc_context"}
    effects: dict[int, dict[str, float]] = {}
    for episode, task in donor_task.items():
        try:
            correct_base = by_key[(episode, 0, "abc_context", "unpatched")]
            wrong_base = by_key[(episode, 0, "abc_wrong_context", "unpatched")]
            wrong_patched = by_key[(episode, 0, "abc_wrong_context", CONDITION)]
            correct_patched = by_key[(episode, 0, "abc_context", CONDITION)]
        except KeyError:
            continue
        quad = (correct_base, wrong_base, wrong_patched, correct_patched)
        if not all(row["parsed"] for row in quad):
            continue
        sign = 1.0 if task["target_strong"] else -1.0
        forward = sign * (wrong_patched["implied_slope"] - wrong_base["implied_slope"])
        reverse = sign * (correct_base["implied_slope"] - correct_patched["implied_slope"])
        effects[episode] = {
            "forward": float(forward),
            "reverse": float(reverse),
            "symmetric": 0.5 * float(forward + reverse),
        }
    return effects


def load_example(mean_effect: float) -> dict:
    """Pick and describe the displayed episode.

    The rule is fixed in advance: among sealed episodes where the patch moves
    the forecast toward the donor in *both* directions, take the one whose
    symmetric effect is nearest the reported mean, so the walked-through case
    is representative rather than the largest available.
    """
    tasks = read_jsonl(TASKS)
    generations = read_jsonl(GENERATIONS)
    effects = episode_effects(tasks, generations)
    sealed = {
        r["episode"]
        for r in tasks
        if r["c_cases"] == 0 and r["arm"] == "abc_context" and r["split"] == "test"
    }
    eligible = [
        episode
        for episode, value in effects.items()
        if episode in sealed and value["forward"] > 0 and value["reverse"] > 0
    ]
    if not eligible:
        raise SystemExit("no sealed episode moves toward the donor in both directions")
    episode = min(eligible, key=lambda e: abs(effects[e]["symmetric"] - mean_effect))

    def one(rows: list[dict], description: str) -> dict:
        if len(rows) != 1:
            raise SystemExit(f"{description}: expected one row, found {len(rows)}")
        return rows[0]

    def task(arm: str) -> dict:
        return one(
            [
                r
                for r in tasks
                if r["episode"] == episode and r["c_cases"] == 0 and r["arm"] == arm
            ],
            f"example {arm} prompt",
        )

    def generation(arm: str, condition: str) -> dict:
        return one(
            [
                r
                for r in generations
                if r["episode"] == episode
                and r["c_cases"] == 0
                and r["recipient_arm"] == arm
                and r["condition"] == condition
            ],
            f"example {arm}/{condition} forecast",
        )

    donor, recipient = task("abc_context"), task("abc_wrong_context")
    if donor["target_symbol"] == recipient["target_symbol"]:
        raise SystemExit("the example prompts do not carry opposite City C labels")

    donor_base = generation("abc_context", "unpatched")
    recipient_base = generation("abc_wrong_context", "unpatched")
    recipient_self = generation("abc_wrong_context", "self_selected_window")
    recipient_patched = generation("abc_wrong_context", CONDITION)
    for row in (donor_base, recipient_base, recipient_self, recipient_patched):
        if not row["parsed"]:
            raise SystemExit("an example forecast did not parse")
    if recipient_self["predicted_poll"] != recipient_base["predicted_poll"]:
        raise SystemExit("the example self-patch control is not a no-op")

    layers = list(recipient_patched["patched_hidden_state_layers"])
    if not layers:
        raise SystemExit("the example patch records no layers")

    # City A carries `strong_symbol` when it is the strong reference.
    if donor["strong_reference_city"] == "A":
        a_symbol, b_symbol = donor["strong_symbol"], donor["weak_symbol"]
    else:
        a_symbol, b_symbol = donor["weak_symbol"], donor["strong_symbol"]
    return {
        "episode": episode,
        "a_symbol": str(a_symbol),
        "b_symbol": str(b_symbol),
        "a_slope": float(donor["reference_a_ols"]),
        "b_slope": float(donor["reference_b_ols"]),
        "donor_symbol": str(donor["target_symbol"]),
        "recipient_symbol": str(recipient["target_symbol"]),
        "start": float(donor["query_starting_poll"]),
        "news": int(donor["query_net_news"]),
        "true_slope": float(donor["target_slope"]),
        "donor_poll": float(donor_base["predicted_poll"]),
        "donor_slope": float(donor_base["implied_slope"]),
        "unpatched_poll": float(recipient_base["predicted_poll"]),
        "unpatched_slope": float(recipient_base["implied_slope"]),
        "patched_poll": float(recipient_patched["predicted_poll"]),
        "patched_slope": float(recipient_patched["implied_slope"]),
        "layers": layers,
        "effect": effects[episode]["symmetric"],
    }


def load_geometry() -> dict:
    """Layer count and label-token span, read from the frozen protocol."""
    protocol = json.loads(PROTOCOL.read_text())
    preflight = json.loads(PREFLIGHT.read_text())
    controls = " ".join(protocol["causal_patching"]["controls"])
    match = re.search(r"hidden-state layers 0--(\d+)", controls)
    if not match:
        raise SystemExit("the protocol does not state the causally effective layers")
    if not preflight["intervention"]["differences_exactly_city_c_symbol_span"]:
        raise SystemExit("the two prompts differ outside the City C label span")
    span = int(preflight["prefix"]["label_span_token_count"])
    if span != int(preflight["intervention"]["differing_token_count"]):
        raise SystemExit("the differing tokens are not the label span")
    return {"top_layer": int(match.group(1)), "label_tokens": span}


def stage_panel(axis, number, title, x, y, width, height):
    """A pastel panel with a numbered badge, marking one step of the walkthrough."""
    stage = STAGES[number]
    if stage["fill"]:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.0,rounding_size=0.012",
                linewidth=0.0,
                facecolor=stage["fill"],
                transform=axis.transAxes,
                zorder=0,
            )
        )
    badge_x, badge_y = x + 0.026, y + height - fh(0.105)
    axis.plot(
        [badge_x],
        [badge_y],
        marker="o",
        markersize=10.0,
        color=stage["badge"],
        transform=axis.transAxes,
        zorder=2,
    )
    axis.text(
        badge_x,
        badge_y,
        str(number),
        fontsize=6.1,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        transform=axis.transAxes,
        zorder=3,
    )
    axis.text(
        badge_x + 0.026,
        badge_y,
        title,
        fontsize=7.0,
        fontweight="bold",
        color=INK,
        ha="left",
        va="center",
        transform=axis.transAxes,
        zorder=3,
    )
    return badge_y


def prompt_card(axis, x, y, width, height, title, example, *, donor: bool):
    """One prompt, drawn so the single differing line is the salient one."""
    accent = DONOR if donor else RECIPIENT
    fill = DONOR_FILL if donor else RECIPIENT_FILL
    symbol = example["donor_symbol"] if donor else example["recipient_symbol"]
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            linewidth=0.9,
            edgecolor=accent,
            facecolor="white",
            transform=axis.transAxes,
            zorder=2,
        )
    )
    axis.text(
        x + width / 2,
        y + height - fh(0.070),
        title,
        fontsize=6.4,
        fontweight="bold",
        color=accent,
        ha="center",
        va="center",
        transform=axis.transAxes,
        zorder=3,
    )
    axis.text(
        x + width / 2,
        y + height - fh(0.155),
        f"City A  ·  {example['a_symbol']}     "
        f"City B  ·  {example['b_symbol']}",
        fontsize=6.0,
        color=NEUTRAL,
        ha="center",
        va="center",
        transform=axis.transAxes,
        zorder=3,
    )
    axis.add_patch(
        FancyBboxPatch(
            (x + 0.010, y + fh(0.013)),
            width - 0.020,
            fh(0.090),
            boxstyle="round,pad=0.003,rounding_size=0.008",
            linewidth=0.0,
            facecolor=fill,
            transform=axis.transAxes,
            zorder=3,
        )
    )
    axis.text(
        x + width / 2,
        y + fh(0.058),
        f"City C  \u00b7  {symbol}",
        fontsize=7.0,
        fontweight="bold",
        color=accent,
        ha="center",
        va="center",
        transform=axis.transAxes,
        zorder=4,
    )


def layer_stack(axis, x, width, bottom, top, example, geometry, *, donor: bool):
    """A schematic residual stream, layer 0 at the top and the last layer at the
    bottom, so the whole figure reads downward from prompt to forecast.

    Nine rows stand in for the full depth: row 0 is layer 0, rows 1 and 7 are
    elisions, rows 3-5 are the patched window, row 8 is the last layer.
    """
    accent = DONOR if donor else RECIPIENT
    layers = example["layers"]
    rows = 9
    pitch = (top - bottom) / rows
    cell_h = pitch * 0.70
    window_rows = {3, 4, 5}
    elision_rows = {1, 7}

    def row_center(row: int) -> float:
        return top - pitch * (row + 0.5)

    for row in range(rows):
        centre = row_center(row)
        if row in elision_rows:
            for offset in (-0.009, 0.0, 0.009):
                axis.plot(
                    [x + width / 2],
                    [centre + offset],
                    marker="o",
                    markersize=0.9,
                    color=accent,
                    transform=axis.transAxes,
                    zorder=3,
                )
            continue
        patched = row in window_rows
        axis.add_patch(
            FancyBboxPatch(
                (x, centre - cell_h / 2),
                width,
                cell_h,
                boxstyle="round,pad=0.0,rounding_size=0.004",
                linewidth=0.9 if patched else 0.7,
                edgecolor=DONOR if patched else accent,
                facecolor=DONOR_FILL if patched else "white",
                transform=axis.transAxes,
                zorder=3,
            )
        )
    axis.text(
        x + width / 2,
        top + fh(0.015),
        "City C label-token states",
        fontsize=5.7,
        color=accent,
        fontweight="bold",
        ha="center",
        va="bottom",
        transform=axis.transAxes,
        zorder=3,
    )
    # The two stacks are aligned, so one set of layer labels serves both.
    if donor:
        for row, text in (
            (0, "layer 0"),
            (4, f"layers {layers[0]}\u2013{layers[-1]}"),
            (8, f"layer {geometry['top_layer']}"),
        ):
            axis.text(
                x - 0.012,
                row_center(row),
                text,
                fontsize=5.8,
                color=DONOR if row == 4 else NEUTRAL,
                fontweight="bold" if row == 4 else "normal",
                ha="right",
                va="center",
                transform=axis.transAxes,
                zorder=3,
            )
    return {
        "window_y": row_center(4),
        "shallow_y": row_center(1),
        "deep_y": row_center(7),
    }


def patch_icon(axis, x, y, width, height, layer_count: int) -> None:
    """The patch, as a glyph: a copy of the donor's states for the layer window.

    The badge holds one small cell per patched layer, so it reads as the same
    thing that is highlighted inside the two stacks.
    """
    offset = 0.006
    for dx, dy, lw, z in ((offset, offset, 0.8, 5), (0.0, 0.0, 1.0, 6)):
        axis.add_patch(
            FancyBboxPatch(
                (x - width / 2 + dx, y - height / 2 + dy),
                width,
                height,
                boxstyle="round,pad=0.0,rounding_size=0.008",
                linewidth=lw,
                edgecolor=DONOR,
                facecolor="white",
                transform=axis.transAxes,
                zorder=z,
            )
        )
    inner_w = width * 0.58
    region = height * 0.62
    pitch = region / layer_count
    cell_h = pitch * 0.64
    for index in range(layer_count):
        axis.add_patch(
            FancyBboxPatch(
                (x - inner_w / 2, y - region / 2 + pitch * index + (pitch - cell_h) / 2),
                inner_w,
                cell_h,
                boxstyle="round,pad=0.0,rounding_size=0.002",
                linewidth=0.6,
                edgecolor=DONOR,
                facecolor=DONOR_FILL,
                transform=axis.transAxes,
                zorder=7,
            )
        )


DONOR_X, RECIPIENT_X, CARD_W = 0.060, 0.584, 0.236
PANEL_X, PANEL_W = 0.006, 0.988
FIG_W, FIG_H = 5.50, 2.84


def fy(top_inches: float) -> float:
    """Axes fraction for a position given in inches from the top."""
    return 1.0 - top_inches / FIG_H


def fh(inches: float) -> float:
    """Axes fraction for a vertical span given in inches."""
    return inches / FIG_H


def draw_prompts(axis, example: dict) -> None:
    stage_panel(
        axis, 1, "Two prompts, one label apart", PANEL_X, fy(0.58), PANEL_W, fh(0.56)
    )
    card_y, card_h = fy(0.540), fh(0.325)
    prompt_card(axis, DONOR_X, card_y, CARD_W, card_h, "DONOR RUN", example, donor=True)
    prompt_card(
        axis, RECIPIENT_X, card_y, CARD_W, card_h, "RECIPIENT RUN", example, donor=False
    )
    axis.text(
        0.5 * (DONOR_X + CARD_W + RECIPIENT_X),
        card_y + card_h / 2,
        f"identical except for\n{example['donor_symbol']} / {example['recipient_symbol']}",
        fontsize=6.2,
        color=INK,
        ha="center",
        va="center",
        linespacing=1.35,
        transform=axis.transAxes,
        zorder=3,
    )


def draw_patch(axis, example: dict, geometry: dict) -> None:
    stage_panel(axis, 2, "The patch", PANEL_X, fy(1.69), PANEL_W, fh(1.05))
    stack_w = 0.072
    bottom, top = fy(1.54), fy(0.94)
    donor_x = DONOR_X + CARD_W / 2 - stack_w / 2
    recipient_x = RECIPIENT_X + CARD_W / 2 - stack_w / 2
    donor_geom = layer_stack(
        axis, donor_x, stack_w, bottom, top, example, geometry, donor=True
    )
    recipient_geom = layer_stack(
        axis, recipient_x, stack_w, bottom, top, example, geometry, donor=False
    )

    start = donor_x + stack_w + 0.055
    end = recipient_x - 0.055
    mid = 0.5 * (start + end)
    axis.add_patch(
        FancyArrowPatch(
            (start, donor_geom["window_y"]),
            (end, recipient_geom["window_y"]),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.6,
            color=DONOR,
            transform=axis.transAxes,
            zorder=4,
        )
    )
    layers = example["layers"]
    patch_icon(axis, mid, donor_geom["window_y"], 0.048, fh(0.145), len(layers))
    axis.text(
        mid,
        donor_geom["window_y"] + fh(0.100),
        "write the donor's vectors over the recipient's",
        fontsize=6.6,
        fontweight="bold",
        color=DONOR,
        ha="center",
        va="bottom",
        transform=axis.transAxes,
        zorder=3,
    )
    axis.text(
        mid,
        donor_geom["window_y"] - fh(0.100),
        f"$h_{{\\ell,t}} \\leftarrow h^{{\\mathrm{{donor}}}}_{{\\ell,t}}$   for   "
        f"$\\ell \\in \\{{{layers[0]},{layers[1]},{layers[2]}\\}}$",
        fontsize=6.2,
        color=NEUTRAL,
        ha="center",
        va="top",
        transform=axis.transAxes,
        zorder=3,
    )

    # What the recipient run keeps, replaces, and recomputes, top to bottom.
    notes_x = recipient_x + stack_w + 0.016
    for y, text, color, weight in (
        (recipient_geom["shallow_y"], "the recipient's own,\nunchanged", NEUTRAL, "normal"),
        (recipient_geom["window_y"], "replaced", DONOR, "bold"),
        (recipient_geom["deep_y"], "recomputed from\nthe patched states", DONOR, "normal"),
    ):
        axis.text(
            notes_x,
            y,
            text,
            fontsize=5.8,
            color=color,
            fontweight=weight,
            ha="left",
            va="center",
            linespacing=1.3,
            transform=axis.transAxes,
            zorder=3,
        )
    axis.text(
        mid,
        fy(1.57),
        "all other layers and positions run unchanged",
        fontsize=5.8,
        color=NEUTRAL,
        ha="center",
        va="top",
        transform=axis.transAxes,
        zorder=3,
    )


def draw_outcome(axis, example: dict) -> None:
    """The rationale the recipient gives, before and after the patch."""
    stage_panel(
        axis, 3, "What the recipient then says", PANEL_X, fy(2.17), PANEL_W, fh(0.42)
    )
    rows = (
        (
            fy(1.960),
            "before:",
            RECIPIENT,
            f"\u201cCity C shares the {example['recipient_symbol']} label "
            f"with City B\u201d",
        ),
        (
            fy(2.085),
            "after:",
            DONOR,
            f"\u201cCity C shares the {example['donor_symbol']} label "
            f"with City A\u201d",
        ),
    )
    for y, tag, color, quote in rows:
        axis.text(
            DONOR_X,
            y,
            tag,
            fontsize=6.6,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=axis.transAxes,
            zorder=3,
        )
        axis.text(
            DONOR_X + 0.075,
            y,
            quote,
            fontsize=7.0,
            style="italic",
            color=color,
            ha="left",
            va="center",
            transform=axis.transAxes,
            zorder=3,
        )


def draw_scale(axis, example: dict) -> None:
    """Where each forecast lands between the episode's two references."""
    stage_panel(
        axis,
        4,
        "Implied responsiveness of City C, in poll points per news point",
        PANEL_X,
        fy(2.81),
        PANEL_W,
        fh(0.58),
    )
    low = min(example["b_slope"], example["unpatched_slope"])
    high = max(example["a_slope"], example["patched_slope"])
    left, right, rule_y = 0.060, 0.948, fy(2.62)

    def place(slope: float) -> float:
        pad = 0.14 * (high - low)
        lo, hi = low - pad, high + pad
        return left + (slope - lo) / (hi - lo) * (right - left)

    axis.plot(
        [left, right],
        [rule_y, rule_y],
        color=RULE,
        linewidth=0.9,
        transform=axis.transAxes,
        zorder=1,
    )
    for slope, symbol, city in (
        (example["b_slope"], example["b_symbol"], "City B"),
        (example["a_slope"], example["a_symbol"], "City A"),
    ):
        x = place(slope)
        axis.plot(
            [x, x],
            [rule_y, rule_y + fh(0.048)],
            color=RULE,
            linewidth=0.8,
            linestyle=(0, (2.2, 1.8)),
            transform=axis.transAxes,
            zorder=1,
        )
        axis.text(
            x,
            rule_y + fh(0.055),
            f"{symbol} reference ({city}) {slope:.2f}",
            fontsize=6.0,
            color=NEUTRAL,
            ha="center",
            va="bottom",
            transform=axis.transAxes,
            zorder=1,
        )
    before_x = place(example["unpatched_slope"])
    after_x = place(example["patched_slope"])
    axis.add_patch(
        FancyArrowPatch(
            (before_x + 0.012, rule_y),
            (after_x - 0.012, rule_y),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.5,
            color=DONOR,
            transform=axis.transAxes,
            zorder=3,
        )
    )
    for slope, x, color, label in (
        (example["unpatched_slope"], before_x, RECIPIENT, "before"),
        (example["patched_slope"], after_x, DONOR, "after"),
    ):
        axis.plot(
            [x],
            [rule_y],
            marker="o",
            markersize=5.4,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            transform=axis.transAxes,
            zorder=4,
        )
        axis.text(
            x,
            rule_y - fh(0.055),
            f"{label} {slope:.2f}",
            fontsize=6.2,
            fontweight="bold",
            color=color,
            ha="center",
            va="top",
            transform=axis.transAxes,
            zorder=4,
        )


def main() -> None:
    conditions = load()
    example = load_example(conditions["k0"]["mean"])
    geometry = load_geometry()

    # Authored at the width it is included at, so nothing is rescaled.
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    axis = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.set_axis_off()
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)

    draw_prompts(axis, example)
    draw_patch(axis, example, geometry)
    draw_outcome(axis, example)
    draw_scale(axis, example)

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, facecolor="white")
    fig.savefig(OUTPUT_PNG, dpi=260, facecolor="white")
    print(f"episode {example['episode']}, symmetric effect {example['effect']:+.3f}")
    print(f"Wrote {OUTPUT_PDF}")
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
