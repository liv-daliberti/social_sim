#!/usr/bin/env python3
"""Pipeline status for Experiment 1.

Shows Stage 2–5 completion per model and the next command to run for each gap.

Usage (from exp1_prospective/):
    python status.py
    python status.py --target 100      # change the per-model target (default 100)
    python status.py --models gpt llama3.1-8b qwen2.5-72b   # filter to specific models
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_ROOT   = Path(__file__).resolve().parent
_IF_DIR = _ROOT / "data" / "initial_forecasts"
_CF_DIR = _ROOT / "data" / "counterfactuals"
_UF_DIR = _ROOT / "data" / "updated_forecasts"
_RES_DIR = _ROOT / "data" / "results"
_CANONICAL = _ROOT / "data" / "selected_markets" / "diverse_2026-06-09.jsonl"

# Models tracked by default (in display order).
DEFAULT_MODELS = [
    "gpt-5.4",
    "claude-opus-4-8",
    "DeepSeek-V4-Pro",
    "llama3.1-8b",
    "llama3.1-70b",
    "llama3.3-70b",
    "qwen2.5-7b",
    "qwen2.5-14b",
    "qwen2.5-32b",
    "qwen2.5-72b",
]

# Models the user considers "target" for full pipeline.
TARGET_MODELS = {
    "gpt-5.4", "llama3.1-8b", "llama3.1-70b", "llama3.3-70b",
    "qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b", "qwen2.5-72b",
}


def _canonical_ids() -> set[str]:
    """Return the set of normalised task_ids in the canonical 100-market file."""
    ids: set[str] = set()
    if not _CANONICAL.exists():
        return ids
    for line in _CANONICAL.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            tid = rec.get("task_id", "")
            if tid:
                ids.add(_norm_tid(tid))
        except json.JSONDecodeError:
            pass
    return ids


def _norm_tid(tid: str) -> str:
    return re.sub(r"_\d{4}-\d{2}-\d{2}$", "", tid)


# ── Stage 2 ───────────────────────────────────────────────────────────────────

def _stage2_counts(canon: set[str]) -> dict[str, int]:
    """Return {model_name: canonical-market count} from initial_forecasts/."""
    counts: dict[str, int] = {}
    for path in sorted(_IF_DIR.glob("forecasts_*.jsonl"), reverse=True):
        if "old" in path.name:
            continue
        mani = path.with_name(path.stem + ".manifest.json")
        model_name = None
        if mani.exists():
            try:
                model_name = json.loads(mani.read_text()).get("model") or None
            except Exception:
                pass
        if not model_name:
            m = re.match(r"forecasts_(.+)_\d{4}-\d{2}-\d{2}$", path.stem)
            model_name = m.group(1).replace(":", "-") if m else None
        if not model_name:
            continue
        model_name = model_name.replace(":", "-")
        if model_name in counts:
            continue  # keep only the latest file per model
        tids: set[str] = set()
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("error") or rec.get("yes_prob") is None:
                        continue
                    t = _norm_tid(rec.get("task_id", ""))
                    if t and t in canon:
                        tids.add(t)
                except json.JSONDecodeError:
                    pass
        counts[model_name] = len(tids)
    return counts


# ── Stage 3 ───────────────────────────────────────────────────────────────────

def _stage3_count(canon: set[str]) -> int:
    """Return number of canonical markets that have CF packets."""
    tids: set[str] = set()
    for path in sorted(_CF_DIR.glob("counterfactuals_*.jsonl"), reverse=True):
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("cf_index") is not None:
                        t = _norm_tid(rec.get("task_id", ""))
                        if t in canon:
                            tids.add(t)
                except json.JSONDecodeError:
                    pass
    return len(tids)


# ── Stage 4 ───────────────────────────────────────────────────────────────────

def _stage4_counts(canon: set[str]) -> dict[str, int]:
    """Return {model_name: canonical-market count} from updated_forecasts/."""
    by_model: dict[str, set] = {}
    for path in sorted(_UF_DIR.glob("updated_*.jsonl"), reverse=True):
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("error") or not rec.get("direction"):
                        continue
                    mn = rec.get("forecast_model", "").replace(":", "-")
                    if not mn:
                        m = re.match(r"updated_(.+)_\d{4}-\d{2}-\d{2}$", path.stem)
                        mn = m.group(1) if m else ""
                    tid = _norm_tid(rec.get("task_id", ""))
                    if mn and tid and tid in canon:
                        by_model.setdefault(mn, set()).add(tid)
                except json.JSONDecodeError:
                    pass
    return {mn: len(tids) for mn, tids in by_model.items()}


# ── Stage 5 ───────────────────────────────────────────────────────────────────

def _stage5_status() -> tuple[bool, str | None]:
    """Return (has_results, latest_date_string)."""
    files = sorted(_RES_DIR.glob("consistency_report_*.json"), reverse=True)
    if not files:
        return False, None
    return True, files[0].name


# ── latest updated file per model (for resume) ────────────────────────────────

def _latest_uf_file(slug: str) -> Path | None:
    candidates = sorted(_UF_DIR.glob(f"updated_{slug}_*.jsonl"), reverse=True)
    return candidates[0] if candidates else None


# ── display ───────────────────────────────────────────────────────────────────

def _bar(n: int, total: int, width: int = 12) -> str:
    if total == 0:
        return "─" * width
    filled = int(width * n / total)
    return "█" * filled + "░" * (width - filled)


def _cell(n: int, target: int) -> str:
    if n == 0:
        return f"{'0':>3}/{target} ❌"
    if n >= target:
        return f"{n:>3}/{target} ✅"
    return f"{n:>3}/{target} 🔄"


def main() -> None:
    ap = argparse.ArgumentParser(description="Show Exp 1 pipeline status.")
    ap.add_argument("--target",  type=int, default=100,
                    help="Expected market count per model (default 100).")
    ap.add_argument("--models",  nargs="*", default=None,
                    help="Filter to specific model slugs.")
    ap.add_argument("--commands", action="store_true",
                    help="Print next-step commands for incomplete models.")
    args = ap.parse_args()

    target    = args.target
    canon     = _canonical_ids()
    s2        = _stage2_counts(canon)
    s3_count  = _stage3_count(canon)
    s4        = _stage4_counts(canon)
    s5_ok, s5_file = _stage5_status()

    models = args.models or DEFAULT_MODELS

    col_w = max(len(m) for m in models) + 2

    # ── header ────────────────────────────────────────────────────────────────
    print()
    print(f"{'Model':<{col_w}}  {'Stage 2':^14}  {'Stage 3':^14}  {'Stage 4':^14}  Stage 5")
    print(f"{'':─<{col_w}}  {'':─<14}  {'':─<14}  {'':─<14}  {'':─<7}")

    for mn in models:
        mn_disp = mn
        s2n = s2.get(mn, 0)
        s4n = s4.get(mn, 0)

        s2_cell = _cell(s2n, target)
        s3_cell = _cell(s3_count, target)
        s4_cell = _cell(s4n, target)
        s5_cell = ("✅" if s5_ok else "❌")

        marker = " *" if mn in TARGET_MODELS else "  "
        print(f"{mn_disp:<{col_w}}{marker} {s2_cell:<14}  {s3_cell:<14}  {s4_cell:<14}  {s5_cell}")

    print()
    print(f"  Stage 3 CF count: {s3_count} markets  (shared, model-agnostic)")
    if s5_file:
        print(f"  Stage 5 latest:   {s5_file}")
    print(f"  * = target models (all should reach {target}/100 end-to-end)")
    print()

    if not args.commands:
        print("  Run with --commands to see next steps for each gap.")
        return

    # ── next-step commands ─────────────────────────────────────────────────────
    gaps: list[str] = []
    for mn in models:
        if mn not in TARGET_MODELS:
            continue
        slug = mn.replace(":", "-")
        s2n  = s2.get(mn, 0)
        s4n  = s4.get(mn, 0)

        if s2n < target:
            missing = target - s2n
            if mn in ("llama3.1-70b", "llama3.3-70b"):
                tag = mn.replace("llama3.1-", "llama3.1:").replace("llama3.3-", "llama3.3:")
                # These errored before; run fresh on canonical diverse set
                gaps.append(
                    f"# Stage 2 — {mn}  ({s2n}/{target} done, {missing} to go)\n"
                    f"python agent/run_local_forecast.py \\\n"
                    f"  --model {tag} \\\n"
                    f"  --input data/selected_markets/diverse_2026-06-09.jsonl \\\n"
                    f"  --k 3 --timeout 3600 --no-third-turn --max-tokens 1000\n"
                )
            elif mn == "qwen2.5-72b":
                gaps.append(
                    f"# Stage 2 — {mn}  ({s2n}/{target} done, {missing} to go)\n"
                    f"python agent/run_local_forecast.py \\\n"
                    f"  --model qwen2.5:72b \\\n"
                    f"  --input data/selected_markets/diverse_topup_30.jsonl \\\n"
                    f"  --out-file data/initial_forecasts/forecasts_qwen2.5-72b_2026-06-14.jsonl \\\n"
                    f"  --k 3 --timeout 1800\n"
                )
            else:
                tag = mn.replace("-", ":", 1) if ":" not in mn and "." not in mn else mn
                gaps.append(
                    f"# Stage 2 — {mn}  ({s2n}/{target} done, {missing} to go)\n"
                    f"python agent/run_local_forecast.py --model {tag} --k 3\n"
                )

        if s4n < target:
            missing = target - s4n
            if mn == "gpt-5.4":
                gaps.append(
                    f"# Stage 4 — {mn}  ({s4n}/{target} done, {missing} to go)\n"
                    f"python agent/gpt_updated_forecast.py --verbose\n"
                    f"  # Requires AZURE_AI_API_KEY env var\n"
                )
            elif mn in ("llama3.1-70b", "llama3.3-70b"):
                tag = mn.replace("llama3.1-", "llama3.1:").replace("llama3.3-", "llama3.3:")
                gaps.append(
                    f"# Stage 4 — {mn}  ({s4n}/{target} done)  ⚠ needs Stage 2 first\n"
                    f"python agent/local_updated_forecast.py \\\n"
                    f"  --model {tag} --timeout 3600\n"
                )
            elif mn in ("qwen2.5-72b",):
                gaps.append(
                    f"# Stage 4 — {mn}  ({s4n}/{target} done, {missing} to go)\n"
                    f"python agent/local_updated_forecast.py \\\n"
                    f"  --model qwen2.5:72b --timeout 1800\n"
                    f"  # auto-resumes from latest updated_{slug}_*.jsonl\n"
                )
            else:
                tag_map = {
                    "llama3.1-8b": "llama3.1:8b",
                    "qwen2.5-7b":  "qwen2.5:7b",
                    "qwen2.5-14b": "qwen2.5:14b",
                    "qwen2.5-32b": "qwen2.5:32b",
                }
                tag = tag_map.get(mn, mn)
                gaps.append(
                    f"# Stage 4 — {mn}  ({s4n}/{target} done, {missing} to go)\n"
                    f"python agent/local_updated_forecast.py --model {tag}\n"
                    f"  # auto-resumes from latest updated_{slug}_*.jsonl\n"
                )

    if not s5_ok or True:  # always show Stage 5 command
        gaps.append(
            "# Stage 5 — consistency evaluation (run after Stage 4 complete)\n"
            "python agent/evaluate_consistency.py\n"
        )

    if gaps:
        print("─── Next steps ──────────────────────────────────────────────────")
        for g in gaps:
            print(g)
    else:
        print("✅ All target models complete through Stage 4. Run Stage 5 next.")
        print("python agent/evaluate_consistency.py")


if __name__ == "__main__":
    main()
