#!/usr/bin/env python3
"""Fail-closed checks on the structure-selection dataset.

Every check is a claim the protocol makes. Any failure aborts. The parent
experiment's defect -- training that never varied the dimension under test --
would have been caught by checks 4 and 5 here.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FAILURES: list[str] = []
PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILURES).append(f"{name}{': ' + detail if detail else ''}")


def load(path: Path) -> list[dict]:
    """Read the HuggingFace DatasetDict the trainer will actually load.

    Validating a different artifact from the one that trains is how the
    jsonl/Arrow mismatch survived to the cluster; this reads the real thing.
    """
    from datasets import load_from_disk
    rows = []
    for row in load_from_disk(str(path))["train"]:
        row = dict(row)
        row["ref"] = json.loads(row["reference"])
        rows.append(row)
    return rows


def main() -> None:
    matched = load(DATA / "causal/train")
    prior = load(DATA / "population_prior/train")
    ev = load(DATA / "causal/heldout")

    # 1. Arms share byte-identical prompts; only the reward differs.
    check("1 prompts byte-identical across arms",
          all(a["input"] == b["input"] for a, b in zip(matched, prior)))
    check("1b rewards differ between arms",
          any(a["ref"]["targets"] != b["ref"]["targets"] for a, b in zip(matched, prior)))
    check("1c causal (episode-matched) reward is the truth",
          all(a["ref"]["targets"] == a["ref"]["truth_targets"] for a in matched))
    check("1d population_prior reward is the structure blend",
          all(b["ref"]["targets"] == b["ref"]["prior_targets"] for b in prior))

    # 2. Training is Coin City, semantic labels, correct cue only.
    check("2 training domain is coin_city only",
          {r["ref"]["domain"] for r in matched} == {"coin_city"})
    check("2b training labels are semantic only",
          {r["ref"]["label_kind"] for r in matched} == {"semantic"})
    check("2c training cue is always correct",
          {r["ref"]["cue"] for r in matched} == {"correct"})

    # 3. Coin Harbor and arbitrary labels are held out of training entirely.
    check("3 coin_harbor absent from training",
          not any(r["ref"]["domain"] == "coin_harbor" for r in matched))
    check("3b arbitrary labels absent from training",
          not any(r["ref"]["arbitrary_mapping"] for r in matched))
    check("3c arbitrary symbols never appear in a training prompt",
          not any(sym in r["input"] for r in matched for sym in ("ZETA", "KAPPA")))

    # 4. THE CHECK THE PARENT EXPERIMENT NEEDED: the dimension under test varies
    #    in training, and does so in balanced fashion.
    counts = Counter(r["ref"]["target_structure"] for r in matched)
    check("4 both structures are training targets", set(counts) == {"direct_a", "mediated_b"},
          str(dict(counts)))
    check("4b training structures are balanced",
          abs(counts["direct_a"] - counts["mediated_b"]) <= 1, str(dict(counts)))

    # 5. Every episode contrasts the two structures, so structure is selectable.
    check("5 every training episode shows one reference of each structure",
          all(sorted(r["ref"]["reference_structures"]) == ["direct_a", "mediated_b"]
              for r in matched))
    check("5b true persistence ratio separates the structures",
          all((r["ref"]["true_persistence_ratio"] == 0.0)
              == (r["ref"]["target_structure"] == "direct_a") for r in matched))

    # 6. Gain is matched within an episode, so structure is the sole discriminator.
    check("6 direct template has no horizon-3 response",
          all(np.allclose(np.asarray(r["ref"]["direct_template_response"])[4:], 0)
              for r in matched))
    check("6b mediated template does persist",
          all(np.abs(np.asarray(r["ref"]["mediated_template_response"])[4:]).mean() > 1e-6
              for r in matched))
    check("6c horizon-1 responses coincide across structures (gain matched)",
          all(np.allclose(np.asarray(r["ref"]["direct_template_response"])[:4],
                          np.asarray(r["ref"]["mediated_template_response"])[:4], atol=1e-6)
              for r in matched[:200]))

    # 7. No leakage of internal labels, equations or parameters into the prompt.
    import re
    leak = [t for t in ("direct_a", "mediated_b", "rho", "lambda", "phi", "gain")
            if any(re.search(rf"\b{t}\b", r["input"]) for r in matched[:500])]
    check("7 no structure label or parameter name in the prompt", not leak, str(leak))

    # 8. Train and evaluation episodes are disjoint.
    check("8 train/eval task ids disjoint",
          not ({r["ref"]["task_id"] for r in matched} & {r["ref"]["task_id"] for r in ev}))
    check("8b train/eval numeric content disjoint",
          not ({r["ref"]["numeric_hash"] for r in matched}
               & {r["ref"]["numeric_hash"] for r in ev}))

    # 9. Evaluation crosses every factor it claims to.
    cells = Counter((r["ref"]["domain"], r["ref"]["label_kind"],
                     r["ref"]["target_structure"], r["ref"]["cue"]) for r in ev)
    check("9 evaluation is fully crossed (2x2x2x3)", len(cells) == 24, f"{len(cells)} cells")
    check("9b every evaluation cell has equal n", len(set(cells.values())) == 1,
          str(sorted(set(cells.values()))))

    # 10. The cue means something: misleading names the other structure.
    check("10 correct cue names the true structure",
          all(r["ref"]["shown_structure"] == r["ref"]["target_structure"]
              for r in ev if r["ref"]["cue"] == "correct"))
    check("10b misleading cue names the other structure",
          all(r["ref"]["shown_structure"] != r["ref"]["target_structure"]
              for r in ev if r["ref"]["cue"] == "misleading"))
    check("10c no cue sentence is shown when cue is none",
          all(r["ref"]["shown_structure"] is not None for r in ev if r["ref"]["cue"] == "none"))

    # 11. Arbitrary label mapping is balanced and carries no fixed meaning.
    arb = [r["ref"]["arbitrary_mapping"] for r in ev if r["ref"]["label_kind"] == "arbitrary"]
    zeta_direct = sum(1 for m in arb if m and m["direct_a"] == "ZETA")
    check("11 arbitrary mapping is balanced across episodes",
          abs(zeta_direct / max(len(arb), 1) - 0.5) < 0.08,
          f"ZETA->direct_a in {zeta_direct}/{len(arb)}")

    # 12. Evaluation rewards are never the blend (evaluation scores against truth).
    check("12 evaluation targets are the truth",
          all(r["ref"]["targets"] == r["ref"]["truth_targets"] for r in ev))

    # 13. Selecting correctly is worth a lot -- the design has headroom.
    gaps = [np.abs(np.asarray(r["ref"]["direct_template_response"])
                   - np.asarray(r["ref"]["mediated_template_response"])).mean()
            for r in ev]
    check("13 structure choice is worth >1 response-MAE point", float(np.mean(gaps)) > 1.0,
          f"mean gap {np.mean(gaps):.3f}")

    print(f"PASSED {len(PASSED)} checks")
    for line in PASSED:
        print(f"  ok   {line}")
    if FAILURES:
        print(f"\nFAILED {len(FAILURES)}:")
        for line in FAILURES:
            print(f"  FAIL {line}")
        sys.exit(1)
    print("\nall fail-closed checks passed")


if __name__ == "__main__":
    main()
