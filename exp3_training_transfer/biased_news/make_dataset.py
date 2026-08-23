#!/usr/bin/env python3
"""Build OAT-format train/eval datasets for the Exp-2 biased-news world.

Each row is one city/episode from the frozen news-response generative process
(``exp3_training_transfer/biased_news/worlds.py``):

  * ``input``      – the information-parity prompt (Appendix 11.4): the structure
                     is disclosed, the agent sees (news, poll) weeks plus one held-out
                     test shock, and must end with a JSON poll prediction.
  * ``reference``  – a JSON string carrying the noise-free target poll and the
                     metadata the reward / recovery readout needs
                     (``target``, ``last_poll``, ``test_news``, ``g``).

Train and eval cities are drawn from DISJOINT seed ranges, so the eval set is a
genuine held-out sample of the world (new latent g draws, new histories), not a
re-scoring of trained cities.

Usage (from repo root, with the oat_conda env):
    .runtime/oat_conda/bin/python exp3_training_transfer/biased_news/make_dataset.py \
        --n-train 4000 --n-eval 250 --out exp3_training_transfer/biased_news/data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from worlds import (
    generate_episode, OPINION_INIT, PHI, G_LO, G_HI,
)

# Disjoint seed bands keep train and held-out cities fully separate.
TRAIN_SEED_BASE = 0
EVAL_SEED_BASE = 10_000_000

# The four held-out shocks (the Experiment-2 four-shock recovery probe).
SHOCKS = [-10, -5, 5, 10]


def build_prompt(news_k, polls_k, shock):
    """Verbatim copy of the Experiment-2 information-parity prompt
    from the archived parity probe, inlined so dataset generation depends only
    on the maintained world module rather than an evaluation client chain."""
    head = [
        "You are tracking a local election campaign with a weekly tracking poll.",
        "Each week reports the net news (a signed number: positive = good news for the "
        "candidate, negative = bad) and the resulting poll (support, 0-100).",
        "",
        "How this city works (the same for every week):",
        f"  - This city has a fixed news-responsiveness g, a number between {G_LO} and {G_HI} "
        "(you do not know which).",
        "  - Each week the underlying opinion moves by about g times that week's net news, "
        "plus a small random drift of about 1 point.",
        "  - The reported poll is that opinion plus survey noise of about 2 points.",
        "  - Low g = sticky coverage (news barely moves the poll); high g = news moves it "
        "nearly one-for-one.",
        "",
    ]
    k = len(news_k)
    if k == 0:
        body = ["No weeks have been observed yet for this city, so g could be anything in its range."]
    else:
        body = ["| Week | Net news | Poll |", "|------|----------|------|"]
        body += [f"| {i:>4} | {n:>+5d} | {p:>4} |" for i, (n, p) in enumerate(zip(news_k, polls_k), 1)]
        body += ["", "From the history above, estimate this city's g (poll points moved per +1 point "
                 "of net news)."]
    tail = [
        f"Given that next week's net news will be {shock:+d}, predict next week's poll.",
        "Reason in AT MOST two short sentences. Do NOT write a week-by-week table or show "
        "arithmetic. Then immediately give the answer.",
        'Output EXACTLY ONE JSON object on its own line and then STOP: '
        '{"rationale": "<=12 words", "points_per_news": <number>, '
        '"predicted_poll": <integer 0-100>}',
    ]
    return "\n".join(head + body + [""] + tail)


def _target_for_shock(ep: dict, shock: float) -> float:
    """Noise-free E[poll_{T+1}] under an arbitrary test shock (same mean-reverting
    transition as the engine), reusing the episode's hidden last opinion o_T."""
    o_T = ep["opinion_traj"][-1]
    o = OPINION_INIT + PHI * (o_T - OPINION_INIT) + ep["g"] * shock
    return round(max(0.0, min(100.0, o)), 3)


def _row(ep: dict, seed: int, shock: float) -> dict:
    prompt = build_prompt(ep["news"], ep["polls"], int(shock))
    reference = {
        "target": _target_for_shock(ep, shock),  # noise-free reward target
        "last_poll": ep["last_poll"],
        "test_news": int(shock),
        "g": ep["g"],                              # hidden latent (recovery readout only)
        "seed": seed,
    }
    return {"input": prompt, "reference": json.dumps(reference)}


def build_train(n: int) -> list[dict]:
    """One row per city, on the episode's own (random-sign) test shock."""
    rows = []
    for i in range(n):
        seed = TRAIN_SEED_BASE + i
        ep = generate_episode(seed=seed)
        rows.append(_row(ep, seed, ep["test_news"]))
    return rows


def build_eval_multishock(n: int) -> list[dict]:
    """Four rows per held-out city (shocks -10/-5/+5/+10) so the offline reader can
    recover the exact Experiment-2 four-shock slope ghat per city."""
    rows = []
    for i in range(n):
        seed = EVAL_SEED_BASE + i
        ep = generate_episode(seed=seed)
        for s in SHOCKS:
            rows.append(_row(ep, seed, s))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=250)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "data_v2")
    args = ap.parse_args()

    from datasets import Dataset, DatasetDict

    train = build_train(args.n_train)
    held_out = build_eval_multishock(args.n_eval)

    args.out.mkdir(parents=True, exist_ok=True)
    # OAT loads each split with load_data_from_disk_or_hf and indexes ["train"].
    DatasetDict({"train": Dataset.from_list(train)}).save_to_disk(str(args.out / "train"))
    DatasetDict({"train": Dataset.from_list(held_out)}).save_to_disk(str(args.out / "heldout"))

    print(f"wrote {len(train)} train rows -> {args.out/'train'}")
    print(f"wrote {len(held_out)} held-out rows -> {args.out/'heldout'}")
    # Show one example for sanity.
    print("\n--- example prompt ---\n" + train[0]["input"][:600])
    print("\n--- example reference ---\n" + train[0]["reference"])


if __name__ == "__main__":
    main()
