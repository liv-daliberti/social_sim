# Native-cell supervised learnability

Frozen 2026-09-22 UTC, before any native-cell training run. Exploratory mechanism
diagnosis, not a confirmatory effectiveness claim.

## Question

The parent protocol trains only on Coin City with semantic labels. Supervised
training at 2e-5 reaches 97.9% forecast structure accuracy in that cell and
39.6% (Coin City / arbitrary), 45.8% (Coin Harbor / semantic) and 50.0%
(Coin Harbor / arbitrary) in the held-out cells, while reference identification
is 100% in every cell and for every checkpoint tested. Two accounts fit:

1. **Semantic shortcut.** Training learned cue-word-to-pattern associations that
   arbitrary symbols and a new surface domain break.
2. **Binding limit.** The interface cannot bind an arbitrary symbol, or a
   Coin Harbor reference, to a numeric response pattern under any training.

These make opposite predictions when the recipe is trained *inside* each cell.

## Design

For each of three cells -- Coin City/arbitrary, Coin Harbor/semantic,
Coin Harbor/arbitrary -- generate 4,800 training episodes with cue always
correct, matching the parent's row count, k cycling and structure cycling.
Episodes are drawn from seed bands 62M/63M/64M, disjoint from the parent's
training band (61M) and evaluation band (71M+); prompt-level disjointness from
the parent training set is verified and recorded. Train the unmodified 2e-5
recipe (same base revision, LoRA rank 32 / alpha 64, seven projection modules,
one epoch, effective batch 16, constant schedule, completion-only loss) at
seeds 42, 43 and 44 per cell. Nine runs.

Evaluate every resulting checkpoint on the existing frozen structure diagnostic,
which scores all four domain/label cells. This yields native-cell accuracy and
cross-cell transfer from the same pass.

## Interpretation, fixed in advance

* Native accuracy high in a cell (comparable to the parent's 97.9%) means that
  cell is learnable and the parent's failure there is a **transfer** failure.
* Native accuracy low in a cell means a **representational or interface** limit
  that supervision inside the cell does not overcome.
* Reference identification is expected to stay at 100% throughout; if it does
  not, the cell's result is not interpretable as a use failure and is reported
  as such.

No hyperparameter is chosen using any evaluation result. The 1e-6 arm is not
run: it did not learn the task in the parent cell, so extending it would only
sharpen a null. Results are reported per seed, never pooled only.
