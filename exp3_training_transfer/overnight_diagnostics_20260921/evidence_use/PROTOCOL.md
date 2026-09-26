# Frozen mechanism-family evidence-use and fresh evaluation protocol

Registered before any new model output on these examples, 2026-09-21. Existing
results motivated this diagnostic; it is an exploratory follow-up, not a new
independent confirmation of the original claim. No original data or paper files
are modified.

The primary evidence intervention changes only the target trajectory's hidden
gain from the 20th to the 80th percentile of its world's native uniform support.
Each pair has identical input history, calibration trajectories, domain, baseline,
mechanism, scenario grid and exact Gaussian noise draws. Target observations and
latent terminal states are recomputed under the changed gain, then the ten
noise-free future forecasts are recomputed with the same simulator. Every changed
observation is therefore coherent. The disclosed description gives the same gain
range on both sides; it does not reveal the actual gain. Within pairs the surface
text changes only where target observations change. Rounded displayed outcomes
retain the established two-decimal renderer. Full-precision exogenous noise and
simulator parameters are saved for replay. Calibration hidden gains remain fixed.

The fixed set contains 16 fresh pairs for each of all 12 catalog mechanisms at
k=9: eight seen mechanism identities (128 pairs) and four heldout compositions /
parameter extrapolation identities (64 pairs). Their native supports are retained,
including the extrapolation world's 1.02--1.28 gain support. Both disclosure
conditions use the same numerical episodes. Seed base 950000000 with 1000000
world stride is separate from all original training/evaluation and fresh ordinary
evaluation seeds. The sample size, pair orientation and support are not changed
based on observed model outputs.

The primary outcome is paired response-change MAE: for each horizon subtract the
zero-shock forecast from each nonzero-shock forecast (eight contrasts), take the
high-minus-low change, and compare it with the corresponding simulator change.
This removes additive baseline or last-observation shortcuts that can improve
forecast levels without tracking the relationship. Report its no-change baseline,
tracking slope through the origin (ideal 1; no response to evidence 0), fraction
of simulator effect recovered, and direction accuracy on nonzero simulator
changes. Forecast-level changes and per-side forecast/response MAE are secondary.
The eight seen and four heldout mechanism groups are reported separately, with
world-specific summaries. A law/graph intervention is not included: gain alone is
a clean intervention on the displayed episode-specific relationship.

Inference uses the original Qwen ChatML wrapper, syntax-only forecast-array
constraint, greedy decoding, 192 output tokens, model length 3072, and decode seed
20260921. We evaluate the untrained Qwen3-4B-Instruct-2507 and all available final
step_00301 matched (causal_family) and fixed-population-prior-trained adapters.
The checkpoint manifest records actual weight-file hashes and missing weight
files, rather than accepting metadata-only directories. No checkpoint is selected
by its performance on this new set. Results do not substitute the population prior
for a shuffled-target training control. Checkpoint-seed uncertainty is limited by
weight availability; do not treat pairs as independent training replicates.

If either side fails strict parsing, the complete pair is retained with zero
predicted change and counted as failed. Valid-only sensitivity is reported. The
primary comparison uses paired seed IDs available in both trained arms, with
training seeds resampled and world-stratified episode resampling within seed
(2000 bootstrap draws, seed 20260921). Report the individual seed results, not
only a pooled mean. Interpret uncertainty cautiously with fewer than five seeds.
Base comparisons have episode uncertainty only.

A second, separate fresh ordinary transfer evaluation uses the existing generator
and unchanged true-target prompt at k=3,6,9, 60 episodes per heldout world (720
rows per disclosure). Seed base 921000000 with 1000000 world stride is separate
from gain-pair seeds. These JSONL/Hugging Face datasets are shared by the shuffled
control and matched checkpoint evaluation. Historical online training evaluation
can remain unchanged to preserve the training recipe; this new evaluation is an
endpoint comparison and is never used for model or training-budget selection.

Freeze artifacts contain exact JSONL SHA256 hashes, checkpoint hashes, source
hashes, counts, and UTC creation time. A completed output must preserve the frozen
manifest hash; resume checks may skip only complete checkpoint outputs with the
same data/checkpoint hashes. No prediction is read during design/data freeze.
