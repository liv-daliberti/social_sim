# Disclosed arithmetic/interface followup, v1

Prepared after inspecting the Qwen3-8B base v1 diagnostic and before any output
from this followup. This is a response to a diagnostic finding, not a replacement
for the fixed four-interface study. Base v1 identifies the intended reference
perfectly in every cell but fails to apply exact supplied response summaries;
for example, it emits nonzero horizon-3 changes when the selected pattern's
supplied horizon-3 coefficient is exactly zero. Prose oracle assistance worsens
its response MAE in all four cells. This motivates one fixed, explicitly
disclosed check; no further prompt search is planned.

Use the first eight paired replicates of each of the four domain × label cells,
selected by replicate index, never model performance. Keep both target-cue
versions and exact v1 reference evidence/targets. There are 32 episode pairs,
64 cue tasks × two interfaces = 128 prompts/model:

1. v1 selected-reference/pattern interface plus the explicit calculation
   `forecast = resting_level + shock * coefficient(horizon)` immediately before
   the output instruction, with the selected horizon-1/3 coefficients repeated.
2. The same task and explicit calculation, retaining reference backgrounds,
   target cue, exact selected response summary, and scenarios, but removing
   reference trajectories. This is a calculator-only use check with the same
   numerical target. It does not test learning structure from trajectories.

Compare Qwen3-8B base, matched RL seed42, and stronger Qwen3-32B base. Preserve
v1 decoding, seeds, syntax constraints, and scoring. Report all results, label
this check as outcome-motivated, and retain v1 findings. Formula coefficients
are eight-decimal per-unit effects, not the ten gold forecasts. A construction
invariant independently checks arithmetic reconstructs all targets to 1e-6.

Launch only if matched seed42 v1 assistance does not reduce Coin City/semantic
response MAE by 25%, AND stronger Qwen3-32B does not resolve the v1 interface
(original cue-change calibration >0.5 or at least one oracle response MAE at
most half its original MAE in Coin City/semantic). The evaluated evidence and
launch decision will be saved to gate.json. This followup does not silently
change the separately frozen SFT launch gate. Success localizes the problem to
task presentation/instruction use; failure under this budget still does not
establish that the model cannot learn the arithmetic.

The frozen v1 evaluator is reused unchanged with this artifact directory.
Short inference allocations have one A6000 for 8B base plus matched42, and two
A6000s for 32B; maximum one hour each. New model loads are necessary because
the v1 inference jobs do not change after results are observed.
