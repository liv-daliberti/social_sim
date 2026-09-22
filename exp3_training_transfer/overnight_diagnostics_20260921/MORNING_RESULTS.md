# Morning results — September 22, 2026

Checked at approximately 10:45 a.m. Eastern. Component inference, evidence-use
inference, both supervised pilots and their component evaluations are complete.
The fixed shuffled-target training comparison is still running.

## Main new learnability result

On the fixed k=0 paired-cue diagnostic, forecast SFT at learning rate2e-5 learns
in-domain semantic structure-dependent forecasts, while the existing RL recipe
and lower-rate SFT do not show comparable cue use. Both SFT conditions are one
seed42, one pass over4800 original prompts,300 optimizer updates. RL uses roughly
2400 updates; these are prompt-presentation-matched diagnostic recipes, not an
isolated or seed-generalized comparison of learning algorithms.

| Coin City, semantic labels | Structure accuracy | Response MAE | Paired cue calibration |
|---|---:|---:|---:|
| Matched RL seed42 |50.0%|1.077|0.005|
| SFT1e-6 seed42 |58.3%|1.253|0.166|
| SFT2e-5 seed42 |97.9%|0.658|1.413|

Calibration0 means no cue response;1 means the simulator-sized change. The
higher-rate SFT overreacts on average, so this is not perfect numerical mastery.
Its structure accuracy falls to39.6% with arbitrary labels in Coin City,45.8%
with semantic labels in Coin Harbor, and50.0% with arbitrary labels in Coin
Harbor. Reference identification remains100% in all cells. This locates a
transfer boundary in forecast composition despite intact cue-name matching.

The initial launch gate was closed; the explicitly recorded outcome-informed
amendment used stronger-model conditional numerical ability to justify the two
unchanged exploratory pilots. No additional selection-only training was needed.
See learnability/COMPONENT_COMPARISON.md and its raw-response audit.

## Existing-checkpoint diagnosis

All8models×768prompts completed, all parsed, and reference identification100%
in both domains and label conditions. Across3matched RL seeds, original City
semantic cue calibration averages0.000; arbitrary0.017. Supplying exact patterns
does not rescue the8B forecasts. Stronger untrained32B has partial numerical
ability with a selected pattern supplied, often returning correct deviations
without the requested resting baseline. Frozen clipped scores are retained along
with an explicitly post-hoc raw-response/offset audit. The task is not simply
unlearnable, nor is cue-name identification the missing ability.

## Mechanism evidence use

Coherent low/high gain interventions replay identical noise and regenerate the
observations. On the two retained training seeds45/46, matched training reduces
paired-change error relative to fixed-population-prior training on seen
mechanisms (mean improvements0.195 disclosed,0.386 undisclosed). No consistent
held-out advantage:−0.019 disclosed,+0.007 undisclosed. Held-out change error
is near or worse than predicting no change. These are exploratory two-seed
results and narrow the interpretation of the positive ordinary-forecast result.
See evidence_use/analysis/RESULTS.md and evidence_use.pdf.

## Running control campaign

All20fixed trainings are running and advancing, without detected startup errors
or training-argument drift:10shuffled controls and10matched comparators. Six
matched reruns replace deleted weights; four make the retained-seed comparisons
use the same available hardware. Prompts, budget and exact compatible target
vector multiset are preserved; assignment is the experimental difference.
Controls are at roughly202–214of300rounds; matched runs116–213. Current pace
suggests controls this afternoon and the full comparison later today/tonight,
subject to runtime. No shuffled-control effect can yet be reported. CPU
finalizer31445499 waits for the complete roster; monitor_status.json records
all20cells. No manuscript claims have been changed.
