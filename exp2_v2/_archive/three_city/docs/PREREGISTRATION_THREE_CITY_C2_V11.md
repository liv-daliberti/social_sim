# Preregistration: identifiable structural-choice paper rerun (C2 v11)

## Status and design history

The completed v9 experiment remains a pilot. V10 changed only the clue, but a
pre-collection design audit showed that its intended analogue choice was not
reliably identified by the data: the median displayed A/B outcome separation
was 2.34 poll points, and the profile-nearest city matched the true
response-nearest city in only 58.3% of episodes. V10 received zero model calls
and is retained as an unrun rejected design.

V11 is a new versioned rerun selected before collection. It retains continuous
city profiles and noisy outcomes but makes the analogue-selection structure
identifiable. No v9, v10, or v11 responses will be pooled.

## Confirmatory arms

1. **City C evidence only (`c_only`)**: the established A prompt template.
2. **A/B/C information (`abc`)**: the established B prompt template.
3. **A/B/C plus structural-choice clue (`abc_structural_clue`)**: all task
   information is identical to B, followed by exactly this fixed block:

> **Structural clue:** One of Cities A and B is a more relevant analogue for
> City C than the other. The displayed regional-profile indices are informative
> about which reference is more relevant, although the resemblance is imperfect
> and City C may still respond differently.

The clue discloses that an analogue-selection problem exists. It does **not**
identify A or B, report profile distances, reveal a hidden label, use target
outcomes, or prescribe a pooling formula. The model must infer the relevant
reference from the displayed information.

The primary contrast is `abc - c_only`, measuring access to cross-city
information. The secondary contrast is `abc_structural_clue - abc`, measuring
the effect of directing the model toward the visible analogue-selection
structure.

## Frozen continuous numerical design

Each episode samples:

- a low reference profile continuously from [0.10, 0.30];
- a high reference profile continuously from [0.70, 0.90];
- City C continuously 8%–22% of the A/B span from one endpoint;
- an episode intercept continuously from [0.20, 0.38];
- an episode profile slope continuously from [1.00, 1.40];
- independent city deviations with SD 0.12 response units; and
- independent case noise with SD 4.0 poll points.

There are no discrete city response types. Which endpoint is nearer to C and
which letter receives the high profile are balanced in a 2 × 2 design, with 30
episodes in each cell. The continuous profile relationship makes the nearer
reference informative, while city and case noise preserve uncertainty.

Each reference city supplies eight completed cases. City C evidence accumulates
at every experimental round:

| Evidence round | Completed City C cases |
|---:|---:|
| 1 | 1 |
| 2 | 2 |
| 3 | 4 |
| 4 | 8 |
| 5 | 16 |

Before any model calls, the frozen tasks must pass all of these gates:

- minimum A/B profile-index separation at least 39.5 points;
- minimum endpoint-choice strength at least 0.54, where zero is the midpoint
  and one is an endpoint;
- profile-nearest and true-response-nearest agreement at least 90%;
- median displayed A/B expected-poll separation at least 5.0 points;
- early observed City C evidence identifies the profile-nearest reference above
  chance and reaches at least 85% by round 5;
- A/B/C estimator MAE beats C-only in rounds 1–4; and
- the estimator gap is below 0.12 poll points by round 5, with both estimators
  below 0.90 MAE.

## Collection

There are 120 episodes × 5 rounds = 600 tasks per arm, 1,800 calls per model,
and 5,400 calls across Claude Opus 4.8, DeepSeek V4 Pro, and GPT-5.4.

No calls may begin until source hashes, prompt-difference checks, balance and
identifiability gates, numerical benchmarks, and the exact annotated prompt
preview pass validation. The prompt preview will be shown for approval before
launch.
