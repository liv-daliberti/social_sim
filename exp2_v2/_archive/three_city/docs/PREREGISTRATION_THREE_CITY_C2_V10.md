# Preregistration: structural-choice clue paper rerun (C2 v10)

## Status and relation to the pilot

The completed v9 experiment is retained as a pilot. Its generic relevance
sentence was chosen before collection but produced an inconsistent secondary
contrast. The v10 wording was selected after inspecting v9, so v10 is a new,
versioned full rerun; v9 and v10 responses will not be pooled.

The numerical data-generating process, 120 frozen seeds, five evidence rounds,
and A/B prompt arms are identical to v9. Only the third prompt arm changes.

## Confirmatory arms

1. **City C evidence only (`c_only`)**: byte-identical to v9 A.
2. **A/B/C information (`abc`)**: byte-identical to v9 B.
3. **A/B/C plus structural-choice clue (`abc_structural_clue`)**: all task
   information is byte-identical to B, followed by exactly this fixed block:

> **Structural clue:** One of Cities A and B is a more relevant analogue for
> City C than the other. The displayed regional-profile indices are informative
> about which reference is more relevant, although the resemblance is imperfect
> and City C may still respond differently.

The clue discloses that a structural choice exists. It does **not** identify A
or B, report profile distances, reveal a hidden class, use target outcomes, or
prescribe a pooling formula. The model must infer which reference is more
relevant from the displayed information.

The primary contrast is `abc - c_only`, measuring access to cross-city
information. The secondary contrast is `abc_structural_clue - abc`, measuring
the effect of directing the model to infer the more relevant reference.

## Frozen numerical design

The DGP is exactly the natural continuous-profile v9 design:

- no discrete city types;
- continuously sampled, nonextreme profile scores;
- target score bracketed by the two reference scores;
- independent city-level Gaussian deviation SD 0.18 response units;
- independent case noise SD 4.0 poll points;
- eight completed cases for each reference city;
- 120 episodes at seed offset 91,000.

| Evidence round | Completed City C cases |
|---:|---:|
| 1 | 1 |
| 2 | 2 |
| 3 | 4 |
| 4 | 8 |
| 5 | 16 |

The frozen estimator projection remains:

| Round | City C estimator MAE | Matched A/B/C MAE | Gap |
|---:|---:|---:|---:|
| 1 | 3.319 | 1.525 | 1.794 |
| 2 | 2.397 | 1.451 | 0.947 |
| 3 | 1.688 | 1.223 | 0.465 |
| 4 | 1.149 | 0.982 | 0.167 |
| 5 | 0.728 | 0.708 | 0.020 |

## Collection

There are 120 episodes × 5 rounds = 600 tasks per arm, 1,800 calls per model,
and 5,400 calls across Claude Opus 4.8, DeepSeek V4 Pro, and GPT-5.4.

No calls may begin until source hashes, prompt-difference checks, numerical
gates, and the exact annotated prompt preview pass validation. The prompt
preview will be shown for approval before launch.
