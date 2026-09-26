# Preregistration: noisier structural-choice rerun (C2 v12)

## Design history

V11 was stopped after 4,373 of 5,400 responses. Its partial responses are
retained as a post-hoc pilot and will not be pooled with v12. After inspecting
partial v11 results, v12 was explicitly designed to raise absolute forecast
error and the early information-value gap while retaining convergence and an
identifiable continuous analogue structure. V12 therefore uses a fresh seed
range beginning at 93,000.

## Frozen manipulation

The three prompt arms and exact structural clue are unchanged from v11:

> **Structural clue:** One of Cities A and B is a more relevant analogue for
> City C than the other. The displayed regional-profile indices are informative
> about which reference is more relevant, although the resemblance is imperfect
> and City C may still respond differently.

The clue does not identify A or B. Arm A receives City C evidence only; arm B
also receives the exact A/B reference records; arm C receives the same records
as B plus the exact clue.

## Numerical changes selected after v11

- case-noise SD increases from 4.0 to 6.0 poll points;
- city-deviation SD increases from 0.12 to 0.16 response units;
- continuous profile-slope range increases from [1.00, 1.40] to [1.55, 1.95];
- all 120 episode seeds are fresh.

All other structural choices remain: continuous profiles, C located 8%–22% of
the A/B span from one endpoint, exact factorial balance, eight cases per
reference, and 1, 2, 4, 8, and 16 cumulative City C cases at rounds 1–5.

Before collection, frozen tasks must show: profile/response analogue agreement
at least 90%; median displayed A/B separation at least 7 poll points; observed
analogue identification at least 65% in round 1 and 85% in round 5; C-only and
matched MAE both above their v11 counterparts; a round-1 estimator gap above
2.5 points; and a round-5 gap below 0.20 points.

There are 600 tasks per arm, 1,800 calls per model, and 5,400 calls across the
same three models. V12 is a post-v11 redesign and is reported as such.
