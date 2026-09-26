# Shuffled-target control status
Comparator: same_hardware.

Updated 2026-09-23T02:02:30.305003+00:00.
Completed endpoint/decode cells: 40/40; missing: 0.
Final five-seed comparison complete.

| Disclosure | Decode | Seeds | Shuffled − matched response MAE | 95% Student-t interval |
|---|---|---:|---:|---|
| disclosed | greedy | 5/5 | +0.1202 | [-0.0268, +0.2672] |
| disclosed | stochastic | 5/5 | +0.1551 | [+0.0050, +0.3053] |
| undisclosed | greedy | 5/5 | +0.0533 | [-0.0216, +0.1282] |
| undisclosed | stochastic | 5/5 | +0.0935 | [+0.0407, +0.1464] |

Positive favors matched training. All comparisons use the frozen fresh test; no historical test scores substitute for missing endpoints.
