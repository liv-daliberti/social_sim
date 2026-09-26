# Shuffled-target control, seven-seed extension

Reported alongside the frozen five-seed roster in results.json, never in place of it.

Updated 2026-09-24T10:23:40.025404+00:00.
Completed endpoint/decode cells: 56/56; missing: 0.

| Disclosure | Decode | Seeds | Shuffled − matched response MAE | 95% Student-t interval |
|---|---|---:|---:|---|
| disclosed | greedy | 7/7 | +0.1009 | [+0.0058, +0.1959] |
| disclosed | stochastic | 7/7 | +0.1108 | [-0.0072, +0.2288] |
| undisclosed | greedy | 7/7 | +0.0647 | [-0.0002, +0.1296] |
| undisclosed | stochastic | 7/7 | +0.0756 | [-0.0025, +0.1538] |

Positive favors matched training.
