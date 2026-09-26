# Shuffled-target control, eight-seed extension (seed 49)

Reported alongside the frozen five-seed roster (results.json) and the seven-seed
extension (results_ext4748.json), never in place of either. Seed 49 was added after
the seven-seed result was known (freeze_seed49.json); the eight-seed estimate is
outcome-informed, its intervals do not carry nominal coverage, and nothing here is a
confirmatory test.

Updated 2026-09-25T23:43:14.900645+00:00.
Completed endpoint/decode cells: 64/64; missing: 0.

| Disclosure | Decode | Seeds | Shuffled − matched response MAE | 95% Student-t interval | Hierarchical 95% | Webb p | Seeds > 0 |
|---|---|---:|---:|---|---|---:|---:|
| disclosed | greedy | 8/8 | +0.0996 | [+0.0200, +0.1792] | [+0.0378, +0.1650] | 0.0183 | 7/8 |
| disclosed | stochastic | 8/8 | +0.1041 | [+0.0040, +0.2041] | [+0.0160, +0.1955] | 0.0479 | 6/8 |
| undisclosed | greedy | 8/8 | +0.0511 | [-0.0119, +0.1142] | [+0.0034, +0.1043] | 0.0968 | 6/8 |
| undisclosed | stochastic | 8/8 | +0.0609 | [-0.0132, +0.1350] | [-0.0141, +0.1419] | 0.1005 | 6/8 |

Positive favors matched training.

Seed-level effects (stochastic, response MAE):
- disclosed: s42: +0.2610, s43: -0.0468, s44: +0.1422, s45: +0.1953, s46: +0.2238, s47: +0.0493, s48: -0.0492, s49: +0.0568
- undisclosed: s42: +0.1320, s43: +0.0789, s44: +0.0272, s45: +0.1259, s46: +0.1036, s47: -0.0914, s48: +0.1534, s49: -0.0425
