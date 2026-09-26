# Completed Qwen3 reasoning follow-up: independent review

All 640 planned records are present. Disabled thinking has 320 valid probabilities; enabled thinking has 300 valid, five truncated baselines, and 15 blocked updates. All failures remain in planned binary denominators.

| Metric | Disabled | Enabled | Enabled minus disabled [95% paired family interval] |
|---|---:|---:|---:|
| Correct signs | 25/40 | 30/40 | +12.5 pp [-2.5, 27.5] |
| Paired reversals | 6/20 | 12/20 | +30.0 pp [5.0, 55.0] |
| Broken absolute movement, common pairs | 14.74 pp | 7.63 pp | -7.11 pp [-16.32, 1.32] |

Binary comparisons retain all 20 families. Broken movement has 19 complete common pairs; its missing radio-family measurement is not imputed. Marginal broken movement is 14.05 pp over 20 disabled trials and 7.63 pp over 19 enabled trials. Original broken stability is not met for disabled thinking and remains indeterminate for enabled thinking.

The symmetric format amendment reparsed 103 existing disabled outputs and generated 198 previously unattempted updates. It generated no baselines and regenerated no received answers. Enabled recovery changed no sampled outcomes. Its failed baselines are all four radio contexts (`dev_16`) and positive carton transfer (`dev_06`).

Independent review passed 8,578 checks, including original and recovered parsers, immutable source/manifest hashes, per-record lineage, exact prompts/chat serialization, own-baseline priors, request seeds, matched inference settings, all planned denominators, paired family vectors, and paired bootstrap estimates/intervals. The initial frozen comparison definitions remain unchanged.

Paper integration adds the completed methods/results and generated reasoning table. The existing direction table is unchanged; its counts were checked against the newly generated table. No paper build, inference, API call, or running-code edit was performed.

- [Independent audit and source hashes](reasoning_independent_audit.json)
- [Complete paired comparison](reasoning_comparison.json)
- [Readable paired comparison](reasoning_comparison.md)
