# Independent robustness review: qwen2_5_72b_instruct

Technical audit: **PASS**. This is not a scientific pass/fail gate or permission to select a model.

All 20 parent families and four variants remain grouped; numeric coverage is 1,280 records and categorical coverage 960 records. Raw answers, exact prompts, own-context priors, separately serialized one-user-turn categorical chats, deterministic seeds, checkpoint metadata, and frozen source hashes were checked without model inference. Parent-bootstrap point estimates and intervals were recomputed independently.

| Variant | Numeric sign / 40 | Numeric pair / 20 | Direction sign / 40 | Direction pair / 20 | Broken unchanged / 20 | Broken mean absolute (pp) | Blocked signed priors / 40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| repaired_base | 28 | 8 | 29 | 9 | 11 | 15.50 | 0 |
| name_only | 27 | 8 | 30 | 10 | 10 | 16.50 | 0 |
| paraphrase | 25 | 5 | 27 | 7 | 12 | 16.50 | 0 |
| resample | 27 | 7 | 30 | 10 | 10 | 16.25 | 0 |

| Edit vs repaired base | New-news update discrepancy mean absolute (pp) | Within 2 pp / 80 | Label agreement / 80 | Both labels correct / 60 | Excess discrepancy vs resample (pp; 95% interval) |
|---|---:|---:|---:|---:|---|
| name_only | 5.34 | 47 | 72 | 38 | 1.28; [-0.8125000000000007, 3.5624999999999987] |
| paraphrase | 5.80 | 44 | 70 | 36 | 1.74; [-0.6875, 4.188437499999996] |
| resample | 4.06 | 47 | 75 | 39 | 0.00; [0.0, 0.0] |

| Variant | No-news absolute drift (pp) | Repeated-news absolute drift (pp) |
|---|---:|---:|
| repaired_base | 0.000 | 0.188 |
| name_only | 0.000 | 0.063 |
| paraphrase | 0.000 | 0.188 |
| resample | 0.000 | 0.025 |

## Numerical instability versus categorical judgments

| Comparison | Update difference over 2 pp / 80 | Of these, categorical labels agree | Of these, categorical labels disagree |
|---|---:|---:|---:|
| name_only | 33 | 30 | 3 |
| paraphrase | 36 | 31 | 5 |
| resample | 33 | 30 | 3 |

These counts separate numerical update instability from changes in the standalone categorical judgment; they do not condition away failures in the primary outcomes. Numerical differences with agreeing labels can include consistently wrong interpretations, which the joint-correctness column exposes. Compare every edit with the identical-text resample before attributing discrepancy to wording.

Broken-link label accuracy and broken numerical movement provide separate evidence about relevance judgment. If categorical judgments also fail while no-news drift is small, ordinary numerical re-elicitation alone does not explain the observed pattern. Endpoint blocking can contribute to signed numeric failures but cannot mechanically force an erroneous standalone label. An excess-discrepancy interval crossing zero establishes neither an edit effect nor equivalence.


## Scientific interpretation limits

- A stable wrong label is agreement, not successful interpretation; the table keeps joint correctness separate and excludes masked new news only from correctness.
- The resample repeats the exact visible text with another request seed. Its nonzero discrepancy is a descriptive sampling reference. A single resample and descriptive family intervals do not isolate a causal naming or paraphrase effect.
- Endpoint-blocked priors remain unconditional numeric failures. Their counts quantify one mechanical restriction, not a reason to remove difficult items. Standalone categorical errors persist independently of numerical endpoint constraints.
- Compare broken-link categorical failures with broken numerical movement and control drift before attributing failures to probability elicitation alone. Agreement can coexist with unstable numerical updates; the JSON retains that cross-tab with observed denominators.
- Repairs and all edit variants are development observations within the original 20 families. They are not 80 independent families, do not replace original results, and cannot validate the proposed fresh cohort. Future freezing criteria concern materials and execution, not these observed success rates.
