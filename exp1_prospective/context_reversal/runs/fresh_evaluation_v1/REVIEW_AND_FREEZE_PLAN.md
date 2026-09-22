# One fresh evaluation, frontier spending capped at $25

Status: prospective design; no evaluation material is frozen and no API calls
are authorized by this file itself. User authorization permits up to $25 for one
frozen evaluation after local development. The operational target is at most $24
under a complete-cohort worst-case calculation, leaving at least $1 within the
user's ceiling. Previous pilot outputs remain exploratory and are never pooled
into the fresh sample.

## Fresh cohort

Target 80 genuinely new narrative/mechanism families, allocated prospectively
across eight mechanism strata (ten each). Each family requires a distinct causal
scenario/intervention/outcome pathway; renaming or paraphrasing a template does
not create another independent family. The candidate strata in `FAMILY_BLUEPRINT.md` are measurement reliability; task
precedence and slack; complementarity and redundancy; procedural gates;
intermediary allocation; network routing; feedback and control; and dependent
or selected evidence. The final mechanism specification and authored examples
follow development checks;
it must identify new mechanisms versus mechanisms already represented in the
20-family development set. Author no held-out target responses during development.

The fixed frontier cohort uses three scored contexts (supporting, opposing and
broken connection), each with one own-context baseline and three independent
updates (new news, no news, repeated news): 960 planned records for 80 families.
The masked descriptive context is omitted prospectively from the paid evaluation.
Identical evidence appears under each of the three relationships. Counterbalance
names, role-row order, surface phrasing and direction of the evidence change.
Explicitly resolve physical destinations and competing pathways. Priors and
outcomes must not be logically forced to 0 or 1 by the stated facts; no supplied
numeric prior replaces each model's own baseline.

The original three large local models supply the main evaluation. GPT-5.6 Sol is
one separately identified frontier deployment. No repeated frontier development,
model sweep or target-output-based family selection is permitted. Once frozen,
all planned families and all failures remain in reported denominators. The
finite authored cohort is the target of inference; do not claim representative
coverage of real-world forecasting.

## Independent screening of new material

The completed original human review is not repeated. Root asked whether the new
families should receive human or separate-local-model review. In the absence of
a different preference, use blinded local-model screening and label it accurately
as model review, not human validation. Candidate reviewer checkpoints are
Mistral-Small-24B-Instruct-2501 and Qwen3-14B, separate from the target checkpoints.
Record reviewer identities, settings, prompts and raw judgments. Reviewers see
only the scenario and message, without intended labels, sibling contexts, author
rationales or target-model outputs. Judge direction, whether the news bears on
the outcome, and specific ambiguity or missing information. Review every new
family and all its scored contexts, including clear/successful cases.

Adjudicate material defects before target inference. Disagreement alone is not a
reason to discard a difficult item, and target-model correctness never enters
adjudication. Retain all candidate IDs, judgments, repairs and disposition rules.
If a material change follows screening, version it and document verification of
the change. Model screening has different limitations from independent human
validation and must remain explicit in any resulting claim.

## Freeze and scoring

Freeze the full family list, exact visible prompts, independent review provenance,
label key, controls, target model IDs/snapshots, API settings, scoring code,
exclusion/disposition rules and budget certificate before any target responses.
Hash every artifact. Final family count must be fixed before outputs; any change
from the target 80 is justified by the ex-ante budget/material process and recorded
before collection. Do not stop or add families based on observed performance.

Primary: both signed context updates correct, averaged over all planned families.
Secondary: unconditional direction accuracy and broken-link signed/absolute
movement with the original 2-point equivalence criterion. Report control drift,
endpoint blocking, invalid responses and all denominators. Direction-only local
judgments are a separate companion. Report per-mechanism results, macro averages
and leave-one-mechanism-out sensitivity. Family bootstrap intervals are descriptive;
mechanism-level dependence and the small number of mechanism strata limit generalization.
A binary proportion at n=80 has a worst-case normal-approximation 95% half-width
of roughly 11 percentage points before additional dependence.

## Budget and transport

Official pricing checked 2026-09-21:
https://developers.openai.com/api/docs/models/gpt-5.6-sol
https://developers.openai.com/api/docs/pricing
https://developers.openai.com/api/docs/guides/batch

GPT-5.6 Sol standard input/output are $4/$20 per million tokens; reserve at the
$5 input cache-write ceiling. Batch input/output are $2/$10, with $2.50 input
cache-write ceiling. Batch is preferred to preserve the pilot's low reasoning
setting and 2,048-token output allowance. For 960 requests, output-only maximum
is $19.6608 in Batch; certified total input must fit the remaining budget. This arithmetic is an
illustration, not a certificate for the proposed 80 families: the actual prompts
have not been authored or counted, and conservative pre-baseline update bounds
may be substantially larger than their eventual token counts.
Two submission stages are necessary: context baselines, then three branches
using only their own saved baseline. These are stages of one fixed evaluation.

Require a certified full-plan worst-case reservation within $24 before submission;
expected cost based on prior short outputs is not sufficient. Use a persistent
ledger, no automatic SDK retries, and retain unknown-result charges/reservations.
A timeout or malformed response is not permission to regenerate a sampled output.
Reconcile actual usage but do not spend released budget on extra target families.
If the complete cohort cannot be certified under the ceiling, revise the design
before freezing; never weaken the cap or silently truncate the planned cohort.
Current shell has no OPENAI_API_KEY configured; credential setup is a separate
execution prerequisite, never stored in the protocol or output artifacts.
