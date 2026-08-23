# Development-family editor handoff

## Scope and independence

The input is the complete GPT-5.6 Sol development pool:

`data/development/development_candidates_readable_v3.jsonl`

It contains 60 families across 20 markets. Its pre-edit SHA-256 is
`e6d12cf788c2b85bfac2c9f1cffc6e32cdbfec318d43e4fe15bb7126f2b760d9`.
The structural and lexical audit in
`reports/development_candidate_audit_readable_v3.json`
passes with zero errors and zero warnings.

Earlier pools are preserved for provenance but are superseded. The original pool
triggered 73 readability warnings across 29 families. Readable v3 was regenerated
under the stricter limits below and is the only pool authorized for editing and
development ratings.

One human editor must inspect every family. The editor:

- must not serve on either later development rating panel;
- must not inspect target-model outputs;
- must work without an AI assistant or web search; and
- should use one deidentified editor code consistently across all 60 families.

Do not edit the raw-generation log. Preserve candidate IDs, market fields, split,
family index, surface-valence assignment, generator ID, and prompt hash.

## Per-family checklist

For each JSONL row, inspect and, where necessary, correct:

1. The initial event model contains the relevant actors, at least two distinct
   mechanisms, and the main uncertainty.
2. The evidence is plausible synthetic news, 20--45 words including its
   headline, and does not directly or indirectly announce the market outcome.
3. The evidence alone does not reveal a fixed update direction. It contains none
   of the registered forbidden cues.
4. Positive and negative connection texts use the exact stored prefix and suffix and
   differ only in the short decisive clause.
5. The positive chain increases the market outcome, the negative chain reverses
   it, and the broken chain genuinely severs the evidence-to-outcome link.
6. Every chain has at least two edges, its nodes and edge signs agree, and the
   connection is readable without adding unstated causal steps.
7. Directional connections are 20--40 words; broken connections are 15--28
   words; decisive clauses are 3--12 words. A connection has at most two
   sentences, no sentence exceeds 24 words, and it contains no semicolon or
   all-caps abbreviation.
8. A general adult without subject-matter expertise should understand the exact
   causal connection on one read. Replace jargon with ordinary words; meeting a
   word count alone is not sufficient.

If a family cannot be repaired cleanly, record the reason in `generation.notes`
and leave its status as `draft_for_human_validation`; the gate must fail rather
than silently accepting it.

For every completed family, set:

```json
{
  "status": "human_edited",
  "editor_id": "deidentified-editor-code",
  "edited_at": "ISO-8601 timestamp"
}
```

`status`, `editor_id`, and `edited_at` belong inside the existing `generation`
object. Marking a family `human_edited` certifies that it was actually inspected,
even if no textual correction was required.

## Gate after editing

Run:

```bash
python exp1_prospective/indirect_interventions/scripts/audit_lexical_leakage.py \
  --development
```

Proceed only when the report has `pass=true`, `human_edited_count=60`, and
`ready_for_ratings=true`. The blinded packet builder independently enforces the
same provenance requirement and creates nothing while any family is unedited.
