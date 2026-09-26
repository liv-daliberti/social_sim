# Human-review UI pilot v2: coordinator handoff

This v2 packet supersedes the original six-reviewer pilot. Do not collect or
analyze ratings with `pilot_review_packet.json` or `pilot_reviewers.json`: each
full-context reviewer in that legacy assignment saw every evidence family under
all three bridges, revealing the manipulation through repetition.

The v2 UI pilot is market-disjoint. Its nine full-context and three shortcut
reviewers each see three distinct markets. The production development allocator
uses the same invariant with 27 full-context and 9 shortcut reviewers across 20
markets.

## What this pilot can establish

This three-family pilot checks the candidate design, blinding, rating workflow,
and acceptance gates. It is separate from the frozen 100-market Experiment 1
sample. Even a clean pass does not authorize a paper result or target-model calls;
the accepted design must next be scaled to a frozen core-market packet.

Before collecting responses, determine whether your institution requires review
or consent for colleague annotations. The app does not request names, email
addresses, IP addresses, or demographic information. Web-server infrastructure
may still create ordinary access logs.

## Reviewer assignments

Give each code to one person. Do not send reviewers the private candidate file,
this handoff, or the assignment JSON.

| Reviewer | Cases | Access code |
|---|---:|---|
| Full 1 | 3 | `maple-7-harbor-92` |
| Full 2 | 3 | `cobalt-4-meadow-61` |
| Full 3 | 3 | `cedar-8-comet-35` |
| Full 4 | 3 | `granite-2-lantern-47` |
| Full 5 | 3 | `willow-5-compass-83` |
| Full 6 | 3 | `saffron-1-forest-64` |
| Full 7 | 3 | `indigo-8-summit-26` |
| Full 8 | 3 | `copper-3-valley-75` |
| Full 9 | 3 | `juniper-6-aster-41` |
| Shortcut 1 | 3 | `amber-3-river-74` |
| Shortcut 2 | 3 | `violet-6-orbit-28` |
| Shortcut 3 | 3 | `silver-9-garden-43` |

The first nine codes receive the full-context panel; the last three receive the
evidence-only shortcut panel. Describe every assignment simply as a forecasting
review; do not announce the two-panel comparison.

Suggested invitation:

> We are piloting a short forecasting-annotation instrument. Please use the link
> and individual code below, work independently, and judge only the hypothetical
> material shown. Do not search the web, use an AI assistant, or discuss items
> with other reviewers. Your responses save after each page, so you may return
> with the same code.

## Start a local or shared server

For a local check:

```bash
python exp1_prospective/indirect_interventions/review_app/app.py --port 5060
```

For colleagues on the same trusted network:

```bash
SECRET_KEY='use-a-long-random-value' \
REVIEW_ADMIN_TOKEN='use-a-different-random-value' \
python exp1_prospective/indirect_interventions/review_app/app.py \
  --host 0.0.0.0 --port 5060
```

Share `http://HOSTNAME-OR-IP:5060/` and one code per reviewer. For an
internet-facing service, terminate HTTPS at a trusted proxy or hosting service,
store the SQLite database on persistent storage, and run the documented one-worker
Gunicorn command rather than Flask's development server.

The administrator view is
`/admin?token=YOUR_REVIEW_ADMIN_TOKEN`. Keep that URL private.

## Completion and decision

The expected total is 36 annotations: 27 full-context ratings and 9 shortcut
ratings. After all twelve rows show complete:

```bash
python exp1_prospective/indirect_interventions/scripts/export_annotations.py
python exp1_prospective/indirect_interventions/scripts/validate_annotations.py \
  --output exp1_prospective/indirect_interventions/data/annotations/pilot_validation.json
```

The validator exits nonzero unless all ratings are present and every full-context
and shortcut gate passes. A failed item should be revised and sent through a new,
freshly blinded pilot; do not silently relabel it or inspect target-model behavior.
