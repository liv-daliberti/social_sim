# Indirect-intervention human-review pilot

The pilot documented below remains a development-only interface check. The full
20-market development and 100-market production workflow is implemented in
`RUNBOOK.md`; its fail-closed gates do not treat this three-family pilot as a
paper result.

This subtree implements the development and human-validation stages of
`../INDIRECT_INTERVENTION_PLAN.md` without modifying the frozen direct-packet
experiment.

The first packet is intentionally small. It contains three development-only
mechanism families drawn from markets outside the frozen 100-market core. Each
family has one evidence item and three full-context variants:

- a bridge expected to increase the YES probability;
- a minimally reversed bridge expected to decrease it; and
- a broken bridge expected to make the evidence immaterial.

The evidence headline and body are byte-identical within a family. A separate
shortcut panel sees only the market and evidence. Intended labels, family roles,
generator metadata, and proposed causal chains remain in the private candidate
file and are never serialized into the browser-facing review packet.

The reviewer view is deliberately minimal: question, short new information,
plain-language “How it connects” text, direction, and confidence. The initial
event model remains private, and detailed resolution rules are collapsed unless
the reviewer chooses to open them. Reviewer-facing connections are limited to
40 words and two short sentences, with no semicolons or all-caps abbreviations.

The original six-reviewer assignment was invalid because every full-context
reviewer saw all three bridge variants of each family. It is preserved only as a
negative regression fixture. Pilot v2 and every production packet enforce at
most one task per market per reviewer; every task still receives three ratings.

## Pilot boundary

The pilot tests the interface, blinding, candidate schema, and reviewer workflow.
Its candidates are drafts, not a frozen instrument. They cannot support a paper
claim until independent ratings satisfy the gates in the full protocol and a new
core-market packet is frozen before target-model calls.

## Rebuild the packet

From the repository root:

```bash
python exp1_prospective/indirect_interventions/scripts/build_pilot_candidates.py
python exp1_prospective/indirect_interventions/scripts/blind_annotation_export.py
python exp1_prospective/indirect_interventions/scripts/audit_review_packet.py
```

The private file is `data/development/pilot_candidates.jsonl`. Do not distribute
it to reviewers. Share only the website and one assigned access code per person.

## Run the review website

```bash
python exp1_prospective/indirect_interventions/review_app/app.py --port 5060
```

For colleagues on the same network, add `--host 0.0.0.0` and share the host URL.
For an internet-facing deployment, set `SECRET_KEY` and `REVIEW_ADMIN_TOKEN`, use
a persistent `INDIRECT_REVIEW_DB` path, and run one Gunicorn worker:

```bash
SECRET_KEY='replace-me' REVIEW_ADMIN_TOKEN='replace-me-too' \
gunicorn exp1_prospective.indirect_interventions.review_app.app:app \
  --bind 0.0.0.0:5060 --workers 1 --threads 8 --timeout 120
```

SQLite WAL mode supports concurrent reviewers in one process. Do not use multiple
Gunicorn workers against the same SQLite file.

Reviewer codes are generated in `data/development/pilot_v2_reviewers.json`. The
default v2 pilot provides nine full-context codes and three shortcut codes. Each
reviewer sees three distinct markets, and the server never shows which panel
another code belongs to.

## Rebuild the appendix screenshots

The manuscript screenshots are generated from the running Flask templates and
the blinded browser packet, not from a mockup:

```bash
python exp1_prospective/indirect_interventions/scripts/capture_review_screenshots.py
```

The script starts a localhost-only temporary server, uses headless Firefox, and
writes the login and full-context images to both manuscript figure directories.
It uses a temporary empty annotation database and does not expose private labels
or reviewer codes in either image.

## Export and validate

The admin page is `/admin?token=$REVIEW_ADMIN_TOKEN`. It reports completion and
downloads JSON or CSV. The same export can be produced offline:

```bash
python exp1_prospective/indirect_interventions/scripts/export_annotations.py
python exp1_prospective/indirect_interventions/scripts/validate_annotations.py
```

The validator joins annotations to intended labels only offline. It reports
coverage, unanimous direction agreement, confidence, and shortcut leakage.
Missing ratings fail closed.
