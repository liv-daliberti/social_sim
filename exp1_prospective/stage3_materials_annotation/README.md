# Stage 3 materials annotation

This directory contains a small human-annotation study of the synthetic evidence
used in Experiment 1, Stage 3. The unit of analysis is the evidence packet, not
the annotator.

Eight volunteer adults rate 18 frozen packets drawn from 18 distinct markets:

- six intended to increase YES;
- six intended to decrease YES; and
- six intended to have no material effect.

Every annotator rates every packet in a separately randomized order. The final
five judgments cover conditional direction, confidence, clarity, plausibility,
and whether the text is usable as an explicit hypothetical premise. A self-review
pilot removed semantic-directness, source-verification, and free-text questions
before registered collection because they added burden without entering the
retention gates.

The study does not compare people assigned to premise-acceptance and
source-verification treatments. It validates the text materials used by the model
experiment.

## Build

From the repository root:

```bash
python exp1_prospective/stage3_materials_annotation/build_packet.py
```

The builder deterministically selects and blinds the items, creates eight
reviewer-specific CSV packets, and records hashes in `generated_v7/manifest.json`.
Each item includes the exact summary from its frozen June 10 initial forecast;
the manifest records and hashes that source. Do not distribute
`generated_v7/private_key.jsonl`. The earlier generated directories preserve the
superseded pilot materials.

## Pilot the browser workflow as `self_review`

The browser interface includes a dedicated practice reviewer. From the
repository root, set private deployment secrets and start the site:

```bash
STAGE3_ANNOTATION_SECRET_KEY=CHOOSE_A_LONG_RANDOM_SECRET \
STAGE3_ANNOTATION_ADMIN_TOKEN=CHOOSE_A_DIFFERENT_ADMIN_TOKEN \
STAGE3_ANNOTATION_PRACTICE_ONLY=1 \
python exp1_prospective/stage3_materials_annotation/review_site/app.py \
  --host 127.0.0.1 --port 5050
```

Open the site and enter reviewer code `self`. This uses the exact item display,
questions, randomized ordering, saving, and completion flow that registered
reviewers receive. Practice rows are tagged `cohort=practice`, exported through
a separate endpoint, and are never part of the eight-reviewer validation input.
The practice-only switch prevents registered reviewer codes from authenticating
during the pilot deployment.

Administrative exports are available at:

- `/export/practice.csv?token=ADMIN_TOKEN`
- `/export/registered.csv?token=ADMIN_TOKEN`

The application stores only reviewer codes, consent timestamps, item responses,
and response timing in `data/reviews_v6.sqlite3`. It does not collect names, email
addresses, demographics, or IP addresses.

## Deploy the registered survey on Render

The repository-level `render-stage3.yaml` Blueprint creates a free Python web
service and a free PostgreSQL database in Render's Virginia region. PostgreSQL
keeps responses across web-service sleeps and redeployments. The production
configuration disables the public `self` practice account and requires eight
private, randomly generated reviewer codes.

Generate the two secret values without writing them to disk:

```bash
python exp1_prospective/stage3_materials_annotation/generate_render_secrets.py
```

In Render, create a Blueprint from this repository and select
`render-stage3.yaml` as the Blueprint path. Supply the printed values for:

- `STAGE3_ANNOTATION_ADMIN_TOKEN`; and
- `STAGE3_ANNOTATION_REVIEWER_CODES`.

For an existing seven-reviewer deployment, leave
`STAGE3_ANNOTATION_REVIEWER_CODES` unchanged and set the separate
`STAGE3_ANNOTATION_REVIEWER_08_CODE` secret to the new private code before
redeploying. This preserves every existing login and database row. A fresh
eight-reviewer deployment may instead put all eight codes in the JSON setting
and leave the separate setting empty.

Render generates the Flask session secret and database connection. Do not put
any of these secret values in Git. Give each reviewer only the public survey URL
and their own code.

Private administrative endpoints are:

- `/admin/status?token=ADMIN_TOKEN`
- `/export/registered.csv?token=ADMIN_TOKEN`

Never share either administrative URL. The free Render PostgreSQL plan expires
30 days after creation and has no managed backups. Download the registered CSV
after collection and upgrade or remove the database when the study is complete.

## Collect registered reviews

After the self-review pilot is accepted, give each volunteer one reviewer code
from `annotator_01` through `annotator_08`. The website displays the exact
information screen and questions in `ANNOTATION_FORM.md` and selects the
corresponding frozen randomized order. Export the registered cohort without
changing any coded values.

## Analyze

```bash
python exp1_prospective/stage3_materials_annotation/analyze_annotations.py \
  --responses PATH/TO/combined_responses.csv
```

The script requires all 144 registered ratings, retains ambiguous judgments, and
reports item-level agreement and validation gates. It does not model annotator
traits or report participant-level treatment effects.

The appendix language in `PAPER_TEMPLATE.tex` must remain outside the compiled
paper until collection and validation are complete.
