#!/usr/bin/env python3
"""Recovery tests for ideal, cue-invariant, and invalid diagnostic outputs."""
import copy
import json
from pathlib import Path
from score import score

ROOT=Path(__file__).resolve().parent
rows=[json.loads(x) for x in (ROOT/'diagnostic.jsonl').read_text().splitlines()]
for row in rows:
    ref=row['reference']
    row['output']=json.dumps({'reference':ref['selected_reference']} if ref['interface']=='selection_only' else {'forecasts':ref['truth_targets']})
ideal=score(rows)
assert all(cell['accuracy']==1 for cell in ideal)
assert all(abs(cell['cue_change_calibration']-1)<1e-5 and cell['response_mae']<1e-5 for cell in ideal if cell['interface']!='selection_only')
for row in rows:
    ref=row['reference']
    row['output']=json.dumps({'reference':1} if ref['interface']=='selection_only' else {'forecasts':[50 if ref['domain']=='coin_city' else 180]*10})
invariant=score(rows)
assert all(cell['accuracy']==.5 and cell['both_cues_correct_rate']==0 for cell in invariant if cell['interface']=='selection_only')
assert all(cell['cue_change_calibration']==0 for cell in invariant if cell['interface']!='selection_only')
for row in rows:row['output']='invalid'
invalid=score(rows)
assert all(cell['parse_rate']==0 and cell['accuracy']==0 for cell in invalid)
assert all(cell['valid_cue_pairs']==0 and cell['cue_change_calibration'] is None and cell['response_mae']==(100 if cell['domain']=='coin_city' else 280) for cell in invalid if cell['interface']!='selection_only')
print('Passed ideal, cue-invariant, constant-selection, and all-invalid scoring recovery checks.')
