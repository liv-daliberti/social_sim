#!/bin/bash
set -uo pipefail
REPO=/n/fs/similarity/social_sim
ROOT="$REPO/exp3_training_transfer/overnight_diagnostics_20260921/shuffled_control"
source "$REPO/.runtime/oat_env.sh"
"$REPO/.runtime/oat_conda/bin/python" "$ROOT/monitor.py" > "$ROOT/final_scheduler_status.log" 2>&1
"$REPO/.runtime/oat_conda/bin/python" "$ROOT/analyze.py" --require-complete 2>&1 | tee "$ROOT/final_analysis.log"
STATUS=${PIPESTATUS[0]}
"$REPO/.runtime/oat_conda/bin/python" - "$ROOT" "$STATUS" <<'PY'
from pathlib import Path
import json,sys
from datetime import datetime,timezone
r=Path(sys.argv[1]);code=int(sys.argv[2]);p=r/'results.json';x=json.loads(p.read_text()) if p.is_file() else {}
(r/'finalization_status.json').write_text(json.dumps({'finished_at':datetime.now(timezone.utc).isoformat(),'analysis_exit_code':code,'complete':bool(code==0 and x.get('complete')),'missing_cells':x.get('incomplete_cells'),'note':'False completeness or a nonzero analysis exit preserves missing/failed cells; no partial roster is declared final.'},indent=2)+'\n')
PY
exit "$STATUS"
