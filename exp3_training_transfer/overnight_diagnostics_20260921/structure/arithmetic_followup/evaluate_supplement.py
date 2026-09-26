#!/usr/bin/env python3
"""Reuse the exact v1 inference engine/scorer on a separately frozen followup."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import evaluate
# Only the immutable artifact/output directory changes; inference settings stay fixed.
evaluate.ROOT=ROOT
if __name__=='__main__':evaluate.main()
