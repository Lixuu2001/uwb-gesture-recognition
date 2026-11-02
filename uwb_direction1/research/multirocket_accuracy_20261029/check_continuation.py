"""Metadata-only transition guards; no UWB files or real subprocesses."""
import json
from pathlib import Path
from continuation import decision
cases=[('training',True,False,'wait'),('training',False,False,'blocked'),
 ('incomplete',False,False,'blocked'),('complete',True,True,'wait'),
 ('complete',False,False,'blocked'),('complete',False,True,'ready'),
 ('incomplete',True,False,'wait'),('initializing',True,False,'wait')]
for state,alive,receipts,expected in cases:assert decision(state,alive,receipts)==expected
r=dict(state='pass',checks_passed=len(cases),synthetic_only=True,project_arrays_or_labels_read=False,
 unfinished_or_missing_upstream_never_launches=True,complete_receipts_and_absent_worker_required=True,
 new_real_analysis_started=False)
p=Path(__file__).with_name('CONTINUATION_SYNTHETIC_CHECK.json');assert not p.exists();p.write_text(json.dumps(r,indent=2));print(json.dumps(r))
