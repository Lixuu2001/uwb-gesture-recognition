"""Guards block unpublished/unfinished follow-on UWB work before data access."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from readiness_guards import *
class Guard(unittest.TestCase):
 def test_missing_audit_does_not_read_diagnostic_summary(self):
  with tempfile.TemporaryDirectory() as d:
   with patch('readiness_guards.read',side_effect=AssertionError('must not read summary')):
    with self.assertRaisesRegex(RuntimeError,'audit is not complete'):prior_closed(d)
 def test_failed_audit_does_not_read_summary(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'COMPLETE_AUDIT.json').write_text(json.dumps(dict(state='incomplete')))
   with self.assertRaisesRegex(RuntimeError,'not passing'):prior_closed(p)
 def test_empty_experiment_template_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'protocol.json';p.write_text(json.dumps(dict(seed=None,epochs=None,batch_size=None,variant=None,cost_gate_passed=False)))
   with patch('readiness_guards.prior_closed',return_value=True):
    with self.assertRaisesRegex(RuntimeError,'Unset'):real_entry_ready(p,Path(d)/'absent_marker.json',d)
 def test_marker_mismatch_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'protocol.json';p.write_text(json.dumps(dict(seed=1,epochs=1,batch_size=1,variant='author_multivariate_OS_CNN',cost_gate_passed=True)));m=Path(d)/'marker.json';m.write_text(json.dumps(dict(state='remote_preregistration_exact_readback_verified',protocol_sha256='wrong')))
   with patch('readiness_guards.prior_closed',return_value=True):
    with self.assertRaisesRegex(RuntimeError,'registration'):real_entry_ready(p,m,d)
if __name__=='__main__':unittest.main()
