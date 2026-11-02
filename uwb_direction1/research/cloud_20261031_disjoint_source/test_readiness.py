"""Scientific boundary guards; temporary synthetic evidence only."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import readiness_guards as g
class Tests(unittest.TestCase):
 def test_active_missing_audit_does_not_read_json_or_summary(self):
  with tempfile.TemporaryDirectory() as d,patch.object(g,'read',side_effect=AssertionError('Must not open diagnostic summary')):
   with self.assertRaisesRegex(RuntimeError,'not complete'):g.prior_closed(d)
 def test_failed_or_partial_audit_refuses_before_later_files(self):
  for a in [dict(state='incomplete'),dict(state='pass',training_updates_replayed_exact=1)]:
   with tempfile.TemporaryDirectory() as d:
    p=Path(d);(p/'COMPLETE_AUDIT.json').write_text(json.dumps(a))
    with self.assertRaisesRegex(RuntimeError,'not passing|partial'):g.prior_closed(p)
 def test_unset_configuration_and_wrong_marker_refuse(self):
  with tempfile.TemporaryDirectory() as d,patch.object(g,'prior_closed',return_value=True):
   p=Path(d);(p/'protocol.json').write_text(json.dumps(dict(seed=None,epochs=None,batch_size=None)))
   with self.assertRaisesRegex(RuntimeError,'Unset'):g.real_entry_ready(p/'protocol.json',p/'missing',p)
   (p/'protocol.json').write_text(json.dumps(dict(seed=7,epochs=2,batch_size=2,variant='author_default_DCNN_2L',cost_gate_passed=True,input_preprocessing='synthetic_fixture_only')));(p/'marker.json').write_text(json.dumps(dict(state='remote_preregistration_exact_readback_verified',protocol_sha256='wrong')))
   with self.assertRaisesRegex(RuntimeError,'preregistration'):g.real_entry_ready(p/'protocol.json',p/'marker.json',p)
if __name__=='__main__':unittest.main()
