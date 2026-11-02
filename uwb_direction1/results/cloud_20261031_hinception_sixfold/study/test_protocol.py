"""Metadata/guard/source-gate tests only; no new UWB arrays or fitting."""
import ast,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import readiness,aggregate
B=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_all_source_roles_counts_steps_and_no_users7_8(self):
  cfg=json.loads((B/'PROTOCOL.json').read_bytes());sizes={1:1200,2:1000,3:1200,4:1200,5:1200,6:1100}
  for held in range(1,7):
   validation=held%6+1;train=set(range(1,7))-{held,validation};counts=[sum(sizes[h] for h in train),sizes[validation],sizes[held]];self.assertEqual(counts,cfg['counts'][str(held)]);self.assertEqual(len(train),4);self.assertEqual(sum(counts),6900);self.assertNotIn(held,train);self.assertNotIn(validation,train)
  self.assertEqual([100*((cfg['counts'][str(h)][0]+63)//64) for h in [2,3,4,5,6]],[7400,7100,7100,7200,7200])
  self.assertEqual(sum(cfg['counts'][str(h)][0]*100 for h in [2,3,4,5,6]),2290000)
 def test_gate_needs_mean_wins_and_worst_for_both_controls(self):
  cfg=json.loads((B/'PROTOCOL.json').read_bytes());rows=[dict(hinception=.96,localcnn=.94,gap=.94) for _ in range(6)];self.assertTrue(aggregate.gate(rows,cfg)[1]);rows[0]['gap']=.99;rows[0]['hinception']=.96;self.assertFalse(aggregate.gate(rows,cfg)[1]);rows=[dict(hinception=.941,localcnn=.94,gap=.94) for _ in range(6)];self.assertFalse(aggregate.gate(rows,cfg)[1])
 def test_missing_registration_refuses_before_prerequisite_reads(self):
  seen=[]
  def missing(p):seen.append(str(p));raise FileNotFoundError(str(p))
  with patch('readiness.read',side_effect=missing):
   with self.assertRaises(FileNotFoundError):readiness.ready({})
  self.assertEqual(len(seen),1);self.assertTrue(seen[0].endswith('REMOTE_PREREGISTRATION.json'))
 def test_no_diagnostic_label_indexing_in_producer_or_auditor(self):
  for name in ['run_fold.py','audit_fold.py']:
   text=(B/name).read_text();self.assertNotIn('diagnostic_labels_read=True',text);self.assertNotIn('labels[ids[ordered[role]]]',text);self.assertNotIn('scores[role]',text);self.assertNotIn('4700',text);self.assertNotIn('7400',text);compile(text,name,'exec')
 def test_worker_and_readiness_use_same_registration_state(self):
  for name in ['run_fold.py','readiness.py']:
   self.assertIn("'remote_source_confirmation_exact_readback_verified'",(B/name).read_text())
 def test_aggregate_refuses_partial_fold_before_label_prediction_load(self):
  import numpy as np
  cfg=json.loads((B/'PROTOCOL.json').read_bytes())
  def fake_read(p):
   if str(p).endswith('PROTOCOL.json'):return cfg
   if str(p).endswith('COMPLETE_AUDIT.json'):return dict(state='incomplete',diagnostic_labels_read=False)
   raise AssertionError('Unexpected later evidence read')
  with patch('aggregate.ready',return_value=True),patch('aggregate.read',side_effect=fake_read),patch.object(np,'load',side_effect=AssertionError('Must not decode predictions/labels')) as load:
   with self.assertRaises(AssertionError):aggregate.main()
   load.assert_not_called()
if __name__=='__main__':unittest.main()
