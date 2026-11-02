"""Synthetic arithmetic/refusal tests; never call a timed model/profile."""
import unittest,tempfile,copy,sys
from pathlib import Path
from unittest.mock import patch
from cost_tools import validate,select
import profile_cost
class Cost(unittest.TestCase):
 def fixture(self):return dict(epoch_candidates=[3,1],batch_size=8,seed=7,input_preprocessing='synthetic_test_only',budget_seconds=43200,memory_limit_KiB=1024,prior_workspace='synthetic_fixture')
 def timings(self):return dict(training_seconds=[.1,.2],evaluation_seconds=[.05,.1],keras_save_seconds=[.01,.02],snapshot_save_seconds=[.005,.01],max_RSS_KiB=500)
 def test_formula_and_original_minibatch_rule(self):
  p=self.fixture();r=select(p,**self.timings());expected=2*(3*588*.2+(3*125+125+150)*.1+4*.02+10*.01)+120
  self.assertEqual(r['estimated_seconds_by_epoch_budget']['3'],expected);self.assertEqual(r['chosen_epochs'],3)
  p['batch_size']=1000;r=select(p,**self.timings());self.assertEqual(r['source_actual_minibatch'],470);self.assertEqual(r['training_batches'],10);self.assertEqual(r['validation_batches'],3)
 def test_fixed_order_gate_and_exact_resource_close(self):
  p=self.fixture();first=select(p,**self.timings());p['budget_seconds']=(first['estimated_seconds_by_epoch_budget']['3']+first['estimated_seconds_by_epoch_budget']['1'])/2
  self.assertEqual(select(p,**self.timings())['chosen_epochs'],1)
  p['budget_seconds']=1;r=select(p,**self.timings());self.assertIsNone(r['chosen_epochs']);self.assertEqual(r['state'],'resource_closed_exact_policy');self.assertFalse(r['accuracy_measured'])
  p=self.fixture();times=self.timings();times['max_RSS_KiB']=1025;self.assertIsNone(select(p,**times)['chosen_epochs'])
 def test_invalid_policy_or_measurement_rejected(self):
  for key,value in [('epoch_candidates',[1,1]),('seed',True),('batch_size',0),('input_preprocessing',''),('budget_seconds',float('nan')),('memory_limit_KiB',True)]:
   p=self.fixture();p[key]=value
   with self.assertRaises(ValueError):validate(p)
  for key,value in [('training_seconds',[]),('evaluation_seconds',[float('inf')]),('snapshot_save_seconds',[0]),('max_RSS_KiB',float('nan'))]:
   t=self.timings();t[key]=value
   with self.assertRaises(ValueError):select(self.fixture(),**t)
 def test_current_prior_refusal_before_policy_output_or_engine_import(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);policy=root/'missing_policy.json';out=root/'should_not_exist';before='source_engine' in sys.modules
   with patch.object(Path,'read_bytes',side_effect=AssertionError('Must refuse before policy/evidence read')):
    with self.assertRaisesRegex(RuntimeError,'audit not complete'):profile_cost.main(policy,out,root)
   self.assertFalse(out.exists());self.assertEqual('source_engine' in sys.modules,before)
if __name__=='__main__':unittest.main()
