"""Pure arithmetic and pre-access refusal; not an actual formal cost profile."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from cost_tools import validate,select
import profile_cost
class Tests(unittest.TestCase):
 def policy(self):return dict(epoch_candidates=[5,2,1],batch_size=64,seed=7,input_preprocessing='explicit_RMS_adaptation',budget_seconds=43200,memory_limit_KiB=10000,prior_workspace='synthetic_prior')
 def timing(self):return dict(training_seconds=[2.,1.],evaluation_seconds=[.5,.25],keras_save_seconds=[.1],snapshot_save_seconds=[.2],max_RSS_KiB=9000)
 def test_independent_arithmetic_and_order(self):
  p=self.policy();r=select(p,**self.timing());expected=2*(5*74*2.+(5*16+16+19)*.5+6*.1+9*.2)+120;self.assertEqual(r['estimated_seconds_by_epoch_budget']['5'],expected);self.assertEqual(r['chosen_epochs'],5);p['budget_seconds']=r['estimated_seconds_by_epoch_budget']['2'];self.assertEqual(select(p,**self.timing())['chosen_epochs'],2)
 def test_no_feasible_budget_or_memory_does_not_shrink(self):
  p=self.policy();p['budget_seconds']=1;self.assertIsNone(select(p,**self.timing())['chosen_epochs']);p=self.policy();t=self.timing();t['max_RSS_KiB']=10001;self.assertIsNone(select(p,**t)['chosen_epochs']);self.assertEqual(p['batch_size'],64)
 def test_bad_timings_or_unset_policy_refuse(self):
  with self.assertRaises(ValueError):validate({})
  t=self.timing();t['training_seconds']=[float('nan')]
  with self.assertRaises(ValueError):select(self.policy(),**t)
 def test_previous_active_refuses_before_policy_read_or_output(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'not_created'
   with patch.object(profile_cost,'prior_closed',side_effect=RuntimeError('prior audit active')),patch.object(Path,'read_bytes',side_effect=AssertionError('No policy read')):
    with self.assertRaisesRegex(RuntimeError,'prior audit active'):profile_cost.main(Path(d)/'absent_policy',out,Path(d)/'prior')
   self.assertFalse(out.exists())
if __name__=='__main__':unittest.main()
