import unittest
from cost_math import select
P=dict(epochs=2000,batch_size=16,seed=20261029,budget_seconds=43200,memory_limit_KiB=14680064)
def run(p=P,train=.01,rss=1000):return select(p,[train]*8,[.005]*4,[.01]*3,[.02]*3,rss)
class Tests(unittest.TestCase):
 def test_independent_example(self):
  r=run();self.assertEqual([r[k] for k in ['training_batches','validation_batches','diagnostic_batches']],[294,63,75]);self.assertAlmostEqual(r['estimated_seconds'],13341.56);self.assertTrue(r['resource_gate_passed'])
 def test_time_memory_threshold(self):
  self.assertFalse(run(train=.04)['resource_gate_passed']);self.assertFalse(run(rss=14680065)['resource_gate_passed'])
 def test_invalid_measurement_refused(self):
  for v in [float('nan'),float('inf'),0,-1]:
   with self.assertRaises(AssertionError):run(train=v)
 def test_changed_budget_seed_batch_and_missing_measurement_refused(self):
  for p in [dict(P,epochs=100),dict(P,batch_size=128),dict(P,seed=1)]:
   with self.assertRaises(AssertionError):run(p=p)
  with self.assertRaises(AssertionError):select(P,[.01]*7,[.005]*4,[.01]*3,[.02]*3,1000)
if __name__=='__main__':unittest.main()
