import math,unittest
from cost_math import select
P=dict(epochs=250,batch_size=128,seed=20261029,budget_seconds=43200,memory_limit_KiB=14680064)
def run(p=P,train=.2,rss=1000):return select(p,[train]*8,[.03]*4,[.01]*3,[.02]*3,rss)
class Tests(unittest.TestCase):
 def test_independent_arithmetic(self):
  r=run();self.assertEqual([r[k] for k in ['training_batches','validation_batches','diagnostic_batches']],[37,8,10]);self.assertTrue(math.isclose(r['estimated_seconds'],3966.26,rel_tol=0,abs_tol=1e-9));self.assertTrue(r['resource_gate_passed'])
 def test_time_and_memory_close(self):
  self.assertFalse(run(train=3)['resource_gate_passed']);self.assertFalse(run(rss=14680065)['resource_gate_passed'])
 def test_nonfinite_measurement_refused(self):
  with self.assertRaises(AssertionError):run(train=float('nan'))
 def test_budget_rescue_and_shape_refused(self):
  with self.assertRaises(AssertionError):run(p=dict(P,epochs=100))
  with self.assertRaises(AssertionError):select(P,[.2]*7,[.03]*4,[.01]*3,[.02]*3,1000)
if __name__=='__main__':unittest.main()
