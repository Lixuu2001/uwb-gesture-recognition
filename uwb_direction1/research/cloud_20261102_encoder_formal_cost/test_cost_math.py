import copy,unittest
from cost_math import select
P=dict(epochs=100,batch_size=12,seed=20261029,budget_seconds=43200,memory_limit_KiB=14680064)
class Boundaries(unittest.TestCase):
 def call(self,p=P,t=0.1,rss=100):return select(p,[t]*8,[.02]*4,[.03]*3,[.04]*3,rss)
 def test_complete_scale_and_saves(self):
  r=self.call();self.assertEqual(r['estimated_seconds'],2*(100*392*.1+8584*.02+101*.03+204*.04)+120);self.assertTrue(r['resource_gate_passed'])
 def test_memory_inclusive(self):self.assertTrue(self.call(rss=14680064)['resource_gate_passed']);self.assertFalse(self.call(rss=14680065)['resource_gate_passed'])
 def test_close_resource(self):self.assertFalse(self.call(t=2)['resource_gate_passed']);self.assertTrue(self.call(t=2)['no_smaller_epoch_batch_fallback'])
 def test_no_reduced_policy(self):
  for k,v in [('epochs',99),('batch_size',16),('seed',7)]:
   p=copy.deepcopy(P);p[k]=v
   with self.assertRaises(AssertionError):self.call(p)
 def test_invalid_timings(self):
  for t in [0,-1,float('inf'),float('nan'),True]:
   with self.assertRaises(AssertionError):self.call(t=t)
if __name__=='__main__':unittest.main()
