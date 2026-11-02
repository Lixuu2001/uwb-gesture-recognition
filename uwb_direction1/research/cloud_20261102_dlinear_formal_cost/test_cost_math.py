import unittest,copy
from cost_math import select
P=dict(epochs=10,batch_size=32,seed=20261029,budget_seconds=43200,memory_limit_KiB=14680064)
class Boundaries(unittest.TestCase):
 def runbound(self,p=P,t=.1,rss=100):return select(p,[t]*8,[.02]*4,[.03]*3,[.04]*3,rss)
 def test_full_scale(self):self.assertEqual(self.runbound()['estimated_seconds'],2*(1470*.1+710*.02+11*.03+24*.04)+120)
 def test_budget_failure(self):self.assertFalse(self.runbound(t=20)['resource_gate_passed'])
 def test_memory_boundary(self):self.assertTrue(self.runbound(rss=14680064)['resource_gate_passed']);self.assertFalse(self.runbound(rss=14680065)['resource_gate_passed'])
 def test_no_reduced_policy(self):
  for k,v in [('epochs',9),('batch_size',64),('seed',7)]:
   q=copy.deepcopy(P);q[k]=v
   with self.assertRaises(AssertionError):self.runbound(p=q)
 def test_invalid_time(self):
  for t in [0,-1,float('nan'),float('inf'),True]:
   with self.assertRaises(AssertionError):self.runbound(t=t)
if __name__=='__main__':unittest.main()
