import unittest,copy
from cost_math import select
P=dict(samples=100,batch_size=64,seed=20261029,threads=1,budget_seconds=43200,memory_limit_KiB=14680064)
class Boundaries(unittest.TestCase):
 def call(self,p=P,t=.1,rss=100):return select(p,[t]*8,[.2]*3,[.3]*3,rss)
 def test_both_roles_control_and_full_independent_replay(self):self.assertEqual(self.call()['estimated_seconds'],2*(3535*.1+3*.2+3*.3)+120)
 def test_budget(self):self.assertFalse(self.call(t=7)['resource_gate_passed'])
 def test_memory_boundary(self):self.assertTrue(self.call(rss=14680064)['resource_gate_passed']);self.assertFalse(self.call(rss=14680065)['resource_gate_passed'])
 def test_changed_policy(self):
  for n,v in [('samples',99),('batch_size',32),('seed',7),('threads',2)]:
   p=copy.deepcopy(P);p[n]=v
   with self.assertRaises(AssertionError):self.call(p=p)
 def test_invalid(self):
  for v in [0,-1,True,float('nan'),float('inf')]:
   with self.assertRaises(AssertionError):self.call(t=v)
if __name__=='__main__':unittest.main()
