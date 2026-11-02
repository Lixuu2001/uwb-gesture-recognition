import unittest
from cost_math import select
P=dict(embedding_width=512,C=1.,seed=20261029,threads=4,batch_size=16,synthetic_train_size=4700,encoder_bytes=46830571)
class Test(unittest.TestCase):
 def test_pass_shared_encoder(self):self.assertTrue(select(P,100,[1]*3,[1]*3,[1]*3,1000,1000,200000000)['resource_gate_passed'])
 def test_large_cost(self):self.assertFalse(select(P,15000,[1]*3,[1]*3,[1]*3,1000,1000,200000000)['resource_gate_passed'])
 def test_memory_and_exact_shared_storage(self):
  self.assertFalse(select(P,100,[1]*3,[1]*3,[1]*3,13000000,1000,200000000)['resource_gate_passed'])
  self.assertFalse(select(P,100,[1]*3,[1]*3,[1]*3,1000,1000,100000000)['resource_gate_passed'])
 def test_scope_and_nonfinite(self):
  with self.assertRaises(AssertionError):select(dict(P,C=2),100,[1]*3,[1]*3,[1]*3,1000,1000,200000000)
  with self.assertRaises(AssertionError):select(P,float('nan'),[1]*3,[1]*3,[1]*3,1000,1000,200000000)
if __name__=='__main__':unittest.main(verbosity=2)
