"""Synthetic ID logging must not alter updates/RNG; wrong identity cannot restore."""
import unittest
from source_engine import *
class State(unittest.TestCase):
 def test_observer_does_not_change_updates_or_RNG(self):
  rng=np.random.RandomState(12);x=torch.from_numpy(rng.randn(8,567,90).astype('float32'));y=torch.tensor([0,1,2,3,4,5,6,11]);ids=torch.arange(8)
  plain=[(x[i:i+4],y[i:i+4]) for i in [0,4]];tagged=[(x[i:i+4],y[i:i+4],ids[i:i+4]) for i in [0,4]]
  a=build(567,90,12,21);r=a.train_epoch(plain);gold=a.snapshot({'synthetic':True},{'epoch':1});observed=[]
  b=build(567,90,12,21);r2=b.train_epoch(tagged,lambda index,sample,loss:observed.append((index,sample[2].tolist(),loss)));equal(r,r2);equal(gold,b.snapshot({'synthetic':True},{'epoch':1}));self.assertEqual([n for _,row,_ in observed for n in row],list(range(8)))
 def test_wrong_identity_refuses_restore_before_mutation(self):
  a=build(567,90,12,21);saved=a.snapshot({'fixed':'identity'},{'epoch':0});before=a.snapshot({'fixed':'identity'},{'epoch':0})
  with self.assertRaises(AssertionError):a.restore(saved,expected_identity={'fixed':'wrong'})
  equal(before,a.snapshot({'fixed':'identity'},{'epoch':0}))
if __name__=='__main__':
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);unittest.main()
