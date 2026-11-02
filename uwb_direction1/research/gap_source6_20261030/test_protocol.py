"""Synthetic protocol/serialization checks; no UWB labels/maps or new fitting."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
import train
from torch.nn import functional as F

class ProtocolTests(unittest.TestCase):
 def test_roles_all_six_and_counts(self):
  cfg=json.loads((train.B/'PROTOCOL.json').read_text());users=np.repeat(np.arange(1,7),[1200,1000,1200,1200,1200,1100])
  for h in range(1,7):
   v,tr,va,te=train.fold_indices(users,h)
   self.assertEqual(v,h%6+1);self.assertEqual([len(tr),len(va),len(te)],cfg['counts'][str(h)])
   self.assertEqual(set(users[tr]),set(range(1,7))-{h,v})
   self.assertFalse(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
 def test_gate_independent_conditions(self):
  cfg=json.loads((train.B/'PROTOCOL.json').read_text())
  self.assertTrue(train.gate([.02]*6,cfg))
  self.assertFalse(train.gate([.005]*6,cfg))
  self.assertFalse(train.gate([.05,.05,.05,-.001,-.001,-.001],cfg))
  self.assertFalse(train.gate([.04,.04,.04,.04,.04,-.021],cfg))
 def test_freeze_guard_and_tamper_before_scores(self):
  cfg={'new_held_users':[2,3]}
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   with self.assertRaises(FileNotFoundError):train.require_freeze(p,cfg)
   for h in [2,3]:
    (p/f'held{h}').mkdir();(p/f'held{h}/predictions_unscored.npz').write_bytes(b'synthetic')
   train.write(p/'ALL_NEW_PREDICTIONS_FROZEN.json',{str(h):train.sha(p/f'held{h}/predictions_unscored.npz') for h in [2,3]})
   train.require_freeze(p,cfg)
   (p/'held3/predictions_unscored.npz').write_bytes(b'changed')
   with self.assertRaises(AssertionError):train.require_freeze(p,cfg)
 def test_metrics_missing_classes(self):
  self.assertEqual(train.macro(np.array([0,0,2,2]),np.array([0,1,2,2])),.75)
 def test_author_gap_and_stochastic_optimizer_resume(self):
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
  m=train.make_model('gap',20261029);self.assertEqual(sum(p.numel() for p in m.parameters()),617612)
  opt=train.StableLookahead(m.parameters());x=torch.randn(4,90,567);y=F.one_hot(torch.arange(4),12).float()
  def step(model,o):
   model.train();o.zero_grad();loss=F.binary_cross_entropy_with_logits(model(x,warmup=True),y);loss.backward()
   self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
   torch.nn.utils.clip_grad_norm_(model.parameters(),2.);o.step()
  for _ in range(7):step(m,opt)
  checkpoint=copy.deepcopy({'model':m.state_dict(),'optimizer':opt.state_dict(),'rng':torch.get_rng_state()})
  for _ in range(6):step(m,opt)
  clone=train.make_model('gap',20261029);other=train.StableLookahead(clone.parameters());clone.load_state_dict(checkpoint['model']);other.load_state_dict(checkpoint['optimizer']);torch.set_rng_state(checkpoint['rng'])
  for _ in range(6):step(clone,other)
  self.assertEqual(other.count,opt.count)
  self.assertTrue(all(torch.equal(v,clone.state_dict()[k]) for k,v in m.state_dict().items()))
  self.assertTrue(all(torch.equal(v,other.slow[k]) for k,v in opt.slow.items()))
 def test_atomic_checkpoint_and_log_roundtrip(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);train.save(p/'state.pt',{'x':torch.arange(4)});self.assertTrue(torch.equal(torch.load(p/'state.pt')['x'],torch.arange(4)))
   train.write(p/'state.json',{'state':'synthetic'});self.assertEqual(json.loads((p/'state.json').read_text())['state'],'synthetic')

if __name__=='__main__':unittest.main()
