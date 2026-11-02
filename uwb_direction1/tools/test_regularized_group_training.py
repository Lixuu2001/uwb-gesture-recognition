"""Synthetic end-to-end interruption and held-label isolation check."""
import contextlib,io,json,tempfile
from pathlib import Path
import numpy as np
import torch
from regularized_group_source import fit,audit_run
from regularized_groups import GroupRisk
from circular_signal import SignalNet

def same(a,b):
 if isinstance(a,torch.Tensor):assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:same(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):same(x,y)
 else:assert a==b

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);rng=np.random.default_rng(42)
 cache=rng.normal(size=(128,3,6,90,63)).astype('float32');users=np.repeat(np.arange(1,9),16);y=np.tile(np.arange(16)%12,8)
 c=dict(seed=2026,lr=.001,weight_decays={'dro_reg':.01},epochs=2,min_lr=.00001,tail_start_epoch=1,batch=16,per_group=4,group_eta=.01)
 with tempfile.TemporaryDirectory(prefix='uwb-group-training-test-') as tmp,contextlib.redirect_stdout(io.StringIO()):
  root=Path(tmp);identity={'test':'synthetic_not_accuracy_evidence'}
  fit(root/'full',c,identity,cache,y,users,3,'dro_reg')
  fit(root/'resume',c,identity,cache,y,users,3,'dro_reg',stop_after=1)
  fit(root/'resume',c,identity,cache,y,users,3,'dro_reg')
  x=torch.load(root/'full/progress.pt',weights_only=False);z=torch.load(root/'resume/progress.pt',weights_only=False)
  for key in ['model','optimizer','scheduler','rng','risk','tail','best','best_ce','best_epoch','step']:same(x[key],z[key])
  for e in [1,2]:
   a=np.load(root/'full'/f'updates_{e:02d}.npz');b=np.load(root/'resume'/f'updates_{e:02d}.npz')
   for k in a.files:np.testing.assert_array_equal(a[k],b[k])
  changed=y.copy();changed[users==3]=(changed[users==3]+5)%12
  fit(root/'changed',c,identity,cache,changed,users,3,'dro_reg');z=torch.load(root/'changed/progress.pt',weights_only=False)
  for key in ['model','optimizer','scheduler','rng','risk','tail','best','best_ce','best_epoch','step']:same(x[key],z[key])
  audit=audit_run(root/'full',cache,y,users,c)
 # Equal-size groups + uniform weights agree with plain CE gradients on real architecture.
 torch.manual_seed(2026);net=SignalNet();x=torch.from_numpy(cache[:8,:,[0,2,3,4]]);target=torch.from_numpy(y[:8]).long();groups=torch.arange(4).repeat_interleave(2)
 logits=net(x);loss=torch.nn.functional.cross_entropy(logits,target,reduction='none');a=GroupRisk().loss(loss,groups)[0]
 g=torch.autograd.grad(a,list(net.parameters()),retain_graph=True);gg=torch.autograd.grad(loss.mean(),list(net.parameters()))
 for v,w in zip(g,gg):torch.testing.assert_close(v,w,rtol=2e-5,atol=2e-6)
 print(json.dumps(dict(passed=True,synthetic=True,checks=['full_optimizer_rng_q_tail_resume_bitwise','all_update_ids_and_q_resume_bitwise','held_labels_cannot_change_training_or_selection','functional_forward_and_q_audit','uniform_group_risk_matches_CE_gradient'],audit=audit)))
if __name__=='__main__':main()
