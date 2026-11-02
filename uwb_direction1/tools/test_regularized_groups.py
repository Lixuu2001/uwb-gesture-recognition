import json,copy
import numpy as np
import torch
from regularized_groups import GroupRisk,balanced_order

def main():
 torch.set_num_threads(1);torch.manual_seed(12);q=np.ones(4)/4;r=GroupRisk(eta=.01,robust=True)
 for i in range(30):
  losses=torch.rand(16,dtype=torch.float64,requires_grad=True)*10;groups=torch.arange(4).repeat_interleave(4);means=np.array([losses.detach().numpy()[j*4:j*4+4].mean() for j in range(4)])
  q=q*np.exp(.01*means);q=q/q.sum();loss,_,got=r.loss(losses,groups);np.testing.assert_allclose(got.numpy(),q,rtol=1e-13,atol=1e-13)
  grad=torch.autograd.grad(loss,losses)[0];np.testing.assert_allclose(grad.numpy(),np.repeat(q/4,4),rtol=1e-13,atol=1e-13)
 s=r.state();r2=GroupRisk(eta=.01,robust=True);r2.load(s);x=torch.rand(16,dtype=torch.float64);assert torch.equal(r.loss(x,groups)[0],r2.loss(x,groups)[0])
 uniform=GroupRisk();zero=GroupRisk(eta=0,robust=True);assert torch.equal(uniform.loss(x,groups)[0],zero.loss(x,groups)[0]);torch.testing.assert_close(uniform.loss(x,groups)[0],x.mean())
 before=r.state()
 try:r.loss(x,torch.zeros(16,dtype=torch.long));raise AssertionError('Missing groups accepted')
 except ValueError:assert torch.equal(before['logq'],r.logq)
 try:r.loss(x*float('nan'),groups);raise AssertionError('NaN accepted')
 except ValueError:assert torch.equal(before['logq'],r.logq)
 huge=GroupRisk(eta=1,robust=True);huge.loss(torch.tensor([1e6,1,2,3],dtype=torch.float64),torch.arange(4));assert torch.isfinite(huge.logq).all() and abs(float(huge.logq.exp().sum())-1)<1e-12
 users=np.repeat(np.arange(1,9),np.array([3,3,32,32,32,17,32,32]));ids=np.flatnonzero(np.isin(users,[3,4,5,6]));order,g=balanced_order(ids,users,2026,7,1,8)
 assert set(order.ravel())==set(ids) and np.all(order>=6)
 for b in order:
  assert len(set(b))==32
  assert all(sum(users[b]==u)==8 for u in g)
 np.testing.assert_array_equal(order,balanced_order(ids,users,2026,7,1,8)[0])
 changed=users.copy();changed[~np.isin(np.arange(len(users)),ids)]=99;np.testing.assert_array_equal(order,balanced_order(ids,changed,2026,7,1,8)[0])
 print(json.dumps(dict(passed=True,checks=['30_numpy_exp_updates_and_detached_gradients','q_resume','eta0_uniform_equivalence','missing_group_and_nonfinite_rejection','large_loss_log_stability','balanced_source_coverage_and_reproducibility','outside_source_user_isolation'])))
if __name__=='__main__':main()
