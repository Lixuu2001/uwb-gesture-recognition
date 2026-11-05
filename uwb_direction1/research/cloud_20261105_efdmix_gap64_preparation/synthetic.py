"""Actual author EFDMix/full native checks, exact sorting and independent gradient routing."""
from pathlib import Path
import copy,time,tempfile,importlib.util,json,random
from common import B,ready
import engine as e
def author():
 spec=importlib.util.spec_from_file_location('reference_efdmix',str(B/'author/efdmix.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.EFDMix
def main():
 ready();start=time.monotonic();rng=e.np.random.default_rng(650);x=rng.normal(size=(64,90,567)).astype('float32');y=e.np.arange(64)%12;xx=e.torch.from_numpy(x);target=e.torch.nn.functional.one_hot(e.torch.as_tensor(y),12).float()
 m,o,g=e.make();initial=copy.deepcopy(e.capture(m,o,g,0,0));assert e.semantic(initial['model'])==e.c['matched_initial_model_state_sha256'];m._efdmix_unit.p=1.;m.train();z=m(xx);ours=copy.deepcopy(e.capture(m,o,g,0,0));e.torch.nn.functional.binary_cross_entropy_with_logits(z,target).backward();grad={n:p.grad.clone() for n,p in m.named_parameters()}
 n=e.native.make_model('gap',e.c['seed']);no=e.native.StableLookahead(n.parameters(),lr=.001,weight_decay=.0001);ng=e.torch.Generator().manual_seed(e.c['seed']);unit=author()(p=1.,alpha=.1,mix='random');e.restore(initial,n,no,ng);n.train();feature=n.feature_extractor.instance_encoder[0](xx.transpose(1,2));styled=unit(feature.unsqueeze(-1)).squeeze(-1);actual=n.head(n.feature_extractor.instance_encoder[1](styled).mean(-1));e.torch.testing.assert_close(z,actual,rtol=0,atol=0);assert e.semantic(ours)==e.semantic(e.capture(n,no,ng,0,0));e.torch.nn.functional.binary_cross_entropy_with_logits(actual,target).backward()
 for name,p in n.named_parameters():e.torch.testing.assert_close(grad[name],p.grad,rtol=0,atol=0)
 m.eval();n.eval();before=copy.deepcopy(e.capture(m,o,g,0,0));a=m(xx);b=n(xx);e.torch.testing.assert_close(a,b,rtol=0,atol=0);assert e.semantic(before)==e.semantic(e.capture(m,o,g,0,0))
 plain=e.native.make_model('gap',e.c['seed']);po=e.native.StableLookahead(plain.parameters(),lr=.001,weight_decay=.0001);pg=e.torch.Generator().manual_seed(e.c['seed']);e.restore(initial,plain,po,pg);plain.train();assert not e.torch.allclose(z,plain(xx),rtol=1e-6,atol=1e-7)
 # Independent scalar rank matching and its published straight-through VJP.
 # Numerical finite differences would intentionally NOT equal this stop-gradient.
 a=e.torch.from_numpy(rng.normal(size=(7,3,9,1)));a[:,:,-3:]=0.;a.requires_grad_();w=e.torch.from_numpy(rng.normal(size=(7,3,9,1)));unit=author()(p=1.,alpha=.1,mix='random');random.seed(8);e.torch.manual_seed(8);pr=random.getstate();tr=e.torch.get_rng_state().clone();out=unit(a);post=(random.getstate(),e.torch.get_rng_state().clone());random.setstate(pr);e.torch.set_rng_state(tr);random.random();values,order=e.torch.sort(a.view(7,3,9));lam=e.torch.distributions.Beta(.1,.1).sample((7,1,1));perm=e.torch.randperm(7);inverse=e.torch.empty_like(order);inverse.scatter_(-1,order,e.torch.arange(9).view(1,1,9).expand(7,3,9));expected=e.torch.empty_like(a);vjp=w.clone()
 for bi in range(7):
  coefficient=float((1-lam[bi,0,0]).item())
  for ci in range(3):
   for ti in range(9):
    donor=int(perm[bi]);donor_t=int(order[donor,ci,inverse[bi,ci,ti]]);expected[bi,ci,ti,0]=a[bi,ci,ti,0]+(a[donor,ci,donor_t,0]-a[bi,ci,ti,0])*coefficient;vjp[donor,ci,donor_t,0]+=w[bi,ci,ti,0]*coefficient
 e.torch.testing.assert_close(out,expected,rtol=0,atol=0);e.torch.testing.assert_close(e.torch.autograd.grad((out*w).sum(),a)[0],vjp,rtol=1e-14,atol=1e-14);assert random.getstate()==post[0] and e.torch.equal(e.torch.get_rng_state(),post[1])
 # Fixed points and ties are retained; deactivation/eval consumes no RNG.
 unit.set_activation_status(False);pr=random.getstate();tr=e.torch.get_rng_state().clone();assert unit(a) is a and random.getstate()==pr and e.torch.equal(tr,e.torch.get_rng_state())
 m,o,g=e.make();branches=[]
 for _ in range(5):m.train();branches.append(e.step(m,o,x,y,1)['efdmix_applied'])
 saved=copy.deepcopy(e.capture(m,o,g,1,5));digest=e.semantic(saved);m.train();left=e.step(m,o,x,y,1);after=copy.deepcopy(e.capture(m,o,g,1,6));assert o.count==6 and o.slow;n,no,ng=e.make();e.restore(saved,n,no,ng);n.train();right=e.step(n,no,x,y,1);assert left==right and e.semantic(after)==e.semantic(e.capture(n,no,ng,1,6)) and e.semantic(saved)==digest
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'native.pt';e.torch.save(after,p);disk=e.torch.load(p,map_location='cpu',weights_only=True);m.train();left=e.step(m,o,x,y,1);last=copy.deepcopy(e.capture(m,o,g,1,7));n,no,ng=e.make();e.restore(disk,n,no,ng);n.train();right=e.step(n,no,x,y,1);assert left==right and e.semantic(last)==e.semantic(e.capture(n,no,ng,1,7))
 n.train();assert e.step(n,no,x[:28],y[:28],1)['forward_batch']==28
 q=dict(state='pass',parameters=617612,alpha=.1,probability=.5,mix='random',insertion=0,seed=20261029,actual_author_and_manual_full_native_logits_BN_gradients_allRNG_exact=True,scalar_rank_matching_ties_and_donor_VJP_verified=True,content_gradient_identity_plus_non_detached_donor_branch=True,evaluation_clean_native_exact_no_RNG=True,deactivated_identity_no_RNG=True,style_effect_reaches_logits=True,native_Lookahead_slow6_disk_model_optimizer_allRNG_restore_exact=True,tail28_passed=True,probability_branches=branches,no_UWB_arrays_or_labels_opened=True,new_UWB_fit_started=False,DSU_diagnostic_read=False,users7_8_labels_used=False,wall_seconds=time.monotonic()-start)
 (B/'SYNTHETIC_RESULT.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
if __name__=='__main__':main()
