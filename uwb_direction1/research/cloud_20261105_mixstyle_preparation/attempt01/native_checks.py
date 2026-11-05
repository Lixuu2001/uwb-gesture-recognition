"""Full native model/actual author/axis/gradient/recovery checks on synthetic signals only."""
from pathlib import Path
import copy,time,tempfile,importlib.util,json,random
from common import B,ready
import engine as e
def author():
    spec=importlib.util.spec_from_file_location('reference_author_mixstyle',str(B/'author/mixstyle.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod.MixStyle
def main():
    ready();t=time.monotonic();rng=e.np.random.default_rng(650);x=rng.normal(size=(64,90,567)).astype('float32');y=e.np.arange(64)%12;xx=e.torch.from_numpy(x);target=e.torch.nn.functional.one_hot(e.torch.as_tensor(y),12).float()
    m,o,g=e.make();initial=copy.deepcopy(e.capture(m,o,g,0,0));assert e.semantic(initial['model'])==e.c['matched_initial_model_state_sha256']
    m._mixstyle_unit.p=1.;m.train();z=m(xx);ours=copy.deepcopy(e.capture(m,o,g,0,0));loss=e.torch.nn.functional.binary_cross_entropy_with_logits(z,target);loss.backward();grads={n:p.grad.clone() for n,p in m.named_parameters()}
    # Actual author class on the first complete block; no installed hook in this reference.
    n=e.native.make_model('gap',e.c['seed']);no=e.native.StableLookahead(n.parameters(),lr=.001,weight_decay=.0001);ng=e.torch.Generator().manual_seed(e.c['seed']);unit=author()(p=1.,alpha=.1,eps=1e-6,mix='random');e.restore(initial,n,no,ng);n.train()
    feature=n.feature_extractor[0](xx.transpose(1,2));styled=unit(feature.unsqueeze(-1)).squeeze(-1);actual=n.head(n.feature_extractor[1](styled).mean(-1));e.torch.testing.assert_close(z,actual,rtol=0,atol=0);assert e.semantic(ours)==e.semantic(e.capture(n,no,ng,0,0))
    ref=e.torch.nn.functional.binary_cross_entropy_with_logits(actual,target);ref.backward()
    for name,p in n.named_parameters():e.torch.testing.assert_close(grads[name],p.grad,rtol=0,atol=0)
    # Eval is a bit-exact clean baseline and consumes no Python/Torch RNG.
    m.eval();n.eval();before=(e.torch.get_rng_state().clone(),random.getstate());a=m(xx);b=n(xx);e.torch.testing.assert_close(a,b,rtol=0,atol=0);assert e.torch.equal(before[0],e.torch.get_rng_state()) and before[1]==random.getstate()
    # Perturbation reaches full native logits; no claim that statistics are pure user style.
    plain=e.native.make_model('gap',e.c['seed']);po=e.native.StableLookahead(plain.parameters(),lr=.001,weight_decay=.0001);pg=e.torch.Generator().manual_seed(e.c['seed']);e.restore(initial,plain,po,pg);plain.train();clean=plain(xx);assert not e.torch.allclose(z,clean,rtol=1e-6,atol=1e-7)
    # Independent closed-form detached-statistics Jacobian and exact author RNG/arithmetic.
    a=e.torch.from_numpy(rng.normal(size=(7,3,9,1))).requires_grad_();w=e.torch.from_numpy(rng.normal(size=(7,3,9,1)));unit=author()(p=1.,alpha=.1,eps=1e-6);random.seed(8);e.torch.manual_seed(8);pr=random.getstate();tr=e.torch.get_rng_state().clone();v=unit(a);post=(random.getstate(),e.torch.get_rng_state().clone());random.setstate(pr);e.torch.set_rng_state(tr);random.random();mu=a.mean((2,3),keepdim=True).detach();sig=(a.var((2,3),keepdim=True)+1e-6).sqrt().detach();lam=e.torch.distributions.Beta(.1,.1).sample((7,1,1,1));perm=e.torch.randperm(7);sm=sig*lam+sig[perm]*(1-lam);expected=(a-mu)/sig*sm+mu*lam+mu[perm]*(1-lam);e.torch.testing.assert_close(v,expected,rtol=0,atol=0);e.torch.testing.assert_close(e.torch.autograd.grad((v*w).sum(),a)[0],w*sm/sig,rtol=1e-14,atol=1e-14);assert random.getstate()==post[0] and e.torch.equal(e.torch.get_rng_state(),post[1])
    # The actual p=.5 engine reconstructs hook behavior and full native lazy optimizer/RNG.
    m,o,g=e.make();branches=[]
    for j in range(5):m.train();branches.append(e.step(m,o,x,y,1)['mixstyle_applied'])
    saved=copy.deepcopy(e.capture(m,o,g,1,5));original=e.semantic(saved);m.train();left=e.step(m,o,x,y,1);after=copy.deepcopy(e.capture(m,o,g,1,6));assert o.count==6 and o.slow
    n,no,ng=e.make();e.restore(saved,n,no,ng);n.train();right=e.step(n,no,x,y,1);assert left==right and e.semantic(after)==e.semantic(e.capture(n,no,ng,1,6)) and e.semantic(saved)==original
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'native.pt';e.torch.save(after,p);disk=e.torch.load(p,map_location='cpu',weights_only=True);m.train();left=e.step(m,o,x,y,1);last=copy.deepcopy(e.capture(m,o,g,1,7));n,no,ng=e.make();e.restore(disk,n,no,ng);n.train();right=e.step(n,no,x,y,1);assert left==right and e.semantic(last)==e.semantic(e.capture(n,no,ng,1,7))
    n.train();tail=e.step(n,no,x[:28],y[:28],1);assert tail['forward_batch']==28
    q=dict(state='pass_conditional_full_native_MixStyle_synthetic_only',parameters=617612,alpha=.1,probability=.5,seed=20261029,matched_native_initialization=True,actual_unmodified_author_class_first_complete_block_logits_BN_gradients_allRNG_exact=True,eval_clean_baseline_bit_exact_no_RNG=True,style_effect_not_canceled_at_logits=True,independent_detached_stats_Jacobian_and_author_RNG_verified=True,full_native_optimizer_and_allRNG_disk_resume_across_slow6_exact=True,hook_reconstructed_from_frozen_code=True,tail28_passed=True,actual_probability_branches=branches,no_UWB_arrays_or_labels_opened=True,new_UWB_fit_started=False,CutMix_diagnostic_read=False,users7_8_labels_used=False,wall_seconds=time.monotonic()-t);(B/'SYNTHETIC_RESULT.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
if __name__=='__main__':main()
