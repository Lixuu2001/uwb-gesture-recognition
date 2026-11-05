"""Full617612parameter native model checks, only synthetic signals and labels."""
from pathlib import Path
import copy,json,tempfile,time
import engine as e
from common import ready
from prototype import batch,native_bce,to_author_layout,from_author_layout
from author_reference import author_functions
B=Path(__file__).resolve().parent
def main():
    ready();started=time.monotonic();rng=e.np.random.default_rng(650);ref=author_functions()
    x=e.torch.from_numpy(rng.normal(size=(64,90,567)).astype('float32'));y=e.torch.arange(64)%12
    m,o,g=e.make();initial=copy.deepcopy(e.capture(m,o,g,0,0));assert e.semantic(initial['model'])==e.c['matched_initial_model_state_sha256']
    # Force the mixed branch ONLY in this equivalence fixture; prospective p=.8 remains fixed.
    m.train();xx,a,b,lam,meta=batch(x,y,1.,1.);assert meta['mixed']
    before={n:int(z.num_batches_tracked) for n,z in m.named_modules() if isinstance(z,e.torch.nn.modules.batchnorm._BatchNorm)}
    z=m(xx);loss=native_bce(z,a,b,lam);loss.backward();grads={n:p.grad.clone() for n,p in m.named_parameters()};buffers=copy.deepcopy(m.state_dict());rng_after=e.torch.get_rng_state().clone()
    assert before and all(int(dict(m.named_modules())[n].num_batches_tracked)==k+1 for n,k in before.items())
    n,no,ng=e.make();e.restore(initial,n,no,ng);n.train();assert float(e.torch.rand(1).item())<1.
    actual,ya,yb,author_lam=ref['_apply_cutmix'](to_author_layout(x),y,1.);assert author_lam==lam
    actual_logits=n(from_author_layout(actual));e.torch.testing.assert_close(z,actual_logits,rtol=0,atol=0);assert e.semantic(buffers)==e.semantic(n.state_dict()) and e.torch.equal(rng_after,e.torch.get_rng_state())
    hard_a=e.torch.nn.functional.one_hot(ya,12).float();hard_b=e.torch.nn.functional.one_hot(yb,12).float()
    independent=lam*e.torch.nn.functional.binary_cross_entropy_with_logits(actual_logits,hard_a)+(1-lam)*e.torch.nn.functional.binary_cross_entropy_with_logits(actual_logits,hard_b)
    e.torch.testing.assert_close(loss,independent,rtol=1e-6,atol=1e-7);independent.backward();maximum=0.
    for name,p in n.named_parameters():
        e.torch.testing.assert_close(grads[name],p.grad,rtol=2e-5,atol=2e-7);maximum=max(maximum,float((grads[name]-p.grad).abs().max()))
    # The actual prospective p=.8 engine, full optimizer and all RNG recover exactly.
    m,o,g=e.make();branches=[]
    for _ in range(5):m.train();branches.append(e.step(m,o,x.numpy(),y.numpy(),1)['mixed'])
    saved=copy.deepcopy(e.capture(m,o,g,5,5));saved_digest=e.semantic(saved);m.train();left=e.step(m,o,x.numpy(),y.numpy(),1);after=copy.deepcopy(e.capture(m,o,g,6,6))
    assert o.count==6 and o.slow and e.semantic(saved)==saved_digest
    n,no,ng=e.make();e.restore(saved,n,no,ng);n.train();right=e.step(n,no,x.numpy(),y.numpy(),1);assert left==right and e.semantic(after)==e.semantic(e.capture(n,no,ng,6,6)) and e.semantic(saved)==saved_digest
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'native_synthetic.pt';e.torch.save(after,p);loaded=e.torch.load(p,map_location='cpu',weights_only=True)
        m.train();left=e.step(m,o,x.numpy(),y.numpy(),1);finish=copy.deepcopy(e.capture(m,o,g,7,7));n,no,ng=e.make();e.restore(loaded,n,no,ng);n.train();right=e.step(n,no,x.numpy(),y.numpy(),1);assert left==right and e.semantic(finish)==e.semantic(e.capture(n,no,ng,7,7))
    # A real pipeline has a raw28 tail; exercise that actual forward/update shape too.
    n.train();tail=e.step(n,no,x[:28].numpy(),y[:28].numpy(),1);assert tail['effective_forward_batch']==28
    q=dict(state='pass',parameters=617612,raw_batch=64,forward_batch=64,alpha=1.,probability=.8,seed=20261029,matched_native_noMixup_initial_model_exact=True,actual_unmodified_radar_author_helper_native_logits_BN_buffers_Torch_RNG_exact=True,native_mixed_branch_forced_only_for_fixture_coverage=True,each_native_BN_one_update=True,native_OvRBCE_gradient_agrees_with_independent_weighted_hard_targets=True,max_gradient_absolute_difference=maximum,full_optimizer_slow_state_and_all_RNG_disk_resume_exact=True,saved_optimizer_fixture_immutable=True,actual_tail_batch28_forward_and_optimizer_update_passed=True,prospective_probability_branches=branches,no_UWB_arrays_or_labels_opened=True,new_UWB_fit_started=False,current_RegMixup_configuration_untouched=True,wall_seconds=time.monotonic()-started)
    (B/'SYNTHETIC_RESULT.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
if __name__=='__main__':main()
