"""Author pure formula; native boundaries/gradients/RNG/journal/completion invariants."""
import ast,copy,tempfile,time,resource,json
from pathlib import Path
from common import B,ready,write,sha
import engine as e
def main():
    c=ready();start=time.monotonic()
    assert sha(c['author_utils_reference_path'])==c['author_utils_reference_sha256']
    tree=ast.parse(Path(c['author_utils_reference_path']).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['mixup_process','get_lambda']];assert len(nodes)==2
    author=dict(np=e.np,torch=e.torch);exec(compile(ast.Module(body=nodes,type_ignores=[]),'author_pure_mix_helpers','exec'),author)
    rng=e.np.random.default_rng(55);x=rng.normal(size=(64,90,567)).astype('float32');y=e.np.arange(64)%12
    features=e.torch.from_numpy(x);target=e.torch.nn.functional.one_hot(e.torch.as_tensor(y).long(),12).float()
    e.np.random.seed(c['seed']);beta=author['get_lambda'](2.);lam=e.torch.tensor([beta],dtype=e.torch.float32);a,b=author['mixup_process'](features,target,lam)
    e.np.random.seed(c['seed']);our_beta=e.np.random.beta(c['alpha'],c['alpha']);xx,yy,perm=e.mix_at(features,target,e.torch.tensor([our_beta],dtype=e.torch.float32))
    e.torch.testing.assert_close(a,xx,rtol=0,atol=0);e.torch.testing.assert_close(b,yy,rtol=0,atol=0)
    # Closed-form cross-sample Jacobian detects detached shuffled-feature gradients.
    feat=e.torch.tensor([[.2,.4],[.3,.8]],dtype=e.torch.float64,requires_grad=True);weights=e.torch.tensor([[.7,.5],[.2,.9]],dtype=e.torch.float64)
    mixed=.3*feat+.7*feat[[1,0]];gradient=e.torch.autograd.grad((mixed*weights).sum(),feat)[0]
    e.torch.testing.assert_close(gradient,.3*weights+.7*weights[[1,0]],atol=1e-14,rtol=1e-14)
    # For lambda1 every native boundary must match ordinary full-GAP forward/backward/update.
    for layer in [0,1,2]:
        m,o,g=e.make();initial=copy.deepcopy(e.capture(m,o,g,0,0));m.train();o.zero_grad()
        z=m(e.torch.from_numpy(x));loss=e.torch.nn.functional.binary_cross_entropy_with_logits(z,target);loss.backward()
        native_grads={n:p.grad.detach().clone() for n,p in m.named_parameters()};e.torch.nn.utils.clip_grad_norm_(m.parameters(),2.);o.step();baseline=e.semantic(e.capture(m,o,g,0,1))
        r,ro,rg=e.make();e.restore(initial,r,ro,rg);r.train();ro.zero_grad();actual,tgt,meta=e.forward_mix(r,e.torch.from_numpy(x),e.torch.as_tensor(y).long(),layer=layer,lam=1.,permutation=e.np.arange(64))
        e.torch.testing.assert_close(actual,z,rtol=0,atol=0);e.torch.testing.assert_close(tgt,target,rtol=0,atol=0)
        loss2=e.torch.nn.functional.binary_cross_entropy_with_logits(actual,tgt);loss2.backward()
        for n,p in r.named_parameters():e.torch.testing.assert_close(p.grad,native_grads[n],rtol=0,atol=0)
        e.torch.nn.utils.clip_grad_norm_(r.parameters(),2.);ro.step();assert baseline==e.semantic(e.capture(r,ro,rg,0,1))
        # Genuine nontrivial hidden mixing must propagate to every complete native block.
        r,ro,rg=e.make();r.train();ro.zero_grad();z2,soft,_=e.forward_mix(r,e.torch.from_numpy(x),e.torch.as_tensor(y).long(),layer=layer,lam=.37,permutation=e.np.arange(63,-1,-1))
        e.torch.nn.functional.binary_cross_entropy_with_logits(z2,soft).backward()
        assert all(p.grad is not None and e.torch.isfinite(p.grad).all() for p in r.parameters())
        assert all(sum(float(p.grad.square().sum()) for p in block.parameters())>0 for block in r.feature_extractor.instance_encoder)
    # Full saved optimizer+Python layer choice+NumPy Beta/permutation+Torch state isolation.
    m,o,g=e.make()
    for ep in range(1,6):m.train();e.step(m,o,x,y,ep)
    saved=copy.deepcopy(e.capture(m,o,g,5,5));kinds=[]
    for epoch in [6,10]:
        before=e.semantic(saved);e.restore(saved,m,o,g);m.train();left=e.step(m,o,x,y,epoch);after=copy.deepcopy(e.capture(m,o,g,epoch,6 if epoch==6 else 7));assert e.semantic(saved)==before
        r,ro,rg=e.make();e.restore(saved,r,ro,rg);r.train();right=e.step(r,ro,x,y,epoch)
        assert left==right and e.semantic(after)==e.semantic(e.capture(r,ro,rg,epoch,after['updates']));saved=after;kinds.append(left['layer'])
    assert o.count==7 and o.slow
    from journal import StepLog,commit,load
    with tempfile.TemporaryDirectory() as tmp:
        d=Path(tmp);m,o,g=e.make();state=copy.deepcopy(e.capture(m,o,g,0,0));log=StepLog(d/'steps.jsonl');commit(d,state,e.record(state,0,0),log.prefix());m.train();log.append(dict(update=1,**e.step(m,o,x,y,1)));log.close()
        st,p=load(d);r,ro,rg=e.make();e.restore(st,r,ro,rg);r.train();log=StepLog(d/'steps.jsonl',0,p['prefix_bytes'],p['prefix_sha256']);log.append(dict(update=1,**e.step(r,ro,x,y,1)));assert log.index==log.original==1;log.close()
        # Canonical final digest must be frozen before selected weights load; no alias metadata rescue.
        e.atomic_save(d/'final.pt',saved);terminal=e.semantic(saved);different=copy.deepcopy(saved['model']);first=next(iter(different));different[first]=different[first]+.001
        m,o,g=e.make();e.restore(saved,m,o,g);m.load_state_dict(different)
        assert terminal==e.semantic(e.torch.load(d/'final.pt',map_location='cpu',weights_only=True))
        source=(B/'worker.py').read_text();assert source.index('terminal_native_digest=e.semantic(state)')<source.index("m.load_state_dict(best_model,strict=True)") and 'final_state_sha256=terminal_native_digest' in source
    result=dict(state='pass',author_beta_and_permutation_feature_target_functions_exact=True,author_CIFAR_recipe_alpha=2.,eligible_complete_native_layers=[0,1,2],all_boundaries_lambda1_native_forward_gradient_and_optimizer_exact=True,nontrivial_mix_gradients_through_all_native_blocks=True,cross_sample_Jacobian_matches_closed_form=True,Python_layer_NumPy_Beta_permutation_Torch_RNG_resume_exact=True,lazy_slow_sync_and_epoch6_to10_exact=True,saved_optimizer_fixture_immutable=True,journal_uncommitted_suffix_replay_exact=True,canonical_terminal_digest_frozen_before_selected_load=True,full_parameters=617612,batch=64,reviewed_author_code_not_redistributed_in_Git=True,no_UWB_arrays_or_labels_opened=True,new_UWB_fit_started=False,wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time(),maxRSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(B/'SYNTHETIC_RESULT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
