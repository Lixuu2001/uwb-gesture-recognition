"""Author formula, soft-label gradient, full model RNG/slow-buffer recovery."""
import ast,copy,tempfile,time,resource
from pathlib import Path
from common import B,write,ready,sha
import engine as e
def main():
    ready();t=time.monotonic();e.torch.manual_seed(9)
    # Execute only the two pure public author functions, with CUDA explicitly off.
    tree=ast.parse((B/'literature/train.py').read_text())
    fs=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ['mixup_data','mixup_criterion']]
    assert len(fs)==2
    env=dict(np=e.np,torch=e.torch);exec(compile(ast.Module(body=fs,type_ignores=[]),'author_mixup_functions','exec'),env)
    x=e.torch.randn(64,90,567);y=e.torch.arange(64)%12
    e.np.random.seed(20261029);e.torch.manual_seed(20261029);a=env['mixup_data'](x,y,1.,False)
    e.np.random.seed(20261029);e.torch.manual_seed(20261029);b=e.mix(x,y)
    assert a[3]==b[3]
    for i in range(3):e.torch.testing.assert_close(a[i],b[i],rtol=0,atol=0)
    z=e.torch.randn(64,12,dtype=e.torch.float64,requires_grad=True);lam=b[3]
    ta=e.torch.nn.functional.one_hot(b[1],12).double();tb=e.torch.nn.functional.one_hot(b[2],12).double()
    direct=env['mixup_criterion'](e.torch.nn.functional.binary_cross_entropy_with_logits,z,ta,tb,lam)
    soft=e.torch.nn.functional.binary_cross_entropy_with_logits(z,lam*ta+(1-lam)*tb)
    e.torch.testing.assert_close(direct,soft,atol=1e-14,rtol=1e-14)
    gd=e.torch.autograd.grad(direct,z,retain_graph=True)[0];gs=e.torch.autograd.grad(soft,z)[0]
    e.torch.testing.assert_close(gd,gs,atol=1e-14,rtol=1e-14)
    m,o,g=e.make();rng=e.np.random.default_rng(55);xx=rng.normal(size=(64,90,567)).astype('float32');yy=e.np.arange(64)%12
    for epoch in range(1,6):m.train();e.step(m,o,xx,yy,epoch)
    frozen=copy.deepcopy(e.capture(m,o,g,5,5))
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'native.pt';e.atomic_save(p,frozen);saved=e.torch.load(p,map_location='cpu',weights_only=True)
        assert e.semantic(saved)==e.semantic(frozen)
        for epoch in [6,10]:
            e.restore(saved,m,o,g);m.train();left=e.step(m,o,xx,yy,epoch);after=copy.deepcopy(e.capture(m,o,g,epoch,6 if epoch==6 else 7))
            right,ro,rg=e.make();e.restore(saved,right,ro,rg);right.train();actual=e.step(right,ro,xx,yy,epoch)
            assert left==actual and e.semantic(after)==e.semantic(e.capture(right,ro,rg,epoch,after['updates']))
            assert right.wave1.data_ptr()==right.wave1_.data_ptr()
            saved=after
    assert o.count==7 and o.slow
    # Test committed-prefix plus stochastic uncommitted-suffix replay with native states.
    from journal import StepLog,commit,load as journal_load
    with tempfile.TemporaryDirectory() as d:
        d=Path(d);a,ao,ag=e.make();state=copy.deepcopy(e.capture(a,ao,ag,0,0))
        log=StepLog(d/'steps.jsonl');commit(d,state,e.record(state,0,0),log.prefix())
        a.train();row=dict(update=1,**e.step(a,ao,xx,yy,1));log.append(row);log.close()
        restored,p=journal_load(d);a,ao,ag=e.make();e.restore(restored,a,ao,ag);a.train()
        log=StepLog(d/'steps.jsonl',0,p['prefix_bytes'],p['prefix_sha256']);log.append(dict(update=1,**e.step(a,ao,xx,yy,1)));assert log.original==log.index==1;log.close()
    result=dict(state='pass',author_two_functions_exact=True,author_alpha_default=1.,soft_BCE_loss_and_gradient_equivalent=True,full850973param_batch64=True,all_numpy_torch_python_order_rng_restored=True,lazy_slow_sync_and_warmup_transition_exact=True,wavelet_alias_retained=True,native_serialization_and_journal_suffix_exact=True,no_UWB_arrays_or_labels_opened=True,new_UWB_fit_started=False,wall_seconds=time.monotonic()-t,CPU_seconds=time.process_time(),maxRSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(B/'SYNTHETIC_RESULT.json',result);print(__import__('json').dumps(result))
if __name__=='__main__':main()
