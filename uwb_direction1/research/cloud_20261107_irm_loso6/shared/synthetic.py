"""All-role fullnative IRM100/500 checks; no UWB input, target or actual ID values."""
from pathlib import Path
import ast,copy,tempfile,json,time
from common import B,PHASE,ready,write
import engine as e
t=e.torch

def pinned_defaults():
    registry=ast.parse((PHASE/'author_reference/domainbed/hparams_registry.py').read_text())
    calls={n.args[0].value:ast.literal_eval(n.args[1]) for n in ast.walk(registry) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_hparam' and len(n.args)>1 and isinstance(n.args[0],ast.Constant) and n.args[0].value in ['irm_lambda','irm_penalty_anneal_iters']}
    assert calls==dict(irm_lambda=100.,irm_penalty_anneal_iters=500)
    tree=ast.parse((PHASE/'author_reference/domainbed/algorithms.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='IRM')
    class ERM(t.nn.Module):
        def __init__(self,a,b,c,hparams):
            super().__init__();self.hparams=hparams;self.network=t.nn.Linear(12,12,bias=False)
            with t.no_grad():self.network.weight.copy_(t.eye(12))
            self.optimizer=t.optim.Adam(self.network.parameters(),lr=.001,weight_decay=.0001)
    class F:cross_entropy=staticmethod(e.native_bce)
    ns=dict(ERM=ERM,torch=t,F=F,autograd=t.autograd)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'pinned_MIT_DomainBed_IRM','exec'),ns)
    for count,weight in [(0,1.),(499,1.),(500,100.),(501,100.)]:
        obj=ns['IRM'](None,None,4,dict(irm_lambda=100.,irm_penalty_anneal_iters=500,lr=.001,weight_decay=.0001));obj.update_count.fill_(count);old=obj.optimizer
        x=t.randn(16,12);y=t.arange(16)%12;u=t.tensor(e.c['train_users']).repeat_interleave(4)
        actual=e.irm_objective(x.clone().requires_grad_(),y,u,weight)[0]
        r=obj.update([(x[u==v],y[u==v]) for v in e.c['train_users']])
        t.testing.assert_close(actual.detach(),t.tensor(r['loss']),atol=2e-6,rtol=2e-6)
        assert (obj.optimizer is not old)==(count==500) and obj.update_count.item()==count+1
    return True

def derivatives():
    z=t.linspace(-2,2,156,dtype=t.float64).reshape(13,12).requires_grad_();y=t.arange(13)%12
    u=t.tensor([e.c['train_users'][j] for j in [0]*4+[1]*3+[2]*5+[3]])
    loss,cl,p,d=e.irm_objective(z,y,u,100.);ref,rp=e.reference_objective(z,y,u,100.)
    t.testing.assert_close(p,rp,rtol=1e-12,atol=1e-14)
    a=t.autograd.grad(loss,z,retain_graph=True)[0];b=t.autograd.grad(ref,z)[0];t.testing.assert_close(a,b,rtol=1e-11,atol=1e-13)
    assert d['omitted_penalty_domains']==[e.c['train_users'][3]] and len(d['valid_training_domains'])==3
    assert t.equal(cl,e.native_bce(z,y))
    assert t.autograd.gradcheck(lambda q:e.irm_objective(q,y,u,100.)[0],(z,),fast_mode=True)
    zz=t.randn(2,12,requires_grad=True);yy=t.tensor([0,1]);uu=t.tensor(e.c['train_users'][:2]);ll,cl,pp,dd=e.irm_objective(zz,yy,uu,100.)
    assert pp.item()==0 and not dd['valid_training_domains'];assert t.equal(ll,cl);ll.backward();assert t.isfinite(zz.grad).all()
    return True

def actual_scalar_boundary():
    p=t.nn.Parameter(t.tensor([.3,-.8]));o=e.ScheduledNativeOptimizer([p])
    for j in range(501):
        assert o.penalty_weight()==(1. if j<500 else 100.)
        reset=o.reset_if_due();assert reset==(j==500)
        if reset:assert o.count==0 and not o.slow and not o.inner.fast.state
        o.zero_grad();((p-.2)**2).sum().backward();o.step()
        if j==499:
            saved=copy.deepcopy(o.state_dict());q=t.nn.Parameter(p.detach().clone());oo=e.ScheduledNativeOptimizer([q]);oo.load_state_dict(saved);assert oo.reset_if_due()
            postreset=copy.deepcopy(oo.state_dict());restored=e.ScheduledNativeOptimizer([q]);restored.load_state_dict(postreset);assert not restored.reset_if_due() and restored.count==0
    assert o.algorithm_updates==501 and o.count==1 and o.reset_events==[500]
    oo=e.ScheduledNativeOptimizer([p]);oo.load_state_dict(copy.deepcopy(o.state_dict()));assert not oo.reset_if_due()
    return True

def native_lifecycle():
    m,o,g=e.make();assert sum(p.numel() for p in m.parameters())==617612 and e.semantic(m.state_dict())==e.c['matched_initial_model_state_sha256']
    rng=e.np.random.default_rng(931);x=rng.normal(size=(8,90,567)).astype('float32');y=e.np.arange(8)%12;u=e.np.array([e.c['train_users'][j] for j in [0,0,0,1,1,2,2,3]])
    initial=copy.deepcopy(e.capture(m,o,g,0,0));m.train();z=m(t.from_numpy(x));ref,_=e.reference_objective(z,t.as_tensor(y),t.as_tensor(u),1.)
    o.zero_grad();ref.backward();t.nn.utils.clip_grad_norm_(m.parameters(),2.);grad=[p.grad.detach().clone() for p in m.parameters()];buffers={k:v.detach().clone() for k,v in m.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k};after_rng=t.get_rng_state().clone()
    e.restore(initial,m,o,g);m.train();r=e.step(m,o,x,y,1,u)
    for a,p in zip(grad,m.parameters()):t.testing.assert_close(a,p.grad,atol=2e-6,rtol=2e-4)
    assert all(t.equal(v,m.state_dict()[k]) for k,v in buffers.items()) and t.equal(after_rng,t.get_rng_state())
    assert r['completed_updates_after']==1 and r['irm_penalty_weight']==1. and not r['optimizer_reset_this_update'] and r['classification_loss']==r['global_BCE']
    e.restore(initial,m,o,g)
    for _ in range(6):m.train();e.step(m,o,x,y,1,u)
    assert o.count==6 and o.slow
    saved=copy.deepcopy(e.capture(m,o,g,1,6))
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'native.pt';e.atomic_save(p,saved);stored=t.load(p,map_location='cpu',weights_only=True)
        m.train();a=e.step(m,o,x,y,2,u);expected=copy.deepcopy(e.capture(m,o,g,1,7));e.restore(stored,m,o,g);m.train();b=e.step(m,o,x,y,2,u)
        assert a==b and e.semantic(e.capture(m,o,g,1,7))==e.semantic(expected)
    # The following500 fixture is not500 actual native updates; scalar test above checks actual501.
    o.algorithm_updates=500;o.inner.count=500;saved=copy.deepcopy(e.capture(m,o,g,1,500));m.train();a=e.step(m,o,x,y,2,u);expected=copy.deepcopy(e.capture(m,o,g,1,501))
    e.restore(saved,m,o,g);m.train();b=e.step(m,o,x,y,2,u);assert a==b and a['optimizer_reset_this_update'] and o.count==1 and e.semantic(e.capture(m,o,g,1,501))==e.semantic(expected)
    m.eval();before=e.semantic(e.capture(m,o,g,1,501))
    with t.inference_mode():assert m(t.from_numpy(x)).shape==(8,12)
    assert before==e.semantic(e.capture(m,o,g,1,501))
    n=e.c['counts'][0]%64;assert n in [20,28,56];tail=rng.normal(size=(n,90,567)).astype('float32');m.train();r=e.step(m,o,tail,e.np.arange(n)%12,2,e.np.array(e.c['train_users'])[e.np.arange(n)%4]);assert r['forward_batch']==n
    return True

def main():
    ready();start=time.monotonic();checks={}
    for name,fn in [('pinned_default_class_schedule_and_reset',pinned_defaults),('independent_dummy_scale_autograd_gradcheck_missing_singletons',derivatives),('actual_scalar501_and_boundary_restore',actual_scalar_boundary),('full617612_native_gradients_BN_RNG_optimizer_slow6_disk7_eval_tail',native_lifecycle)]:checks[name]=fn()
    write(B/'SYNTHETIC_RESULT.json',dict(state='pass',checks=checks,wall_seconds=time.monotonic()-start,no_UWB_arrays_or_labels_opened=True,new_UWB_fit_started=False,users7_8_labels_used=False))
    print(json.dumps(dict(state='pass',checks=checks,wall_seconds=time.monotonic()-start)))
if __name__=='__main__':main()
