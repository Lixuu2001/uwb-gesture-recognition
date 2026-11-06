"""Synthetic-only VREx equation, default schedule, native gradients and recovery checks."""
from pathlib import Path
import ast,copy,json,tempfile,time
from common import B,ready,write,sha
import engine as e
t=e.torch

def defaults_and_pinned():
    registry=ast.parse((B/'author_reference/domainbed/hparams_registry.py').read_text())
    calls={n.args[0].value:ast.literal_eval(n.args[1]) for n in ast.walk(registry) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_hparam' and len(n.args)>1 and isinstance(n.args[0],ast.Constant) and n.args[0].value in ['vrex_lambda','vrex_penalty_anneal_iters']}
    assert calls==dict(vrex_lambda=10.,vrex_penalty_anneal_iters=500)
    tree=ast.parse((B/'author_reference/domainbed/algorithms.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VREx')
    # Execute the actual pinned MIT class; substitute only its CE slot with native BCE.
    class ERM(t.nn.Module):
        def __init__(self,a,b,c,hparams):
            super().__init__();self.hparams=hparams;self.network=t.nn.Linear(12,12,bias=False)
            with t.no_grad():self.network.weight.copy_(t.eye(12))
            self.optimizer=t.optim.Adam(self.network.parameters(),lr=.001,weight_decay=.0001)
    class F:
        cross_entropy=staticmethod(e.native_bce)
    ns=dict(ERM=ERM,torch=t,F=F)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'pinned_MIT_DomainBed_VREx','exec'),ns)
    for count,weight in [(0,1.),(499,1.),(500,10.),(501,10.)]:
        obj=ns['VREx'](None,None,4,dict(vrex_lambda=10.,vrex_penalty_anneal_iters=500,lr=.001,weight_decay=.0001))
        obj.update_count.fill_(count);old=obj.optimizer
        x=t.randn(11,12);y=t.arange(11)%12;u=t.tensor([3]*4+[4]*3+[5]*3+[6])
        pieces=[(x[u==v],y[u==v]) for v in [3,4,5,6]]
        z=x.detach().clone().requires_grad_();loss,mean,var,_=e.vrex_objective(z,y,u,weight)
        result=obj.update(pieces)
        t.testing.assert_close(loss.detach(),t.tensor(result['loss']),rtol=1e-6,atol=1e-7)
        assert (obj.optimizer is not old)==(count==500) and obj.update_count.item()==count+1
    return True

def independent_derivatives():
    z=t.linspace(-2,2,60,dtype=t.float64).reshape(5,12).requires_grad_();y=t.tensor([0,2,4,6,8]);u=t.tensor([3,3,4,5,6]);beta=10.
    loss,mean,var,risks=e.vrex_objective(z,y,u,beta)
    # Independent analytic gradient for equally weighted domain BCE risks.
    target=t.nn.functional.one_hot(y,12).to(t.float64);expected=t.zeros_like(z)
    detached=risks.detach();mu=detached.mean();domains=[3,4,5,6]
    for k,v in enumerate(domains):
        mask=u==v;n=int(mask.sum());factor=(1+2*beta*(detached[k]-mu))/len(domains)
        expected[mask]=factor*(z.detach()[mask].sigmoid()-target[mask])/(n*12)
    actual=t.autograd.grad(loss,z)[0];t.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-13)
    assert t.autograd.gradcheck(lambda a:e.vrex_objective(a,y,u,beta)[0],(z,),fast_mode=True)
    for dom in [t.full((5,),3),t.tensor([3,3,3,3,4])]:
        _,m,p,r=e.vrex_objective(z,y,dom,1.)
        assert len(r)==len(t.unique(dom)) and t.isfinite(p)
        if len(r)==1:assert p.item()==0 and t.autograd.grad(p,z,retain_graph=True)[0].abs().sum()==0
    # Domainmean is intentionally different from sampleweighted BCE on unequal groups.
    assert not t.isclose(mean,e.native_bce(z,y),rtol=0,atol=1e-8)
    return True

def small_real_boundary():
    p=t.nn.Parameter(t.tensor([.3,-.8]));o=e.ScheduledNativeOptimizer([p])
    for j in range(501):
        w=o.penalty_weight();reset=o.reset_if_due()
        assert w==(1. if j<500 else 10.) and reset==(j==500)
        if j==500:assert o.count==0 and not o.slow and not o.inner.fast.state
        o.zero_grad();((p-.2)**2).sum().backward();o.step()
        if j==499:
            saved=copy.deepcopy(o.state_dict());q=t.nn.Parameter(p.detach().clone());oo=e.ScheduledNativeOptimizer([q]);oo.load_state_dict(saved)
            assert e.semantic(oo.state_dict())==e.semantic(saved)
            assert oo.reset_if_due() and oo.algorithm_updates==500 and oo.count==0
    assert o.algorithm_updates==501 and o.reset_events==[500] and o.count==1
    saved=copy.deepcopy(o.state_dict());q=t.nn.Parameter(p.detach().clone());oo=e.ScheduledNativeOptimizer([q]);oo.load_state_dict(saved)
    assert not oo.reset_if_due() and oo.reset_events==[500]
    return True

def native_comparison():
    m,o,g=e.make();assert sum(p.numel() for p in m.parameters())==617612
    assert e.semantic(m.state_dict())==e.c['matched_initial_model_state_sha256']
    rng=e.np.random.default_rng(931);x=rng.normal(size=(8,90,567)).astype('float32');y=e.np.arange(8)%12;u=e.np.array([3,3,3,4,4,5,5,6])
    initial=copy.deepcopy(e.capture(m,o,g,0,0));m.train()
    # Independent native graph and explicit source risks; compare every gradient and BN.
    z=m(t.from_numpy(x));risks=t.zeros(4)
    labels=t.as_tensor(y,dtype=t.long)
    for k,v in enumerate([3,4,5,6]):risks[k]=e.native_bce(z[t.as_tensor(u==v)],labels[t.as_tensor(u==v)])
    mean=risks.mean();loss=mean+((risks-mean)**2).mean();o.zero_grad();loss.backward()
    t.nn.utils.clip_grad_norm_(m.parameters(),2.);grads=[p.grad.detach().clone() for p in m.parameters()];o.step()
    expected=copy.deepcopy(e.capture(m,o,g,0,1))
    e.restore(initial,m,o,g);m.train();r=e.step(m,o,x,y,1,u)
    assert r['completed_updates_after']==1 and r['vrex_penalty_weight']==1. and not r['optimizer_reset_this_update']
    for a,b in zip(grads,[p.grad for p in m.parameters()]):assert t.equal(a,b),'full native clipped gradient mismatch'
    assert e.semantic(e.capture(m,o,g,0,1))==e.semantic(expected),'native model/BN/optimizer/allRNG mismatch'
    # Native single-domain control has zero variance and exactly the original native BCE update.
    e.restore(initial,m,o,g);m.train();z=m(t.from_numpy(x));ref_loss=e.native_bce(z,labels);o.zero_grad();ref_loss.backward();t.nn.utils.clip_grad_norm_(m.parameters(),2.);o.step();reference=copy.deepcopy(e.capture(m,o,g,0,1))
    e.restore(initial,m,o,g);m.train();r=e.step(m,o,x,y,1,e.np.full(8,3));assert r['penalty']==0
    assert e.semantic(e.capture(m,o,g,0,1))==e.semantic(reference)
    # Slow-step6 and disk-step7 using complete native backbone and the real wrapper.
    e.restore(initial,m,o,g)
    for j in range(6):m.train();e.step(m,o,x,y,1,u)
    assert o.count==6 and o.slow and o.algorithm_updates==6
    saved=copy.deepcopy(e.capture(m,o,g,1,6))
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'native.pt';e.atomic_save(path,saved);stored=t.load(path,map_location='cpu',weights_only=True)
        m.train();a=e.step(m,o,x,y,2,u);expected=copy.deepcopy(e.capture(m,o,g,1,7))
        e.restore(stored,m,o,g);m.train();b=e.step(m,o,x,y,2,u)
        assert a==b and e.semantic(e.capture(m,o,g,1,7))==e.semantic(expected)
    m.eval();before=e.semantic(e.capture(m,o,g,1,7))
    with t.inference_mode():z=m(t.from_numpy(x))
    assert e.semantic(e.capture(m,o,g,1,7))==before and z.shape==(8,12)
    # A real native tail28 graph reaches all parameters at the frozen input dimensions.
    tail=rng.normal(size=(28,90,567)).astype('float32');m.train();e.step(m,o,tail,e.np.arange(28)%12,2,e.np.arange(28)%4+3)
    assert all(p.grad is not None and t.isfinite(p.grad).all() for p in m.parameters())
    return True

def main():
    ready();start=time.monotonic();checks={}
    for name,fn in [('pinned_class_defaults_schedule_and_Adam_reset',defaults_and_pinned),('independent_analytic_gradient_gradcheck_singletons_domainmean',independent_derivatives),('actual_small_optimizer_501_updates_restore_500_boundary',small_real_boundary),('full617612_native_BN_optimizer_allRNG_slow6_disk7_eval_tail28',native_comparison)]:checks[name]=fn()
    write(B/'SYNTHETIC_RESULT.json',dict(state='pass',checks=checks,wall_seconds=time.monotonic()-start,no_UWB_arrays_or_labels_opened=True,new_UWB_fit_started=False,users7_8_labels_used=False))
    print(json.dumps(dict(state='pass',checks=checks,wall_seconds=time.monotonic()-start)))
if __name__=='__main__':main()
