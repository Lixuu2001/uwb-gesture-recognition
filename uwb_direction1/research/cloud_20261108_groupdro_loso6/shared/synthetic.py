"""Generated full-native engine/q/gradient/disk checks for each frozen LOSO role set."""
from pathlib import Path
import copy,tempfile,time,json
from common import B,PHASE,ready,write
import engine as e
t=e.torch
def derivatives():
    from groupdro_core import GroupDROState
    for k in [4,3,2,1]:
        z=t.linspace(-2,2,156,dtype=t.float64).reshape(13,12).requires_grad_();y=t.arange(13)%12;u=e.np.array(e.c['train_users'])[e.np.arange(13)%k]
        q=GroupDROState(e.c['train_users'],dtype=t.float64);a,cl,d=q.objective(z,y,u)
        grad,loss,refq,counts=e.reference_objective(z,y,u,e.c['train_users'],e.np.ones(4))
        t.testing.assert_close(t.autograd.grad(a,z)[0],grad,rtol=1e-11,atol=1e-13);t.testing.assert_close(q.q,refq,rtol=1e-12,atol=1e-14)
        assert abs(float(a.detach())-loss)<1e-13 and counts==d['domain_counts']
        before=e.semantic(q.state_dict())
        try:q.objective(z,y,e.np.full(13,e.c['held']))
        except AssertionError:pass
        else:raise AssertionError('held domain accepted')
        assert e.semantic(q.state_dict())==before
        if k==1:t.testing.assert_close(a,cl,rtol=1e-14,atol=1e-14)
    return True
def native_lifecycle():
    m,o,g=e.make();assert e.semantic(m.state_dict())==e.c['matched_initial_model_state_sha256'] and o.robust.domains==tuple(e.c['train_users'])
    assert o.native.count==o.robust.updates==0 and not o.native.slow and not o.native.fast.state
    rng=e.np.random.default_rng(931);x=rng.normal(size=(8,90,567)).astype('float32');y=e.np.arange(8)%12;u=e.np.array(e.c['train_users'])[[0,0,0,1,1,2,2,3]]
    initial=copy.deepcopy(e.capture(m,o,g,0,0));m.train();z=m(t.from_numpy(x));grad,loss,q,counts=e.reference_objective(z,y,u,e.c['train_users'],o.robust.q.numpy())
    grads=t.autograd.grad(z,tuple(m.parameters()),grad_outputs=grad)
    for p,a in zip(m.parameters(),grads):p.grad=a
    t.nn.utils.clip_grad_norm_(m.parameters(),2.);expectedgrads=[p.grad.detach().clone() for p in m.parameters()]
    buffers={k:v.clone() for k,v in m.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k};afterrng=t.get_rng_state().clone()
    e.restore(initial,m,o,g);m.train();r=e.step(m,o,x,y,1,u)
    for a,p in zip(expectedgrads,m.parameters()):t.testing.assert_close(a,p.grad,rtol=2e-4,atol=2e-6)
    t.testing.assert_close(o.robust.q,q,rtol=0,atol=1e-7)
    assert r['domain_counts']==counts and abs(loss-r['loss'])<2e-6 and all(t.equal(v,m.state_dict()[k]) for k,v in buffers.items()) and t.equal(afterrng,t.get_rng_state())
    e.restore(initial,m,o,g)
    for j in range(7):m.train();e.step(m,o,x,y,1,u)
    assert o.native.count==o.robust.updates==7 and o.native.slow
    saved=copy.deepcopy(e.capture(m,o,g,1,7))
    with tempfile.TemporaryDirectory() as folder:
        p=Path(folder)/'native.pt';e.atomic_save(p,saved);stored=t.load(p,map_location='cpu',weights_only=True)
        m.train();a=e.step(m,o,x,y,2,u);expected=copy.deepcopy(e.capture(m,o,g,1,8));e.restore(stored,m,o,g);m.train();b=e.step(m,o,x,y,2,u)
        assert a==b and e.semantic(e.capture(m,o,g,1,8))==e.semantic(expected)
    m.eval();before=e.semantic(e.capture(m,o,g,1,8))
    with t.inference_mode():assert m(t.from_numpy(x)).shape==(8,12)
    assert before==e.semantic(e.capture(m,o,g,1,8))
    n=e.c['counts'][0]%64;assert n in [20,28,56];tail=rng.normal(size=(n,90,567)).astype('float32');m.train()
    r=e.step(m,o,tail,e.np.arange(n)%12,2,e.np.array(e.c['train_users'])[e.np.arange(n)%3]);assert r['forward_batch']==n and sum(r['present'])==3 and o.robust.updates==9
    return True
def main():
    ready();start=time.monotonic();checks=dict(independent_float64_NumPy_q_BCE_logit_derivative_4_3_2_1groups_and_held_rejection=derivatives(),full617612_native_gradients_BN_RNG_slow6_disk7_exact8_eval_and_missing_group_native_tail=native_lifecycle())
    write(B/'SYNTHETIC_RESULT.json',dict(state='pass',checks=checks,wall_seconds=time.monotonic()-start,no_UWB_arrays_or_labels_opened=True,new_UWB_fit_started=False,users7_8_labels_used=False));print(json.dumps(dict(state='pass',checks=checks)))
if __name__=='__main__':main()
