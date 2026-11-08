"""Generated-only equations, detached fullnative gradients, q/RNG/disk/slow restore and fixed cost profile."""
from pathlib import Path
import sys,json,copy,random,time,tempfile,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from groupdro_core import GroupDROState,NativeGroupDROOptimizer
P=Path(__file__).resolve().parent;sys.path.insert(0,'/workspace/uwb-timemil-20261029');import adapter as native
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
DOM=(3,4,5,6);SEED=20261029
def semantic(x):
    h=hashlib.sha256()
    def walk(v):
        if isinstance(v,torch.Tensor):h.update(str((tuple(v.shape),str(v.dtype))).encode());h.update(v.detach().contiguous().numpy().tobytes())
        elif isinstance(v,dict):
            for k,z in v.items():h.update(str(k).encode());walk(z)
        elif isinstance(v,(list,tuple)):
            for z in v:walk(z)
        else:h.update(str(v).encode())
    walk(x);return h.hexdigest()
def factory(ps):return native.StableLookahead(ps,lr=.001,weight_decay=.0001)
def make(robust=True):
    random.seed(SEED);np.random.seed(SEED);m=native.make_model('gap',SEED);assert sum(p.numel() for p in m.parameters())==617612
    return m,NativeGroupDROOptimizer(m.parameters(),DOM,factory) if robust else factory(list(m.parameters())),torch.Generator().manual_seed(SEED+77)
def batch(g,n=64):
    return torch.randn(n,90,567,generator=g),torch.randint(0,12,(n,),generator=g),torch.as_tensor(DOM)[torch.randint(0,4,(n,),generator=g)]
def step(m,o,x,y,d,robust=True):
    m.train();z=m(x);loss=o.robust.objective(z,y,d)[0] if robust else F.binary_cross_entropy_with_logits(z,F.one_hot(y,12).float());o.zero_grad();loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());torch.nn.utils.clip_grad_norm_(m.parameters(),2.);o.step();return float(loss.detach())
def capture(m,o,g):
    n=np.random.get_state();return copy.deepcopy(dict(model=m.state_dict(),optimizer=o.state_dict(),order_rng=g.get_state(),torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=(n[0],torch.from_numpy(n[1].copy()),n[2],n[3],n[4]),training=m.training))
def restore(s,m,o,g):
    m.load_state_dict(s['model']);o.load_state_dict(s['optimizer']);g.set_state(s['order_rng']);torch.set_rng_state(s['torch_rng']);random.setstate(s['python_rng']);n=s['numpy_rng'];np.random.set_state((n[0],n[1].numpy(),n[2],n[3],n[4]));m.train(s['training'])
def reference(z,y,d,q):
    a=z.detach().numpy().astype(np.float64);b=y.numpy();u=d.numpy();hot=np.eye(12)[b];per=(np.logaddexp(0,a)-hot*a).mean(1);counts=np.array([(u==k).sum() for k in DOM]);risk=np.array([per[u==k].mean() if n else 0. for k,n in zip(DOM,counts)]);present=counts>0;q=q.astype(np.float64).copy();q[present]*=np.exp(.01*risk[present]);q/=q.sum();weights=q.copy()
    if not present.all():weights[present]/=weights[present].sum()
    scale=np.array([weights[DOM.index(int(k))]/counts[DOM.index(int(k))] for k in u]);grad=(1/(1+np.exp(-a))-hot)/12*scale[:,None]
    return float(np.dot(weights[present],risk[present])),q,grad
def main():
    started=time.monotonic();cpu=time.process_time();torch.manual_seed(91);q=GroupDROState(DOM,dtype=torch.float64);max_equation=0.
    for n,present in [(64,4),(20,2),(28,3),(56,4)]:
        z=torch.randn(n,12,dtype=torch.float64,requires_grad=True);y=torch.arange(n)%12;d=torch.as_tensor(DOM)[torch.arange(n)%present];before=q.q.numpy().copy();r,rq,rg=reference(z,y,d,before);loss,base,_=q.objective(z,y,d);grad=torch.autograd.grad(loss,z)[0];max_equation=max(max_equation,float(abs(loss-r)));np.testing.assert_allclose(q.q.numpy(),rq,rtol=0,atol=2e-15);np.testing.assert_allclose(grad.numpy(),rg,rtol=2e-13,atol=2e-15)
    # Literal pinned source equation when all4 source groups are present.
    q=GroupDROState(DOM);z=torch.randn(64,12,requires_grad=True);y=torch.arange(64)%12;d=torch.as_tensor(DOM)[torch.arange(64)%4];r=F.binary_cross_entropy_with_logits(z,F.one_hot(y,12).float(),reduction='none').mean(1);risks=torch.stack([r[d==k].mean() for k in DOM]);qq=torch.ones(4)
    for j in range(4):qq[j]*=(.01*risks[j].detach()).exp()
    qq/=qq.sum();expected=torch.dot(qq,risks);actual,_,_=q.objective(z,y,d);torch.testing.assert_close(q.q,qq,rtol=1e-7,atol=1e-7);torch.testing.assert_close(actual,expected,rtol=1e-7,atol=1e-7)
    m,o,g=make();x,y,d=batch(g);before=o.robust.q.numpy().copy();z=m(x);ref_loss,refq,ref_grad=reference(z,y,d,before);loss,_,_=o.robust.objective(z,y,d);pars=tuple(m.parameters());a=torch.autograd.grad(loss,pars,retain_graph=True);b=torch.autograd.grad(z,pars,grad_outputs=torch.from_numpy(ref_grad).to(z.dtype));worst=0.
    for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,rtol=2e-4,atol=2e-6);worst=max(worst,float((aa-bb).abs().max()))
    # Fresh actual native state; slow step6 and serialized step7 -> exact step8.
    m,o,g=make()
    for _ in range(7):x,y,d=batch(g);step(m,o,x,y,d)
    assert o.native.count==o.robust.updates==7
    s=capture(m,o,g)
    with tempfile.TemporaryDirectory(prefix='groupdro_generated_only_',dir=str(P)) as td:
        f=Path(td)/'state.pt';torch.save(s,f);disk=torch.load(f,weights_only=True);assert semantic(s)==semantic(disk)
        x,y,d=batch(g);step(m,o,x,y,d);expected=capture(m,o,g)
        m2,o2,g2=make();restore(disk,m2,o2,g2);xx,yy,dd=batch(g2);assert torch.equal(x,xx) and torch.equal(y,yy) and torch.equal(d,dd);step(m2,o2,xx,yy,dd);assert semantic(expected)==semantic(capture(m2,o2,g2))
    # Predetermined eight interleaved full B64 steps per arm, discard first3 warmups.
    baseline,bo,bg=make(False);robust,ro,rg=make(True);data_g=torch.Generator().manual_seed(SEED+8);timings={'native':[],'GroupDRO':[]}
    for i in range(8):
        x,y,d=batch(data_g)
        for arm in (['native','GroupDRO'] if i%2==0 else ['GroupDRO','native']):
            model,opt=(baseline,bo) if arm=='native' else (robust,ro);w=time.monotonic();c=time.process_time();step(model,opt,x,y,d,arm=='GroupDRO');timings[arm].append(dict(wall=time.monotonic()-w,CPU=time.process_time()-c))
    ratio=max(1.,*(max(r[k] for r in timings['GroupDRO'][3:])/max(r[k] for r in timings['native'][3:]) for k in ['wall','CPU']));reserve=1.25*31189.589142517*ratio+600
    result=dict(state='generated_core_fullnative_independent_q_gradient_and_complete_state_restore_passed',real_UWB_values_or_gesture_labels_or_IDs_read=False,new_real_fit_started=False,parameters=617612,shape=[64,90,567],default_eta=.01,allpresent_literal_pinned_equation_passed=True,missing_group_cases=[2,3,4],max_float64_equation_abs_difference=max_equation,full617612_parameter_reference_gradients_passed=True,max_native_gradient_abs_difference=worst,native_AdamWLookahead_slow6_disk7_exact8_q_allRNG_mode_restored=True,profile_protocol_fixed_before_measure=True,timings=timings,conservative_ratio=ratio,projected_full6_train_replay_reserve_seconds=reserve,capacity_gate_passed=reserve<=43200,remaining_complete_loader_worker_supervisor_role_all6_audits_needed=True,complete_runner_implemented=False,wall=time.monotonic()-started,CPU=time.process_time()-cpu)
    (P/'GENERATED_CORE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
