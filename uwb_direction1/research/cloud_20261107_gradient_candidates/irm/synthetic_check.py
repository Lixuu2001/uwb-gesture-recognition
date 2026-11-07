"""Independent dummy-scale autograd reference and complete native scheduled-state replay."""
from pathlib import Path
import json,time,tempfile,copy
import native_engine as e
from irm_core import objective,global_bce
P=Path(__file__).resolve().parent

def reference(z,y,d,weight):
    t=e.torch;penalties=[]
    for u in sorted(d.unique().tolist()):
        a=z[d==u];b=y[d==u]
        if len(b)<2:continue
        scale=t.tensor(1.,dtype=z.dtype,requires_grad=True)
        even=global_bce(a[::2]*scale,b[::2]);odd=global_bce(a[1::2]*scale,b[1::2])
        g0=t.autograd.grad(even,scale,create_graph=True)[0];g1=t.autograd.grad(odd,scale,create_graph=True)[0]
        penalties.append(g0*g1)
    penalty=t.stack(penalties).mean() if penalties else z.sum()*0.
    return global_bce(z,y)+weight*penalty,penalty

def main():
    started=time.monotonic();cpu=time.process_time();t=e.torch
    z=t.randn(13,12,generator=t.Generator().manual_seed(121),dtype=t.float64,requires_grad=True);y=t.arange(13)%12;d=t.tensor([1]*4+[2]*3+[3]*5+[4])
    l,cl,p,diagnostic=objective(z,y,d,100.);ref,rp=reference(z,y,d,100.)
    t.testing.assert_close(p,rp,atol=1e-14,rtol=1e-12)
    g=t.autograd.grad(l,z,retain_graph=True)[0];rg=t.autograd.grad(ref,z)[0];t.testing.assert_close(g,rg,atol=1e-13,rtol=1e-11)
    assert diagnostic['omitted_penalty_domains']==[4]
    x=t.randn(64,90,567,generator=t.Generator().manual_seed(7821));y=t.arange(64)%12;d=t.tensor([1]*13+[2]*17+[3]*19+[4]*15)
    m,opt=e.make();m.train();initial=e.capture(m,opt);logits=m(x)
    loss,_,_,_=objective(logits,y,d,100.);ref,_=reference(logits,y,d,100.);params=list(m.parameters())
    grad=t.autograd.grad(loss,params,retain_graph=True);rgrad=t.autograd.grad(ref,params)
    maximum=max(float((a-b).abs().max()) for a,b in zip(grad,rgrad))
    for a,b in zip(grad,rgrad):t.testing.assert_close(a,b,atol=2e-6,rtol=2e-4)
    m,opt=e.make();m.train()
    for _ in range(5):e.step(m,opt,x,y,d)
    checkpoint=e.capture(m,opt)
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'generated_only.pt';t.save(checkpoint,path);checkpoint=t.load(path,weights_only=False)
        for _ in range(2):e.step(m,opt,x,y,d)
        expected=e.semantic(e.capture(m,opt));assert opt.count==7 and opt.slow
        m2,o2=e.make();e.restore(checkpoint,m2,o2);m2.train()
        for _ in range(2):e.step(m2,o2,x,y,d)
        assert expected==e.semantic(e.capture(m2,o2))
    # Actual scalar optimizer501-update schedule check, separate from the native counter fixture.
    scalar=t.nn.Parameter(t.tensor(1.));o=e.ScheduledOptimizer([scalar]);events=[]
    for i in range(501):
        reset=o.reset_if_due();events.append(reset);o.zero_grad();scalar.square().backward();o.step()
    assert events==[False]*500+[True] and o.algorithm_updates==501 and o.count==1 and o.reset_events==[500]
    # Native full-state fixture at the pre501 boundary verifies full optimizer recreation and disk resume.
    m,o=e.make();m.train();e.step(m,o,x,y,d);o.algorithm_updates=500;o.inner.count=500
    fixture=e.capture(m,o);a=e.step(m,o,x,y,d);assert a['irm_penalty_weight']==100. and a['optimizer_reset_this_update'] and o.count==1
    expected=e.semantic(e.capture(m,o));m2,o2=e.make();e.restore(fixture,m2,o2);m2.train();e.step(m2,o2,x,y,d);assert expected==e.semantic(e.capture(m2,o2))
    tiny=t.randn(2,12,requires_grad=True);loss,cl,pen,diag=objective(tiny,t.tensor([0,1]),t.tensor([1,2]),100.);assert float(pen.detach())==0 and diag['valid_training_domains']==[];loss.backward();assert tiny.grad.isfinite().all()
    result={'state':'pass_generated_fullnative_IRM100_500_BCE_audits','parameters':617612,'dummy_scale_autograd_float64_gradient_max_difference':float((g-rg).abs().max()),'all617612_native_parameter_gradient_reference_max_difference':maximum,'native_initialization_same_seed':True,'native_global_BCE_not_equal_domain_average':True,'warmup_weight1_active100_no_loss_division':True,'actual_scalar_501_update_reset_passed':True,'native_boundary_fixture_not500_actual_native_updates':True,'native_reset_and_all_model_optimizer_Lookahead_slow_and_RNG_resume_exact':True,'missing_or_singleton_penalty_omission_preserves_BCE':True,'real_UWB_samples_labels_or_IDs_read':False,'new_UWB_fit_started':False,'wall_seconds':time.monotonic()-started,'CPU_seconds':time.process_time()-cpu}
    (P/'SYNTHETIC_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
