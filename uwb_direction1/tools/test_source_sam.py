"""Independent analytic SAM gradient, stochastic semantics, resume and held isolation."""
import contextlib,copy,io,json,tempfile
from pathlib import Path
import numpy as np
import torch
from source_sam import source_step
from source_sam_pilot import fit,arm_config
from swad_source_pilot import fit as original_fit
from audit_source_sam import audit_run
from test_temporal_scale import same


def numeric_checks():
    # Independent NumPy softmax CE gradient and normalized perturbation.
    x=np.array([[1.,-.4],[-.3,.8],[.5,.2]]);y=np.array([0,2,1]);w=np.array([[.1,.4],[-.5,.2],[.3,-.2]])
    def grad(w):
        z=x@w.T;pr=np.exp(z-z.max(1,keepdims=True));pr/=pr.sum(1,keepdims=True)
        pr[np.arange(3),y]-=1
        return pr.T@x/3
    g=grad(w);rho=.05;lr=.013;pert=w+rho*g/np.linalg.norm(g);expected=w-lr*grad(pert)
    net=torch.nn.Linear(2,3,bias=False).double();net.weight.data.copy_(torch.from_numpy(w));opt=torch.optim.SGD(net.parameters(),lr=lr)
    source_step(net,opt,torch.from_numpy(x),torch.from_numpy(y),rho)
    np.testing.assert_allclose(net.weight.detach().numpy(),expected,rtol=1e-12,atol=1e-12)
    # Dropout mask alignment and single logical RNG advancement.
    torch.manual_seed(87);net=torch.nn.Sequential(torch.nn.Dropout(.3),torch.nn.Linear(2,3)).double()
    snapshots=[];net[0].register_forward_hook(lambda m,i,o:snapshots.append(o.detach().clone()))
    opt=torch.optim.AdamW(net.parameters(),lr=.001);before=torch.get_rng_state()
    source_step(net,opt,torch.from_numpy(x),torch.from_numpy(y),rho);after=torch.get_rng_state()
    assert len(snapshots)==2;assert torch.equal(*snapshots)
    torch.set_rng_state(before);net[0](torch.from_numpy(x));assert torch.equal(after,torch.get_rng_state())
    # Rejected second pass must not leave perturbed weights or change optimizer.
    saved=copy.deepcopy(net.state_dict());ostate=copy.deepcopy(opt.state_dict())
    calls=[0]
    def fail(m,i,o):
        calls[0]+=1
        if calls[0]==2:raise RuntimeError('intentional second-pass failure')
    handle=net.register_forward_hook(fail)
    try:source_step(net,opt,torch.from_numpy(x),torch.from_numpy(y),rho)
    except RuntimeError as e:assert 'intentional' in str(e)
    else:raise AssertionError('expected injected exception')
    handle.remove();same(saved,net.state_dict());same(ostate,opt.state_dict())
    # Zero gradient is finite; plain rho0 matches one-pass AdamW including dropout.
    zero=torch.nn.Linear(2,3,bias=False).double();o=torch.optim.SGD(zero.parameters(),lr=.1);zstate=copy.deepcopy(zero.state_dict())
    source_step(zero,o,torch.zeros(3,2,dtype=torch.float64),torch.from_numpy(y),rho);same(zstate,zero.state_dict())
    a=copy.deepcopy(net);b=copy.deepcopy(net);a._forward_hooks.clear();b._forward_hooks.clear()
    oa=torch.optim.AdamW(a.parameters(),lr=.001);ob=torch.optim.AdamW(b.parameters(),lr=.001)
    torch.manual_seed(423);source_step(a,oa,torch.from_numpy(x),torch.from_numpy(y),0);rng=torch.get_rng_state()
    torch.manual_seed(423);ob.zero_grad();torch.nn.functional.cross_entropy(b(torch.from_numpy(x)),torch.from_numpy(y)).backward();ob.step()
    same(a.state_dict(),b.state_dict());same(oa.state_dict(),ob.state_dict());same(rng,torch.get_rng_state())


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);numeric_checks()
    rng=np.random.default_rng(998);users=np.repeat(np.arange(3,9),3);y=np.tile(np.arange(3),6)
    cache=rng.normal(size=(18,3,6,90,63)).astype(np.float16)
    c={'seed':2026,'batch':8,'epochs':3,'lr':.001,'weight_decay':.001,'min_lr':1e-5,'tail_start_epoch':2,'rho':.05,'n_converge':3,'n_tolerance':6,'tolerance_ratio':.3}
    with tempfile.TemporaryDirectory(prefix='uwb-sam-test-') as td,contextlib.redirect_stdout(io.StringIO()):
        root=Path(td)
        original_fit(root/'old',c,{},cache,y,users,3,stop_after=2)
        fit(root/'plain',c,{},cache,y,users,3,'shared128',stop_after=2)
        a=torch.load(root/'old/progress.pt',weights_only=False);b=torch.load(root/'plain/progress.pt',weights_only=False)
        for k in ['model','optimizer','scheduler','rng','tail','best','best_ce','best_epoch','step']:same(a[k],b[k])
        for arm in ['sam24','erm48']:
            full=root/(arm+'full');res=root/(arm+'resume');iso=root/(arm+'isolate')
            fit(full,c,{},cache,y,users,3,arm)
            fit(res,c,{},cache,y,users,3,arm,stop_after=2);torch.rand(713);fit(res,c,{},cache,y,users,3,arm)
            a=torch.load(full/'progress.pt',weights_only=False);b=torch.load(res/'progress.pt',weights_only=False)
            for k in ['model','optimizer','scheduler','rng','tail','best','best_ce','best_epoch','step']:same(a[k],b[k])
            same(torch.load(full/'selected.pt',weights_only=False),torch.load(res/'selected.pt',weights_only=False))
            changed=cache.copy();changed[users==3]*=2;labels=y.copy();labels[users==3]=(labels[users==3]+5)%12
            fit(iso,c,{},changed,labels,users,3,arm)
            same(torch.load(full/'selected.pt',weights_only=False),torch.load(iso/'selected.pt',weights_only=False))
            audit_run(full,cache,y,users,arm_config(c,arm))
    print(json.dumps({'synthetic_only':True,'numpy_analytic_softmax_sam_gradient_verified':True,'same_dropout_mask_rng_once':True,'second_pass_exception_restores_parameters_and_preserves_optimizer':True,'zero_gradient_finite':True,'rho0_equals_plain_adamw_including_rng':True,'old_trainer_plain_trajectory_bitwise_equal':True,'both_arms_resume_model_adamw_scheduler_rng_tail_selection_exact':True,'held_inputs_and_labels_do_not_affect_selected_weights':True,'both_arms_independent_inference_roles_order_selection_audit':True},indent=2))

if __name__=='__main__':main()
