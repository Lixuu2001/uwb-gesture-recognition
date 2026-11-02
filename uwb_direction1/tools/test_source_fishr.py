"""Higher derivatives, EMA normalization, stochastic semantics, resume and isolation."""
import contextlib,copy,io,json,tempfile
from pathlib import Path
import numpy as np
import torch
from source_fishr import VarianceEMA,source_step,balanced_order,permuted_domains
from source_fishr_pilot import fit
from audit_source_fishr import audit_run
from check_linear_gradient_variance import run as analytic_checks
from circular_signal import SignalNet
from test_temporal_scale import same


def numeric_checks():
    analytic_checks()
    ema=VarianceEMA(.95);prev=np.zeros((4,7));rng=np.random.default_rng(21)
    for step in range(1,4):
        x=rng.uniform(size=(4,7));t=torch.tensor(x,requires_grad=True)
        out=ema.update(t,2*t)['real'];prev=.95*prev+.05*x;smooth=prev/(1-.95**step);center=smooth-smooth.mean(0)
        np.testing.assert_allclose(out.detach(),np.square(center).mean(),rtol=1e-12,atol=1e-12)
        grad=torch.autograd.grad(out,t)[0].numpy();expected=2*center/center.size*.05/(1-.95**step)
        np.testing.assert_allclose(grad,expected,rtol=1e-12,atol=1e-12)
        assert all(not v.requires_grad for v in ema.values.values())
    clone=VarianceEMA(.95);clone.load_state_dict(ema.state_dict());same(ema.state_dict(),clone.state_dict())
    x=torch.randn(8,3,4,90,63);y=torch.tensor([0,1,0,2,2,1,0,2]);d=torch.tensor([3,3,4,4,5,5,6,6])
    shifted=permuted_domains(y,d,2026,3,1)
    for k in y.unique():same(torch.sort(d[y==k])[0],torch.sort(shifted[y==k])[0])
    net=SignalNet();plain=copy.deepcopy(net);net.train();plain.train();a=torch.optim.AdamW(net.parameters(),lr=.001);b=torch.optim.AdamW(plain.parameters(),lr=.001)
    c={'seed':2026,'ema_beta':.95,'warmup_epochs':1,'calibration_norm_ratio':.1,'scale_bounds':[.0001,10000.]}
    torch.manual_seed(23);source_step(net,a,x,y,d,VarianceEMA(.95),c,3,1,1,'balanced_erm');rng_state=torch.get_rng_state()
    torch.manual_seed(23);b.zero_grad(set_to_none=True);torch.nn.functional.cross_entropy(plain(x),y).backward();b.step()
    same(net.state_dict(),plain.state_dict());same(a.state_dict(),b.state_dict());same(rng_state,torch.get_rng_state())


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);numeric_checks()
    rng=np.random.default_rng(998);users=np.repeat(np.arange(3,9),3);y=np.tile(np.arange(3),6);cache=rng.normal(size=(18,3,6,90,63)).astype(np.float16)
    c={'seed':2026,'batch':8,'per_domain':2,'epochs':3,'lr':.001,'weight_decay':.001,'min_lr':1e-5,'tail_start_epoch':2,'ema_beta':.95,'warmup_epochs':1,'calibration_norm_ratio':.1,'scale_bounds':[.0001,10000.],'arms':['balanced_erm','real_variance','permuted_variance']}
    calibrations=[];warm=[]
    with tempfile.TemporaryDirectory(prefix='uwb-fishr-test-') as td,contextlib.redirect_stdout(io.StringIO()):
        root=Path(td)
        for arm in c['arms']:
            full=root/(arm+'full');res=root/(arm+'resume');iso=root/(arm+'isolate')
            fit(full,c,{},cache,y,users,3,arm)
            fit(res,c,{},cache,y,users,3,arm,stop_after=2);torch.rand(713);fit(res,c,{},cache,y,users,3,arm)
            a=torch.load(full/'progress.pt',weights_only=False);b=torch.load(res/'progress.pt',weights_only=False)
            for k in ['model','optimizer','scheduler','rng','tail','best','best_ce','best_epoch','step','ema']:same(a[k],b[k])
            same(torch.load(full/'selected.pt',weights_only=False),torch.load(res/'selected.pt',weights_only=False))
            changed=cache.copy();changed[users==3]*=2;labels=y.copy();labels[users==3]=(labels[users==3]+5)%12
            fit(iso,c,{},changed,labels,users,3,arm)
            same(torch.load(full/'selected.pt',weights_only=False),torch.load(iso/'selected.pt',weights_only=False))
            same(a['ema'],torch.load(iso/'progress.pt',weights_only=False)['ema'])
            audit_run(full,cache,y,users,c);calibrations.append(a['ema']['calibration']);warm.append(torch.load(full/'segment_01.pt',weights_only=False)['average'])
        assert all(x==calibrations[0] for x in calibrations)
        for x in warm:same(x,warm[0])
    print(json.dumps({'state':'passed','synthetic_only':True,'independent_analytic_higher_derivative_and_finite_difference_tests':True,'ema_conventional_bias_correction_and_current_derivative_matches_numpy':True,'ema_history_detached':True,'permutation_preserves_class_domain_counts':True,'ordinary_step_model_adamw_dropout_rng_bitwise':True,'all3_arms_resume_model_adamw_scheduler_rng_ema_scale_tail_selection_exact':True,'held_inputs_and_labels_do_not_affect_weights_or_ema_or_scale':True,'all3_arms_independent_inference_roles_order_selection_audit':True,'same_warmup_states_and_calibration_across_arms':True},indent=2))

if __name__=='__main__':main()
