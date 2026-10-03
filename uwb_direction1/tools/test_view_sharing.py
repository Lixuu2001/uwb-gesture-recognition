"""Behavioral tests: shared trajectory equivalence, resume and held isolation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from audit_restored import dump
from circular_signal import SignalNet, get_batch
from swad_source_pilot import fit as original_fit
from view_sharing import make_model
from view_sharing_source import fit
from audit_view_sharing import functional_forward, audit_run


def same(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(8197);users=np.repeat(np.arange(3,9),4)
    y=np.tile(np.arange(4),6);cache=rng.normal(size=(24,3,6,90,63)).astype(np.float16)
    x=get_batch(cache,np.arange(3),'legacy');receipt={'synthetic_only':True}
    torch.manual_seed(2026);original=SignalNet();ref_rng=torch.get_rng_state()
    shared=make_model('shared128',2026);same(original.state_dict(),shared.state_dict());same(ref_rng,torch.get_rng_state())
    untied=make_model('untied128',2026);same(ref_rng,torch.get_rng_state())
    for e in untied.encoders:same(e.state_dict(),shared.encoder.state_dict())
    assert len({e[0].layers[0].weight.data_ptr() for e in untied.encoders})==3
    shared.eval();untied.eval()
    with torch.no_grad():assert torch.allclose(shared(x),untied(x),atol=3e-6,rtol=3e-6)
    for arm in ['shared128','untied128','shared177']:
        n=make_model(arm,2026);same(ref_rng,torch.get_rng_state());n.eval()
        with torch.no_grad():assert torch.equal(n(x),functional_forward(x,n.state_dict()))
    c={'seed':2026,'batch':8,'epochs':4,'lr':.001,'weight_decay':.001,'min_lr':1e-5,
       'tail_start_epoch':2,'n_converge':3,'n_tolerance':6,'tolerance_ratio':.3}
    with tempfile.TemporaryDirectory(prefix='uwb-sharing-test-') as td, contextlib.redirect_stdout(io.StringIO()):
        root=Path(td)
        original_fit(root/'old',c,{},cache,y,users,3,stop_after=3)
        fit(root/'shared',c,{},cache,y,users,3,'shared128',stop_after=3)
        old=torch.load(root/'old/progress.pt',weights_only=False);new=torch.load(root/'shared/progress.pt',weights_only=False)
        for key in ['model','optimizer','scheduler','rng','tail','best','best_ce','best_epoch','step']:same(old[key],new[key])
        receipt['original_trainer_shared_trajectory_bitwise_equal']=True
        for arm in ['untied128','shared177']:
            full=root/(arm+'full');resume=root/(arm+'resume');isolate=root/(arm+'isolate')
            fit(full,c,{},cache,y,users,3,arm)
            fit(resume,c,{},cache,y,users,3,arm,stop_after=2)
            torch.rand(500)
            fit(resume,c,{},cache,y,users,3,arm)
            p=torch.load(full/'progress.pt',weights_only=False);q=torch.load(resume/'progress.pt',weights_only=False)
            for key in ['model','optimizer','scheduler','rng','tail','best','best_ce','best_epoch','step']:same(p[key],q[key])
            same(torch.load(full/'selected.pt',weights_only=False),torch.load(resume/'selected.pt',weights_only=False))
            changed=cache.copy();changed[users==3]=rng.normal(size=changed[users==3].shape).astype(np.float16)*2
            labels=y.copy();labels[users==3]=(labels[users==3]+5)%12
            fit(isolate,c,{},changed,labels,users,3,arm)
            same(torch.load(full/'selected.pt',weights_only=False),torch.load(isolate/'selected.pt',weights_only=False))
            assert json.loads((full/'result.json').read_text())['arm']==arm
            if arm=='untied128':
                assert not torch.equal(p['model']['encoders.0.0.layers.0.weight'],p['model']['encoders.1.0.layers.0.weight'])
            audit_run(full,cache,y,users,c)
        receipt.update(resume_model_optimizer_rng_selection_bitwise_equal_both_arms=True,
                       held_input_and_label_perturbation_does_not_change_selected_weights=True,
                       functional_logits_and_artifact_audit_both_arms=True,
                       independent_encoder_storage_and_initialization_verified=True)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
