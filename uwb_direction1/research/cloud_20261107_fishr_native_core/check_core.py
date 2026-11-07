"""Synthetic-only first/second derivative and pinned-author EMA/distance parity."""
from pathlib import Path
from collections import OrderedDict
import ast, hashlib, io, json, time
import torch
from torch import nn
from torch.nn import functional as F
from fishr_core import PublishedEMA, native_head_gradients, variance_penalty, schedule, DEFAULTS
ROOT=Path(__file__).resolve().parent

def main():
    started=time.monotonic();cpu=time.process_time();torch.set_num_threads(1);torch.manual_seed(20261029)
    author=ROOT/'author_reference/domainbed/lib/misc.py';raw=author.read_bytes()
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()=='f9e6497a82fed9a4e1d70346ad708c2c9df2ad9f'
    tree=ast.parse(raw.decode());defs=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['MovingAverage','l2_between_dicts']]
    scope={'torch':torch};exec(compile(ast.Module(body=defs,type_ignores=[]),str(author),'exec'),scope)
    # Native two-linear/128/ReLU/dropout.2/12 head, all 18060 parameters.
    head=nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,12)).double().train()
    z=torch.randn(9,128,dtype=torch.float64,requires_grad=True);y=torch.randint(12,(9,));groups=torch.tensor([0,0,1,1,1,2,3,3,3])
    logits,analytic=native_head_gradients(head,z,y)
    reference=OrderedDict((name,[]) for name,_ in head.named_parameters())
    params=list(head.parameters())
    for i in range(len(y)):
        loss=F.binary_cross_entropy_with_logits(logits[i],F.one_hot(y[i],12).double(),reduction='mean')
        values=torch.autograd.grad(loss,params,retain_graph=True,create_graph=True)
        for (name,_),value in zip(head.named_parameters(),values):reference[name].append(value.flatten())
    reference=OrderedDict((k,torch.stack(v)) for k,v in reference.items())
    first=max(float((analytic[k]-reference[k]).abs().max().detach()) for k in analytic)
    assert first < 1e-11
    ours=variance_penalty(analytic,groups,PublishedEMA(4))
    moments=[]
    author_emas=[scope['MovingAverage'](.95,oneminusema_correction=True) for _ in range(4)]
    for group in range(4):
        current=OrderedDict()
        for name,g in reference.items():
            samples=g[groups==group];current[name]=(samples-samples.mean(0,keepdim=True)).pow(2).mean(0)
        moments.append(author_emas[group].update(current))
    mean=OrderedDict((k,torch.stack([d[k] for d in moments]).mean(0)) for k in reference)
    expected=sum(scope['l2_between_dicts'](d,mean) for d in moments)/4
    assert abs(float((ours-expected).detach())) < 1e-13
    ga=torch.autograd.grad(ours,[z]+params,retain_graph=True)
    gb=torch.autograd.grad(expected,[z]+params,retain_graph=True)
    second=max(float((a-b).abs().max()) for a,b in zip(ga,gb));assert second<1e-10
    # Five updates verify the author's fixed denominator and detached history.
    ema=PublishedEMA(1);ref=scope['MovingAverage'](.95,oneminusema_correction=True);ema_error=0.
    for step in range(5):
        v=torch.rand(7,dtype=torch.float64,requires_grad=True)
        a=ema.update(0,OrderedDict(w=v))['w'];b=ref.update(OrderedDict(w=v))['w'].reshape(-1)
        ema_error=max(ema_error,float((a-b).abs().max().detach()));assert torch.equal(a,b)
        grad=torch.autograd.grad(a.sum(),v)[0];assert torch.allclose(grad,torch.ones_like(grad),atol=1e-13,rtol=0)
        assert all(not x.requires_grad for x in ema.data[0].values())
    saved=io.BytesIO();torch.save(ema.state_dict(),saved);saved.seek(0);resumed=PublishedEMA(1);resumed.load_state_dict(torch.load(saved,weights_only=True))
    v=torch.rand(7,dtype=torch.float64,requires_grad=True)
    assert torch.equal(ema.update(0,OrderedDict(w=v))['w'],resumed.update(0,OrderedDict(w=v))['w'])
    # Explicit native mixed-batch transfer: absent groups keep their EMA; singleton
    # population variance remains finite; one represented group has connected zero.
    e=PublishedEMA(4);g=OrderedDict(w=torch.randn(4,5,dtype=torch.float64,requires_grad=True));value=variance_penalty(g,torch.tensor([0,0,2,2]),e)
    assert e.updates==[1,0,1,0] and torch.isfinite(value)
    one=variance_penalty(g,torch.zeros(4,dtype=torch.long),PublishedEMA(4));assert one.item()==0 and torch.autograd.grad(one,g['w'])[0].abs().max()==0
    assert schedule(1499)==dict(weight=0.,reset_before_this_update=False)
    assert schedule(1500)==dict(weight=1000.,reset_before_this_update=True)
    assert schedule(1501)==dict(weight=1000.,reset_before_this_update=False)
    result=dict(state='pass_synthetic_core_only_not_native_LOSO_training',seed=20261029,
        native_classifier_parameters=18060,first_derivative_max_abs_error=first,
        penalty_abs_error=abs(float((ours-expected).detach())),second_derivative_max_abs_error=second,
        five_update_pinned_author_EMA_bitwise_equal=True,EMA_max_abs_error=ema_error,
        fixed_one_minus_EMA_gradient_compensation_verified=True,checkpoint_state_continuation_exact=True,
        missing_group_transfer_and_singleton_connectedzero_finite=True,synthetic_schedule_boundary=True,
        defaults=DEFAULTS,source_author_git_blob_sha1='f9e6497a82fed9a4e1d70346ad708c2c9df2ad9f',
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'fishr_core.py',Path(__file__),author]},
        real_UWB_data_or_labels_read=False,optimizer_steps=0,native_backbone_worker_implemented=False,
        fitting_preregistered=False,new_background_training_started=False,wall=time.monotonic()-started,CPU=time.process_time()-cpu)
    (ROOT/'CORE_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
