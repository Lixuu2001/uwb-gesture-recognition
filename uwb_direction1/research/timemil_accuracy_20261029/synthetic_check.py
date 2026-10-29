"""No UWB reads; wavelet, optimizer/resume, full-width gradient and CPU cost checks."""
from adapter import make_model,StableLookahead,cpu_compatible_wavelet
import copy,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
B=Path(__file__).resolve().parent
def optimizer_check():
    # Independent scalar/vector implementation of author's update rule.
    p=torch.nn.Parameter(torch.tensor([.5,-.25],dtype=torch.float64));q=torch.nn.Parameter(p.detach().clone())
    wrapped=StableLookahead([p]);fast=torch.optim.AdamW([q],lr=.001,weight_decay=.0001);slow=None
    for step in range(1,15):
        wrapped.zero_grad();fast.zero_grad();p.grad=torch.tensor([.1*step,-.2],dtype=p.dtype);q.grad=p.grad.clone();wrapped.step();fast.step()
        if step%6==0:
            if slow is None:slow=q.detach().clone()
            slow.add_(q-slow,alpha=.5)
            with torch.no_grad():q.copy_(slow)
        torch.testing.assert_close(p,q,atol=0,rtol=0)
    # Resume after a synchronization, exercising saved slow buffers and counters.
    r=torch.nn.Parameter(p.detach().clone());resume=StableLookahead([r]);resume.load_state_dict(copy.deepcopy(wrapped.state_dict()))
    for step in range(15,25):
        for v,opt in [(p,wrapped),(r,resume)]:opt.zero_grad();v.grad=torch.tensor([.1*step,-.2],dtype=v.dtype);opt.step()
        torch.testing.assert_close(p,r,atol=0,rtol=0)
def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    resource.setrlimit(resource.RLIMIT_CPU,(580,590));start=time.monotonic();optimizer_check()
    scale=torch.tensor([[.7],[1.4]],dtype=torch.float64);shift=torch.tensor([[.2],[-.5]],dtype=torch.float64)
    actual=cpu_compatible_wavelet((2,19),scale,shift).numpy()
    x=np.linspace(-9,9,19)[None,:]-shift.numpy();s=scale.numpy();expected=(2/(np.sqrt(3)*np.pi**.25))*(1-(x/s)**2)*np.exp(-.5*(x/s)**2)/np.sqrt(np.abs(s))
    np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-13)
    models={a:make_model(a) for a in ['timemil','gap']};f=models['timemil'];g=models['gap']
    assert all(torch.equal(v,g.feature_extractor.state_dict()[k]) for k,v in f.feature_extractor.state_dict().items())
    assert all(torch.equal(v,g.head.state_dict()[k]) for k,v in f._fc2.state_dict().items())
    # Preserve and explicitly report the author's distinct Parameters sharing storage.
    alias=all(getattr(f,n).data_ptr()==getattr(f,n+'_').data_ptr() for n in ['wave1','wave2','wave3']);assert alias
    x=torch.randn(64,90,567);y=F.one_hot(torch.arange(64)%12,num_classes=12).float();results={}
    for arm,m in models.items():
        opt=StableLookahead(m.parameters());durations=[];unused=None
        for i in range(4):
            t=time.monotonic();m.train();opt.zero_grad();pred=m(x,warmup=i<2);assert pred.shape==(64,12)
            loss=F.binary_cross_entropy_with_logits(pred,y);assert torch.isfinite(loss);loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            unused=[n for n,p in m.named_parameters() if p.grad is None];torch.nn.utils.clip_grad_norm_(m.parameters(),2.);opt.step()
            assert all(torch.isfinite(p).all() for p in m.parameters());durations.append(time.monotonic()-t)
        m.eval()
        with torch.no_grad():
            t=time.monotonic();pred=m(x);forward=time.monotonic()-t
        results[arm]=dict(parameters=sum(p.numel() for p in m.parameters()),seconds_per_train_batch64=float(np.mean(durations[1:])),seconds_per_eval_batch64=forward,unused_parameter_names=unused)
    estimate=2*300*sum(74*v['seconds_per_train_batch64']+16*v['seconds_per_eval_batch64'] for v in results.values())
    output=dict(state='pass',synthetic_only=True,project_arrays_or_labels_read=False,paper_or_code_exact_replication=False,
        author_wavelet_storage_aliases_preserved=alias,backbone_and_head_initialization_matched=True,
        optimizer_matches_author_rule=True,optimizer_resume_bitwise_exact=True,wavelet_numpy_formula_matches=True,
        full_width567_input=True,batch_size=64,embed=128,threads=1,arms=results,
        prospective_epochs=300,estimated_pair_seconds_with_margin2=estimate,cost_gate_12h_passed=estimate<43200,
        scientific_protocol_registered=False,new_real_training_started=False,wall_seconds=time.monotonic()-start,
        adapter_sha256=hashlib.sha256((B/'adapter.py').read_bytes()).hexdigest())
    p=B/'SYNTHETIC_CHECK.json';assert not p.exists();p.write_text(json.dumps(output,indent=2));print(json.dumps(output))
if __name__=='__main__':main()
