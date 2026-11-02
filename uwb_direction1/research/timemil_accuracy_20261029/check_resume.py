"""Stochastic full-model restart check, including aliases and slow buffers."""
from adapter import make_model,StableLookahead
import copy,json,resource
from pathlib import Path
import torch
from torch.nn import functional as F
def main():
    resource.setrlimit(resource.RLIMIT_CPU,(280,290));torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    m=make_model();o=StableLookahead(m.parameters());x=torch.randn(4,90,567);y=F.one_hot(torch.arange(4)%12,12).float()
    def update(model,opt):
        model.train();opt.zero_grad();loss=F.binary_cross_entropy_with_logits(model(x,warmup=True),y);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
    for _ in range(7):update(m,o)
    saved=copy.deepcopy(dict(model=m.state_dict(),optimizer=o.state_dict(),rng=torch.get_rng_state()))
    for _ in range(6):update(m,o)
    clone=make_model();other=StableLookahead(clone.parameters());clone.load_state_dict(saved['model']);other.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
    for _ in range(6):update(clone,other)
    assert all(torch.equal(v,clone.state_dict()[k]) for k,v in m.state_dict().items())
    assert all(getattr(clone,k).data_ptr()==getattr(clone,k+'_').data_ptr() for k in ['wave1','wave2','wave3'])
    assert o.count==other.count==13
    assert all(torch.equal(v,other.slow[k]) for k,v in o.slow.items())
    result=dict(state='pass',synthetic_only=True,project_arrays_or_labels_read=False,full_model_all_tensors_bitwise_equal=True,
        optimizer_steps=13,resume_from_step=7,slow_buffer_sync_boundary_crossed=True,wavelet_aliases_retained=True,dropout_rng_restored=True)
    p=Path(__file__).with_name('RESUME_CHECK.json');assert not p.exists();p.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
