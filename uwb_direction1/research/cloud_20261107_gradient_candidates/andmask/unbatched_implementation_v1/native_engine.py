"""Generated-fixture runnable native617612 AND-mask step; no data/label loading or real fit entry."""
import sys,copy,random,hashlib
sys.path.insert(0,'/workspace/uwb-timemil-20261029')
import adapter as native
import torch
import numpy as np
from andmask_core import gradients_from_pooled_logits
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
SEED=20261029

def make():
    random.seed(SEED);np.random.seed(SEED)
    m=native.make_model('gap',SEED)
    assert sum(p.numel() for p in m.parameters())==617612
    return m,native.StableLookahead(m.parameters(),lr=.001,weight_decay=.0001)

def step(m,opt,x,y,domains,arm='andmask'):
    assert m.training and arm in ['andmask','native']
    logits=m(x);opt.zero_grad()
    if arm=='native':
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.nn.functional.one_hot(y,12).to(logits.dtype));loss.backward();diagnostic={}
    else:
        loss,grads,diagnostic=gradients_from_pooled_logits(logits,y,domains,m.parameters())
        for p,g in zip(m.parameters(),grads):p.grad=g
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),2.)
    opt.step();assert all(torch.isfinite(p).all() for p in m.parameters())
    return dict(global_BCE=float(loss.detach()),preclip_gradient_norm=float(norm),**diagnostic)

def capture(m,opt):
    return copy.deepcopy(dict(model=m.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state()))

def restore(state,m,opt):
    state=copy.deepcopy(state);m.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])

def semantic(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):h.update(str((v.shape,v.dtype)).encode());h.update(v.detach().numpy().tobytes())
        elif isinstance(v,np.ndarray):h.update(str((v.shape,v.dtype)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k,x in v.items():h.update(str(k).encode());visit(x)
        elif isinstance(v,(tuple,list)):
            for x in v:visit(x)
        else:h.update(str(v).encode())
    visit(value);return h.hexdigest()
