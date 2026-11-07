"""Generated-fixture runnable native617612 IRM step; no data/label loading or real fit entry."""
import sys,copy,random,hashlib
sys.path.insert(0,'/workspace/uwb-timemil-20261029')
import adapter as native
import torch
import numpy as np
from irm_core import objective
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
SEED=20261029

def make():
    random.seed(SEED);np.random.seed(SEED)
    m=native.make_model('gap',SEED)
    assert sum(p.numel() for p in m.parameters())==617612
    return m,ScheduledOptimizer(m.parameters())

class ScheduledOptimizer:
    def __init__(self,params):
        self.params=list(params);self.algorithm_updates=0;self.reset_events=[];self.inner=self.fresh()
    def fresh(self):return native.StableLookahead(self.params,lr=.001,weight_decay=.0001)
    @property
    def count(self):return self.inner.count
    @property
    def slow(self):return self.inner.slow
    def zero_grad(self):self.inner.zero_grad()
    def reset_if_due(self):
        if self.algorithm_updates==500 and not self.reset_events:
            self.inner=self.fresh();self.reset_events=[500];return True
        return False
    def step(self):
        assert self.algorithm_updates<500 or self.reset_events==[500]
        self.inner.step();self.algorithm_updates+=1
    def state_dict(self):return dict(native_optimizer=self.inner.state_dict(),algorithm_updates=self.algorithm_updates,reset_events=list(self.reset_events))
    def load_state_dict(self,s):
        self.algorithm_updates=s['algorithm_updates'];self.reset_events=list(s['reset_events']);assert self.reset_events==([] if self.algorithm_updates<=500 else [500]);self.inner=self.fresh();self.inner.load_state_dict(copy.deepcopy(s['native_optimizer']))
        assert self.count==(self.algorithm_updates-500 if self.reset_events else self.algorithm_updates)

def step(m,opt,x,y,domains,arm='irm'):
    assert m.training and arm in ['irm','native']
    logits=m(x);weight=1. if opt.algorithm_updates<500 else 100.
    if arm=='native':
        classification=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.nn.functional.one_hot(y,12).to(logits.dtype));loss=classification;penalty=logits.sum()*0.;diagnostic={}
    else:loss,classification,penalty,diagnostic=objective(logits,y,domains,weight)
    did_reset=opt.reset_if_due();opt.zero_grad();loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),2.);opt.step();assert all(torch.isfinite(p).all() for p in m.parameters())
    return dict(loss=float(loss.detach()),global_BCE=float(classification.detach()),penalty=float(penalty.detach()),irm_penalty_weight=weight,optimizer_reset_this_update=did_reset,algorithm_updates=opt.algorithm_updates,reset_events=list(opt.reset_events),preclip_gradient_norm=float(norm),**diagnostic)

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
