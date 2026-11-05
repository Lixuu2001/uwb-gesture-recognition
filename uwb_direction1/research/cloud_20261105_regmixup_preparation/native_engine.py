"""Conditional full-native GAP RegMixup transfer; synthetic use only until registered."""
from pathlib import Path
import copy,hashlib,json,random,sys
B=Path(__file__).resolve().parent
c=json.loads((B/'PROBE_CONFIG.json').read_text())
sys.path.insert(0,c['native_source'])
import adapter as native
import numpy as np
import torch
from prototype import batch,native_ovr_bce_transfer
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ready():
    assert c['synthetic_only'] and not c['new_UWB_fit_allowed']
    for p,h in c['native_dependencies'].items():assert sha(p)==h,p
    for p,h in json.loads((B/'NATIVE_PROBE_HASHES.json').read_text()).items():assert sha(B/p)==h,p
def make():
    random.seed(c['seed']);np.random.seed(c['seed'])
    m=native.make_model('gap',c['seed']);assert sum(p.numel() for p in m.parameters())==617612
    return m,native.StableLookahead(m.parameters(),lr=.001,weight_decay=.0001),torch.Generator().manual_seed(c['seed'])
def semantic(value):
    h=hashlib.sha256()
    def walk(x):
        if isinstance(x,torch.Tensor):h.update(str((tuple(x.shape),str(x.dtype))).encode());h.update(x.detach().contiguous().cpu().numpy().tobytes())
        elif isinstance(x,dict):
            for k,v in x.items():h.update(str(k).encode());walk(v)
        elif isinstance(x,(tuple,list)):
            for v in x:walk(v)
        else:h.update(str(x).encode())
    walk(value);return h.hexdigest()
def capture(m,o,g,epoch,updates):
    n=np.random.get_state()
    return dict(model=m.state_dict(),optimizer=o.state_dict(),order_rng=g.get_state(),torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=(n[0],torch.from_numpy(n[1].copy()),n[2],n[3],n[4]),epoch=epoch,updates=updates,model_training=m.training)
def restore(st,m,o,g):
    m.load_state_dict(st['model'],strict=True);o.load_state_dict(copy.deepcopy(st['optimizer']));g.set_state(st['order_rng']);torch.set_rng_state(st['torch_rng']);random.setstate(st['python_rng']);n=st['numpy_rng'];np.random.set_state((n[0],n[1].numpy(),n[2],n[3],n[4]));m.train(st['model_training'])
def step(m,o,x,y):
    assert m.training
    o.zero_grad();xx,a,b,lam,p=batch(x,y,c['alpha'],c['alpha'])
    logits=m(xx);loss=native_ovr_bce_transfer(logits,a,b,lam)
    assert logits.shape==(2*len(x),12) and torch.isfinite(loss)
    loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    torch.nn.utils.clip_grad_norm_(m.parameters(),2.);o.step();assert all(torch.isfinite(p).all() for p in m.parameters())
    return dict(loss=float(loss.detach()),lambda_value=lam,permutation_sha256=semantic(p),mixed_input_sha256=semantic(xx),effective_forward_batch=len(xx))
