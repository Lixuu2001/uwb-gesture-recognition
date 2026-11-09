"""Passive dense tail averaging on the unchanged native optimizer trajectory."""
from pathlib import Path
import os,sys,json,hashlib,copy,random,time
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/workspace/uwb-timemil-20261029')
import adapter as native
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    if tmp.exists():raise FileExistsError(tmp)
    with tmp.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def semantic(x):
    h=hashlib.sha256()
    def walk(v):
        if isinstance(v,torch.Tensor):h.update(str((tuple(v.shape),str(v.dtype))).encode());h.update(v.detach().contiguous().cpu().numpy().tobytes())
        elif isinstance(v,dict):
            for k,z in v.items():h.update(str(k).encode());walk(z)
        elif isinstance(v,(list,tuple)):
            for z in v:walk(z)
        else:h.update(str(v).encode())
    walk(x);return h.hexdigest()
def atomic_save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    if tmp.exists():raise FileExistsError('Preserve orphan before explicit recovery: '+str(tmp))
    torch.save(x,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    tmp.replace(p)
def verify_sources():
    for n,h in read(ROOT/'NATIVE_DEPENDENCIES.json')['files'].items():
        if sha(n)!=h:raise ValueError('Native dependency changed: '+n)
def make():
    verify_sources();random.seed(20261029);np.random.seed(20261029)
    m=native.make_model('gap',20261029)
    if sum(p.numel() for p in m.parameters())!=617612:raise ValueError('Native capacity changed')
    if semantic(m.state_dict())!='2aae73c58e38ca324027e888955b130a82fbaece7167df3465522ec51986d2d3':raise ValueError('Native initial model changed')
    return m,native.StableLookahead(m.parameters(),lr=.001,weight_decay=.0001),Tail(m)
class Tail:
    """Exact parameter arithmetic of pinned SWAD AveragedModel; no model mutation."""
    def __init__(self,m):self.model=copy.deepcopy(m);self.n=0;self.start=None;self.end=None
    @torch.no_grad()
    def add(self,m,step):
        before=semantic(m.state_dict());rng=torch.get_rng_state().clone()
        for a,b in zip(self.model.parameters(),m.parameters()):
            if self.n==0:a.copy_(b.detach())
            else:a.copy_(a+(b.detach()-a)/(self.n+1))
        if self.n==0:self.start=step
        self.n+=1;self.end=step
        if semantic(m.state_dict())!=before or not torch.equal(rng,torch.get_rng_state()):raise ValueError('Passive collector mutated trajectory')
    def state_dict(self):return dict(model=self.model.state_dict(),n=self.n,start=self.start,end=self.end)
    def load_state_dict(self,x):self.model.load_state_dict(x['model']);self.n=x['n'];self.start=x['start'];self.end=x['end']
def order(n,epoch):return torch.randperm(n,generator=torch.Generator().manual_seed(20261029+epoch)).numpy()
def bce(z,y):return torch.nn.functional.binary_cross_entropy_with_logits(z,torch.nn.functional.one_hot(torch.as_tensor(y,dtype=torch.long),12).to(z.dtype))
def step(m,o,t,x,y,collect):
    m.train();z=m(torch.from_numpy(x));loss=bce(z,y)
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    o.zero_grad();loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()):raise ValueError('Native gradient missing/nonfinite')
    torch.nn.utils.clip_grad_norm_(m.parameters(),2.);o.step()
    if collect:t.add(m,o.count)
    return dict(loss=float(loss.detach()),update=o.count,model_sha256=semantic(m.state_dict()),optimizer_sha256=semantic(o.state_dict()),tail_sha256=semantic(t.state_dict()),logits_sha256=semantic(z))
def capture(m,o,t,epoch,best_key=None,best_model=None,best_epoch=None):
    n=np.random.get_state()
    return copy.deepcopy(dict(model=m.state_dict(),optimizer=o.state_dict(),tail=t.state_dict(),epoch=epoch,updates=o.count,
        best_key=best_key,best_model=best_model,best_epoch=best_epoch,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
        numpy_rng=(n[0],torch.from_numpy(n[1].copy()),n[2],n[3],n[4]),modes={k:v.training for k,v in m.named_modules()}))
def restore(s,m,o,t):
    m.load_state_dict(s['model']);o.load_state_dict(copy.deepcopy(s['optimizer']));t.load_state_dict(s['tail'])
    torch.set_rng_state(s['torch_rng']);random.setstate(s['python_rng']);n=s['numpy_rng'];np.random.set_state((n[0],n[1].numpy(),n[2],n[3],n[4]))
    for k,v in m.named_modules():v.training=s['modes'][k]
    if o.count!=s['updates']:raise ValueError('Optimizer count mismatch')
