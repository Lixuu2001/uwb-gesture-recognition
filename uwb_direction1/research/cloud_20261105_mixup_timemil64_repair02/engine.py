"""Full native TimeMIL, standard input Mixup, complete reproducible RNG state."""
import sys,random,hashlib,os,copy
from common import protocol
c=protocol();sys.path.insert(0,c['native_source'])
import adapter as native
import torch
import numpy as np
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def make():
    random.seed(c['seed']);np.random.seed(c['seed'])
    m=native.make_model('timemil',c['seed']);assert sum(p.numel() for p in m.parameters())==850973
    return m,native.StableLookahead(m.parameters(),lr=.001,weight_decay=.0001),torch.Generator().manual_seed(c['seed'])
def semantic(value):
    h=hashlib.sha256()
    def walk(x):
        if isinstance(x,torch.Tensor):
            h.update(str((tuple(x.shape),str(x.dtype))).encode());h.update(x.detach().contiguous().cpu().numpy().tobytes())
        elif isinstance(x,dict):
            for k,v in x.items():h.update(str(k).encode());walk(v)
        elif isinstance(x,(list,tuple)):
            for v in x:walk(v)
        else:h.update(str(x).encode())
    walk(value);return h.hexdigest()
def capture(m,opt,g,epoch,updates,best_key=None,best_model=None,best_epoch=0):
    n=np.random.get_state()
    return dict(model=m.state_dict(),optimizer=opt.state_dict(),order_rng=g.get_state(),torch_rng=torch.get_rng_state(),
        python_rng=random.getstate(),numpy_rng=(n[0],torch.from_numpy(n[1].copy()),n[2],n[3],n[4]),
        epoch=epoch,updates=updates,best_key=best_key,best_model=best_model,best_epoch=best_epoch,model_training=m.training)
def restore(state,m,opt,g):
    m.load_state_dict(state['model'],strict=True);opt.load_state_dict(copy.deepcopy(state['optimizer']));g.set_state(state['order_rng'])
    torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng']);n=state['numpy_rng']
    np.random.set_state((n[0],n[1].numpy(),n[2],n[3],n[4]));m.train(state['model_training'])
def atomic_save(path,state):
    from pathlib import Path
    import time
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    if tmp.exists():
        d=path.parent/'orphans';d.mkdir(exist_ok=True);tmp.rename(d/(tmp.name+'.'+str(time.time_ns())))
    torch.save(state,tmp)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    tmp.replace(path)

def order(n,epoch):return torch.randperm(n,generator=torch.Generator().manual_seed(c['seed']+epoch)).numpy()
def mix(x,y):
    lam=float(np.random.beta(c['alpha'],c['alpha']))
    perm=torch.randperm(len(x))
    return lam*x+(1-lam)*x[perm,:],y,y[perm],lam,perm
def step(m,opt,x,y,epoch):
    assert m.training
    opt.zero_grad();xx=torch.from_numpy(x);yy=torch.as_tensor(y,dtype=torch.long)
    mixed,a,b,lam,perm=mix(xx,yy)
    target=lam*torch.nn.functional.one_hot(a,12).float()+(1-lam)*torch.nn.functional.one_hot(b,12).float()
    z=m(mixed,warmup=epoch<10)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(z,target)
    assert torch.isfinite(loss);loss.backward()
    assert sorted(n for n,p in m.named_parameters() if p.grad is None)==sorted(c['expected_unused'])
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    torch.nn.utils.clip_grad_norm_(m.parameters(),2.);opt.step()
    assert all(torch.isfinite(p).all() for p in m.parameters())
    return dict(loss=float(loss.detach()),lam=lam,permutation_sha256=hashlib.sha256(perm.numpy().tobytes()).hexdigest(),mixed_input_sha256=hashlib.sha256(mixed.numpy().tobytes()).hexdigest())
def macro(y,p):return float(np.mean([np.mean(p[y==k]==k) for k in np.unique(y)]))
def evaluate(m,maps,positions,y=None):
    m.eval();chunks=[]
    with torch.inference_mode():
        for i in range(0,len(positions),c['batch']):chunks.append(m(torch.from_numpy(np.array(maps[positions[i:i+c['batch']]],copy=True))).numpy())
    z=np.concatenate(chunks);assert z.shape==(len(positions),12) and np.isfinite(z).all()
    if y is None:return z
    bce=torch.nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(z),torch.nn.functional.one_hot(torch.as_tensor(y,dtype=torch.long),12).float())
    return z,macro(y,z.argmax(1)),float(bce)
def record(state,epoch,updates,macro=None,loss=None,scores=None):
    return dict(epoch=epoch,updates=updates,validation_macro=macro,validation_loss=loss,validation_logits_sha256=None if scores is None else hashlib.sha256(scores.tobytes()).hexdigest(),state_sha256=semantic(state),model_sha256=semantic(state['model']),optimizer_sha256=semantic(state['optimizer']),best_epoch=state['best_epoch'],best_key=None if state['best_key'] is None else list(state['best_key']))
