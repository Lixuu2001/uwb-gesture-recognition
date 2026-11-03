"""Generic-role loops reuse native model/updates, with exact resumable state."""
import sys,copy,hashlib,random,os
_arm=None
def setup(arm):
    global _arm,torch,np,native
    from common import config
    c,p=config()
    if _arm is not None:
        assert _arm==arm,'Member families require isolated processes'
        return
    assert arm in ['timemil','shuffle'];_arm=arm
    key='TimeMIL' if arm=='timemil' else 'ShuffleNet'
    sys.path.insert(0,p[key]['source'])
    if arm=='timemil':
        import adapter as native
        import torch
        import numpy as np
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    else:
        import native_adam100 as native
        from native_adam100 import torch,np
    assert torch.get_num_threads()==p[key]['threads'] and torch.get_num_interop_threads()==1
def make(arm):
    setup(arm);random.seed(20261029);np.random.seed(20261029)
    if arm=='timemil':m=native.make_model('timemil',20261029);opt=native.StableLookahead(m.parameters(),lr=.001,weight_decay=.0001)
    else:m=native.make();opt=native.optimizer(m)
    assert sum(p.numel() for p in m.parameters())==(850973 if arm=='timemil' else 1265904)
    return m,opt,torch.Generator().manual_seed(20261029)
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
    m.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);g.set_state(state['order_rng'])
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
def order(arm,n,epoch,g):
    return torch.randperm(n,generator=torch.Generator().manual_seed(20261029+epoch) if arm=='timemil' else g).numpy()
def raw(arm,maps,positions):
    x=np.asarray(maps[positions])
    if arm=='shuffle':x=np.ascontiguousarray(x.reshape(len(x),90,3,189).transpose(0,2,1,3))
    return np.array(x,copy=True)
def step(arm,m,opt,x,y,epoch):
    assert m.training
    if arm=='shuffle':return native.step(m,opt,x,y)
    from common import config
    c,p=config();opt.zero_grad();target=torch.nn.functional.one_hot(torch.as_tensor(y,dtype=torch.long),12).float()
    logits=m(torch.from_numpy(x),warmup=epoch<10)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target);assert torch.isfinite(loss);loss.backward()
    unused=sorted(n for n,q in m.named_parameters() if q.grad is None)
    assert unused==sorted(c['expected_TimeMIL_unused_gradients'])
    assert all(q.grad is None or torch.isfinite(q.grad).all() for q in m.parameters())
    torch.nn.utils.clip_grad_norm_(m.parameters(),2.);opt.step()
    assert all(torch.isfinite(q).all() for q in m.parameters())
    return float(loss.detach())
def evaluate(arm,m,maps,positions,y=None):
    from data import macro
    m.eval();chunks=[];batch=64 if arm=='timemil' else 20
    with torch.inference_mode():
        for start in range(0,len(positions),batch):
            x=raw(arm,maps,positions[start:start+batch])
            z=m(torch.from_numpy(x)) if arm=='timemil' else m(native.preprocess(x))
            chunks.append(z.numpy())
    scores=np.concatenate(chunks);assert scores.shape==(len(positions),12) and np.isfinite(scores).all()
    if y is None:return scores
    z=torch.from_numpy(scores);target=torch.as_tensor(y,dtype=torch.long)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(z,torch.nn.functional.one_hot(target,12).float()) if arm=='timemil' else torch.nn.functional.cross_entropy(z,target)
    return scores,macro(y,scores.argmax(1)),float(loss)
def record(state,epoch,updates,macro=None,loss=None,scores=None):
    return dict(epoch=epoch,updates=updates,validation_macro=macro,validation_loss=loss,
        validation_logits_sha256=None if scores is None else hashlib.sha256(scores.tobytes()).hexdigest(),
        state_sha256=semantic(state),model_sha256=semantic(state['model']),optimizer_sha256=semantic(state['optimizer']),
        order_rng_sha256=semantic(state['order_rng']),torch_rng_sha256=semantic(state['torch_rng']),
        best_epoch=state['best_epoch'],best_key=None if state['best_key'] is None else list(state['best_key']))
