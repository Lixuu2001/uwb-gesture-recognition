"""Published GroupDRO fixedeta.01 component transfer on complete native GAP backbone."""
import sys,random,hashlib,os,copy
from common import B,PHASE,protocol
sys.path.insert(0,str(PHASE))
from groupdro_core import NativeGroupDROOptimizer
from groupdro_reference import reference as reference_objective
c=protocol();sys.path.insert(0,c['native_source'])
import adapter as native
import torch
import numpy as np
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def make():
    random.seed(c['seed']);np.random.seed(c['seed'])
    m=native.make_model('gap',c['seed']);assert sum(p.numel() for p in m.parameters())==617612
    return m,NativeGroupDROOptimizer(m.parameters(),c['train_users'],lambda ps:native.StableLookahead(ps,lr=.001,weight_decay=.0001)),torch.Generator().manual_seed(c['seed'])
def native_bce(logits,labels):
    target=torch.nn.functional.one_hot(labels,12).to(dtype=logits.dtype)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
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
    assert opt.robust.updates==opt.native.count==updates
    n=np.random.get_state()
    return dict(model=m.state_dict(),optimizer=opt.state_dict(),order_rng=g.get_state(),torch_rng=torch.get_rng_state(),
        python_rng=random.getstate(),numpy_rng=(n[0],torch.from_numpy(n[1].copy()),n[2],n[3],n[4]),
        epoch=epoch,updates=updates,best_key=best_key,best_model=best_model,best_epoch=best_epoch,model_training=m.training)
def restore(state,m,opt,g):
    m.load_state_dict(state['model'],strict=True);opt.load_state_dict(copy.deepcopy(state['optimizer']));assert opt.robust.updates==opt.native.count==state['updates'];g.set_state(state['order_rng'])
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
def step(m,opt,x,y,epoch,domains):
    assert m.training
    labels=torch.as_tensor(y,dtype=torch.long);logits=m(torch.from_numpy(x));count=opt.robust.updates
    loss,global_bce,details=opt.robust.objective(logits,labels,domains)
    assert logits.shape==(len(x),12) and torch.isfinite(loss)
    opt.zero_grad();loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    torch.nn.utils.clip_grad_norm_(m.parameters(),2.);opt.step();assert all(torch.isfinite(p).all() for p in m.parameters())
    return dict(loss=float(loss.detach()),weighted_source_BCE=float(loss.detach()),global_BCE=float(global_bce.detach()),completed_updates_before=count,completed_updates_after=opt.robust.updates,group_users=list(details['domain_users']),domain_counts=details['domain_counts'],source_risks=details['source_risks'].tolist(),group_q=details['q'].tolist(),present=details['present'].tolist(),logits_sha256=semantic(logits),forward_batch=len(x))

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
