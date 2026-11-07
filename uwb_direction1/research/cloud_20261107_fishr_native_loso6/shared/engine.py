"""Published Fishr defaults on complete native GAP and global BCE, all native head gradients."""
import sys,random,hashlib,os,copy
from common import B,PHASE,protocol
sys.path.insert(0,str(PHASE))
from fishr_core import native_head_gradients,PublishedEMA,variance_penalty
c=protocol();sys.path.insert(0,c['native_source'])
import adapter as native
import torch
import numpy as np
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def make():
    random.seed(c['seed']);np.random.seed(c['seed'])
    m=native.make_model('gap',c['seed']);assert sum(p.numel() for p in m.parameters())==617612
    return m,ScheduledNativeOptimizer(m.parameters()),torch.Generator().manual_seed(c['seed'])
def native_bce(logits,labels):
    target=torch.nn.functional.one_hot(labels,12).to(dtype=logits.dtype)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
# Whole native optimizer and all Fishr state are checkpointed together.
class ScheduledNativeOptimizer:
    def __init__(self,parameters):
        self.params=list(parameters);self.inner=self.fresh();self.algorithm_updates=0;self.reset_events=[]
        self.ema=PublishedEMA(4,c['fishr_ema'])
    def fresh(self):return native.StableLookahead(self.params,lr=.001,weight_decay=.0001)
    @property
    def count(self):return self.inner.count
    @property
    def slow(self):return self.inner.slow
    def zero_grad(self):self.inner.zero_grad()
    def penalty_weight(self):return c['fishr_lambda'] if self.algorithm_updates>=c['fishr_penalty_anneal_iters'] else c['fishr_warmup_penalty_weight']
    def reset_if_due(self):
        boundary=c['fishr_reset_at_completed_updates']
        if self.algorithm_updates==boundary and not self.reset_events:
            self.inner=self.fresh();self.reset_events=[boundary];return True
        assert self.algorithm_updates<boundary or self.reset_events==[boundary]
        return False
    def step(self):
        assert self.algorithm_updates<c['fishr_reset_at_completed_updates'] or self.reset_events==[c['fishr_reset_at_completed_updates']]
        self.inner.step();self.algorithm_updates+=1
    def state_dict(self):return dict(native_optimizer=self.inner.state_dict(),algorithm_updates=self.algorithm_updates,reset_events=list(self.reset_events),fishr_EMA=self.ema.state_dict())
    def load_state_dict(self,state):
        n=state['algorithm_updates'];events=state['reset_events'];boundary=c['fishr_reset_at_completed_updates']
        valid=events==[] if n<boundary else events in [[],[boundary]] if n==boundary else events==[boundary]
        assert isinstance(n,int) and n>=0 and valid
        self.inner=self.fresh();self.inner.load_state_dict(copy.deepcopy(state['native_optimizer']));self.algorithm_updates=n;self.reset_events=list(events)
        self.ema.load_state_dict(state['fishr_EMA'])
        assert self.count==(n-boundary if events else n)
def local_domains(domains,n):
    dom=torch.as_tensor(domains,dtype=torch.long)
    assert dom.shape==(n,) and set(dom.tolist())<=set(c['train_users'])
    return torch.as_tensor([c['train_users'].index(int(u)) for u in dom],dtype=torch.long)

def native_features_logits(m,x):
    features=m.feature_extractor(x.transpose(1,2)).mean(-1)
    return features,m.head(features)
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
    assert opt.algorithm_updates==updates
    n=np.random.get_state()
    return dict(model=m.state_dict(),optimizer=opt.state_dict(),order_rng=g.get_state(),torch_rng=torch.get_rng_state(),
        python_rng=random.getstate(),numpy_rng=(n[0],torch.from_numpy(n[1].copy()),n[2],n[3],n[4]),
        epoch=epoch,updates=updates,best_key=best_key,best_model=best_model,best_epoch=best_epoch,model_training=m.training)
def restore(state,m,opt,g):
    m.load_state_dict(state['model'],strict=True);opt.load_state_dict(copy.deepcopy(state['optimizer']));assert opt.algorithm_updates==state['updates'];g.set_state(state['order_rng'])
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
    xx=torch.from_numpy(x);labels=torch.as_tensor(y,dtype=torch.long)
    features=m.feature_extractor(xx.transpose(1,2)).mean(-1)
    logits,grads=native_head_gradients(m.head,features,labels)
    groups=local_domains(domains,len(x));count=opt.algorithm_updates;weight=opt.penalty_weight()
    classification=native_bce(logits,labels);penalty=variance_penalty(grads,groups,opt.ema)
    # Keep native warmup update bitwise; no 0*penalty branch in backward.
    loss=classification if weight==0. else classification+weight*penalty
    assert logits.shape==(len(x),12) and torch.isfinite(loss) and torch.isfinite(penalty)
    did_reset=opt.reset_if_due();opt.zero_grad()
    loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),2.);opt.step();assert all(torch.isfinite(p).all() for p in m.parameters())
    return dict(loss=float(loss.detach()),classification_loss=float(classification.detach()),global_BCE=float(classification.detach()),penalty=float(penalty.detach()),fishr_penalty_weight=weight,completed_updates_before=count,completed_updates_after=opt.algorithm_updates,optimizer_reset_this_update=did_reset,optimizer_reset_events=list(opt.reset_events),valid_training_domains=len(groups.unique()),fishr_EMA_updates=list(opt.ema.updates),fishr_EMA_sha256=semantic(opt.ema.state_dict()),classifier_gradient_coordinates=18060,preclip_gradient_norm=float(norm),logits_sha256=semantic(logits),forward_batch=len(x))

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
