"""Published SWAD LossValley default3/6/.3 dense native component transfer on complete native GAP backbone."""
import sys,random,hashlib,os,copy
from common import B,PHASE,protocol
sys.path.insert(0,str(PHASE))
from swad_core import NativeSWADState
from bn_calibration import calibrate
c=protocol();sys.path.insert(0,c['native_source'])
import adapter as native
import torch
import numpy as np
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
def make():
    random.seed(c['seed']);np.random.seed(c['seed'])
    m=native.make_model('gap',c['seed']);assert sum(p.numel() for p in m.parameters())==617612
    return m,NativeOptimizerAndSWAD(m),torch.Generator().manual_seed(c['seed'])
class NativeOptimizerAndSWAD:
    def __init__(self,m):
        self.model=m;self.native=native.StableLookahead(m.parameters(),lr=.001,weight_decay=.0001);self.swad=NativeSWADState(m)
    def zero_grad(self):self.native.zero_grad()
    def step(self,epoch):
        assert self.native.count==self.swad.updates
        self.native.step();self.swad.post_step(self.model,self.native.count,epoch)
    def epoch_end(self,epoch,vm,vl):self.swad.epoch_end(self.model,epoch,vm,vl)
    def state_dict(self):return dict(native_optimizer=self.native.state_dict(),swad=self.swad.state_dict())
    def load_state_dict(self,state):
        self.native=native.StableLookahead(self.model.parameters(),lr=.001,weight_decay=.0001);self.native.load_state_dict(copy.deepcopy(state['native_optimizer']));self.swad.load_state_dict(copy.deepcopy(state['swad']),self.model)
        assert self.native.count==self.swad.updates
STEP_DIGEST_COST=[]
EVAL_BATCH_COST=[]
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
    assert opt.swad.updates==opt.native.count==updates and opt.swad.last_epoch==epoch
    n=np.random.get_state()
    return dict(model=m.state_dict(),optimizer=opt.state_dict(),order_rng=g.get_state(),torch_rng=torch.get_rng_state(),
        python_rng=random.getstate(),numpy_rng=(n[0],torch.from_numpy(n[1].copy()),n[2],n[3],n[4]),
        epoch=epoch,updates=updates,best_key=best_key,best_model=best_model,best_epoch=best_epoch,model_training=m.training)
def restore(state,m,opt,g):
    m.load_state_dict(state['model'],strict=True);opt.load_state_dict(copy.deepcopy(state['optimizer']));assert opt.swad.updates==opt.native.count==state['updates'] and opt.swad.last_epoch==state['epoch'];g.set_state(state['order_rng'])
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
    import time
    assert m.training and set(np.asarray(domains).tolist())<=set(c['train_users'])
    labels=torch.as_tensor(y,dtype=torch.long);logits=m(torch.from_numpy(x));count=opt.native.count
    loss=native_bce(logits,labels);assert logits.shape==(len(x),12) and torch.isfinite(loss)
    opt.zero_grad();loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    torch.nn.utils.clip_grad_norm_(m.parameters(),2.);opt.step(epoch);assert all(torch.isfinite(p).all() for p in m.parameters())
    wt=time.monotonic();ct=time.process_time();seg=semantic(opt.swad.segment.module.state_dict());tail=semantic(opt.swad.tail.module.state_dict()) if opt.swad.tail.n_averaged.item() else None
    STEP_DIGEST_COST.append(dict(wall=time.monotonic()-wt,CPU=time.process_time()-ct))
    return dict(loss=float(loss.detach()),global_BCE=float(loss.detach()),completed_updates_before=count,completed_updates_after=opt.native.count,segment_n=int(opt.swad.segment.n_averaged),segment_start=opt.swad.segment.start_step,segment_end=opt.swad.segment.end_step,segment_model_sha256=seg,tail_n=int(opt.swad.tail.n_averaged),tail_model_sha256=tail,logits_sha256=semantic(logits),forward_batch=len(x))

def macro(y,p):return float(np.mean([np.mean(p[y==k]==k) for k in np.unique(y)]))
def evaluate(m,maps,positions,y=None):
    m.eval();chunks=[]
    with torch.inference_mode():
        for i in range(0,len(positions),c['batch']):
            import time
            xx=torch.from_numpy(np.array(maps[positions[i:i+c['batch']]],copy=True));wt=time.monotonic();ct=time.process_time();chunks.append(m(xx).numpy());EVAL_BATCH_COST.append(dict(n=len(xx),wall=time.monotonic()-wt,CPU=time.process_time()-ct))
    z=np.concatenate(chunks);assert z.shape==(len(positions),12) and np.isfinite(z).all()
    if y is None:return z
    bce=torch.nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(z),torch.nn.functional.one_hot(torch.as_tensor(y,dtype=torch.long),12).float())
    return z,macro(y,z.argmax(1)),float(bce)
def record(state,epoch,updates,macro=None,loss=None,scores=None):
    return dict(epoch=epoch,updates=updates,validation_macro=macro,validation_loss=loss,validation_logits_sha256=None if scores is None else hashlib.sha256(scores.tobytes()).hexdigest(),state_sha256=semantic(state),model_sha256=semantic(state['model']),optimizer_sha256=semantic(state['optimizer']),native_reference_best_epoch=state['best_epoch'],swad_converge_step=state['optimizer']['swad']['converge_step'],swad_dead=state['optimizer']['swad']['dead_valley'],best_key=None if state['best_key'] is None else list(state['best_key']))

def legacy_checkpoint(path,expected_sha256):
    from common import sha
    assert sha(path)==expected_sha256
    unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(path);assert set(unsafe)<={'torch.torch_version.TorchVersion'}
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):return torch.load(path,map_location='cpu',weights_only=True)

def make_outputs(m,opt,ids,users,loc,maps,best_model,best_epoch):
    before=semantic(dict(model=m.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state()))
    method=copy.deepcopy(opt.swad);swad=method.finalize();tail=copy.deepcopy(opt.swad.tail)
    assert int(tail.n_averaged)==32*(opt.native.count//64) and tail.start_step==32*(opt.native.count//64)+1 and tail.end_step==opt.native.count
    ranges=dict(swad=dict(start_step=swad.start_step,end_step=swad.end_step,n_averaged=int(swad.n_averaged),converge_step=method.valley.converge_step,dead_valley=method.valley.dead_valley),tail=dict(start_step=tail.start_step,end_step=tail.end_step,n_averaged=int(tail.n_averaged)),bn_only=dict(native_reference_selected_epoch=best_epoch),native_reference=dict(native_reference_selected_epoch=best_epoch))
    states=dict(swad=swad.module.state_dict(),tail=tail.module.state_dict(),bn_only=best_model,native_reference=best_model);result={};costs={}
    for arm,state in states.items():
        model=copy.deepcopy(m);model.load_state_dict(state,strict=True);model.eval();meta=dict(arm=arm,selection=ranges[arm])
        if arm!='native_reference':
            cal,cost=calibrate(model,maps,loc['train'],ids,users,c['train_users'],c['bn_seed'],c['bn_batch']);meta['calibration']=cal;costs[arm]=cost
        z={role:evaluate(model,maps,loc[role]) for role in ['validation','diagnostic']}
        result[arm]=dict(model=copy.deepcopy(model.state_dict()),metadata=meta,scores=z)
    assert before==semantic(dict(model=m.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state()))
    return result,costs
