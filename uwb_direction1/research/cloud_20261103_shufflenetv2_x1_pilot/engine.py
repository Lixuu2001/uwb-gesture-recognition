"""Fixed 40 epoch full-layer learning method; held1 labels are never loaded here."""
import sys,hashlib,json,copy
from pathlib import Path
sys.path.insert(0,'/workspace/uwb-shufflenetv2-x1-source-20261103')
from native_shufflenet import np,torch,make,optimizer,step,preprocess,encoder_digest
def semantic(v):
 h=hashlib.sha256()
 def walk(x):
  if isinstance(x,torch.Tensor):h.update(str((x.shape,x.dtype)).encode());h.update(x.detach().contiguous().numpy().tobytes())
  elif isinstance(x,dict):
   for k,w in x.items():h.update(str(k).encode());walk(w)
  elif isinstance(x,(list,tuple)):
   for w in x:walk(w)
  else:h.update(str(x).encode())
 walk(v);return h.hexdigest()
def raw(maps,ix):
 a=np.asarray(maps[ix]);return np.ascontiguousarray(a.reshape(len(ix),90,3,189).transpose(0,2,1,3),dtype=np.float32)
def evaluate(m,maps,indices):
 m.eval();out=[]
 with torch.inference_mode():
  for start in range(0,len(indices),15):out.append(m(preprocess(raw(maps,indices[start:start+15]))).numpy())
 a=np.concatenate(out);assert a.shape==(len(indices),12) and np.isfinite(a).all();return a
def macro(y,pred,support):
 assert y.shape==pred.shape and set(np.unique(y).tolist())==set(support)
 x=float(np.mean([np.mean(pred[y==k]==k) for k in np.unique(y)]));assert np.isfinite(x);return x
def atomic_checkpoint(path,payload):
 q=path.with_suffix(path.suffix+'.tmp');assert not q.exists();torch.save(payload,q);q.replace(path)
def run_training(maps,loc,y,v,on_step,on_epoch):
 m=make();opt=optimizer(m);orderg=torch.Generator().manual_seed(20261029);augg=torch.Generator().manual_seed(20261029);best=-1.;best_epoch=None;updates=0
 initial_model_sha=encoder_digest(m);best_state=None
 for epoch in range(1,41):
  m.train();order=torch.randperm(4700,generator=orderg).numpy();assert len(np.unique(order))==4700
  for offset in range(0,4700,15):
   positions=order[offset:offset+15];loss=step(m,opt,raw(maps,loc['train'][positions]),y[positions],augg);updates+=1
   on_step(dict(epoch=epoch,offset=offset,update=updates,order_sha256=hashlib.sha256(positions.tobytes()).hexdigest(),augmentation_rng_sha256=semantic(augg.get_state()),loss=loss))
  scores=evaluate(m,maps,loc['validation']);acc=macro(v,scores.argmax(1),range(10));improved=acc>best
  if improved:best=acc;best_epoch=epoch;best_state=copy.deepcopy(m.state_dict())
  state=dict(model=m.state_dict(),optimizer=opt.state_dict(),order_rng=orderg.get_state(),augmentation_rng=augg.get_state(),torch_rng=torch.get_rng_state(),epoch=epoch,updates=updates)
  record=dict(epoch=epoch,updates=updates,validation_macro=acc,selected_epoch=best_epoch,best_validation_macro=best,model_state_sha256=encoder_digest(m),optimizer_state_sha256=semantic(opt.state_dict()),order_rng_sha256=semantic(orderg.get_state()),augmentation_rng_sha256=semantic(augg.get_state()),torch_rng_sha256=semantic(torch.get_rng_state()))
  on_epoch(record,scores,state,best_state,improved)
 assert updates==12560 and best_epoch is not None
 return m,opt,best_state,best_epoch,best,initial_model_sha
