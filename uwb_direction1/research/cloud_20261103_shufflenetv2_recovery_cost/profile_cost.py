from pathlib import Path
import sys,time,json,resource,shutil,hashlib,gc
from guard import B,ready,sha
from cost_math import select
def digest(v):
 import torch
 h=hashlib.sha256()
 def walk(x):
  if isinstance(x,torch.Tensor):h.update(str((x.shape,x.dtype)).encode());h.update(x.detach().contiguous().numpy().tobytes())
  elif isinstance(x,dict):
   for k,w in x.items():h.update(str(k).encode());walk(w)
  elif isinstance(x,(list,tuple)):
   for w in x:walk(w)
  else:h.update(str(x).encode())
 walk(v);return h.hexdigest()
def main():
 p=ready();P=B/'profile01';P.mkdir(exist_ok=False);(P/'DECLARATION_BEFORE_TIMING.json').write_text(json.dumps(dict(policy_sha256=sha(B/'POLICY.json'),cost_hashes_sha256=sha(B/'COST_HASHES.json'),full_network_B15_all_layers=True,actual_UWB_values_labels_fit=False),indent=2)+'\n')
 sys.path.insert(0,p['source']);from native_shufflenet import np,torch,make,optimizer,step,preprocess
 rng=np.random.default_rng(20261029);m=make();opt=optimizer(m);g=torch.Generator().manual_seed(20261029);steps=[];last=[];ev=[];saves=[];loads=[];sizes=[];deleted=[]
 for n in [15]*6+[5]:
  x=rng.normal(size=(n,3,90,189)).astype(np.float32);y=np.arange(n)%12;t=time.monotonic();loss=step(m,opt,x,y,g);dt=time.monotonic()-t
  (steps if n==15 else last).append(dt);print(json.dumps(dict(stage='complete_full_native_SGD_update',batch=n,seconds=dt,synthetic_loss=loss)),flush=True)
 m.eval();x=rng.normal(size=(15,3,90,189)).astype(np.float32)
 for i in range(3):
  t=time.monotonic()
  with torch.inference_mode():scores=m(preprocess(x))
  assert scores.shape==(15,12) and torch.isfinite(scores).all();ev.append(time.monotonic()-t)
 payload=dict(model=m.state_dict(),optimizer=opt.state_dict(),augmentation_generator=g.get_state(),torch_rng=torch.get_rng_state(),epoch=0,updates=7)
 expected=digest(payload)
 for i in range(3):
  path=P/f'disposable_full_checkpoint_{i}.pt';t=time.monotonic();torch.save(payload,path);saves.append(time.monotonic()-t);sizes.append(path.stat().st_size);t=time.monotonic();loaded=torch.load(path,map_location='cpu',weights_only=True);loads.append(time.monotonic()-t);assert digest(loaded)==expected
  deleted.append(dict(name=path.name,bytes=path.stat().st_size,sha256=sha(path),full_state_exact=True));del loaded;gc.collect();path.unlink()
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;free=shutil.disk_usage('/workspace').free;r=select(p,steps,last,ev,saves,loads,max(sizes),rss,free)
 r.update(state='complete_full_ShuffleNetV2_x1_alllayer_SGDM40_B15_synthetic_cost',full_B15_update_seconds=steps,last_B5_update_seconds=last,evaluation_B15_seconds=ev,full_checkpoint_save_seconds=saves,full_checkpoint_load_seconds=loads,peak_RSS_KiB=rss,disposable_checkpoint_records=deleted,model_optimizer_momentum_RNG_checkpoint_exact=True,no_actual_UWB_values_labels_read=True,full40_epochs_not_run=True)
 (P/'COST_RESULT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
