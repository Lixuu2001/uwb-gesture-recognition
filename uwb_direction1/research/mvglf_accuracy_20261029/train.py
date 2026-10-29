"""Fixed source-user pilot, two arms, checkpoint selection on validation only."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='3'
import sys,json,time,hashlib,signal,resource,fcntl,traceback,platform,random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import MVGLF,preprocess
B=Path(__file__).resolve().parent;O=B/'run01';O.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for v in iter(lambda:f.read(4*1024*1024),b''):h.update(v)
 return h.hexdigest()
def save(p,z):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(z,indent=2));t.replace(p)
def checkpoint(p,z):
 t=p.with_suffix('.tmp');torch.save(z,t);os.replace(t,p)
def macro(y,p):return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)]))
class BudgetStop(Exception):pass
def main():
 lock=open(O/'worker.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cfg=json.loads((B/'PROTOCOL.json').read_text());assert cfg['cost_gate_passed'];torch.set_num_threads(3);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 start=time.monotonic();prior=json.loads((O/'STATE.json').read_text()) if (O/'STATE.json').exists() else {};resume='--resume' in sys.argv
 if prior and prior.get('state')=='complete':raise RuntimeError('already complete; do not restart')
 if prior and not resume:raise RuntimeError('existing run: inspect state and use explicit --resume if justified')
 offsets=(float(prior.get('cumulative_wall_seconds',0)),float(prior.get('cumulative_cpu_seconds',0)))
 remaining_wall=cfg['wall_budget_seconds']-offsets[0];remaining_cpu=cfg['cpu_budget_seconds']-offsets[1]
 if remaining_wall<60 or remaining_cpu<180:raise BudgetStop('cumulative budget exhausted')
 def stop(sig,frame):raise BudgetStop(f'signal {sig}')
 for sig in [signal.SIGTERM,signal.SIGALRM,signal.SIGXCPU]:signal.signal(sig,stop)
 resource.setrlimit(resource.RLIMIT_CPU,(int(remaining_cpu)-30,int(remaining_cpu)));signal.setitimer(signal.ITIMER_REAL,remaining_wall)
 state=dict(state='initializing',pid=os.getpid(),resume=resume,completed_arms=prior.get('completed_arms',[]),held_user=1,validation_user=2)
 def record(**kw):
  state.update(kw);state['cumulative_wall_seconds']=offsets[0]+time.monotonic()-start;state['cumulative_cpu_seconds']=offsets[1]+time.process_time();state['observed_unix']=time.time();save(O/'STATE.json',state)
 try:
  identity=dict(protocol_sha256=sha(B/'PROTOCOL.json'),model_sha256=sha(B/'model.py'),runner_sha256=sha(Path(__file__)),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__)
  if (O/'IDENTITY.json').exists():assert json.loads((O/'IDENTITY.json').read_text())==identity
  else:save(O/'IDENTITY.json',identity)
  record()
  root=Path(cfg['data']);observed={n:sha(root/'cache'/n) for n in cfg['input_sha256']};assert observed==cfg['input_sha256'];save(O/'INPUT_HASHES.json',observed)
  with np.load(cfg['source_manifest']) as z:ids=z['ids'];u=z['users']
  assert sha(cfg['source_manifest'])==cfg['source_manifest_sha256'];assert len(ids)==6900 and set(u)==set(range(1,7))
  tr=np.flatnonzero((u!=1)&(u!=2));va=np.flatnonzero(u==2);te=np.flatnonzero(u==1);assert (len(tr),len(va),len(te))==(4700,1000,1200)
  save(O/'split.json',dict(train=ids[tr].tolist(),validation=ids[va].tolist(),diagnostic=ids[te].tolist()))
  assert not(set(ids[tr])&set(ids[va]) or set(ids[tr])&set(ids[te]) or set(ids[te])&set(ids[va]))
  labels=np.load(root/'cache/y_clutterremoved.npy',mmap_mode='r');ytr=torch.from_numpy(np.asarray(labels[ids[tr]]).copy()).long();yva=np.asarray(labels[ids[va]]).copy()
  # Diagnostic labels are not indexed before both arms finish and predictions are frozen.
  record(state='preparing_source_maps',training_labels_read=True,validation_labels_read=True,diagnostic_labels_read=False,users7_8_labels_used=False)
  cache=O/'source_maps.npy'
  if not cache.exists():
   raw=np.load(root/'cache/X_clutterremoved.npy',mmap_mode='r');tmp=cache.with_suffix('.partial.npy');z=np.lib.format.open_memmap(tmp,mode='w+',dtype='float32',shape=(6900,3,64,128))
   with torch.no_grad():
    for b in range(0,6900,64):z[b:b+64]=preprocess(torch.from_numpy(np.array(raw[ids[b:b+64]],copy=True))).numpy()
   z.flush();del z;os.replace(tmp,cache)
  data=torch.from_numpy(np.array(np.load(cache,mmap_mode='r'),copy=True));assert torch.isfinite(data).all();cachehash=sha(cache);save(O/'SOURCE_CACHE.json',dict(sha256=cachehash,shape=list(data.shape),source_only=True))
  def batch(ix):return data[ix].contiguous(memory_format=torch.channels_last)
  def evaluate(model,ix,y=None):
   model.eval();pp=[]
   with torch.no_grad():
    for b in range(0,len(ix),64):pp.append(model(batch(ix[b:b+64])).cpu())
   logits=torch.cat(pp)
   if y is None:return logits
   return logits,macro(y,logits.argmax(1).numpy()),float(F.cross_entropy(logits,torch.from_numpy(y).long()))
  # Dedicated epoch order RNG means the two arms see exactly the same permutations.
  for arm in cfg['arms']:
   d=O/arm;d.mkdir(exist_ok=True)
   if (d/'ARM_COMPLETE.json').exists():continue
   torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed']);m=MVGLF(arm).to(memory_format=torch.channels_last)
   opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);first=1;best=(-1.,-float('inf'),-1)
   if (d/'last.pt').exists():
    z=torch.load(d/'last.pt',map_location='cpu',weights_only=False);assert z['identity']==identity;m.load_state_dict(z['model']);opt.load_state_dict(z['optimizer']);first=z['epoch']+1;best=tuple(z['best']);torch.set_rng_state(z['torch_rng']);np.random.set_state(z['numpy_rng']);random.setstate(z['python_rng'])
   else:checkpoint(d/'initial.pt',dict(model=m.state_dict(),identity=identity,epoch=0))
   for epoch in range(first,cfg['epochs']+1):
    m.train();order=torch.randperm(len(tr),generator=torch.Generator().manual_seed(cfg['seed']+epoch)).numpy();total=0.;epstart=time.monotonic()
    record(state='training',arm=arm,epoch=epoch,batch=0,training_started=True,diagnostic_labels_read=False)
    for b in range(0,len(tr),cfg['batch_size']):
     ji=order[b:b+cfg['batch_size']];opt.zero_grad(set_to_none=True);p=m(batch(tr[ji]));loss=F.cross_entropy(p,ytr[ji]);assert torch.isfinite(loss),'nonfinite training loss';loss.backward()
     if b==0:assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters()),'invalid gradients'
     opt.step();total+=float(loss.detach())*len(ji)
     if not (d/'first_update.pt').exists():
      checkpoint(d/'first_update.pt',dict(model=m.state_dict(),optimizer=opt.state_dict(),epoch=epoch,batch=1,training_ids=ids[tr[ji]].tolist(),identity=identity));record(first_update_sha256=sha(d/'first_update.pt'))
     if b%(cfg['batch_size']*10)==0:record(batch=b//cfg['batch_size']+1,train_loss_partial=total/max(b+len(ji),1))
    _,vm,vl=evaluate(m,va,yva);key=(vm,-vl,-epoch)
    if key>best:best=key;checkpoint(d/'best.pt',dict(model=m.state_dict(),epoch=epoch,validation_macro=vm,validation_loss=vl,identity=identity))
    checkpoint(d/'last.pt',dict(model=m.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=list(best),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),identity=identity))
    row=dict(arm=arm,epoch=epoch,train_loss=total/len(tr),validation_macro=vm,validation_loss=vl,best_validation_macro=best[0],epoch_wall_seconds=time.monotonic()-epstart)
    with open(d/'epochs.jsonl','a') as f:f.write(json.dumps(row)+'\n')
    record(state='epoch_complete',last_checkpoint=str(d/'last.pt'),last_checkpoint_sha256=sha(d/'last.pt'),**row);print(json.dumps(row),flush=True)
   z=torch.load(d/'best.pt',map_location='cpu',weights_only=False);m.load_state_dict(z['model']);logits=evaluate(m,te);np.savez_compressed(d/'predictions_unscored.npz',ids=ids[te],logits=logits.numpy());save(d/'ARM_COMPLETE.json',dict(epochs=cfg['epochs'],selected_epoch=z['epoch'],best_validation_macro=z['validation_macro'],checkpoint_sha256=sha(d/'best.pt'),prediction_sha256=sha(d/'predictions_unscored.npz')));record(completed_arms=state['completed_arms']+[arm]);del m,opt
  record(state='predictions_frozen',prediction_hashes={a:sha(O/a/'predictions_unscored.npz') for a in cfg['arms']},diagnostic_labels_read=False)
  yt=np.asarray(labels[ids[te]]).copy();scores={};details={}
  for arm in cfg['arms']:
   with np.load(O/arm/'predictions_unscored.npz') as z:np.testing.assert_array_equal(z['ids'],ids[te]);p=z['logits'].argmax(1);scores[arm]=macro(yt,p);details[arm]={str(c):float(np.mean(p[yt==c]==c)) for c in np.unique(yt)}
  with np.load(cfg['minirocket_reference']) as z:
   mask=z['users']==1;np.testing.assert_array_equal(z['ids'][mask],ids[te]);np.testing.assert_array_equal(z['y'][mask],yt);scores['minirocket']=macro(yt,z['minirocket'][mask])
  summary=dict(state='complete_source_pilot',scores=scores,per_class_recall=details,full_gate_passed=bool(scores['full']-scores['local']>=.005 and scores['full']>=scores['minirocket']),local_baseline_expansion_eligible=bool(scores['local']>=scores['minirocket']+.01),independent_validation=False,innovation_claim=False,held_user=1,training_users=[3,4,5,6],validation_user=2,users7_8_labels_used=False,epochs=cfg['epochs'],cumulative_wall_seconds=offsets[0]+time.monotonic()-start,cumulative_cpu_seconds=offsets[1]+time.process_time())
  save(O/'SUMMARY.json',summary);record(state='complete',pid=None,diagnostic_labels_read=True);save(O/'FILES.json',{str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file() and p.suffix!='.tmp'});print(json.dumps(summary),flush=True)
 except BaseException as e:
  record(state='incomplete',pid=None,error=repr(e),traceback=traceback.format_exc(),scientific_negative=False);raise
if __name__=='__main__':main()
