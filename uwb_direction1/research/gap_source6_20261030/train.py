"""Fixed source-sixfold known GAP baseline; no automatic invocation or retries."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import fcntl,hashlib,json,math,random,resource,signal,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,'/workspace/uwb-timemil-20261029')
from adapter import make_model,StableLookahead
B=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
 return h.hexdigest()
def write(p,v):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
def save(p,v):
 p=Path(p);q=p.with_suffix('.tmp');torch.save(v,q);q.replace(p)
def macro(y,p): return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)]))
def fold_indices(users,held):
 val=held%6+1
 return val,np.flatnonzero((users!=held)&(users!=val)),np.flatnonzero(users==val),np.flatnonzero(users==held)
def gate(gains,cfg):
 g=cfg['gate'];return bool(np.mean(gains)>=g['mean_gap_minus_localcnn_minimum'] and sum(x>0 for x in gains)>=g['positive_users_minimum'] and min(gains)>=g['worst_user_difference_minimum'])
def require_freeze(out,cfg):
 frozen=json.loads((out/'ALL_NEW_PREDICTIONS_FROZEN.json').read_text())
 assert set(frozen)==set(map(str,cfg['new_held_users']))
 for h in cfg['new_held_users']: assert sha(out/f'held{h}/predictions_unscored.npz')==frozen[str(h)]
def main():
 cfg=json.loads((B/'PROTOCOL.json').read_text())
 assert json.loads(Path(cfg['upstream_state']).read_text())['state']=='complete'
 assert json.loads(Path(cfg['upstream_summary']).read_text())['gap_baseline_expansion_eligible']
 assert json.loads(Path(cfg['upstream_private_backup']).read_text())['state']=='private_readback_verified'
 for k in ('upstream_audit','localcnn_audit'): assert json.loads(Path(cfg[k]).read_text())['state']=='pass'
 out=B/'run01';out.mkdir(exist_ok=False);lock=open(out/'worker.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 start=time.monotonic();state=dict(state='starting',pid=os.getpid(),completed_folds=[1],all_new_diagnostic_predictions_frozen=False,users7_8_labels_used=False)
 def record(**kw):
  state.update(kw);state.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time(),observed_unix=time.time());write(out/'STATE.json',state)
 def stop(sig,frame): raise TimeoutError('budget signal '+str(sig))
 for sig in (signal.SIGTERM,signal.SIGALRM,signal.SIGXCPU): signal.signal(sig,stop)
 resource.setrlimit(resource.RLIMIT_CPU,(43000,43100));signal.setitimer(signal.ITIMER_REAL,43100);record()
 try:
  frozen=json.loads((B/'FROZEN_HASHES.json').read_text())
  for p,h in frozen.items(): assert sha(B/p)==h,p
  identity=dict(protocol_sha256=sha(B/'PROTOCOL.json'),frozen_hashes=frozen,torch=torch.__version__,numpy=np.__version__,python=sys.version);write(out/'IDENTITY.json',identity)
  assert cfg['cost_gate_passed'] and sha(cfg['source_manifest'])==cfg['source_manifest_sha256']
  with np.load(cfg['source_manifest']) as z: ids=z['ids'];users=z['users']
  assert len(ids)==len(np.unique(ids))==6900 and set(users)==set(range(1,7))
  for n,h in cfg['input_sha256'].items(): assert sha(Path(cfg['data'])/'cache'/n)==h
  assert sha(cfg['source_maps'])==cfg['source_maps_sha256']
  data=torch.from_numpy(np.array(np.load(cfg['source_maps'],mmap_mode='r'),copy=True));assert data.shape==(6900,90,567) and torch.isfinite(data).all()
  labels=np.load(Path(cfg['data'])/'cache/y_clutterremoved.npy',mmap_mode='r')
  write(out/'REUSED_HELD1.json',cfg['reused_held1'])
  def evaluate(model,ix,y=None):
   model.eval();chunks=[]
   with torch.no_grad():
    for b in range(0,len(ix),64): chunks.append(model(data[ix[b:b+64]]))
   logits=torch.cat(chunks);assert torch.isfinite(logits).all()
   if y is None: return logits
   return macro(y,logits.argmax(1).numpy()),float(F.binary_cross_entropy_with_logits(logits,F.one_hot(torch.from_numpy(y).long(),12).float()))
  for held in cfg['new_held_users']:
   val,tr,va,te=fold_indices(users,held);assert [len(tr),len(va),len(te)]==cfg['counts'][str(held)]
   d=out/f'held{held}';d.mkdir();write(d/'split.json',dict(train=ids[tr].tolist(),validation=ids[va].tolist(),diagnostic=ids[te].tolist()))
   ytr=torch.from_numpy(np.array(labels[ids[tr]],copy=True)).long();target=F.one_hot(ytr,12).float();yv=np.array(labels[ids[va]],copy=True)
   random.seed(cfg['seed']);np.random.seed(cfg['seed']);model=make_model('gap',cfg['seed']);assert sum(p.numel() for p in model.parameters())==617612
   opt=StableLookahead(model.parameters(),lr=.001,weight_decay=.0001);best=(-1.,-float('inf'),-1)
   save(d/'initial.pt',dict(model=model.state_dict(),identity=identity,epoch=0,held=held))
   for epoch in range(1,cfg['epochs']+1):
    model.train();order=torch.randperm(len(tr),generator=torch.Generator().manual_seed(cfg['seed']+epoch)).numpy();total=0.;epstart=time.monotonic();record(state='training',held_user=held,validation_user=val,epoch=epoch,batch=0)
    for b in range(0,len(tr),64):
     ji=order[b:b+64];opt.zero_grad();logits=model(data[tr[ji]],warmup=epoch<10);loss=F.binary_cross_entropy_with_logits(logits,target[ji]);assert torch.isfinite(loss);loss.backward()
     if b==0: assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
     torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();total+=float(loss.detach())*len(ji)
     if not (d/'first_update.pt').exists(): save(d/'first_update.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),training_ids=ids[tr[ji]].tolist(),held=held,identity=identity))
     if b%(64*10)==0: record(batch=b//64+1)
    vm,vl=evaluate(model,va,yv);key=(vm,-vl,-epoch)
    if key>best: best=key;save(d/'best.pt',dict(model=model.state_dict(),epoch=epoch,validation_macro=vm,validation_loss=vl,identity=identity,held=held))
    save(d/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=list(best),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),identity=identity,held=held))
    row=dict(held=held,epoch=epoch,train_bce=total/len(tr),validation_macro=vm,validation_loss=vl,epoch_seconds=time.monotonic()-epstart)
    with (d/'epochs.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    record(state='epoch_complete',**row);print(json.dumps(row),flush=True)
   best_state=torch.load(d/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(best_state['model']);pred=evaluate(model,te)
   np.savez_compressed(d/'predictions_unscored.npz',ids=ids[te],logits=pred.numpy())
   write(d/'FOLD_COMPLETE.json',dict(held=held,epochs=cfg['epochs'],selected_epoch=best_state['epoch'],checkpoint_sha256=sha(d/'best.pt'),prediction_sha256=sha(d/'predictions_unscored.npz')))
   record(completed_folds=state['completed_folds']+[held]);del model,opt
  write(out/'ALL_NEW_PREDICTIONS_FROZEN.json',{str(h):sha(out/f'held{h}/predictions_unscored.npz') for h in cfg['new_held_users']});require_freeze(out,cfg);record(state='predictions_frozen',all_new_diagnostic_predictions_frozen=True)
  rows=[]
  for held in range(1,7):
   te=np.flatnonzero(users==held);yt=np.array(labels[ids[te]],copy=True);path=Path(cfg['reused_held1']['prediction']) if held==1 else out/f'held{held}/predictions_unscored.npz'
   with np.load(path) as z: np.testing.assert_array_equal(z['ids'],ids[te]);score=macro(yt,z['logits'].argmax(1))
   ref=cfg['cnn_references'][str(held)];assert sha(ref['path'])==ref['sha256']
   with np.load(ref['path']) as z: np.testing.assert_array_equal(z['ids'],ids[te]);baseline=macro(yt,z['logits'].argmax(1))
   rows.append(dict(held=held,gap=score,localcnn=baseline,difference=score-baseline))
  gains=[r['difference'] for r in rows];summary=dict(state='complete_source_confirmation',rows=rows,means={k:float(np.mean([r[k] for r in rows])) for k in ('gap','localcnn')},mean_difference=float(np.mean(gains)),wins=int(sum(v>0 for v in gains)),descriptive_continuation_gate=gate(gains,cfg),independent_validation=False,novel_contribution_validated=False,users7_8_labels_used=False,reused_fold=1,new_trajectories=5,new_epochs=320,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time())
  write(out/'SUMMARY.json',summary);record(state='complete',pid=None);write(out/'FILES.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix!='.tmp'});print(json.dumps(summary),flush=True)
 except BaseException as e:
  record(state='incomplete',pid=None,error=repr(e),traceback=traceback.format_exc(),scientific_negative=False);raise
 finally: lock.close()
if __name__=='__main__': assert sys.argv[1:]==['--run'];main()
