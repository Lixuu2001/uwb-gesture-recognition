"""Fixed, separately registered TimeMIL resource pilot; no automatic invocation."""
from adapter import make_model,StableLookahead
import fcntl,hashlib,json,os,pickle,random,resource,signal,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
B=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for v in iter(lambda:f.read(4*1024*1024),b''):h.update(v)
 return h.hexdigest()
def write(p,v):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def save(p,v):
 p=Path(p);t=p.with_suffix('.tmp');torch.save(v,t);t.replace(p)
def macro(y,p):return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)]))
def main():
 cfg=json.loads((B/'PROTOCOL.json').read_text())
 # Before opening arrays or making a fitting directory, require the prior queue.
 previous=json.loads(Path(cfg['previous_controller_state']).read_text())
 assert previous['state']=='complete_registered_queue','wait registered CNN/MultiRocket queue and audits'
 prior_audit=json.loads(Path(cfg['previous_multirocket_audit']).read_text());assert prior_audit['state']=='pass'
 out=B/cfg.get('run_directory','run01');out.mkdir(exist_ok=False);lock=open(out/'worker.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 started=time.monotonic();state=dict(state='starting',pid=os.getpid(),completed_arms=[],diagnostic_labels_read=False,users7_8_labels_used=False)
 def record(**kw):
  state.update(kw);state.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time(),observed_unix=time.time());write(out/'STATE.json',state)
 def stop(sig,frame):raise TimeoutError('budget signal '+str(sig))
 for s in [signal.SIGTERM,signal.SIGXCPU,signal.SIGALRM]:signal.signal(s,stop)
 resource.setrlimit(resource.RLIMIT_CPU,(43000,43100));signal.setitimer(signal.ITIMER_REAL,43100);record()
 try:
  frozen=json.loads((B/'FROZEN_HASHES.json').read_text())
  for n,h in frozen.items():assert sha(B/n)==h
  identity=dict(frozen_hashes=frozen,protocol_sha256=sha(B/'PROTOCOL.json'),python=sys.version,torch=torch.__version__,numpy=np.__version__);write(out/'IDENTITY.json',identity)
  assert json.loads((B/'RESUME_CHECK.json').read_text())['state']=='pass'
  assert json.loads((B/'PROTOCOL.json').read_text())['cost_gate_passed']
  assert sha(cfg['source_manifest'])==cfg['source_manifest_sha256']
  with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
  tr=np.flatnonzero(np.isin(users,[3,4,5,6]));va=np.flatnonzero(users==2);te=np.flatnonzero(users==1)
  assert len(ids)==len(np.unique(ids))==6900 and set(users)==set(range(1,7)) and (len(tr),len(va),len(te))==(4700,1000,1200)
  write(out/'split.json',dict(train=ids[tr].tolist(),validation=ids[va].tolist(),diagnostic=ids[te].tolist()))
  root=Path(cfg['data'])/'cache'
  for n,h in cfg['input_sha256'].items():assert sha(root/n)==h
  labels=np.load(root/'y_clutterremoved.npy',mmap_mode='r');ytr=torch.from_numpy(np.array(labels[ids[tr]],copy=True)).long();yv=np.array(labels[ids[va]],copy=True)
  raw=np.load(root/'X_clutterremoved.npy',mmap_mode='r');maps=np.lib.format.open_memmap(out/'source_maps.npy',mode='w+',dtype='float32',shape=(6900,90,567))
  with torch.no_grad():
   for b in range(0,6900,64):
    x=torch.from_numpy(np.array(raw[ids[b:b+64]],copy=True));x=x/x.square().mean((2,3),keepdim=True).sqrt().clamp_min(1e-8)
    maps[b:b+64]=x.permute(0,2,1,3).reshape(len(x),90,567).numpy()
  maps.flush();data=torch.from_numpy(np.array(maps,copy=True));del maps
  assert torch.isfinite(data).all();write(out/'SOURCE_CACHE.json',dict(sha256=sha(out/'source_maps.npy'),shape=list(data.shape),sample_ids_sha256=sha(cfg['source_manifest'])))
  target=F.one_hot(ytr,num_classes=12).float()
  def evaluate(model,indices,y=None):
   model.eval();chunks=[]
   with torch.no_grad():
    for b in range(0,len(indices),64):chunks.append(model(data[indices[b:b+64]]))
   logits=torch.cat(chunks);assert torch.isfinite(logits).all()
   if y is None:return logits
   return logits,macro(y,logits.argmax(1).numpy()),float(F.binary_cross_entropy_with_logits(logits,F.one_hot(torch.from_numpy(y).long(),12).float()))
  for arm in ['timemil','gap']:
   d=out/arm;d.mkdir();random.seed(cfg['seed']);np.random.seed(cfg['seed']);model=make_model(arm,cfg['seed']);opt=StableLookahead(model.parameters(),lr=.001,weight_decay=.0001);best=(-1.,-float('inf'),-1)
   save(d/'initial.pt',dict(model=model.state_dict(),identity=identity,epoch=0))
   expected_unused=set(json.loads((B/'SYNTHETIC_CHECK.json').read_text())['arms'][arm]['unused_parameter_names'])
   for epoch in range(1,cfg['epochs']+1):
    model.train();order=torch.randperm(len(tr),generator=torch.Generator().manual_seed(cfg['seed']+epoch)).numpy();total=0.;epstart=time.monotonic();record(state='training',arm=arm,epoch=epoch,batch=0)
    for b in range(0,len(tr),64):
     ji=order[b:b+64];opt.zero_grad();pred=model(data[tr[ji]],warmup=epoch<10);loss=F.binary_cross_entropy_with_logits(pred,target[ji]);assert torch.isfinite(loss);loss.backward()
     if b==0:
      assert set(n for n,p in model.named_parameters() if p.grad is None)==expected_unused
      assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
     torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();total+=float(loss.detach())*len(ji)
     if not (d/'first_update.pt').exists():save(d/'first_update.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),training_ids=ids[tr[ji]].tolist(),identity=identity))
     if b%(64*10)==0:record(batch=b//64+1)
    _,vm,vl=evaluate(model,va,yv);key=(vm,-vl,-epoch)
    if key>best:best=key;save(d/'best.pt',dict(model=model.state_dict(),epoch=epoch,validation_macro=vm,validation_loss=vl,identity=identity))
    save(d/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,best=list(best),torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),identity=identity))
    row=dict(arm=arm,epoch=epoch,train_bce=total/len(tr),validation_macro=vm,validation_bce=vl,epoch_seconds=time.monotonic()-epstart)
    with (d/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    record(state='epoch_complete',**row);print(json.dumps(row),flush=True)
   best_state=torch.load(d/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(best_state['model']);prediction=evaluate(model,te)
   np.savez_compressed(d/'predictions_unscored.npz',ids=ids[te],logits=prediction.numpy())
   write(d/'ARM_COMPLETE.json',dict(epochs=cfg['epochs'],selected_epoch=best_state['epoch'],checkpoint_sha256=sha(d/'best.pt'),prediction_sha256=sha(d/'predictions_unscored.npz')))
   record(completed_arms=state['completed_arms']+[arm]);del model,opt
  freezes={a:sha(out/a/'predictions_unscored.npz') for a in ['timemil','gap']};write(out/'ALL_PREDICTIONS_FROZEN.json',freezes);record(state='predictions_frozen')
  yt=np.array(labels[ids[te]],copy=True);scores={}
  for arm in ['timemil','gap']:
   with np.load(out/arm/'predictions_unscored.npz') as z:np.testing.assert_array_equal(z['ids'],ids[te]);scores[arm]=macro(yt,z['logits'].argmax(1))
  cnn=json.loads(Path(cfg['previous_cnn_summary']).read_text())['scores'];reference=max(cnn['full'],cnn['local'])
  result=dict(state='complete_exploratory_source_pilot',scores=scores,cnn_reference=cnn,gate_passed=bool(scores['timemil']-scores['gap']>=.005 and scores['timemil']>=reference-.005),gap_baseline_expansion_eligible=bool(scores['gap']>=reference+.01),epochs=cfg['epochs'],independent_validation=False,novel_contribution_validated=False,users7_8_labels_used=False,wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time())
  prior=cfg.get('prior_predata_attempt_budget',{});result.update(cumulative_wall_seconds=result['wall_seconds']+prior.get('wall_seconds',0),cumulative_cpu_seconds=result['cpu_seconds']+prior.get('cpu_seconds',0))
  write(out/'SUMMARY.json',result);record(state='complete',pid=None,diagnostic_labels_read=True);write(out/'FILES.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix!='.tmp'});print(json.dumps(result),flush=True)
 except BaseException as e:
  record(state='incomplete',pid=None,error=repr(e),traceback=traceback.format_exc(),scientific_negative=False);raise
 finally:lock.close()
if __name__=='__main__':
 assert sys.argv[1:]==['--run'];main()
