"""Fixed full source OS-CNN CPU100 accuracy pilot; final last, never val-best."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import fcntl,hashlib,json,resource,signal,sys,time,traceback
from pathlib import Path
from ops import torch,np,make,loaders,monitor,predictions,snapshot
from readiness_guards import prior_closed
B=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
 return h.hexdigest()
def write(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
def save(p,v):
 p=Path(p);tmp=p.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(p)
def ready(cfg):return prior_closed(cfg['previous_base'])
def roles(users):return dict(train=np.flatnonzero(np.isin(users,[3,4,5,6])),validation=np.flatnonzero(users==2),diagnostic=np.flatnonzero(users==1))
def macro(y,p):return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)]))
def gate(score,ref):return bool(score-ref+1e-12>=.005)
def main():
 cfg=read(B/'PROTOCOL.json');ready(cfg);assert cfg['seed']==20261029 and cfg['epochs']==100 and cfg['batch_size']==16 and cfg['cost_gate_passed'] and cfg['selection']=='source_final_last'
 marker=read(B/'REMOTE_PREREGISTRATION.json');assert marker['state']=='remote_preregistration_exact_readback_verified' and marker['protocol_sha256']==sha(B/'PROTOCOL.json')
 out=B/'run01';out.mkdir(exist_ok=False);lock=open(out/'worker.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();state=dict(state='starting',pid=os.getpid(),epoch=0,total_steps=0,all_predictions_frozen=False,diagnostic_labels_read=False,users7_8_labels_used=False)
 def record(**kw):state.update(kw);state.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time(),observed_unix=time.time());write(out/'STATE.json',state)
 def stop(sig,frame):raise TimeoutError('budgetsignal'+str(sig))
 for sig in (signal.SIGTERM,signal.SIGALRM,signal.SIGXCPU):signal.signal(sig,stop)
 resource.setrlimit(resource.RLIMIT_CPU,(43000,43100));signal.setitimer(signal.ITIMER_REAL,43100);record()
 try:
  frozen=read(B/'FROZEN_HASHES.json')
  for n,h in frozen.items():assert sha(B/n)==h,n
  identity=dict(protocol_sha256=sha(B/'PROTOCOL.json'),frozen_hashes=frozen,python=sys.version,torch=torch.__version__,numpy=np.__version__);write(out/'IDENTITY.json',identity)
  assert sha(cfg['source_manifest'])==cfg['source_manifest_sha256'] and sha(cfg['source_maps'])==cfg['source_maps_sha256']
  with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
  assert len(ids)==len(np.unique(ids))==6900 and set(users)==set(range(1,7));ix=roles(users);assert {k:len(v) for k,v in ix.items()}==cfg['counts'];write(out/'split.json',{k:ids[v].tolist() for k,v in ix.items()})
  for n,h in cfg['input_sha256'].items():assert sha(Path(cfg['data'])/'cache'/n)==h,n
  maps=np.load(cfg['source_maps'],mmap_mode='r');assert maps.shape==(6900,90,567) and maps.dtype==np.float32
  train=np.ascontiguousarray(maps[ix['train']].transpose(0,2,1));val=np.ascontiguousarray(maps[ix['validation']].transpose(0,2,1));assert np.isfinite(train).all() and np.isfinite(val).all()
  labels=np.load(Path(cfg['data'])/'cache/y_clutterremoved.npy',mmap_mode='r');yt=np.array(labels[ids[ix['train']]],dtype=np.int64,copy=True);yv=np.array(labels[ids[ix['validation']]],dtype=np.int64,copy=True);assert yt.dtype==yv.dtype==np.int64 and np.all((yt>=0)&(yt<12)) and np.all((yv>=0)&(yv<12))
  e=make(cfg['seed']);assert sum(p.numel() for p in e.model.parameters() if p.requires_grad)==1870464;tr,vl=loaders(train,yt,ids[ix['train']],val,yv,ids[ix['validation']]);total=0
  def checkpoint(epoch,batch,phase):return snapshot(e,identity,epoch,total,batch,phase)
  save(out/'initial.pt',checkpoint(0,-1,'before_loader_iteration'));e.model.train();record(state='training')
  for epoch in range(1,101):
   t=time.monotonic()
   def observed(batch,sample,loss):
    nonlocal total
    total+=1;row=dict(epoch=epoch,batch=batch,total_steps=total,training_ids=sample[2].tolist(),loss=loss)
    with (out/'iterations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    if total==1:save(out/'first_update.pt',dict(checkpoint(epoch,batch,'after_optimizer_before_scheduler'),training_ids=row['training_ids']));record(epoch=epoch,batch=batch,total_steps=total)
   stats=e.train_epoch(tr,observed);row=dict(epoch=epoch,total_steps=total,training_records=4700,last_batch_loss=stats['last_batch_loss'],mean_batch_loss_for_logging=float(np.mean(stats['batch_losses'])),learning_rate=stats['learning_rate'],scheduler_bad_epochs=e.scheduler.num_bad_epochs,scheduler_best=float(e.scheduler.best),periodic_monitor=(epoch%50==0))
   if epoch%50==0:
    def eval_record(value):
     with (out/'evaluation_batches.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    row.update(monitor(e,tr,vl,epoch,eval_record));save(out/'source_periodic.pth',e.model.state_dict())
   save(out/'last.pt',checkpoint(epoch,293,'after_epoch_and_periodic_monitor'));row['epoch_seconds']=time.monotonic()-t
   with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   record(epoch=epoch,batch=293,total_steps=total,last_batch_loss=row['last_batch_loss']);print(json.dumps(row),flush=True)
  assert total==29400 and len(e.optimizer.state)==14;save(out/'source_final.pth',e.model.state_dict());write(out/'TRAINING_COMPLETE.json',dict(epochs=100,steps=total,selected_epoch=100,selection='source_final_last',last_checkpoint_sha256=sha(out/'last.pt'),source_final_sha256=sha(out/'source_final.pth'),source_periodic_sha256=sha(out/'source_periodic.pth')))
  frozen_preds={}
  for role in ['validation','diagnostic']:
   values=val if role=='validation' else np.ascontiguousarray(maps[ix[role]].transpose(0,2,1));actual,logits=predictions(e,values,ids[ix[role]]);np.testing.assert_array_equal(actual,ids[ix[role]]);np.savez_compressed(out/(role+'_predictions_unscored.npz'),ids=actual,logits=logits,predictions=logits.argmax(1));frozen_preds[role]=sha(out/(role+'_predictions_unscored.npz'))
  save(out/'final_prediction_state.pt',checkpoint(100,293,'both_predictions_frozen_before_diagnostic_labels'));write(out/'ALL_PREDICTIONS_FROZEN.json',frozen_preds);record(state='predictions_frozen',all_predictions_frozen=True);record(diagnostic_labels_read=True);scores={}
  for role in ['validation','diagnostic']:
   with np.load(out/(role+'_predictions_unscored.npz')) as z:np.testing.assert_array_equal(z['ids'],ids[ix[role]]);scores[role]=macro(np.array(labels[ids[ix[role]]],copy=True),z['predictions'])
  eligible=gate(scores['diagnostic'],cfg['reference_held1_macro']);result=dict(state='complete_exploratory_source_pilot',scores=scores,reference_held1_macro=cfg['reference_held1_macro'],known_baseline_expansion_eligible=eligible,predeclared_route='known_baseline_source_confirmation' if eligible else 'close_exact_configuration',epochs=100,steps=total,selected_epoch=100,selection='source_final_last',training_appearances=470000,source_author2000_and_demo500_epochs_not_run=True,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time());write(out/'SUMMARY.json',result);record(state='complete',pid=None);write(out/'FILES.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix!='.tmp'});print(json.dumps(result),flush=True)
 except BaseException as ex:record(state='incomplete',pid=None,error=repr(ex),traceback=traceback.format_exc(),scientific_negative=False);raise
 finally:lock.close()
if __name__=='__main__':assert sys.argv[1:]==['--run'];main()
