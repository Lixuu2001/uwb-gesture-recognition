"""Fixed full native HInception CPU100 source pilot; held1 scoring follows dual freeze."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import fcntl,hashlib,json,resource,signal,sys,time,traceback
from pathlib import Path
from ops import np,torch,keras,make,training_order,evaluation_order,snapshot,train_step,validation,predict,preprocess,close
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
def ready(cfg):
 sys.path.insert(0,'/workspace/uwb-litemv-engine-20261031');from readiness_guards import prior_closed
 assert prior_closed('/workspace/uwb-oscnn-pilot-20261031')
 cost=read('/workspace/uwb-hinception-cost-20261031/profile01/COST_CHECK.json');assert cost['state']=='pass' and cost['chosen_epochs']==cfg['epochs'] and cost['policy']['seed']==cfg['seed'] and cost['policy']['batch_size']==cfg['batch_size']
 term=read('/workspace/uwb-hinception-cost-20261031/supervisor01/TERMINATION.json');assert term['exit_code']==0 and not term['wall_timeout']
 s=read(cfg['previous_state']);assert s['state']=='complete' and s['pid'] is None
 summary=read(cfg['previous_summary']);assert summary['predeclared_route']=='close_exact_configuration' and summary['reference_held1_macro']==cfg['reference_held1_macro']
 for key in ('previous_termination','previous_audit_termination'):
  t=read(cfg[key]);assert t['exit_code']==0 and not t['wall_timeout']
 assert read(cfg['previous_audit'])['state']=='pass'
 r=read(cfg['previous_private']);assert r['state']=='private_readback_verified' and r['all_completed_models_included'] and r['standalone_complete_modelset_recoverable']
 return True
def roles(users):return dict(train=np.flatnonzero(np.isin(users,[3,4,5,6])),validation=np.flatnonzero(users==2),diagnostic=np.flatnonzero(users==1))
def macro(y,p):return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)]))
def gate(score,reference):return bool(score-reference+1e-12>=.005)
def main():
 cfg=read(B/'PROTOCOL.json');ready(cfg);assert cfg['seed']==20261029 and cfg['epochs']==100 and cfg['batch_size']==64 and cfg['parameters']==851434 and cfg['cost_gate_passed']
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
  identity=dict(protocol_sha256=sha(B/'PROTOCOL.json'),frozen_hashes=frozen,python=sys.version,torch=torch.__version__,numpy=np.__version__,keras=keras.__version__,backend=keras.backend.backend());write(out/'IDENTITY.json',identity)
  assert sha(cfg['source_manifest'])==cfg['source_manifest_sha256'] and sha(cfg['source_maps'])==cfg['source_maps_sha256']
  with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
  assert len(ids)==len(np.unique(ids))==6900 and set(users)==set(range(1,7));ix=roles(users);assert {k:len(v) for k,v in ix.items()}==cfg['counts']
  ordered={role:ix[role][evaluation_order(len(ix[role]),role,cfg['seed'])] for role in ['validation','diagnostic']};write(out/'split.json',{k:ids[v].tolist() for k,v in ix.items()});write(out/'EVAL_BATCH_IDS.json',{role:[ids[v[i:i+64]].tolist() for i in range(0,len(v),64)] for role,v in ordered.items()})
  for n,h in cfg['input_sha256'].items():assert sha(Path(cfg['data'])/'cache'/n)==h,n
  maps=np.load(cfg['source_maps'],mmap_mode='r');assert maps.shape==(6900,90,567) and maps.dtype==np.float32
  train=preprocess(maps[ix['train']]);val=preprocess(maps[ordered['validation']]);assert np.isfinite(train).all() and np.isfinite(val).all()
  labels=np.load(Path(cfg['data'])/'cache/y_clutterremoved.npy',mmap_mode='r');yt=np.array(labels[ids[ix['train']]],copy=True);yv=np.array(labels[ids[ordered['validation']]],copy=True);assert np.all((yt>=0)&(yt<12)) and np.all((yv>=0)&(yv<12));onehot=np.eye(12,dtype='float32')[yt]
  h,m,cb=make(out,cfg['seed']);assert m.count_params()==851434 and len(m.trainable_variables)==50 and len(m.optimizer.variables)==102;best_epoch=None;best_loss=None;total=0
  def checkpoint(epoch,batch):return snapshot(h,m,epoch,total,batch,identity)
  save(out/'initial.pt',checkpoint(0,-1));record(state='training')
  for epoch in range(1,101):
   order=training_order(epoch,cfg['seed']);t=time.monotonic();cb.on_epoch_begin(epoch-1);m.reset_metrics()
   for batch,i in enumerate(range(0,4700,64)):
    ii=order[i:i+64];cb.on_train_batch_begin(batch);logs,batchloss=train_step(m,train[ii],onehot[ii]);assert np.isfinite(list(logs.values())).all() and np.isfinite(batchloss);cb.on_train_batch_end(batch,logs);total+=1
    row=dict(epoch=epoch,batch=batch,total_steps=total,training_ids=ids[ix['train'][ii]].tolist(),native_cumulative_metrics=logs,batch_loss_from_tracker_delta=batchloss)
    with (out/'iterations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    if total==1:save(out/'first_update.pt',dict(checkpoint(epoch,batch),training_ids=row['training_ids']));record(epoch=epoch,batch=batch,total_steps=total)
   epochlogs=dict(logs);vl=validation(m,val,yv);epochlogs.update({'val_'+k:v for k,v in vl.items()});improved=best_loss is None or epochlogs['loss']<best_loss;cb.on_epoch_end(epoch-1,epochlogs)
   if improved:best_epoch=epoch;best_loss=epochlogs['loss'];save(out/'best.pt',checkpoint(epoch,batch))
   assert float(h.callbacks[1].best)==best_loss;save(out/'last.pt',checkpoint(epoch,batch));summary=dict(epoch=epoch,steps=total,training_records=4700,metrics=epochlogs,best_epoch=best_epoch,best_training_loss=best_loss,LR_wait=h.callbacks[0].wait,LR_cooldown=h.callbacks[0].cooldown_counter,current_learning_rate=float(m.optimizer.learning_rate.numpy()),epoch_seconds=time.monotonic()-t)
   with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
   record(epoch=epoch,batch=batch,total_steps=total,best_epoch=best_epoch,best_training_loss=best_loss);print(json.dumps(summary),flush=True)
  cb.on_train_end();assert total==7400 and int(m.optimizer.iterations.numpy())==7400;save(out/'last.pt',checkpoint(100,73));m.save(out/'last_model.keras')
  write(out/'TRAINING_COMPLETE.json',dict(epochs=100,steps=total,best_epoch=best_epoch,best_training_loss=best_loss,best_checkpoint_sha256=sha(out/'best.pt'),last_checkpoint_sha256=sha(out/'last.pt'),native_best_sha256=sha(out/'best_model.keras'),native_last_sha256=sha(out/'last_model.keras')))
  best=torch.load(out/'best.pt',map_location='cpu',weights_only=False);m.set_weights(best['native']['weights']);frozen_preds={}
  for role in ['validation','diagnostic']:
   values=val if role=='validation' else preprocess(maps[ordered[role]]);probs=predict(m,values);np.savez_compressed(out/(role+'_predictions_unscored.npz'),ids=ids[ordered[role]],probabilities=probs,predictions=probs.argmax(1));frozen_preds[role]=sha(out/(role+'_predictions_unscored.npz'))
  write(out/'ALL_PREDICTIONS_FROZEN.json',frozen_preds);save(out/'final_prediction_state.pt',checkpoint(100,73));record(state='predictions_frozen',all_predictions_frozen=True);record(diagnostic_labels_read=True);scores={}
  for role in ['validation','diagnostic']:
   with np.load(out/(role+'_predictions_unscored.npz')) as z:np.testing.assert_array_equal(z['ids'],ids[ordered[role]]);scores[role]=macro(np.array(labels[ids[ordered[role]]],copy=True),z['predictions'])
  eligible=gate(scores['diagnostic'],cfg['reference_held1_macro']);result=dict(state='complete_exploratory_source_pilot',scores=scores,reference_held1_macro=cfg['reference_held1_macro'],known_baseline_expansion_eligible=eligible,predeclared_route='known_baseline_source_confirmation' if eligible else 'close_exact_configuration',epochs=100,steps=total,best_epoch=best_epoch,best_training_loss=best_loss,training_appearances=470000,source_author_epochs1500_not_run=True,original_TensorFlow_reproduction=False,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time());write(out/'SUMMARY.json',result);record(state='complete',pid=None);write(out/'FILES.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix!='.tmp'});print(json.dumps(result),flush=True)
 except BaseException as e:record(state='incomplete',pid=None,error=repr(e),traceback=traceback.format_exc(),scientific_negative=False);raise
 finally:
  if 'cb' in locals():close(cb)
  lock.close()
if __name__=='__main__':assert sys.argv[1:]==['--run'];main()
