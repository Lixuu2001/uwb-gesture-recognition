"""Prospective fixed TS2Vec600/source-user pilot; explicit launch only.

The existing TSLANet must be complete, fully audited, privately recoverable and
closed by its original gate before this runner can read any real arrays.
"""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import fcntl,hashlib,json,random,resource,signal,sys,time,traceback,warnings
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning,FitFailedWarning
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from training_ops import make_model,source_check,STEP,optimizer,checkpoint,batch_indices
from readout_adapter import definition
B=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,indent=2));q.replace(p)
def save(p,value):
 p=Path(p);q=p.with_suffix('.tmp');torch.save(value,q);q.replace(p)
def ready(cfg):
 previous=read(cfg['previous_state'])
 if previous.get('state')!='complete' or previous.get('pid') is not None:raise RuntimeError('not_ready: prior TSLANet incomplete/active')
 term=read(cfg['previous_termination'])
 if term['exit_code']!=0 or term['wall_timeout']:raise RuntimeError('not_ready: prior training supervisor unsuccessful')
 audit=read(cfg['previous_audit']);assert audit['state']=='pass'
 if read(cfg['previous_summary'])['gates']['predeclared_route']!='close_exact_configuration':raise RuntimeError('not_ready: prior eligible route requires source confirmation')
 receipt=read(cfg['previous_final_private_backup']);assert receipt['state']=='private_readback_verified' and receipt['full_experiment_modelset_recoverable_composite'] is True
 return audit
def roles(users):return dict(train=np.flatnonzero(np.isin(users,[3,4,5,6])),validation=np.flatnonzero(users==2),diagnostic=np.flatnonzero(users==1))
def macro(y,p):return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)]))
def gate(score,reference):return bool(score-reference+1e-12>=.005)
def main():
 cfg=read(B/'PROTOCOL.json');ready(cfg);assert cfg['iterations']==600 and cfg['seed']==20261029 and cfg['batch_size']==16 and cfg['cost_gate_passed']
 out=B/'run01';out.mkdir(exist_ok=False);lock=open(out/'worker.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 started=time.monotonic();state=dict(state='starting',pid=os.getpid(),iteration=0,diagnostic_labels_read=False,all_predictions_frozen=False,users7_8_labels_used=False)
 def record(**kw):
  state.update(kw);state.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time(),observed_unix=time.time());write(out/'STATE.json',state)
 def stop(sig,frame):raise TimeoutError('budget signal '+str(sig))
 for sig in (signal.SIGTERM,signal.SIGALRM,signal.SIGXCPU):signal.signal(sig,stop)
 resource.setrlimit(resource.RLIMIT_CPU,(43000,43100));signal.setitimer(signal.ITIMER_REAL,43100);record()
 try:
  frozen=read(B/'FROZEN_HASHES.json')
  for n,h in frozen.items():assert sha(B/n)==h,n
  source_check();identity=dict(protocol_sha256=sha(B/'PROTOCOL.json'),frozen_hashes=frozen,python=sys.version,torch=torch.__version__,numpy=np.__version__);write(out/'IDENTITY.json',identity)
  old=read(cfg['prior_held1_summary']);previous=read(cfg['previous_summary']);reference=max(old['scores']['timemil'],old['scores']['gap'],old['cnn_reference']['full'],old['cnn_reference']['local'],*previous['scores'].values());assert reference==cfg['reference_held1_macro']
  assert sha(cfg['source_manifest'])==cfg['source_manifest_sha256']
  with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
  assert len(ids)==len(np.unique(ids))==6900 and set(users)==set(range(1,7));ix=roles(users);assert {k:len(v) for k,v in ix.items()}==cfg['counts'];write(out/'split.json',{k:ids[v].tolist() for k,v in ix.items()})
  for n,h in cfg['input_sha256'].items():assert sha(Path(cfg['data'])/'cache'/n)==h,n
  assert sha(cfg['source_maps'])==cfg['source_maps_sha256'];maps=np.load(cfg['source_maps'],mmap_mode='r');assert maps.shape==(6900,90,567) and maps.dtype==np.float32
  train=np.array(maps[ix['train']],copy=True);assert np.isfinite(train).all();data=torch.from_numpy(train)
  model=make_model(cfg['seed']);assert sum(p.numel() for p in model._net.parameters())==673472;opt=optimizer(model);assert opt.param_groups[0]['weight_decay']==.01
  save(out/'initial.pt',checkpoint(model,opt,0,dict(epoch=1,offset=0,order_seed=cfg['seed']+1),identity))
  record(state='encoder_training')
  for step in range(1,601):
   index,sampler=batch_indices(step,cfg['seed']);loss=STEP(model,data[index],opt);assert torch.isfinite(loss)
   assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model._net.parameters());model.n_iters=step;model.n_epochs=step//293
   row=dict(iteration=step,sampler=sampler,training_ids=ids[ix['train'][index]].tolist(),loss=float(loss.detach()))
   with (out/'ITERATIONS.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   if step==1:save(out/'first_update.pt',dict(checkpoint(model,opt,step,sampler,identity),training_ids=row['training_ids']))
   if step%50==0 or step==1:
    save(out/'last.pt',checkpoint(model,opt,step,sampler,identity));record(iteration=step,encoder_loss=row['loss']);print(json.dumps(dict(stage='encoder',iteration=step,loss=row['loss'])),flush=True)
  assert int(model.net.n_averaged)==601;write(out/'ENCODER_COMPLETE.json',dict(iterations=600,raw_and_average_checkpoint_sha256=sha(out/'last.pt'),initial_inclusive_average_samples=601))
  record(state='encoding')
  representations={}
  for role in ('train','validation','diagnostic'):
   values=train if role=='train' else np.array(maps[ix[role]],copy=True);representations[role]=model.encode(values,encoding_window='full_series');assert representations[role].shape==(len(ix[role]),320) and np.isfinite(representations[role]).all()
  np.savez_compressed(out/'features.npz',ids=ids,users=users,**representations)
  record(state='source_only_SVM_CV')
  label_map=np.load(Path(cfg['data'])/'cache/y_clutterremoved.npy',mmap_mode='r');yt=np.array(label_map[ids[ix['train']]],copy=True);assert set(yt)==set(range(12));captured=[]
  def traced_grid(*args,**kwargs):
   assert kwargs['n_jobs']==1 and kwargs['cv']==5;search=GridSearchCV(*args,**kwargs);captured.append(search);return search
  folds=[dict(train_ids=ids[ix['train'][tr]].tolist(),validation_ids=ids[ix['train'][va]].tolist()) for tr,va in StratifiedKFold(5).split(representations['train'],yt)];write(out/'SVM_TRAINING_CV_IDS.json',folds)
  with warnings.catch_warnings(record=True) as notices:
   warnings.simplefilter('always');classifier=definition(True,traced_grid)(representations['train'],yt)
  assert len(captured)==1;cv=captured[0];joblib.dump(dict(classifier=classifier,cv_results=cv.cv_results_,best_index=int(cv.best_index_),training_ids=ids[ix['train']],identity=identity),out/'svm.joblib',compress=3)
  notice_rows=[dict(category=n.category.__name__,message=str(n.message)) for n in notices];write(out/'SVM_WARNINGS.json',notice_rows)
  if any(issubclass(n.category,(ConvergenceWarning,FitFailedWarning)) for n in notices):raise RuntimeError('SVM convergence/fit warning: incomplete numerical execution')
  assert np.isfinite(cv.cv_results_['mean_test_score']).all() and classifier.fit_status_==0;np.testing.assert_array_equal(classifier.classes_,np.arange(12))
  for role in ('validation','diagnostic'):
   labels=classifier.predict(representations[role]);decisions=classifier.decision_function(representations[role]);assert labels.shape==(len(ix[role]),) and decisions.shape==(len(ix[role]),12) and np.isfinite(decisions).all()
   np.savez_compressed(out/(role+'_predictions_unscored.npz'),ids=ids[ix[role]],predictions=labels,decisions=decisions)
  frozen_predictions={role:sha(out/(role+'_predictions_unscored.npz')) for role in ('validation','diagnostic')};write(out/'ALL_PREDICTIONS_FROZEN.json',frozen_predictions);record(state='predictions_frozen',all_predictions_frozen=True)
  scores={}
  for role in ('validation','diagnostic'):
   if role=='diagnostic':record(diagnostic_labels_read=True)
   labels=np.array(label_map[ids[ix[role]]],copy=True)
   with np.load(out/(role+'_predictions_unscored.npz')) as z:np.testing.assert_array_equal(z['ids'],ids[ix[role]]);scores[role]=macro(labels,z['predictions'])
  c=float(classifier.C);result=dict(state='complete_exploratory_source_pilot',scores=scores,reference_held1_macro=reference,known_baseline_expansion_eligible=gate(scores['diagnostic'],reference),predeclared_route='known_baseline_source_confirmation' if gate(scores['diagnostic'],reference) else 'close_exact_configuration',selected_C='inf' if np.isinf(c) else c,SVM_selection_scope='Original5foldCV only within trainingusers3-6; no val2/held1labels used for fitting or Cselection',iterations=600,independent_validation=False,novel_contribution_validated=False,users7_8_labels_used=False,wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time())
  write(out/'SUMMARY.json',result);record(state='complete',pid=None);write(out/'FILES.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix!='.tmp'});print(json.dumps(result),flush=True)
 except BaseException as e:record(state='incomplete',pid=None,error=repr(e),traceback=traceback.format_exc(),scientific_negative=False);raise
 finally:lock.close()
if __name__=='__main__':assert sys.argv[1:]==['--run'];main()
