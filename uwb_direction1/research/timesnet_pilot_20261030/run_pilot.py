"""Prospectively fixed original TimesNet gesture preset, source-user pilot."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import fcntl,hashlib,json,resource,signal,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from ops import make,step,early_stopper,snapshot,training_order,evaluation_order,predictions
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
def predict(model,values):
 model.eval();pieces=[]
 with torch.inference_mode():
  for i in range(0,len(values),16):
   x=torch.from_numpy(np.array(values[i:i+16],copy=True));pieces.append(model(x,torch.ones(x.shape[:2]),None,None).numpy())
 return np.concatenate(pieces)
def validation_ce(logits,y):return float(np.mean([float(torch.nn.functional.cross_entropy(torch.from_numpy(logits[i:i+16]),torch.from_numpy(y[i:i+16]).long())) for i in range(0,len(y),16)]))
def main():
 cfg=read(B/'PROTOCOL.json');ready(cfg);assert cfg['seed']==20261029 and cfg['epochs_cap']==30 and cfg['batch_size']==16 and cfg['patience']==10 and cfg['cost_gate_passed']
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
  assert len(ids)==len(np.unique(ids))==6900 and set(users)==set(range(1,7));ix=roles(users);assert {k:len(v) for k,v in ix.items()}==cfg['counts']
  ordered={role:ix[role][evaluation_order(len(ix[role]),role,cfg['seed'])] for role in ['validation','diagnostic']};write(out/'split.json',{k:ids[v].tolist() for k,v in ix.items()});write(out/'EVAL_BATCH_IDS.json',{role:[ids[v[i:i+16]].tolist() for i in range(0,len(v),16)] for role,v in ordered.items()})
  for n,h in cfg['input_sha256'].items():assert sha(Path(cfg['data'])/'cache'/n)==h,n
  maps=np.load(cfg['source_maps'],mmap_mode='r');assert maps.shape==(6900,90,567) and maps.dtype==np.float32
  train=np.array(maps[ix['train']],copy=True);validation=np.array(maps[ordered['validation']],copy=True);assert np.isfinite(train).all() and np.isfinite(validation).all()
  label_map=np.load(Path(cfg['data'])/'cache/y_clutterremoved.npy',mmap_mode='r');yt=np.array(label_map[ids[ix['train']]],copy=True);yv=np.array(label_map[ids[ordered['validation']]],copy=True)
  c,m,o=make(cfg['seed']);assert sum(p.numel() for p in m.parameters())==2433260;stopper=early_stopper(cfg['patience']);best_epoch=None;best_acc=None;total=0
  def checkpoint(epoch,batch):return dict(snapshot(m,o,stopper,epoch,total,dict(epoch=epoch,batch=batch,order_seed=cfg['seed']+epoch)),identity=identity)
  save(out/'initial.pt',checkpoint(0,-1));record(state='training')
  for epoch in range(1,31):
   order=training_order(epoch,cfg['seed']);losses=[];t=time.monotonic();m.train()
   for batch,start0 in enumerate(range(0,4700,16)):
    ii=order[start0:start0+16];loss=step(m,o,torch.from_numpy(train[ii]),torch.from_numpy(yt[ii]));assert torch.isfinite(loss);assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());total+=1;value=float(loss.detach());losses.append(value)
    row=dict(epoch=epoch,batch=batch,total_steps=total,training_ids=ids[ix['train'][ii]].tolist(),loss=value)
    with (out/'iterations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    if total==1:save(out/'first_update.pt',dict(checkpoint(epoch,batch),training_ids=row['training_ids']));record(epoch=epoch,batch=batch,total_steps=total)
   logits=predict(m,validation);ce=validation_ce(logits,yv);acc=float(np.mean(predictions(logits)==yv));m.train();improved=stopper.best_score is None or acc>=stopper.best_score+stopper.delta;stopper(-acc,m,str(out))
   if improved:best_epoch=epoch;best_acc=acc;save(out/'best.pt',checkpoint(epoch,batch))
   save(out/'last.pt',checkpoint(epoch,batch));summary=dict(epoch=epoch,training_loss=float(np.mean(losses)),training_records=4700,steps=total,validation_loss=ce,validation_accuracy=acc,validation_macro=macro(yv,predictions(logits)),best_epoch=best_epoch,best_accuracy=best_acc,stopper_counter=stopper.counter,early_stop=stopper.early_stop,epoch_seconds=time.monotonic()-t)
   with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
   record(epoch=epoch,batch=batch,total_steps=total,**{k:v for k,v in summary.items() if k not in ['epoch','steps']});print(json.dumps(summary),flush=True)
   if stopper.early_stop:break
  assert best_epoch is not None;best=torch.load(out/'best.pt',map_location='cpu',weights_only=False);m.load_state_dict(best['model']);write(out/'TRAINING_COMPLETE.json',dict(epochs=epoch,steps=total,best_epoch=best_epoch,best_accuracy=best_acc,early_stop=stopper.early_stop,best_checkpoint_sha256=sha(out/'best.pt'),last_checkpoint_sha256=sha(out/'last.pt')))
  frozen_preds={}
  for role in ['validation','diagnostic']:
   values=validation if role=='validation' else np.array(maps[ordered[role]],copy=True);logits=predict(m,values);assert logits.shape==(len(ordered[role]),12) and np.isfinite(logits).all();np.savez_compressed(out/(role+'_predictions_unscored.npz'),ids=ids[ordered[role]],logits=logits,predictions=predictions(logits));frozen_preds[role]=sha(out/(role+'_predictions_unscored.npz'))
  write(out/'ALL_PREDICTIONS_FROZEN.json',frozen_preds);record(state='predictions_frozen',all_predictions_frozen=True);record(diagnostic_labels_read=True);scores={}
  for role in ['validation','diagnostic']:
   with np.load(out/(role+'_predictions_unscored.npz')) as z:np.testing.assert_array_equal(z['ids'],ids[ordered[role]]);scores[role]=macro(np.array(label_map[ids[ordered[role]]],copy=True),z['predictions'])
  result=dict(state='complete_exploratory_source_pilot',scores=scores,reference_held1_macro=cfg['reference_held1_macro'],known_baseline_expansion_eligible=gate(scores['diagnostic'],cfg['reference_held1_macro']),predeclared_route='known_baseline_source_confirmation' if gate(scores['diagnostic'],cfg['reference_held1_macro']) else 'close_exact_configuration',epochs=epoch,steps=total,best_epoch=best_epoch,best_validation_accuracy=best_acc,training_appearances=epoch*4700,prediction_batch_context='fixedlabelblindseed+101/102permutations;B16;originalperiodroutingpreserved',users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time());write(out/'SUMMARY.json',result);record(state='complete',pid=None);write(out/'FILES.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix!='.tmp'});print(json.dumps(result),flush=True)
 except BaseException as e:record(state='incomplete',pid=None,error=repr(e),traceback=traceback.format_exc(),scientific_negative=False);raise
 finally:lock.close()
if __name__=='__main__':assert sys.argv[1:]==['--run'];main()
