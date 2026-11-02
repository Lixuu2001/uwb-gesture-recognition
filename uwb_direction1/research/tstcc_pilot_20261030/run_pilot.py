"""Single prospectively registered full-scale TS-TCC source-user experiment."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import fcntl,hashlib,json,resource,signal,sys,time,traceback,warnings
from pathlib import Path
import numpy as np
import torch
from ops import make,step,indices,augmented,save_state
B=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
def save(p,v):
 p=Path(p);tmp=p.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(p)
def ready(cfg):
 prior=read(cfg['previous_state']);assert prior['state']=='complete' and prior['pid'] is None
 summary=read(cfg['previous_summary']);assert summary['predeclared_route']=='close_exact_configuration' and summary['reference_held1_macro']==cfg['reference_held1_macro']
 term=read(cfg['previous_termination']);assert term['exit_code']==0 and not term['wall_timeout']
 assert read(cfg['previous_audit'])['state']=='pass' and read(cfg['previous_audit_receipt'])['state']=='pass'
 receipt=read(cfg['previous_private']);assert receipt['state']=='private_readback_verified' and receipt['modelset_recoverable']
 return True
def roles(users):return dict(train=np.flatnonzero(np.isin(users,[3,4,5,6])),validation=np.flatnonzero(users==2),diagnostic=np.flatnonzero(users==1))
def macro(y,p):return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)]))
def eligible(scores,reference):return any(v-reference+1e-12>=.005 for v in scores.values())
def predict(m,values):
 m.eval();pieces=[]
 with torch.inference_mode():
  for start in range(0,len(values),128):pieces.append(m(torch.from_numpy(np.array(values[start:start+128],copy=True)))[0].numpy())
 return np.concatenate(pieces)
def validation_loss(logits,labels):return float(np.mean([float(torch.nn.functional.cross_entropy(torch.from_numpy(logits[i:i+128]),torch.from_numpy(labels[i:i+128]).long())) for i in range(0,len(labels),128)]))
def main():
 cfg=read(B/'PROTOCOL.json');ready(cfg);assert cfg['epochs']==40 and cfg['batch_size']==128 and cfg['seed']==20261029 and cfg['cost_gate_passed']
 out=B/'run01';out.mkdir(exist_ok=False);lock=open(out/'worker.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);warnings.filterwarnings('once',message='Implicit dimension choice')
 started=time.monotonic();state=dict(state='starting',pid=os.getpid(),completed_phases=[],all_predictions_frozen=False,diagnostic_labels_read=False,users7_8_labels_used=False)
 def record(**kw):state.update(kw);state.update(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time(),observed_unix=time.time());write(out/'STATE.json',state)
 def stop(sig,frame):raise TimeoutError('budget signal'+str(sig))
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
  train=np.array(maps[ix['train']].transpose(0,2,1),copy=True,order='C');assert train.shape==(4700,567,90) and np.isfinite(train).all()
  record(state='fixed_training_augmentation');weak,strong,recipe=augmented(train,cfg['seed']);save(out/'augmentation_recipe.pt',recipe)
  write(out/'AUGMENTATION.json',dict(weak_sha256=hashlib.sha256(weak).hexdigest(),strong_sha256=hashlib.sha256(strong).hexdigest(),recipe_sha256=sha(out/'augmentation_recipe.pt'),count=4700,shape=list(weak.shape),dtype=str(weak.dtype),once_fixed_training_only=True))
  pretrained=None;label_map=np.load(Path(cfg['data'])/'cache/y_clutterremoved.npy',mmap_mode='r');all_predictions={}
  for phase in ('ssl','linear','finetune','scratch'):
   stage=out/phase;stage.mkdir();parts=make(cfg['seed'],phase,pretrained);c,m,t,o,to,s=parts
   assert sum(p.numel() for p in m.parameters())==247500 and sum(p.numel() for p in t.parameters())==313096
   train_labels=np.zeros(4700,dtype=np.int64) if phase=='ssl' else np.array(label_map[ids[ix['train']]],copy=True)
   val_values=None if phase=='ssl' else np.array(maps[ix['validation']].transpose(0,2,1),copy=True,order='C');val_labels=None if phase=='ssl' else np.array(label_map[ids[ix['validation']]],copy=True)
   for key in ('validation_loss','validation_macro','learning_rate','training_loss'):state.pop(key,None)
   save(stage/'initial.pt',save_state(m,t,o,to,s,phase,0,-1,0,identity));record(state='training',phase=phase,epoch=0,batch=0,total_steps=0)
   for epoch in range(1,41):
    losses=[];epoch_start=time.monotonic()
    for batch in range(36):
     index=indices(epoch,batch,cfg['seed']);x=torch.from_numpy(train[index]);y=torch.from_numpy(train_labels[index]);aw=torch.from_numpy(weak[index]) if phase=='ssl' else None;ast=torch.from_numpy(strong[index]) if phase=='ssl' else None
     loss,_=step(m,t,o,to,c,x,y,aw,ast,phase);assert torch.isfinite(loss)
     assert all(p.grad is None or torch.isfinite(p.grad).all() for p in list(m.parameters())+list(t.parameters()))
     total=(epoch-1)*36+batch+1;losses.append(float(loss));row=dict(epoch=epoch,batch=batch,total_steps=total,training_ids=ids[ix['train'][index]].tolist(),loss=float(loss))
     with (stage/'iterations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
     if total==1:save(stage/'first_update.pt',dict(save_state(m,t,o,to,s,phase,epoch,batch,total,identity),training_ids=row['training_ids']));record(epoch=epoch,batch=batch,total_steps=total)
    m.eval();t.eval();summary=dict(epoch=epoch,training_loss=float(np.mean(losses)),steps=epoch*36,records_used=4608)
    if phase!='ssl':
     logits=predict(m,val_values);vl=validation_loss(logits,val_labels);s.step(vl);summary.update(validation_loss=vl,validation_macro=macro(val_labels,logits.argmax(1)),learning_rate=o.param_groups[0]['lr'])
    summary['epoch_seconds']=time.monotonic()-epoch_start
    with (stage/'epochs.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
    ck=save_state(m,t,o,to,s,phase,epoch,35,epoch*36,identity);save(stage/'last.pt',ck)
    if epoch%10==0:save(stage/f'epoch{epoch:02d}.pt',ck)
    record(epoch=epoch,batch=35,total_steps=epoch*36,**{k:v for k,v in summary.items() if k not in ('epoch','steps')});print(json.dumps(dict(phase=phase,**summary)),flush=True)
   if phase=='ssl':pretrained={k:v.detach().clone() for k,v in m.state_dict().items()}
   else:
    for role in ('validation','diagnostic'):
     values=val_values if role=='validation' else np.array(maps[ix[role]].transpose(0,2,1),copy=True,order='C');logits=predict(m,values);np.savez_compressed(stage/(role+'_predictions_unscored.npz'),ids=ids[ix[role]],logits=logits,predictions=logits.argmax(1))
     all_predictions[phase+'/'+role]=sha(stage/(role+'_predictions_unscored.npz'))
   write(stage/'PHASE_COMPLETE.json',dict(phase=phase,epochs=40,optimizer_updates=1440,training_appearances=184320,checkpoint=sha(stage/'last.pt'),selection='last_epoch_only'))
   state['completed_phases'].append(phase);record()
  write(out/'ALL_PREDICTIONS_FROZEN.json',all_predictions);record(state='predictions_frozen',all_predictions_frozen=True)
  scores={};validation={};record(diagnostic_labels_read=True)
  for phase in ('linear','finetune','scratch'):
   for role,destination in [('validation',validation),('diagnostic',scores)]:
    with np.load(out/phase/(role+'_predictions_unscored.npz')) as z:np.testing.assert_array_equal(z['ids'],ids[ix[role]]);destination[phase]=macro(np.array(label_map[ids[ix[role]]],copy=True),z['predictions'])
  result=dict(state='complete_exploratory_source_pilot',scores=scores,validation_scores=validation,reference_held1_macro=cfg['reference_held1_macro'],known_baseline_expansion_eligible=eligible(scores,cfg['reference_held1_macro']),predeclared_route='known_baseline_source_confirmation' if eligible(scores,cfg['reference_held1_macro']) else 'close_exact_configuration',finetune_minus_scratch=scores['finetune']-scores['scratch'],linear_minus_scratch=scores['linear']-scores['scratch'],all_four_full40_epochs_complete=True,training_updates=5760,training_appearances=737280,independent_validation=False,novel_contribution_validated=False,users7_8_labels_used=False,wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time())
  write(out/'SUMMARY.json',result);record(state='complete',pid=None);write(out/'FILES.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix!='.tmp'});print(json.dumps(result),flush=True)
 except BaseException as e:record(state='incomplete',pid=None,error=repr(e),traceback=traceback.format_exc(),scientific_negative=False);raise
 finally:lock.close()
if __name__=='__main__':assert sys.argv[1:]==['--run'];main()
