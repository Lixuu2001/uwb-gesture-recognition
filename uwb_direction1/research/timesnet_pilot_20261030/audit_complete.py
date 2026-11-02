"""Independent reconstruction of the completed fixed TimesNet source pilot."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,sys,tempfile,time
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
 return h.hexdigest()
def require_complete(base):
 out=base/'run01';s=read(out/'STATE.json')
 if s.get('state')!='complete' or s.get('pid') is not None:raise RuntimeError('not_ready: active/incomplete pilot')
 assert s['all_predictions_frozen'] and s['diagnostic_labels_read'] and s['users7_8_labels_used'] is False
 t=read(base/'supervisor01/TERMINATION.json');assert t['exit_code']==0 and not t['wall_timeout'] and t['supervisor_wall_seconds']<=43200 and t['child_cpu_seconds']<=43200
 done=read(out/'TRAINING_COMPLETE.json');assert 1<=done['epochs']<=30 and done['steps']==done['epochs']*294 and 1<=done['best_epoch']<=done['epochs'];assert sha(out/'best.pt')==done['best_checkpoint_sha256'] and sha(out/'last.pt')==done['last_checkpoint_sha256']
 frozen=read(out/'ALL_PREDICTIONS_FROZEN.json');assert set(frozen)=={'validation','diagnostic'}
 for role,h in frozen.items():assert sha(out/(role+'_predictions_unscored.npz'))==h
 return t
def metric(y,p):return float(sum(float(np.mean(p[y==c]==c)) for c in np.unique(y))/len(np.unique(y)))
def audit(output):
 term=require_complete(B);assert not output.exists();start=time.monotonic();cfg=read(B/'PROTOCOL.json');out=B/'run01';hashes={}
 def checked(p,h):assert sha(p)==h,str(p);hashes[str(p)]=h
 for n,h in read(B/'FROZEN_HASHES.json').items():checked(B/n,h)
 for n,h in read(out/'FILES.json').items():checked(out/n,h)
 import torch
 from ops import make,step,early_stopper,snapshot,equal,training_order,evaluation_order,predictions
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 identity=read(out/'IDENTITY.json');assert identity['python']==sys.version and identity['torch']==torch.__version__ and identity['numpy']==np.__version__ and identity['protocol_sha256']==sha(B/'PROTOCOL.json') and identity['frozen_hashes']==read(B/'FROZEN_HASHES.json')
 checked(cfg['source_manifest'],cfg['source_manifest_sha256']);checked(cfg['source_maps'],cfg['source_maps_sha256']);cache=Path(cfg['data'])/'cache'
 for n,h in cfg['input_sha256'].items():checked(cache/n,h)
 with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
 assert len(ids)==len(np.unique(ids))==6900 and set(users)==set(range(1,7));ix=dict(train=np.flatnonzero(np.isin(users,[3,4,5,6])),validation=np.flatnonzero(users==2),diagnostic=np.flatnonzero(users==1));split=read(out/'split.json');parts=read(out/'EVAL_BATCH_IDS.json');ordered={}
 for role,ind in ix.items():np.testing.assert_array_equal(split[role],ids[ind]);assert len(ind)==cfg['counts'][role]
 for role in ['validation','diagnostic']:
  ordered[role]=ix[role][evaluation_order(len(ix[role]),role,cfg['seed'])];assert len(parts[role])==(63 if role=='validation' else 75)
  for got,i in zip(parts[role],range(0,len(ordered[role]),16)):np.testing.assert_array_equal(got,ids[ordered[role][i:i+16]])
 subj=np.load(cache/'subj_clutterremoved.npy',mmap_mode='r');np.testing.assert_array_equal(subj[ids],users);raw=np.load(cache/'X_clutterremoved.npy',mmap_mode='r');maps=np.load(cfg['source_maps'],mmap_mode='r')
 for i in range(0,6900,64):
  x=np.array(raw[ids[i:i+64]],copy=True);rms=np.sqrt(np.mean(x*x,axis=(2,3),keepdims=True,dtype=np.float32));ref=(x/np.maximum(rms,np.float32(1e-8))).transpose(0,2,1,3).reshape(len(x),90,567);np.testing.assert_allclose(ref,maps[i:i+64],rtol=2e-5,atol=2e-6)
 train=np.array(maps[ix['train']],copy=True);validation=np.array(maps[ordered['validation']],copy=True);labels=np.load(cache/'y_clutterremoved.npy',mmap_mode='r');yt=np.array(labels[ids[ix['train']]],copy=True);yv=np.array(labels[ids[ordered['validation']]],copy=True)
 c,m,o=make(cfg['seed']);stopper=early_stopper(cfg['patience']);done=read(out/'TRAINING_COMPLETE.json');rows=[json.loads(x) for x in (out/'iterations.jsonl').read_text().splitlines()];epochs=[json.loads(x) for x in (out/'epochs.jsonl').read_text().splitlines()];assert len(rows)==done['steps'] and len(epochs)==done['epochs'];total=0;best_epoch=None;best_acc=None
 def compare(name,ep,batch):
  saved=torch.load(out/name,map_location='cpu',weights_only=False);expected=dict(snapshot(m,o,stopper,ep,total,dict(epoch=ep,batch=batch,order_seed=cfg['seed']+ep)),identity=identity)
  if name=='first_update.pt':np.testing.assert_array_equal(saved.pop('training_ids'),ids[ix['train'][training_order(1,cfg['seed'])[:16]]])
  equal(saved,expected)
 def forward(values):
  m.eval();pieces=[]
  with torch.inference_mode():
   for i in range(0,len(values),16):
    x=torch.from_numpy(np.array(values[i:i+16],copy=True));pieces.append(m(x,torch.ones(x.shape[:2]),None,None).numpy())
  return np.concatenate(pieces)
 compare('initial.pt',0,-1)
 with tempfile.TemporaryDirectory() as stopdir:
  for epoch in range(1,done['epochs']+1):
   assert not stopper.early_stop;order=training_order(epoch,cfg['seed']);losses=[];m.train()
   for batch,i in enumerate(range(0,4700,16)):
    ii=order[i:i+16];row=rows[total];assert row['epoch']==epoch and row['batch']==batch and row['total_steps']==total+1;np.testing.assert_array_equal(row['training_ids'],ids[ix['train'][ii]])
    loss=step(m,o,torch.from_numpy(train[ii]),torch.from_numpy(yt[ii]));assert float(loss.detach())==row['loss'];losses.append(float(loss.detach()));total+=1
    if total==1:compare('first_update.pt',1,0)
   logits=forward(validation);val_ce=float(np.mean([float(torch.nn.functional.cross_entropy(torch.from_numpy(logits[i:i+16]),torch.from_numpy(yv[i:i+16]).long())) for i in range(0,1000,16)]));acc=float(np.mean(predictions(logits)==yv));m.train();improved=stopper.best_score is None or acc>=stopper.best_score+stopper.delta;stopper(-acc,m,stopdir)
   if improved:best_epoch=epoch;best_acc=acc
   if epoch==done['best_epoch']:compare('best.pt',epoch,batch);equal(torch.load(out/'checkpoint.pth',map_location='cpu',weights_only=False),m.state_dict())
   row=epochs[epoch-1];assert row['epoch']==epoch and row['steps']==total and row['training_records']==4700;assert abs(row['training_loss']-np.mean(losses))<=1e-12 and abs(row['validation_loss']-val_ce)<=1e-12 and row['validation_accuracy']==acc and abs(row['validation_macro']-metric(yv,predictions(logits)))<=1e-12;assert row['best_epoch']==best_epoch and row['best_accuracy']==best_acc and row['stopper_counter']==stopper.counter and row['early_stop']==stopper.early_stop
   shifted=logits.astype(np.float64)-logits.max(1,keepdims=True);nll=np.log(np.exp(shifted).sum(1))-shifted[np.arange(1000),yv];nce=float(np.mean([nll[i:i+16].mean() for i in range(0,1000,16)]));assert abs(nce-val_ce)<2e-5
 assert total==done['steps'] and best_epoch==done['best_epoch'] and best_acc==done['best_accuracy'] and stopper.early_stop==done['early_stop'];assert done['epochs']==30 or stopper.early_stop;compare('last.pt',done['epochs'],293)
 assert len(o.state)==53 and {int(v['step']) for v in o.state.values()}=={total};best=torch.load(out/'best.pt',map_location='cpu',weights_only=False);m.load_state_dict(best['model']);scores={}
 for role in ['validation','diagnostic']:
  values=validation if role=='validation' else np.array(maps[ordered[role]],copy=True);logits=forward(values);decision=predictions(logits)
  with np.load(out/(role+'_predictions_unscored.npz')) as z:np.testing.assert_array_equal(z['ids'],ids[ordered[role]]);np.testing.assert_array_equal(z['logits'],logits);np.testing.assert_array_equal(z['predictions'],decision)
  scores[role]=metric(np.array(labels[ids[ordered[role]]],copy=True),decision)
 summary=read(out/'SUMMARY.json')
 for role in scores:assert abs(scores[role]-summary['scores'][role])<=1e-12
 eligible=bool(scores['diagnostic']-cfg['reference_held1_macro']+1e-12>=.005);assert eligible==summary['known_baseline_expansion_eligible'] and summary['predeclared_route']==('known_baseline_source_confirmation' if eligible else 'close_exact_configuration')
 for p,h in hashes.items():assert sha(p)==h,p
 result=dict(state='pass',scores=scores,producer_scores=summary['scores'],reference_held1_macro=cfg['reference_held1_macro'],known_baseline_expansion_eligible=eligible,training_updates_replayed_exact=total,training_appearances_verified=done['epochs']*4700,epochs_replayed=done['epochs'],selected_epoch_verified=best_epoch,initial_first_best_last_model_RAdam_stopper_all_RNG_bitwise=True,raw_author_selected_checkpoint_equal=True,all6900RMS_inputs_verified=True,labelblind_validation_and_diagnostic_batches_reconstructed=True,all2200final_logits_predictions_exact=True,all_source_validation_epochs_and_stopping_replayed=True,artifact_hashes=hashes,training_termination=term,auditor_sha256=sha(__file__),audit_wall_seconds=time.monotonic()-start,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,limitations='Batch-context Fourier routing is original model behavior, predictions use fixed labelblind partitions. Macro reduction tolerance1e-12; tensors/RAdam/stopper/RNG/logits bitwise. User-disjoint source pilot exploratory; not independent test or originality.')
 output.write_text(json.dumps(result,indent=2));print(json.dumps(dict(state='pass',epochs=done['epochs'],steps=total)))
if __name__=='__main__':assert sys.argv[1:]==['--run'];audit(B/'COMPLETE_AUDIT.json')
