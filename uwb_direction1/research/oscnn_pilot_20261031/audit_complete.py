"""Reconstruct native global-loader shuffle, all updates and eval weight mutation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys,time
from pathlib import Path
from run_pilot import read,sha
B=Path(__file__).resolve().parent
def require_complete(base):
 out=base/'run01';s=read(out/'STATE.json')
 if s.get('state')!='complete' or s.get('pid') is not None:raise RuntimeError('not_ready: active/incomplete pilot')
 assert s['all_predictions_frozen'] and s['diagnostic_labels_read'] and s['users7_8_labels_used'] is False
 t=read(base/'supervisor01/TERMINATION.json');assert t['exit_code']==0 and not t['wall_timeout'] and t['supervisor_wall_seconds']<=43200 and t['child_cpu_seconds']<=43200
 done=read(out/'TRAINING_COMPLETE.json');assert done['epochs']==100 and done['steps']==29400 and done['selected_epoch']==100 and done['selection']=='source_final_last'
 for n,k in [('last.pt','last_checkpoint_sha256'),('source_final.pth','source_final_sha256'),('source_periodic.pth','source_periodic_sha256')]:assert sha(out/n)==done[k],n
 f=read(out/'ALL_PREDICTIONS_FROZEN.json');assert set(f)=={'validation','diagnostic'}
 for role,h in f.items():assert sha(out/(role+'_predictions_unscored.npz'))==h
 return t
def metric(y,p):return float(sum(float((p[y==c]==c).mean()) for c in sorted(set(y.tolist())))/len(set(y.tolist())))
def audit(output):
 term=require_complete(B);assert not output.exists();start=time.monotonic();cfg=read(B/'PROTOCOL.json');out=B/'run01';hashes={}
 def checked(p,h):assert sha(p)==h,str(p);hashes[str(p)]=h
 for n,h in read(B/'FROZEN_HASHES.json').items():checked(B/n,h)
 for n,h in read(out/'FILES.json').items():checked(out/n,h)
 from ops import torch,np,make,loaders,monitor,predictions,snapshot,equal
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 identity=read(out/'IDENTITY.json');assert identity['python']==sys.version and identity['torch']==torch.__version__ and identity['numpy']==np.__version__ and identity['protocol_sha256']==sha(B/'PROTOCOL.json') and identity['frozen_hashes']==read(B/'FROZEN_HASHES.json')
 checked(cfg['source_manifest'],cfg['source_manifest_sha256']);checked(cfg['source_maps'],cfg['source_maps_sha256']);cache=Path(cfg['data'])/'cache'
 for n,h in cfg['input_sha256'].items():checked(cache/n,h)
 with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
 assert len(ids)==len(np.unique(ids))==6900 and set(users)==set(range(1,7));ix=dict(train=np.flatnonzero(np.isin(users,[3,4,5,6])),validation=np.flatnonzero(users==2),diagnostic=np.flatnonzero(users==1));split=read(out/'split.json')
 for role,ind in ix.items():np.testing.assert_array_equal(split[role],ids[ind]);assert len(ind)==cfg['counts'][role]
 subj=np.load(cache/'subj_clutterremoved.npy',mmap_mode='r');np.testing.assert_array_equal(subj[ids],users);raw=np.load(cache/'X_clutterremoved.npy',mmap_mode='r');maps=np.load(cfg['source_maps'],mmap_mode='r')
 for i in range(0,6900,64):
  x=np.array(raw[ids[i:i+64]],copy=True);rms=np.sqrt(np.mean(x*x,axis=(2,3),keepdims=True,dtype=np.float32));ref=(x/np.maximum(rms,np.float32(1e-8))).transpose(0,2,1,3).reshape(len(x),90,567);np.testing.assert_allclose(ref,maps[i:i+64],rtol=2e-5,atol=2e-6)
 train=np.ascontiguousarray(maps[ix['train']].transpose(0,2,1));val=np.ascontiguousarray(maps[ix['validation']].transpose(0,2,1));labels=np.load(cache/'y_clutterremoved.npy',mmap_mode='r');yt=np.array(labels[ids[ix['train']]],dtype=np.int64,copy=True);yv=np.array(labels[ids[ix['validation']]],dtype=np.int64,copy=True)
 e=make(cfg['seed']);tr,vl=loaders(train,yt,ids[ix['train']],val,yv,ids[ix['validation']]);total=0;eval_count=0;rows=[json.loads(x) for x in (out/'iterations.jsonl').read_text().splitlines()];epochs=[json.loads(x) for x in (out/'epochs.jsonl').read_text().splitlines()];evalrows=[json.loads(x) for x in (out/'evaluation_batches.jsonl').read_text().splitlines()];assert len(rows)==29400 and len(epochs)==100 and len(evalrows)==714
 def compare(name,ep,batch,phase):
  got=torch.load(out/name,map_location='cpu',weights_only=False);expected=snapshot(e,identity,ep,total,batch,phase)
  if name=='first_update.pt':assert len(got.pop('training_ids'))==16
  equal(got,expected)
 compare('initial.pt',0,-1,'before_loader_iteration');e.model.train()
 for epoch in range(1,101):
  def observed(batch,sample,loss):
   nonlocal total
   row=rows[total];assert row['epoch']==epoch and row['batch']==batch and row['total_steps']==total+1 and row['loss']==loss;np.testing.assert_array_equal(row['training_ids'],sample[2].numpy());total+=1
   if total==1:
    saved=torch.load(out/'first_update.pt',map_location='cpu',weights_only=False);np.testing.assert_array_equal(saved['training_ids'],sample[2].numpy());compare('first_update.pt',1,0,'after_optimizer_before_scheduler')
  stats=e.train_epoch(tr,observed);row=epochs[epoch-1];assert row['epoch']==epoch and row['total_steps']==total and row['training_records']==4700;assert row['last_batch_loss']==stats['last_batch_loss'] and abs(row['mean_batch_loss_for_logging']-np.mean(stats['batch_losses']))<=1e-12 and row['learning_rate']==stats['learning_rate'] and row['scheduler_bad_epochs']==e.scheduler.num_bad_epochs and row['scheduler_best']==float(e.scheduler.best) and row['periodic_monitor']==(epoch%50==0)
  if epoch%50==0:
   def checked_eval(value):
    nonlocal eval_count
    assert evalrows[eval_count]==value;eval_count+=1
   vals=monitor(e,tr,vl,epoch,checked_eval)
   for k,v in vals.items():assert row[k]==v,k
 assert total==29400 and eval_count==714 and len(e.optimizer.state)==14 and {int(v['step']) for v in e.optimizer.state.values()}=={29400};compare('last.pt',100,293,'after_epoch_and_periodic_monitor');equal(torch.load(out/'source_final.pth',weights_only=False),e.model.state_dict());equal(torch.load(out/'source_periodic.pth',weights_only=False),e.model.state_dict())
 scores={}
 for role in ['validation','diagnostic']:
  values=val if role=='validation' else np.ascontiguousarray(maps[ix[role]].transpose(0,2,1));actual,logits=predictions(e,values,ids[ix[role]]);decision=logits.argmax(1)
  with np.load(out/(role+'_predictions_unscored.npz')) as z:np.testing.assert_array_equal(z['ids'],actual);np.testing.assert_array_equal(z['logits'],logits);np.testing.assert_array_equal(z['predictions'],decision)
  scores[role]=metric(np.array(labels[ids[ix[role]]],copy=True),decision)
 compare('final_prediction_state.pt',100,293,'both_predictions_frozen_before_diagnostic_labels');summary=read(out/'SUMMARY.json')
 for role in scores:assert abs(scores[role]-summary['scores'][role])<=1e-12
 eligible=bool(scores['diagnostic']-cfg['reference_held1_macro']+1e-12>=.005);assert eligible==summary['known_baseline_expansion_eligible'] and summary['predeclared_route']==('known_baseline_source_confirmation' if eligible else 'close_exact_configuration')
 for p,h in hashes.items():assert sha(p)==h,p
 result=dict(state='pass',scores=scores,producer_scores=summary['scores'],reference_held1_macro=cfg['reference_held1_macro'],known_baseline_expansion_eligible=eligible,training_updates_replayed_exact=total,training_appearances_verified=470000,epochs_replayed=100,selection='source_final_last',initial_first_last_finalprediction_model_Adam_scheduler_all_RNG_bitwise=True,source_periodic_and_final_raw_checkpoint_equal=True,all6900RMS_inputs_verified=True,native_global_DataLoader_training_and_periodic_eval_IDs_reconstructed=True,periodic_eval_batches_verified=eval_count,eval_mask_weight_side_effect_replayed=True,all2200final_logits_predictions_exact=True,artifact_hashes=hashes,training_termination=term,auditor_sha256=sha(__file__),audit_wall_seconds=time.monotonic()-start,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,limitations='Unmodified author multivariate OS-CNN architecture, CPU100 budget adaptation not source2000/demo500 reproduction. Source global loader/last-batch scheduling/periodic eval and final last preserved. Exploratory historically-developed source cohort, not independent test or novelty.')
 output.write_text(json.dumps(result,indent=2));print(json.dumps(dict(state='pass',epochs=100,steps=total)))
if __name__=='__main__':assert sys.argv[1:]==['--run'];audit(B/'COMPLETE_AUDIT.json')
