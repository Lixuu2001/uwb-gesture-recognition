"""Independent full fixed CPU100 replay; no selection/parameter search."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys,tempfile,time
from pathlib import Path
from run_fold import read,sha
B=Path(__file__).resolve().parent
def require_complete(base,held):
 out=base/'run01'/('held'+str(held));s=read(out/'STATE.json')
 if s.get('state')!='complete' or s.get('pid') is not None:raise RuntimeError('not_ready: active/incomplete pilot')
 assert s['all_predictions_frozen'] and s['diagnostic_labels_read'] is False and s['users7_8_labels_used'] is False
 t=read(base/'jobs'/('held'+str(held))/'train/TERMINATION.json');assert t['exit_code']==0 and not t['wall_timeout'] and t['supervisor_wall_seconds']<=43200 and t['child_cpu_seconds']<=43200
 done=read(out/'TRAINING_COMPLETE.json');ntrain=read(base/'PROTOCOL.json')['counts'][str(held)][0];assert done['epochs']==100 and done['steps']==100*((ntrain+63)//64) and 1<=done['best_epoch']<=100
 for n,k in [('best.pt','best_checkpoint_sha256'),('last.pt','last_checkpoint_sha256'),('best_model.keras','native_best_sha256'),('last_model.keras','native_last_sha256')]:assert sha(out/n)==done[k],n
 f=read(out/'ALL_PREDICTIONS_FROZEN.json');assert set(f)=={'validation','diagnostic'}
 for role,h in f.items():assert sha(out/(role+'_predictions_unscored.npz'))==h
 return t
def metric(y,p):return float(sum(float((p[y==c]==c).mean()) for c in sorted(set(y.tolist())))/len(set(y.tolist())))
def audit(output,held):
 term=require_complete(B,held);assert not output.exists();start=time.monotonic();cfg=read(B/'PROTOCOL.json');cfg=dict(cfg,counts=dict(zip(['train','validation','diagnostic'],cfg['counts'][str(held)])));NTRAIN=cfg['counts']['train'];out=B/'run01'/('held'+str(held));hashes={}
 def checked(p,h):assert sha(p)==h,str(p);hashes[str(p)]=h
 for n,h in read(B/'FROZEN_HASHES.json').items():checked(B/n,h)
 for n,h in read(out/'FILES.json').items():checked(out/n,h)
 from ops import np,torch,keras,make,training_order,evaluation_order,train_step,validation,predict,snapshot,equal,preprocess,close
 __import__('ops').configure(held,NTRAIN);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 identity=read(out/'IDENTITY.json');assert identity['held']==held;assert identity['python']==sys.version and identity['torch']==torch.__version__ and identity['numpy']==np.__version__ and identity['keras']==keras.__version__ and identity['backend']=='torch' and identity['protocol_sha256']==sha(B/'PROTOCOL.json') and identity['frozen_hashes']==read(B/'FROZEN_HASHES.json')
 checked(cfg['source_manifest'],cfg['source_manifest_sha256']);checked(cfg['source_maps'],cfg['source_maps_sha256']);cache=Path(cfg['data'])/'cache'
 for n,h in cfg['input_sha256'].items():checked(cache/n,h)
 with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
 assert len(ids)==len(np.unique(ids))==6900 and set(users)==set(range(1,7));ix=dict(train=np.flatnonzero(~np.isin(users,[held,held%6+1])),validation=np.flatnonzero(users==(held%6+1)),diagnostic=np.flatnonzero(users==held));split=read(out/'split.json');parts=read(out/'EVAL_BATCH_IDS.json');ordered={}
 for role,ind in ix.items():np.testing.assert_array_equal(split[role],ids[ind]);assert len(ind)==cfg['counts'][role]
 for role in ['validation','diagnostic']:
  ordered[role]=ix[role][evaluation_order(len(ix[role]),role,cfg['seed'])];assert len(parts[role])==(len(ix[role])+63)//64
  for got,i in zip(parts[role],range(0,len(ordered[role]),64)):np.testing.assert_array_equal(got,ids[ordered[role][i:i+64]])
 subj=np.load(cache/'subj_clutterremoved.npy',mmap_mode='r');np.testing.assert_array_equal(subj[ids],users);raw=np.load(cache/'X_clutterremoved.npy',mmap_mode='r');maps=np.load(cfg['source_maps'],mmap_mode='r')
 for i in range(0,6900,64):
  x=np.array(raw[ids[i:i+64]],copy=True);rms=np.sqrt(np.mean(x*x,axis=(2,3),keepdims=True,dtype=np.float32));ref=(x/np.maximum(rms,np.float32(1e-8))).transpose(0,2,1,3).reshape(len(x),90,567);np.testing.assert_allclose(ref,maps[i:i+64],rtol=2e-5,atol=2e-6);native_norm=preprocess(ref);cache_norm=preprocess(maps[i:i+64]);np.testing.assert_allclose(native_norm,cache_norm,rtol=2e-5,atol=2e-6);reference64=ref.astype('float64');std64=reference64.std(axis=1,keepdims=True);reference64=(reference64-reference64.mean(axis=1,keepdims=True))/np.where(std64==0,1.,std64);np.testing.assert_allclose(native_norm,reference64,rtol=2e-5,atol=2e-6)
 train=preprocess(maps[ix['train']]);val=preprocess(maps[ordered['validation']]);labels=np.load(cache/'y_clutterremoved.npy',mmap_mode='r');yt=np.array(labels[ids[ix['train']]],copy=True);yv=np.array(labels[ids[ordered['validation']]],copy=True);onehot=np.eye(12,dtype='float32')[yt]
 done=read(out/'TRAINING_COMPLETE.json');rows=[json.loads(x) for x in (out/'iterations.jsonl').read_text().splitlines()];epochs=[json.loads(x) for x in (out/'epochs.jsonl').read_text().splitlines()];assert len(rows)==100*((NTRAIN+63)//64) and len(epochs)==100;total=0;best_epoch=None;best_loss=None
 with tempfile.TemporaryDirectory() as d:
  h,m,cb=make(d,cfg['seed'])
  def compare(name,ep,batch):
   saved=torch.load(out/name,map_location='cpu',weights_only=False);expected=snapshot(h,m,ep,total,batch,identity)
   if name=='first_update.pt':np.testing.assert_array_equal(saved.pop('training_ids'),ids[ix['train'][training_order(1,cfg['seed'])[:64]]])
   equal(saved,expected)
  compare('initial.pt',0,-1)
  for epoch in range(1,101):
   order=training_order(epoch,cfg['seed']);cb.on_epoch_begin(epoch-1);m.reset_metrics()
   for batch,i in enumerate(range(0,NTRAIN,64)):
    ii=order[i:i+64];row=rows[total];assert row['epoch']==epoch and row['batch']==batch and row['total_steps']==total+1;np.testing.assert_array_equal(row['training_ids'],ids[ix['train'][ii]])
    cb.on_train_batch_begin(batch);logs,batchloss=train_step(m,train[ii],onehot[ii]);cb.on_train_batch_end(batch,logs);equal(logs,row['native_cumulative_metrics']);assert batchloss==row['batch_loss_from_tracker_delta'];total+=1
    if total==1:compare('first_update.pt',1,0)
   epochlogs=dict(logs);vl=validation(m,val,yv);epochlogs.update({'val_'+k:v for k,v in vl.items()});improved=best_loss is None or epochlogs['loss']<best_loss;cb.on_epoch_end(epoch-1,epochlogs)
   if improved:best_epoch=epoch;best_loss=epochlogs['loss']
   if epoch==done['best_epoch']:compare('best.pt',epoch,batch)
   row=epochs[epoch-1];assert row['epoch']==epoch and row['steps']==total and row['training_records']==NTRAIN;equal(epochlogs,row['metrics']);assert row['best_epoch']==best_epoch and row['best_training_loss']==best_loss and row['LR_wait']==h.callbacks[0].wait and row['LR_cooldown']==h.callbacks[0].cooldown_counter and row['current_learning_rate']==float(m.optimizer.learning_rate.numpy())
  cb.on_train_end();assert total==100*((NTRAIN+63)//64) and best_epoch==done['best_epoch'] and best_loss==done['best_training_loss'] and int(m.optimizer.iterations.numpy())==100*((NTRAIN+63)//64);compare('last.pt',100,(NTRAIN+63)//64-1)
  best=torch.load(out/'best.pt',map_location='cpu',weights_only=False);last=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
  from inspection_rng import isolate_inspection_python_rng
  with isolate_inspection_python_rng(h,m,100,total,(NTRAIN+63)//64-1,identity):
   for name,saved in [('best_model.keras',best),('last_model.keras',last)]:
    loaded=keras.models.load_model(out/name);equal(loaded.get_weights(),saved['native']['weights']);equal([v.numpy() for v in loaded.optimizer.variables],saved['native']['optimizer'])
  m.set_weights(best['native']['weights']);scores={}
  for role in ['validation','diagnostic']:
   values=val if role=='validation' else preprocess(maps[ordered[role]]);probs=predict(m,values);decision=probs.argmax(1)
   with np.load(out/(role+'_predictions_unscored.npz')) as z:np.testing.assert_array_equal(z['ids'],ids[ordered[role]]);np.testing.assert_array_equal(z['probabilities'],probs);np.testing.assert_array_equal(z['predictions'],decision)
  compare('final_prediction_state.pt',100,(NTRAIN+63)//64-1);close(cb)
 for p,hsh in hashes.items():assert sha(p)==hsh,p
 result=dict(state='pass',held=held,training_updates_replayed_exact=total,training_appearances_verified=100*NTRAIN,epochs_replayed=100,selected_epoch_verified=best_epoch,all_initial_first_best_last_final_states_exact=True,native_selected_last_weights_optimizer_exact=True,all6900RMS_and_author_znorm_independent_float64_reference_verified=True,evaluation_probabilities_and_IDs_exact=True,evaluation_rows=cfg['counts']['validation']+cfg['counts']['diagnostic'],diagnostic_labels_read=False,users7_8_labels_used=False,artifact_hashes=hashes,training_termination=term,auditor_sha256=sha(__file__),inspection_helper_sha256=sha(B/'inspection_rng.py'),audit_wall_seconds=time.monotonic()-start,independent_validation=False,novel_contribution_validated=False)
 output.write_text(json.dumps(result,indent=2));print(json.dumps(dict(state='pass',held=held,epochs=100,steps=total)))
if __name__=='__main__':assert len(sys.argv)==3 and sys.argv[1]=='--held' and int(sys.argv[2]) in [2,3,4,5,6];held=int(sys.argv[2]);audit(B/'run01'/('held'+str(held))/'COMPLETE_AUDIT.json',held)
