"""Replay every fixed update and all prediction states, never diagnostic labels."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,sys,tempfile,time
from guards import ready,read,sha,require_training_complete
from artifacts import write,isolate_native
B=Path(__file__).resolve().parent
def replay_arrays(c,out,identity,ids,ix,ordered,maps,x,yt,vx,yv):
 from engine import setup,snapshot,epoch,permutation,predict,close,equal,values,np,torch,keras
 done=read(out/'TRAINING_COMPLETE.json');rows=(out/'iterations.jsonl').open();epochs=[json.loads(x) for x in (out/'epochs.jsonl').read_text().splitlines()]
 steps_per=(len(x)+c['batch_size']-1)//c['batch_size'];assert len(epochs)==c['epochs'];values(x,len(x));values(vx,len(vx));total=0;best_epoch=None;best_loss=None
 with tempfile.TemporaryDirectory() as d:
  h,m,cb,settings=setup(d,c['seed'],len(x),c['epochs'],c['batch_size'])
  def compare(name,ep,batch):
   saved=torch.load(out/name,map_location='cpu',weights_only=False)
   if name=='first_update.pt':np.testing.assert_array_equal(saved.pop('training_ids'),ids[ix['train'][permutation(len(x),0,c['seed'])[:c['batch_size']]]])
   equal(saved,snapshot(h,m,settings,identity,ep,total,batch))
  try:
   compare('initial.pt',0,-1)
   for ep in range(c['epochs']):
    def after(actual):
     nonlocal total
     got=json.loads(next(rows));assert got['epoch']==ep+1
     np.testing.assert_array_equal(got['training_ids'],ids[ix['train'][actual['indices']]])
     for key,value in actual.items():
      if key!='indices':equal(got[key],value)
     total+=1;assert total==actual['total_steps']
     if total==1:compare('first_update.pt',1,actual['batch'])
    r=epoch(h,m,cb,settings,x,yt,vx,yv,ep,permutation(len(x),ep,c['seed']),after)
    if r['improved']:best_epoch=ep+1;best_loss=r['logs']['loss']
    if ep+1==done['best_epoch']:compare('best.pt',ep+1,steps_per-1)
    got=epochs[ep];assert got['epoch']==ep+1 and got['steps']==total and got['training_records']==len(x);equal(got['metrics'],r['logs']);assert got['training_unweighted']
    assert got['best_epoch']==best_epoch and got['best_training_loss']==best_loss and got['current_learning_rate']==float(m.optimizer.learning_rate.numpy())
   assert rows.readline()=='';rows.close();cb.on_train_end();assert total==c['epochs']*steps_per and best_epoch==done['best_epoch'] and best_loss==done['best_training_loss'];compare('last.pt',c['epochs'],steps_per-1)
   best=torch.load(out/'best.pt',map_location='cpu',weights_only=False);last=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
   def inspect():
    for name,saved in [('best_model.keras',best),('last_model.keras',last)]:
     loaded=keras.models.load_model(out/name);equal(loaded.get_weights(),saved['model_state']['native']['weights']);equal([v.numpy() for v in loaded.optimizer.variables],saved['model_state']['native']['optimizer'])
   isolate_native(m,inspect);m.set_weights(best['model_state']['native']['weights'])
   for role in ['validation','diagnostic']:
    eval_values=vx if role=='validation' else np.array(maps[ordered[role]],dtype='float32',copy=True,order='C')
    probs=predict(m,eval_values,c['batch_size'])
    with np.load(out/(role+'_predictions_unscored.npz')) as z:np.testing.assert_array_equal(z['ids'],ids[ordered[role]]);np.testing.assert_array_equal(z['scores'],probs);np.testing.assert_array_equal(z['predictions'],probs.argmax(1))
   compare('final_prediction_state.pt',c['epochs'],steps_per-1)
  finally:rows.close();close(cb)
 return {'updates_replayed':total,'epochs_replayed':c['epochs'],'training_appearances':c['epochs']*len(x),'selected_epoch':best_epoch,'all_initial_first_best_last_final_model_Adam_BN_metrics_callbacks_modes_global_layer_RNG_exact':True,'native_best_last_weights_optimizer_exact':True,'both_partition_IDs_and_predictions_exact':True,'diagnostic_labels_read':False}
def main():
 term=require_training_complete();c=ready();assert not (B/'COMPLETE_AUDIT.json').exists();start=time.monotonic();out=B/'run01';hashes={}
 def checked(p,d):assert sha(p)==d,str(p);hashes[str(p)]=d
 for name,d in read(B/'FROZEN_HASHES.json').items():checked(B/name,d)
 for name,d in read(out/'FILES.json').items():checked(out/name,d)
 from engine import np,torch,keras
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 identity=read(out/'IDENTITY.json');assert identity=={'protocol_sha256':sha(B/'PROTOCOL.json'),'frozen_sha256':sha(B/'FROZEN_HASHES.json'),'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'keras':keras.__version__,'backend':'torch'}
 from data import load
 ids,users,ix,ordered,maps,x,yt,vx,yv=load(c);split=read(out/'split.json');parts=read(out/'EVAL_BATCH_IDS.json')
 for role,ind in ix.items():np.testing.assert_array_equal(split[role],ids[ind])
 for role,ind in ordered.items():
  assert len(parts[role])==(len(ind)+c['batch_size']-1)//c['batch_size']
  for batch,i in zip(parts[role],range(0,len(ind),c['batch_size'])):np.testing.assert_array_equal(batch,ids[ind[i:i+c['batch_size']]])
 cache=Path(c['data'])/'cache';subj=np.load(cache/'subj_clutterremoved.npy',mmap_mode='r');np.testing.assert_array_equal(subj[ids],users)
 raw=np.load(cache/'X_clutterremoved.npy',mmap_mode='r')
 # Independent raw signal reference for all source IDs, no gesture labels.
 for i in range(0,len(ids),64):
  raw_batch=np.array(raw[ids[i:i+64]],copy=True);rms=np.sqrt(np.mean(raw_batch*raw_batch,axis=(2,3),keepdims=True,dtype=np.float32));ref=(raw_batch/np.maximum(rms,np.float32(1e-8))).transpose(0,2,1,3).reshape(len(raw_batch),90,567)
  np.testing.assert_allclose(ref,maps[i:i+64],rtol=2e-5,atol=2e-6)
  fp64=raw_batch.astype('float64');rms64=np.sqrt(np.mean(fp64*fp64,axis=(2,3),keepdims=True));ref64=(fp64/np.maximum(rms64,1e-8)).transpose(0,2,1,3).reshape(len(raw_batch),90,567);np.testing.assert_allclose(ref,ref64,rtol=2e-5,atol=2e-6)
 result=replay_arrays(c,out,identity,ids,ix,ordered,maps,x,yt,vx,yv)
 for p,d in hashes.items():assert sha(p)==d,p
 result.update(state='pass',all_source_RMS_independent_raw_and_FP64_reference=True,artifact_hashes=hashes,training_termination=term,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,audit_wall_seconds=time.monotonic()-start,auditor_sha256=sha(__file__))
 write(B/'COMPLETE_AUDIT.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['artifact_hashes','training_termination']}))
if __name__=='__main__':main()
