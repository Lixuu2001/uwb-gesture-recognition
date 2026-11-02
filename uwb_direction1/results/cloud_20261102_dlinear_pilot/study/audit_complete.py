"""Every native-shuffled RAdam update and validation early-stop replay; no diagnostic labels."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,sys,tempfile,time
from guards import ready,read,sha,require_training_complete
from artifacts import write
B=Path(__file__).resolve().parent
def replay_arrays(c,out,identity,ids,ix,ordered,maps,x,y,vx,vy):
 from engine import setup,snapshot,epoch,predict,equal,values,np,torch
 done=read(out/'TRAINING_COMPLETE.json');rows=(out/'iterations.jsonl').open();epochs=[json.loads(x) for x in (out/'epochs.jsonl').read_text().splitlines()];assert len(epochs)==done['epochs'];steps_per=(len(x)+c['batch_size']-1)//c['batch_size'];total=0;best_epoch=None;completed=0
 with tempfile.TemporaryDirectory() as d:
  m,opt,h,s=setup(c['seed'],len(x),c['epochs'],c['batch_size'])
  def compare(name,ep,batch,first_ids=None):
   saved=torch.load(out/name,map_location='cpu',weights_only=False)
   if name=='first_update.pt':np.testing.assert_array_equal(saved.pop('training_ids'),first_ids)
   equal(saved,snapshot(m,opt,h,s,identity,ep,total,batch))
  try:
   compare('initial.pt',0,-1);equal(torch.load(out/'initial_model.pt',weights_only=True),m.state_dict())
   for ep in range(c['epochs']):
    def after(actual):
     nonlocal total
     got=json.loads(next(rows));assert got['epoch']==ep+1;actual_ids=ids[ix['train'][actual['indices']]];np.testing.assert_array_equal(got['training_ids'],actual_ids)
     for key,value in actual.items():
      if key!='indices':equal(got[key],value)
     total+=1;assert total==actual['total_steps']
     if total==1:compare('first_update.pt',1,0,actual_ids)
    r=epoch(m,opt,h,s,x,y,vx,vy,ep,d,after);completed=ep+1
    if r['improved']:best_epoch=completed
    if completed==done['selected_epoch']:compare('best.pt',completed,steps_per-1)
    got=epochs[ep];assert got['epoch']==completed and got['steps']==total and got['training_records']==len(x);equal(got['metrics'],r['logs']);assert got['best_epoch']==best_epoch and got['early_stop']==h.early_stop and got['early_counter']==h.counter and got['learning_rate']==float(opt.param_groups[0]['lr'])
    if h.early_stop:break
   assert completed==done['epochs'] and best_epoch==done['selected_epoch'] and h.early_stop==done['early_stop'] and total==completed*steps_per and (completed==c['epochs'] or h.early_stop);assert rows.readline()=='';compare('last.pt',completed,steps_per-1)
   best=torch.load(out/'best.pt',map_location='cpu',weights_only=False);last=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
   equal(torch.load(out/'checkpoint.pth',weights_only=True),best['model_state']['model']);equal(torch.load(out/'last_model.pt',weights_only=True),last['model_state']['model']);m.load_state_dict(best['model_state']['model'])
   for role in ['validation','diagnostic']:
    a=vx if role=='validation' else np.array(maps[ordered[role]],dtype='float32',copy=True,order='C');p=predict(m,a,c['batch_size'])
    with np.load(out/(role+'_predictions_unscored.npz')) as z:np.testing.assert_array_equal(z['ids'],ids[ordered[role]]);np.testing.assert_array_equal(z['logits'],p);np.testing.assert_array_equal(z['predictions'],p.argmax(1))
   compare('final_prediction_state.pt',completed,steps_per-1)
  finally:rows.close()
 return dict(updates_replayed=total,epochs_replayed=completed,selected_epoch=best_epoch,training_appearances=completed*len(x),source_early_stop_replayed=True,all_initial_first_best_last_final_model_RAdam_modes_global_RNG_and_early_state_exact=True,native_initial_best_last_weights_exact=True,both_partition_IDs_and_logits_exact=True,diagnostic_labels_read=False)
def main():
 term=require_training_complete();c=ready();assert not (B/'COMPLETE_AUDIT.json').exists();start=time.monotonic();out=B/'run01';hashes={}
 def checked(p,d):assert sha(p)==d,str(p);hashes[str(p)]=d
 for name,d in read(B/'FROZEN_HASHES.json').items():checked(B/name,d)
 for name,d in read(out/'FILES.json').items():checked(out/name,d)
 from engine import np,torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 identity=read(out/'IDENTITY.json');assert identity==dict(protocol_sha256=sha(B/'PROTOCOL.json'),frozen_sha256=sha(B/'FROZEN_HASHES.json'),python=sys.version,torch=torch.__version__,numpy=np.__version__)
 from data import load
 ids,users,ix,ordered,maps,x,y,vx,vy=load(c);split=read(out/'split.json');parts=read(out/'EVAL_BATCH_IDS.json')
 for role,ind in ix.items():np.testing.assert_array_equal(split[role],ids[ind])
 for role,ind in ordered.items():
  assert len(parts[role])==(len(ind)+c['batch_size']-1)//c['batch_size']
  for batch,i in zip(parts[role],range(0,len(ind),c['batch_size'])):np.testing.assert_array_equal(batch,ids[ind[i:i+c['batch_size']]])
 cache=Path(c['data'])/'cache';subj=np.load(cache/'subj_clutterremoved.npy',mmap_mode='r');np.testing.assert_array_equal(subj[ids],users);raw=np.load(cache/'X_clutterremoved.npy',mmap_mode='r')
 for i in range(0,len(ids),64):
  a=np.array(raw[ids[i:i+64]],copy=True);rms=np.sqrt(np.mean(a*a,axis=(2,3),keepdims=True,dtype=np.float32));ref=(a/np.maximum(rms,np.float32(1e-8))).transpose(0,2,1,3).reshape(len(a),90,567);np.testing.assert_allclose(ref,maps[i:i+64],rtol=2e-5,atol=2e-6)
  f=a.astype('float64');r64=np.sqrt(np.mean(f*f,axis=(2,3),keepdims=True));ref64=(f/np.maximum(r64,1e-8)).transpose(0,2,1,3).reshape(len(a),90,567);np.testing.assert_allclose(ref,ref64,rtol=2e-5,atol=2e-6)
 result=replay_arrays(c,out,identity,ids,ix,ordered,maps,x,y,vx,vy)
 for p,d in hashes.items():assert sha(p)==d,p
 result.update(state='pass',all_source_RMS_independent_raw_and_FP64_reference=True,artifact_hashes=hashes,training_termination=term,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,audit_wall_seconds=time.monotonic()-start,auditor_sha256=sha(__file__));write(B/'COMPLETE_AUDIT.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['artifact_hashes','training_termination']}))
if __name__=='__main__':main()
