"""All-update source replay/state/log probability audit; no diagnostic labels."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,sys,time
from guards import ready,read,sha,require_training_complete
from artifacts import write
B=Path(__file__).resolve().parent
def replay_arrays(c,out,identity,ids,ix,ordered,maps,x,y,vx,vy):
 from engine import setup,snapshot,epoch,predict,isolate,equal,np,torch
 done=read(out/'TRAINING_COMPLETE.json');m,opt,s=setup(c['seed'],len(x),c['epochs'],c['batch_size']);total=0;stepsper=(len(x)+c['batch_size']-1)//c['batch_size']
 rows=(out/'iterations.jsonl').open();epochs=[json.loads(z) for z in (out/'epochs.jsonl').read_text().splitlines()];assert len(epochs)==c['epochs']
 def compare(name,ep,batch):
  saved=torch.load(out/name,map_location='cpu',weights_only=False)
  if name=='first_update.pt':np.testing.assert_array_equal(saved.pop('training_ids'),ids[ix['train'][:c['batch_size']]])
  equal(saved,snapshot(m,opt,s,identity,ep,total,batch))
 try:
  compare('initial.pt',0,-1)
  initial=torch.load(out/'initial.pt',weights_only=False)
  def check_initial():equal(torch.load(out/'initial_model.pt',weights_only=False),initial['model_state']['model'])
  isolate(m,opt,check_initial)
  for ep in range(c['epochs']):
   def after(actual):
    nonlocal total
    got=json.loads(next(rows));assert got['epoch']==ep+1;np.testing.assert_array_equal(got['training_ids'],ids[ix['train'][actual['indices']]])
    for k,v in actual.items():
     if k!='indices':equal(got[k],v)
    total+=1;assert total==actual['total_steps']
    if total==1:compare('first_update.pt',1,0)
   r=epoch(m,opt,s,x,y,vx,vy,ep,after);got=epochs[ep]
   assert got['epoch']==ep+1 and got['steps']==total and got['training_records']==len(x);equal(got['metrics'],r['logs']);equal(got['learning_rate_before'],r['learning_rate_before']);equal(got['learning_rate_after'],r['learning_rate_after'])
   if ep+1==10:compare('epoch10.pt',10,stepsper-1)
  assert rows.readline()=='' and total==c['epochs']*stepsper and done['selected_epoch']==c['epochs'];compare('last.pt',c['epochs'],stepsper-1)
  last=torch.load(out/'last.pt',weights_only=False)
  def native():equal(torch.load(out/'last_model.pt',weights_only=False),last['model_state']['model'])
  isolate(m,opt,native)
  for role in ['validation','diagnostic']:
   a=vx if role=='validation' else np.array(maps[ordered[role]],dtype='float32',copy=True,order='C');p=predict(m,a,c['batch_size'])
   with np.load(out/(role+'_predictions_unscored.npz')) as z:np.testing.assert_array_equal(z['ids'],ids[ordered[role]]);np.testing.assert_array_equal(z['log_probabilities'],p);np.testing.assert_array_equal(z['predictions'],p.argmax(1))
  compare('final_prediction_state.pt',c['epochs'],stepsper-1)
 finally:rows.close()
 return dict(updates_replayed=total,epochs_replayed=c['epochs'],selected_epoch=c['epochs'],training_appearances=c['epochs']*len(x),all_initial_first_epoch10_last_final_model_Adam_modes_global_RNG_exact=True,native_initial_last_model_weights_exact=True,both_partition_IDs_and_log_probabilities_exact=True,diagnostic_labels_read=False)
def main():
 term=require_training_complete();c=ready();assert not (B/'COMPLETE_AUDIT.json').exists();start=time.monotonic();out=B/'run01';hashes={}
 def checked(p,d):assert sha(p)==d,str(p);hashes[str(p)]=d
 for n,d in read(B/'FROZEN_HASHES.json').items():checked(B/n,d)
 for n,d in read(out/'FILES.json').items():checked(out/n,d)
 from engine import np,torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 ident=read(out/'IDENTITY.json');assert ident==dict(protocol_sha256=sha(B/'PROTOCOL.json'),frozen_sha256=sha(B/'FROZEN_HASHES.json'),python=sys.version,torch=torch.__version__,numpy=np.__version__)
 from data import load
 ids,users,ix,ordered,maps,x,y,vx,vy=load(c);split=read(out/'split.json');parts=read(out/'EVAL_BATCH_IDS.json')
 for k,v in ix.items():np.testing.assert_array_equal(split[k],ids[v])
 for k,v in ordered.items():
  expected=[ids[v[i:i+c['batch_size']]].tolist() for i in range(0,len(v),c['batch_size'])];assert parts[k]==expected
 raw=np.load(Path(c['data'])/'cache/X_clutterremoved.npy',mmap_mode='r')
 for i in range(0,len(ids),64):
  a=np.array(raw[ids[i:i+64]],copy=True);rms=np.sqrt(np.mean(a*a,axis=(2,3),keepdims=True,dtype=np.float32));ref=(a/np.maximum(rms,np.float32(1e-8))).transpose(0,2,1,3).reshape(len(a),90,567);np.testing.assert_allclose(ref,maps[i:i+64],rtol=2e-5,atol=2e-6)
  d=a.astype('float64');rd=(d/np.maximum(np.sqrt(np.mean(d*d,axis=(2,3),keepdims=True)),1e-8)).transpose(0,2,1,3).reshape(len(a),90,567);np.testing.assert_allclose(ref,rd,rtol=2e-5,atol=2e-6)
 r=replay_arrays(c,out,ident,ids,ix,ordered,maps,x,y,vx,vy)
 for p,d in hashes.items():assert sha(p)==d,p
 r.update(state='pass',all_source_RMS_independent_raw_and_FP64_reference=True,artifact_hashes=hashes,training_termination=term,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,audit_wall_seconds=time.monotonic()-start,auditor_sha256=sha(__file__))
 write(B/'COMPLETE_AUDIT.json',r);print(json.dumps({k:v for k,v in r.items() if k not in ['artifact_hashes','training_termination']}))
if __name__=='__main__':main()
