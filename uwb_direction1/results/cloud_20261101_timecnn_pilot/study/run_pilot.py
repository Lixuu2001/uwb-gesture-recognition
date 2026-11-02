"""Fixed real TimeCNN2000 source pilot; no diagnostic scoring or selection."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import fcntl,json,resource,signal,sys,time,traceback
from guards import ready,read,sha
from artifacts import write,save,isolate_native
B=Path(__file__).resolve().parent
def train_arrays(c,out,identity,ids,ix,ordered,maps,x,yt,vx,yv,record):
 from engine import setup,snapshot,epoch,permutation,predict,close,values,np,torch
 values(x,len(x));values(vx,len(vx));h,m,cb,settings=setup(out,c['seed'],len(x),c['epochs'],c['batch_size']);assert m.count_params()==25356
 total=0;best_epoch=None;best_loss=None
 def state(ep,batch):return snapshot(h,m,settings,identity,ep,total,batch)
 save(out/'initial.pt',state(0,-1),torch)
 try:
  for ep in range(c['epochs']):
   start=time.monotonic();order=permutation(len(x),ep,c['seed'])
   def after(row):
    nonlocal total
    total=row['total_steps'];r={k:v for k,v in row.items() if k!='indices'};r.update(epoch=ep+1,training_ids=ids[ix['train'][row['indices']]].tolist())
    with (out/'iterations.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    if total==1:save(out/'first_update.pt',dict(state(1,row['batch']),training_ids=r['training_ids']),torch);record(state='training',epoch=1,batch=row['batch'],total_steps=total)
   r=epoch(h,m,cb,settings,x,yt,vx,yv,ep,order,after)
   if r['improved']:best_epoch=ep+1;best_loss=r['logs']['loss'];save(out/'best.pt',state(ep+1,len(r['batch_records'])-1),torch)
   save(out/'last.pt',state(ep+1,len(r['batch_records'])-1),torch)
   row={'epoch':ep+1,'steps':total,'training_records':len(x),'metrics':r['logs'],'best_epoch':best_epoch,'best_training_loss':best_loss,'training_unweighted':True,'current_learning_rate':float(m.optimizer.learning_rate.numpy()),'epoch_seconds':time.monotonic()-start}
   with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   record(state='training',epoch=ep+1,total_steps=total,best_epoch=best_epoch);print(json.dumps(row),flush=True)
  cb.on_train_end();save(out/'last.pt',state(c['epochs'],(len(x)+c['batch_size']-1)//c['batch_size']-1),torch)
  isolate_native(m,lambda:m.save(out/'last_model.keras'))
  write(out/'TRAINING_COMPLETE.json',{'epochs':c['epochs'],'steps':total,'best_epoch':best_epoch,'best_training_loss':best_loss,**{key:sha(out/name) for name,key in [('best.pt','best_checkpoint_sha256'),('last.pt','last_checkpoint_sha256'),('best_model.keras','native_best_sha256'),('last_model.keras','native_last_sha256')]}})
  best=torch.load(out/'best.pt',map_location='cpu',weights_only=False);m.set_weights(best['model_state']['native']['weights']);frozen={}
  for role in ['validation','diagnostic']:
   eval_values=vx if role=='validation' else np.array(maps[ordered[role]],dtype='float32',copy=True,order='C')
   probs=predict(m,eval_values,c['batch_size']);np.savez_compressed(out/(role+'_predictions_unscored.npz'),ids=ids[ordered[role]],scores=probs,predictions=probs.argmax(1));frozen[role]=sha(out/(role+'_predictions_unscored.npz'))
  write(out/'ALL_PREDICTIONS_FROZEN.json',frozen);save(out/'final_prediction_state.pt',state(c['epochs'],(len(x)+c['batch_size']-1)//c['batch_size']-1),torch)
  return {'state':'complete_unscored_training','epochs':c['epochs'],'steps':total,'best_epoch':best_epoch,'training_appearances':c['epochs']*len(x),'diagnostic_labels_read':False,'users7_8_labels_used':False}
 finally:close(cb)
def main():
 c=ready();out=B/'run01';out.mkdir(exist_ok=False);lock=(out/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 start=time.monotonic();s={'state':'starting','pid':os.getpid(),'epoch':0,'total_steps':0,'all_predictions_frozen':False,'diagnostic_labels_read':False,'users7_8_labels_used':False}
 def record(**kw):s.update(kw);s.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time(),observed_unix=time.time());write(out/'STATE.json',s)
 def stop(sig,frame):raise TimeoutError('budget_signal'+str(sig))
 for sig in [signal.SIGTERM,signal.SIGALRM,signal.SIGXCPU]:signal.signal(sig,stop)
 resource.setrlimit(resource.RLIMIT_CPU,(c['train_CPU_soft'],c['train_CPU_hard']));signal.setitimer(signal.ITIMER_REAL,c['train_wall_seconds']);record()
 try:
  from engine import torch,np,keras
  torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
  from data import load
  identity={'protocol_sha256':sha(B/'PROTOCOL.json'),'frozen_sha256':sha(B/'FROZEN_HASHES.json'),'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'keras':keras.__version__,'backend':'torch'};write(out/'IDENTITY.json',identity)
  ids,users,ix,ordered,maps,x,yt,vx,yv=load(c);write(out/'split.json',{k:ids[v].tolist() for k,v in ix.items()});write(out/'EVAL_BATCH_IDS.json',{role:[ids[v[i:i+c['batch_size']]].tolist() for i in range(0,len(v),c['batch_size'])] for role,v in ordered.items()})
  result=train_arrays(c,out,identity,ids,ix,ordered,maps,x,yt,vx,yv,record);assert result['steps']==588000 and result['training_appearances']==9400000
  write(out/'TRAINING_RESULT.json',result);record(state='complete',pid=None,all_predictions_frozen=True);write(out/'FILES.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['FILES.json','worker.lock'] and p.suffix!='.tmp'})
 except BaseException as e:record(state='incomplete',pid=None,error=repr(e),traceback=traceback.format_exc(),scientific_negative=False);raise
 finally:lock.close()
if __name__=='__main__':main()
