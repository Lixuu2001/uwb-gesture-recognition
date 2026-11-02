"""One full DLinear classification CLI pilot; no diagnostic labels or score."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import fcntl,json,resource,signal,sys,time,traceback,contextlib,io
from guards import ready,read,sha
from artifacts import write,save
B=Path(__file__).resolve().parent
def train_arrays(c,out,identity,ids,ix,ordered,maps,x,y,vx,vy,record):
 from engine import setup,snapshot,epoch,predict,values,np,torch
 values(x,len(x));values(vx,len(vx));m,opt,h,s=setup(c['seed'],len(x),c['epochs'],c['batch_size']);assert sum(p.numel() for p in m.parameters())==628752
 total=0;best_epoch=None;completed=0
 def state(ep,batch):return snapshot(m,opt,h,s,identity,ep,total,batch)
 save(out/'initial.pt',state(0,-1),torch);torch.save(m.state_dict(),out/'initial_model.pt')
 for ep in range(c['epochs']):
  start=time.monotonic()
  def after(row):
   nonlocal total
   total=row['total_steps'];r={k:v for k,v in row.items() if k!='indices'};r.update(epoch=ep+1,training_ids=ids[ix['train'][row['indices']]].tolist())
   with (out/'iterations.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
   if total==1:save(out/'first_update.pt',dict(state(1,0),training_ids=r['training_ids']),torch);record(state='training',epoch=1,total_steps=1)
  r=epoch(m,opt,h,s,x,y,vx,vy,ep,out,after);lastbatch=len(r['batch_records'])-1;completed=ep+1
  if r['improved']:best_epoch=completed;save(out/'best.pt',state(completed,lastbatch),torch)
  save(out/'last.pt',state(completed,lastbatch),torch)
  row=dict(epoch=completed,steps=total,training_records=len(x),metrics=r['logs'],best_epoch=best_epoch,early_stop=r['early_stop'],early_counter=r['early_counter'],learning_rate=float(opt.param_groups[0]['lr']),epoch_seconds=time.monotonic()-start)
  with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  record(state='training',epoch=completed,total_steps=total,best_epoch=best_epoch);print(json.dumps(row),flush=True)
  if h.early_stop:break
 torch.save(m.state_dict(),out/'last_model.pt');best=torch.load(out/'best.pt',map_location='cpu',weights_only=False);m.load_state_dict(best['model_state']['model']);frozen={}
 for role in ['validation','diagnostic']:
  a=vx if role=='validation' else np.array(maps[ordered[role]],dtype='float32',copy=True,order='C');p=predict(m,a,c['batch_size'])
  np.savez_compressed(out/(role+'_predictions_unscored.npz'),ids=ids[ordered[role]],logits=p,predictions=p.argmax(1));frozen[role]=sha(out/(role+'_predictions_unscored.npz'))
 write(out/'ALL_PREDICTIONS_FROZEN.json',frozen);save(out/'final_prediction_state.pt',state(completed,lastbatch),torch)
 write(out/'TRAINING_COMPLETE.json',dict(epochs=completed,max_epochs=c['epochs'],steps=total,selected_epoch=best_epoch,early_stop=h.early_stop,source_selection='validation_micro_accuracy_latest_tie_patience3',checkpoint_hashes={n:sha(out/n) for n in ['initial.pt','first_update.pt','best.pt','last.pt','final_prediction_state.pt','initial_model.pt','checkpoint.pth','last_model.pt']}))
 return dict(state='complete_unscored_training',epochs=completed,max_epochs=c['epochs'],steps=total,selected_epoch=best_epoch,early_stop=h.early_stop,training_appearances=completed*len(x),diagnostic_labels_read=False,users7_8_labels_used=False)
def main():
 c=ready();out=B/'run01';out.mkdir(exist_ok=False);lock=(out/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic()
 s=dict(state='starting',pid=os.getpid(),epoch=0,total_steps=0,all_predictions_frozen=False,diagnostic_labels_read=False,users7_8_labels_used=False)
 def record(**kw):s.update(kw);s.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time(),observed_unix=time.time());write(out/'STATE.json',s)
 def stop(sig,frame):raise TimeoutError('budget_signal'+str(sig))
 for sig in [signal.SIGTERM,signal.SIGALRM,signal.SIGXCPU]:signal.signal(sig,stop)
 resource.setrlimit(resource.RLIMIT_CPU,(c['train_CPU_soft'],c['train_CPU_hard']));signal.setitimer(signal.ITIMER_REAL,c['train_wall_seconds']);record()
 try:
  from engine import np,torch
  torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
  from data import load
  identity=dict(protocol_sha256=sha(B/'PROTOCOL.json'),frozen_sha256=sha(B/'FROZEN_HASHES.json'),python=sys.version,torch=torch.__version__,numpy=np.__version__);write(out/'IDENTITY.json',identity)
  ids,users,ix,ordered,maps,x,y,vx,vy=load(c);write(out/'split.json',{k:ids[v].tolist() for k,v in ix.items()});write(out/'EVAL_BATCH_IDS.json',{k:[ids[v[i:i+c['batch_size']]].tolist() for i in range(0,len(v),c['batch_size'])] for k,v in ordered.items()})
  r=train_arrays(c,out,identity,ids,ix,ordered,maps,x,y,vx,vy,record);assert r['steps']==r['epochs']*147 and r['training_appearances']==r['epochs']*4700 and 1<=r['epochs']<=10 and (r['epochs']==10 or r['early_stop'])
  write(out/'TRAINING_RESULT.json',r);record(state='complete',pid=None,all_predictions_frozen=True);write(out/'FILES.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['FILES.json','worker.lock'] and p.suffix!='.tmp'})
 except BaseException as e:record(state='incomplete',pid=None,error=repr(e),traceback=traceback.format_exc(),scientific_negative=False);raise
 finally:lock.close()
if __name__=='__main__':main()
