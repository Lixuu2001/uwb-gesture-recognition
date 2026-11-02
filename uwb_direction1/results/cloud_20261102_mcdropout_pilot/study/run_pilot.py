import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import fcntl,json,resource,signal,sys,time,traceback
from guards import ready,sha
from artifacts import write
B=Path(__file__).resolve().parent
def main():
 c=ready();out=B/'run01';out.mkdir(exist_ok=False);lock=(out/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic();s=dict(state='starting',pid=os.getpid(),diagnostic_labels_read=False,users7_8_labels_used=False,trained_weights_changed=False)
 def record(**kw):s.update(kw);s.update(observed_unix=time.time(),wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time());write(out/'STATE.json',s)
 def stop(sig,frame):raise TimeoutError('budget signal '+str(sig))
 for sig in [signal.SIGTERM,signal.SIGXCPU,signal.SIGALRM]:signal.signal(sig,stop)
 resource.setrlimit(resource.RLIMIT_CPU,(c['inference_CPU_soft'],c['inference_CPU_hard']));signal.setitimer(signal.ITIMER_REAL,c['inference_wall_seconds']);record()
 try:
  from engine import np,torch,model,produce
  torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
  from data import load
  ids,users,ordered,maps=load(c);identity=dict(protocol_sha256=sha(B/'PROTOCOL.json'),frozen_sha256=sha(B/'FROZEN_HASHES.json'),selected_checkpoint_sha256=sha(c['selected_checkpoint']),python=sys.version,torch=torch.__version__,numpy=np.__version__);write(out/'IDENTITY.json',identity);write(out/'split.json',{k:ids[ix].tolist() for k,ix in ordered.items()});m=model(c['seed'],c['selected_checkpoint']);record(state='inference')
  r=produce(c,out,m,ids,ordered,maps,identity,record,c['reference_predictions']);write(out/'INFERENCE_COMPLETE.json',r);record(state='complete',pid=None,all_predictions_frozen=True);write(out/'FILES.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['FILES.json','worker.lock'] and p.suffix!='.tmp'})
 except BaseException as e:record(state='incomplete',pid=None,error=repr(e),traceback=traceback.format_exc(),scientific_negative=False);raise
 finally:lock.close()
if __name__=='__main__':main()
