import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,sys,time,resource,signal
from guard import ready,sha
from cost_math import select
B=Path(__file__).resolve().parent
def main():
 p=ready();out=B/'profile01';out.mkdir(exist_ok=False)
 (out/'POLICY.json').write_bytes((B/'POLICY.json').read_bytes())
 d=dict(policy_sha256=sha(B/'POLICY.json'),profiler_sha256=sha(__file__),cost_hashes_sha256=sha(B/'COST_HASHES.json'),policy_fixed_before_timing=True,real_UWB_values_labels_or_fit=False)
 (out/'DECLARATION_BEFORE_TIMING.json').write_text(json.dumps(d,indent=2)+'\n');resource.setrlimit(resource.RLIMIT_CPU,(590,600));signal.alarm(610)
 sys.path.insert(0,'/workspace/uwb-dlinear-source-probe-20261102');from native_dlinear import np,torch,make,capture,F,forward,step
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 start=time.monotonic();m,opt,_=make(p['seed']);assert sum(v.numel() for v in m.parameters())==628752
 g=np.random.RandomState(p['seed']);x=torch.from_numpy(g.randn(32,90,567).astype('float32'));y=torch.arange(32)%12
 training=[];evaluation=[];native=[];full=[]
 m.train()
 for _ in range(8):
  t=time.monotonic();_,loss,_=step(m,opt,x,y);training.append(time.monotonic()-t);assert np.isfinite(loss)
 m.eval()
 with torch.no_grad():
  for _ in range(4):
   t=time.monotonic();loss=F.cross_entropy(forward(m,x),y);evaluation.append(time.monotonic()-t);assert np.isfinite(float(loss))
 for i in range(3):
  t=time.monotonic();torch.save(m.state_dict(),out/f'native{i}.pt');native.append(time.monotonic()-t)
  t=time.monotonic();torch.save(capture(m,opt),out/f'full{i}.pt');full.append(time.monotonic()-t)
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;r=select(p,training,evaluation,native,full,rss)
 r.update(policy=p,total_parameters=628752,trainable_tensors=6,full_TSLib_DLinear_classification_CLI=True,training_seconds=training,evaluation_seconds=evaluation,native_save_seconds=native,fullstate_save_seconds=full,max_RSS_KiB=rss,wall_seconds=time.monotonic()-start,synthetic_fixture_only=True,real_UWB_values_labels_or_fit=False,real_pilot_started=False)
 (out/'COST_CHECK.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));signal.alarm(0)
if __name__=='__main__':main()
