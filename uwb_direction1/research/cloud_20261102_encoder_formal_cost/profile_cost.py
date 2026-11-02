"""One full review Encoder100/B12 synthetic timing after remote registration."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,resource,signal,sys,time
from guard import ready,sha
from cost_math import select
B=Path(__file__).resolve().parent
def main():
 p=ready();assert (p['epochs'],p['batch_size'],p['seed'])==(100,12,20261029)
 out=B/'profile01';out.mkdir(exist_ok=False);(out/'POLICY.json').write_bytes((B/'POLICY.json').read_bytes())
 d=dict(policy_sha256=sha(B/'POLICY.json'),profiler_sha256=sha(__file__),cost_hashes_sha256=sha(B/'COST_HASHES.json'),declared_unix=time.time(),policy_fixed_before_timing=True,real_UWB_values_labels_or_fit=False)
 (out/'DECLARATION_BEFORE_TIMING.json').write_text(json.dumps(d,indent=2)+'\n');resource.setrlimit(resource.RLIMIT_CPU,(590,600));signal.alarm(610)
 sys.path.insert(0,'/workspace/uwb-encoder-source-probe-20261102');from native_encoder import np,torch,make,capture
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 start=time.monotonic();h,m=make(out,p['seed']);assert m.count_params()==3613324 and len(m.trainable_weights)==21 and len(m.optimizer.variables)==44
 rng=np.random.RandomState(p['seed']);x=rng.randn(12,90,567).astype('float32');y=np.eye(12,dtype='float32')[np.arange(12)%12]
 training=[];evaluation=[];native=[];full=[]
 for _ in range(8):
  t=time.monotonic();logs=m.train_on_batch(x,y,return_dict=True);training.append(time.monotonic()-t);assert all(np.isfinite(float(z)) for z in logs.values())
 m.reset_metrics()
 for _ in range(4):
  t=time.monotonic();logs=m.test_on_batch(x,y,return_dict=True);evaluation.append(time.monotonic()-t);assert all(np.isfinite(float(z)) for z in logs.values())
 for i in range(3):
  t=time.monotonic();m.save(out/f'native{i}.keras');native.append(time.monotonic()-t)
  t=time.monotonic();torch.save(capture(m),out/f'fullstate{i}.pt');full.append(time.monotonic()-t)
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;r=select(p,training,evaluation,native,full,rss)
 r.update(policy=p,total_parameters=3613324,trainable_parameters=3613324,trainable_tensors=21,Adam_variables=44,full_review_Encoder_128_256_512=True,training_unweighted=True,validation_unweighted=True,training_seconds=training,evaluation_seconds=evaluation,native_save_seconds=native,fullstate_save_seconds=full,max_RSS_KiB=rss,wall_seconds=time.monotonic()-start,synthetic_fixture_only=True,real_UWB_values_labels_or_fit=False,real_pilot_started=False,legacy_TF_numeric_parity=False)
 (out/'COST_CHECK.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));signal.alarm(0)
if __name__=='__main__':main()
