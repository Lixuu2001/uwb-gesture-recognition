"""One preregistered synthetic native full-size MLSTM-FCN cost execution."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import hashlib,json,resource,signal,sys,time
from closure import check
from cost_math import select
B=Path(__file__).resolve().parent
SOURCE=Path('/workspace/uwb-mlstmfcn-source-probe-20261031')
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 check() # before policy read, model import, output creation or any timing
 marker=read(B/'REMOTE_PREREGISTRATION.json');assert marker['state']=='remote_cost_preregistration_exact_readback_verified' and marker['cost_hashes_sha256']==sha(B/'COST_HASHES.json')
 for name,digest in read(B/'COST_HASHES.json').items():assert sha(B/name)==digest
 deps=read(B/'DEPENDENCIES.json')
 for name,digest in deps['native_source_and_runtime_references'].items():assert sha(Path(name))==digest,name
 for name,digest in read(SOURCE/'PREPARATION_HASHES.json')['files'].items():assert sha(SOURCE/name)==digest,name
 policy=read(B/'POLICY.json');assert policy['epochs']==250 and policy['batch_size']==128 and policy['seed']==20261029
 out=B/'profile01';out.mkdir(exist_ok=False);(out/'POLICY.json').write_bytes((B/'POLICY.json').read_bytes())
 declaration={'policy_sha256':sha(B/'POLICY.json'),'profiler_sha256':sha(Path(__file__)),'cost_hashes_sha256':sha(B/'COST_HASHES.json'),'declared_unix':time.time(),'policy_fixed_before_timing':True,'real_UWB_values_labels_or_fit':False}
 (out/'DECLARATION_BEFORE_TIMING.json').write_text(json.dumps(declaration,indent=2)+'\n')
 resource.setrlimit(resource.RLIMIT_CPU,(590,600));signal.alarm(610)
 sys.path.insert(0,str(SOURCE));from native_mlstm import make,capture,np,torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 start=time.monotonic();model=make(policy['seed']);assert model.count_params()==860364 and sum(int(np.prod(v.shape)) for v in model.trainable_variables)==859340
 assert len(model.trainable_variables)==21 and len(model.optimizer.variables)==44
 rng=np.random.RandomState(policy['seed']);x=rng.randn(128,567,90).astype('float32');y=np.eye(12,dtype='float32')[np.arange(128)%12]
 # Native per-row weighting equivalent to the helper's inverse-frequency weights;
 # these counts are synthetic, not UWB class counts.
 fixture=np.arange(4700)%12;counts=np.bincount(fixture,minlength=12);weight=(4700/counts/12)[np.arange(128)%12].astype('float32')
 training=[];evaluation=[];ksave=[];fullsave=[]
 for _ in range(8):
  t=time.monotonic();logs=model.train_on_batch(x,y,sample_weight=weight,return_dict=True);training.append(time.monotonic()-t);assert all(np.isfinite(float(v)) for v in logs.values())
 model.reset_metrics()
 for _ in range(4):
  t=time.monotonic();logs=model.test_on_batch(x,y,return_dict=True);evaluation.append(time.monotonic()-t);assert all(np.isfinite(float(v)) for v in logs.values())
 for i in range(3):
  t=time.monotonic();model.save(out/f'cost_native{i}.keras');ksave.append(time.monotonic()-t)
  t=time.monotonic();state=capture(model);torch.save(state,out/f'cost_fullstate{i}.pt');fullsave.append(time.monotonic()-t)
 peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;result=select(policy,training,evaluation,ksave,fullsave,peak)
 result.update(policy=policy,full_documented_MLSTM_FCN8_128_256_128_SE16_Dropout08=True,total_parameters=860364,trainable_parameters=859340,LSTM_scan_steps=567,native_recurrent_gate='sigmoid',legacy_TF_numeric_parity=False,synthetic_inverse_frequency_sample_weights=True,validation_unweighted=True,training_seconds=training,evaluation_seconds=evaluation,native_save_seconds=ksave,fullstate_save_seconds=fullsave,max_RSS_KiB=peak,wall_seconds=time.monotonic()-start,synthetic_fixture_only=True,real_UWB_values_labels_or_fit=False,real_pilot_started=False)
 (out/'COST_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));signal.alarm(0)
if __name__=='__main__':main()
