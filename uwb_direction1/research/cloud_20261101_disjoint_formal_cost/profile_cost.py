"""Guarded synthetic-only full DCNN2L profiler; no policy chosen or executed.

This CLI needs a later explicit fixed policy after current full audit/private
closure. No actual UWB values, gesture labels or fitting are used by a profile.
"""
import argparse,hashlib,json,resource,signal,sys,time
from pathlib import Path
from cost_tools import validate,select
SOURCE=Path('/workspace/uwb-disjoint-source-probe-20261031');ENGINE=Path('/workspace/uwb-disjoint-engine-20261031')
sys.path.insert(0,str(SOURCE))
from source6_closure import prior_closed

def main(policy_path,out,prior):
 prior_closed(prior)  # before policy read, directory creation, model import/fit
 raw=Path(policy_path).read_bytes();policy=validate(json.loads(raw))
 if Path(policy['prior_workspace']).resolve()!=Path(prior).resolve():raise RuntimeError('Different prior closure identity')
 # Both source and epoch engine are already remotely pinned, without mutation.
 for root in [SOURCE,ENGINE]:
  for name,digest in json.loads((root/'PREPARATION_HASHES.json').read_bytes())['files'].items():
   if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Changed source/engine dependency: '+name)
 out=Path(out);out.mkdir(exist_ok=False);(out/'POLICY.json').write_bytes(raw)
 (out/'DECLARATION_BEFORE_TIMING.json').write_text(json.dumps(dict(policy_sha256=hashlib.sha256(raw).hexdigest(),utility_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),declared_unix=time.time(),policy_fixed_before_timing=True,real_UWB_values_labels_or_fit=False),indent=2)+'\n')
 resource.setrlimit(resource.RLIMIT_CPU,(590,600));signal.alarm(610)
 sys.path.insert(0,str(ENGINE));from source_engine import setup,snapshot,close,np,torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic()
 h,m,cb,settings=setup(out,policy['seed'],4700,max(policy['epoch_candidates']),policy['batch_size']);batch=settings['actual_batch']
 assert m.count_params()==2587468 and sum(int(np.prod(v.shape)) for v in m.trainable_variables)==2586956
 from native_disjoint import create_class_weight
 rng=np.random.RandomState(policy['seed']);x=rng.randn(batch,90,567,1).astype('float32');y=np.eye(12,dtype='float32')[np.arange(batch)%12]
 # Purely synthetic 4700-row label fixture; never actual class counts/labels.
 cw=create_class_weight(np.eye(12,dtype='float32')[np.arange(4700)%12]);weight=np.array([cw[int(c)] for c in y.argmax(1)],dtype='float32')
 training=[];evaluation=[];ksave=[];fullstate=[]
 try:
  for _ in range(8):
   t=time.monotonic();logs=m.train_on_batch(x,y,sample_weight=weight,return_dict=True);training.append(time.monotonic()-t);assert all(np.isfinite(float(v)) for v in logs.values())
  m.reset_metrics()
  for _ in range(4):
   t=time.monotonic();logs=m.test_on_batch(x,y,return_dict=True);evaluation.append(time.monotonic()-t);assert all(np.isfinite(float(v)) for v in logs.values())
  for i in range(3):
   t=time.monotonic();m.save(out/('cost_native'+str(i)+'.keras'));ksave.append(time.monotonic()-t)
   t=time.monotonic();s=snapshot(h,m,settings,dict(synthetic_cost_only=True),0,8);torch.save(s,out/('cost_fullstate'+str(i)+'.pt'));fullstate.append(time.monotonic()-t)
  peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;result=select(policy,training_seconds=training,evaluation_seconds=evaluation,keras_save_seconds=ksave,snapshot_save_seconds=fullstate,max_RSS_KiB=peak)
  result.update(policy=policy,full_author_default_DCNN2L_architecture=True,total_parameters=2587468,trainable_parameters=2586956,original_training_classweight_formula=True,validation_unweighted=True,training_seconds=training,evaluation_seconds=evaluation,keras_save_seconds=ksave,snapshot_save_seconds=fullstate,max_RSS_KiB=peak,wall_seconds=time.monotonic()-start,synthetic_fixture_only=True,preprocessing_choice_recorded_but_no_UWB_preprocessing_executed=True,real_UWB_values_labels_or_fit=False,real_pilot_started=False)
  (out/'COST_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 finally:close(cb);signal.alarm(0)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--policy',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--prior',required=True,type=Path);a=p.parse_args();main(a.policy,a.out,a.prior)
