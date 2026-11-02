"""Guarded synthetic timing utility; no policy/config selected in this file.

A future caller supplies a fixed JSON policy ONLY after prior audit/private
closure. This tool refuses before policy read when that closure is missing.
All inputs below are generated fixtures; no UWB values/labels or fit occurs.
"""
import argparse,hashlib,json,resource,sys,time,signal
from pathlib import Path
from cost_tools import validate,select
ENGINE=Path('/workspace/uwb-litemv-engine-20261031');sys.path.insert(0,str(ENGINE))
from readiness_guards import prior_closed
def main(policy_path,out,prior):
 prior_closed(prior)  # before reading or selecting any next policy/configuration
 policy=validate(json.loads(Path(policy_path).read_bytes()))
 if Path(policy['prior_workspace']).resolve()!=Path(prior).resolve():raise RuntimeError('Different prior closure identity')
 out=Path(out);out.mkdir(exist_ok=False);policy_bytes=Path(policy_path).read_bytes();(out/'POLICY.json').write_bytes(policy_bytes);(out/'DECLARATION_BEFORE_TIMING.json').write_text(json.dumps(dict(policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),utility_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),declared_unix=time.time(),fixed_policy_before_timing=True,real_UWB_values_labels_or_fit=False),indent=2))
 resource.setrlimit(resource.RLIMIT_CPU,(590,600));signal.alarm(610)
 from source_engine import Engine,np,torch,keras,source_znormalization
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();batch=policy['batch_size'];g=np.random.RandomState(policy['seed']);x=g.randn(batch,90,567).astype('float32');y=np.eye(12,dtype='float32')[np.arange(batch)%12]
 if policy['input_preprocessing']=='explicit_RMS_then_author_znormalization_adaptation':x=source_znormalization(x)
 engine=Engine(output=out,length=90,channels=567,classes=12,epochs=1,batch_size=batch,seed=policy['seed'],identity=dict(synthetic_cost_only=True,policy_sha256=hashlib.sha256(policy_bytes).hexdigest()));model=engine.model;assert model.count_params()==889624 and sum(int(np.prod(v.shape)) for v in model.trainable_variables)==621680
 training=[];evaluation=[];ksave=[];snapshot=[]
 for _ in range(8):
  t=time.monotonic();logs=model.train_on_batch(x,y,return_dict=True);training.append(time.monotonic()-t);assert all(np.isfinite(v) for v in logs.values())
 model.reset_metrics()
 for _ in range(4):
  t=time.monotonic();logs=model.test_on_batch(x,y,return_dict=True);evaluation.append(time.monotonic()-t);assert all(np.isfinite(v) for v in logs.values())
 for i in range(3):
  t=time.monotonic();model.save(out/('cost_checkpoint'+str(i)+'.keras'));ksave.append(time.monotonic()-t)
  t=time.monotonic();state=engine.checkpoint(epoch=1,batch=7,total_steps=8,sampler=dict(synthetic_only=True));torch.save(state,out/('cost_fullstate'+str(i)+'.pt'));snapshot.append(time.monotonic()-t)
 result=select(policy,training_seconds=training,evaluation_seconds=evaluation,keras_save_seconds=ksave,snapshot_save_seconds=snapshot,max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);result.update(policy=policy,full_author_LITEMV_architecture=True,total_parameters=889624,trainable_parameters=621680,training_seconds=training,evaluation_seconds=evaluation,keras_save_seconds=ksave,snapshot_save_seconds=snapshot,max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,wall_seconds=time.monotonic()-start,real_UWB_values_labels_or_fit=False,real_pilot_started=False);engine.close();signal.alarm(0);(out/'COST_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--policy',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--prior',required=True,type=Path);a=p.parse_args();main(a.policy,a.out,a.prior)
