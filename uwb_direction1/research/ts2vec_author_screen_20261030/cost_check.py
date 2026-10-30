"""Full-size synthetic resource timing only; never imports dataset loaders."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import hashlib,json,math,resource,time
import numpy as np
import torch
from author_adapter import BASE,source_check,make_model,full_context_update
from readout_adapter import fit_svm
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not (BASE/'COST_CHECK.json').exists();started=time.monotonic();source_check()
 for n,h in json.loads((BASE/'COST_HASHES.json').read_text()).items():assert sha(BASE/n)==h,n
 for n in ('SYNTHETIC_CHECK.json','READOUT_SYNTHETIC_CHECK.json'):assert json.loads((BASE/n).read_text())['state']=='pass'
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 model=make_model();opt=torch.optim.AdamW(model._net.parameters(),lr=model.lr);assert opt.param_groups[0]['weight_decay']==.01
 data=torch.randn(16,90,567,generator=torch.Generator().manual_seed(2048));train_times=[];encode_times=[]
 for i in range(4):
  begin=time.monotonic();full_context_update(model,opt,data);seconds=time.monotonic()-begin
  if i:train_times.append(seconds)
 fake=data.numpy().copy()
 for i in range(4):
  begin=time.monotonic();z=model.encode(fake.copy(),encoding_window='full_series');assert z.shape==(16,320) and np.isfinite(z).all();seconds=time.monotonic()-begin
  if i:encode_times.append(seconds)
 rng=np.random.default_rng(814);features=rng.normal(size=(4700,320));labels=np.arange(4700)%12;query=rng.normal(size=(2200,320));begin=time.monotonic();classifier=fit_svm(features,labels);svm_seconds=time.monotonic()-begin
 begin=time.monotonic();scores=classifier.decision_function(query);predictions=classifier.predict(query);infer_seconds=time.monotonic()-begin;assert scores.shape==(2200,12) and predictions.shape==(2200,) and np.isfinite(scores).all()
 estimate=2*(600*max(train_times)+math.ceil(6900/16)*max(encode_times)+svm_seconds+infer_seconds)+120
 result=dict(state='synthetic_cost_screen_complete',policy_sha256=sha(BASE/'COST_POLICY.json'),parameters=sum(p.numel() for p in model._net.parameters()),encoder_iterations_profiled=4,encoder_iterations_estimated=600,training_fullbatch_seconds=train_times,encoding_fullbatch_seconds=encode_times,SVM_full_training_shape=[4700,320],SVM_original_C_candidates=10,SVM_original_training_CV_folds=5,SVM_jobs=1,SVM_full_fit_seconds=svm_seconds,SVM_2200_inference_seconds=infer_seconds,estimated_total_seconds=estimate,CPU_and_wall_gate_passed=estimate<=43200,real_UWB_values_or_labels_read=False,real_training_started=False,wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limitations='Two full90frame contexts conservatively replace random shorter crop contexts for synthetic timing only. Original masking/dropout/AdamW defaultdecay.01/initial-inclusive averaging preserved. Actual full600fit, SVM on real representations, resumable driver and source accuracy remain unexecuted. Passing estimate is not a runtime guarantee or real-launch permission.')
 (BASE/'COST_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
