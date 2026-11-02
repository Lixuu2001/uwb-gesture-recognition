import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,pickle,time,hashlib,signal,resource,traceback
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import recall_score
from threadpoolctl import threadpool_limits
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'vendor/hydra/code'))
from hydra import SparseScaler
import torch
O=B/'hydra_readout_completion';O.mkdir(exist_ok=False);start=time.monotonic();old=json.loads((B/'hydra_run01/SUMMARY.json').read_text());cpu_budget=int(885-old['cpu_seconds']-10)
def save(p,x):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stop(sig,frame):raise TimeoutError('cumulative HYDRA computation budget')
signal.signal(signal.SIGXCPU,stop);signal.signal(signal.SIGTERM,stop);resource.setrlimit(resource.RLIMIT_CPU,(cpu_budget,cpu_budget+5))
try:
 torch.set_num_threads(1)
 save(O/'state.json',dict(state='running_exact_linear_completion',pid=os.getpid()))
 save(O/'identity.json',dict(code_sha256=sha(Path(__file__)),amendment_sha256=sha(B/'HYDRA_NUMERICAL_AMENDMENT.json'),feature_sha256=sha(B/'hydra_run01/source_features.npz'),original_summary_sha256=sha(B/'hydra_run01/SUMMARY.json'),cpu_budget_remaining=cpu_budget,original_cpu_seconds=old['cpu_seconds']))
 with np.load(B/'hydra_run01/source_features.npz') as z:f=torch.from_numpy(z['features']);ids=z['ids']
 with np.load(B/'run01/source_ids.npz') as z:u=z['users'];y=z['labels'];np.testing.assert_array_equal(ids,z['ids'])
 rows=[]
 with threadpool_limits(limits=1):
  for h in range(1,7):
   tr=np.flatnonzero((u!=h)&(u!=h%6+1));ev=np.flatnonzero((u==h)|(u==h%6+1));te=u[ev]==h
   with open(B/f'hydra_run01/held{h}/readout.pkl','rb') as z:scaler,_=pickle.load(z)
   a=scaler.transform(f[tr]).numpy().astype('float64');e=scaler.transform(f[ev]).numpy().astype('float64')
   w=np.zeros(len(tr))
   for v in np.unique(u[tr]):
    for c in np.unique(y[tr][u[tr]==v]):
     m=(u[tr]==v)&(y[tr]==c);w[m]=1/m.sum()
   w*=len(tr)/w.sum();model=RidgeClassifier(alpha=1,solver='cholesky');model.fit(a,y[tr],sample_weight=w)
   # Verify weighted, centered normal equations before diagnostic scores.
   target=np.where(y[tr,None]==model.classes_[None,:],1.,-1.);ac=a-np.average(a,axis=0,weights=w);bc=target-np.average(target,axis=0,weights=w)
   rhs=ac.T@(bc*w[:,None]);res=ac.T@((ac@model.coef_.T-bc)*w[:,None])+model.coef_.T
   residual=float(np.linalg.norm(res)/max(np.linalg.norm(rhs),1e-30));assert residual<1e-6
   p=model.predict(e)
   with np.load(B/f'hydra_run01/held{h}/predictions.npz') as ref:
    np.testing.assert_array_equal(ids[ev],ref['ids']);mini=ref['minirocket'];raw=ref['raw_ridge']
   score={name:float(recall_score(y[ev][te],pp[te],labels=np.unique(y[ev][te]),average='macro',zero_division=0)) for name,pp in [('hydra',p),('minirocket',mini),('raw_ridge',raw)]}
   d=O/f'held{h}';d.mkdir();np.savez_compressed(d/'predictions.npz',ids=ids[ev],users=u[ev],y=y[ev],hydra=p,minirocket=mini,raw_ridge=raw)
   with open(d/'readout.pkl','wb') as z:pickle.dump((scaler,model),z)
   row=dict(held=h,scores=score,normal_equation_relative_residual=residual);rows.append(row);save(d/'result.json',row);save(O/'state.json',dict(state='solving',completed_folds=len(rows),pid=os.getpid()));print(json.dumps(row),flush=True)
  gains=[r['scores']['hydra']-r['scores']['minirocket'] for r in rows]
  summary=dict(state='complete_numerical_completion',rows=rows,means={k:float(np.mean([r['scores'][k] for r in rows])) for k in ['hydra','minirocket','raw_ridge']},mean_gain_vs_mini=float(np.mean(gains)),wins_vs_mini=int(sum(g>0 for g in gains)),gate_passed=bool(np.mean(gains)>=.01 and sum(g>0 for g in gains)>=4 and min(gains)>=-.02),cpu_seconds=time.process_time(),wall_seconds=time.monotonic()-start,cumulative_cpu_seconds=old['cpu_seconds']+time.process_time(),cumulative_wall_seconds=old['wall_seconds']+time.monotonic()-start,users7_8_labels_used=False,no_new_randomness_or_features=True,strong_cnn_comparison_completed=False)
  save(O/'SUMMARY.json',summary);save(O/'state.json',dict(state='complete',pid=None));save(O/'FILES.json',{str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()});print(json.dumps(summary),flush=True)
except BaseException as e:save(O/'state.json',dict(state='incomplete',error=repr(e),traceback=traceback.format_exc()));raise
