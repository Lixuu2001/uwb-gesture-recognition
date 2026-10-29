import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,resource,signal,hashlib,pickle,traceback
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import recall_score
from threadpoolctl import threadpool_limits
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'vendor/hydra/code'))
from hydra_multivariate import HydraMultivariate
from hydra import SparseScaler
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def classifier(z,y,u,tr,ev):
 scaler=SparseScaler();a=scaler.fit_transform(z[tr]).numpy();e=scaler.transform(z[ev]).numpy()
 w=np.zeros(len(tr))
 for v in np.unique(u[tr]):
  for c in np.unique(y[tr][u[tr]==v]):
   m=(u[tr]==v)&(y[tr]==c);w[m]=1/m.sum()
 w*=len(tr)/w.sum();m=RidgeClassifier(alpha=1,solver='lsqr',tol=1e-4,max_iter=200);m.fit(a,y[tr],sample_weight=w)
 return m.predict(e),scaler,m

def run(mode):
 o=B/('hydra_cost' if mode=='cost' else 'hydra_run01');o.mkdir(exist_ok=False);t0=time.monotonic()
 def stop(sig,frame):raise TimeoutError('CPU/wall limit')
 signal.signal(signal.SIGXCPU,stop);signal.signal(signal.SIGTERM,stop);resource.setrlimit(resource.RLIMIT_CPU,(885,895))
 try:
  torch.set_num_threads(1);torch.set_num_interop_threads(1);cfg=json.loads((B/'HYDRA_PROTOCOL.json').read_text())
  save(o/'identity.json',dict(protocol_sha256=sha(B/'HYDRA_PROTOCOL.json'),runner_sha256=sha(__file__),author_commit=cfg['author_commit'],author_sha256={n:sha(B/'vendor/hydra/code'/n) for n in ['hydra.py','hydra_multivariate.py']},torch=torch.__version__,pid=os.getpid()))
  if mode!='cost':assert json.loads((B/'hydra_cost/COST.json').read_text())['passed']
  torch.manual_seed(cfg['seed']);m=HydraMultivariate(90,567);torch.manual_seed(cfg['seed']);m2=HydraMultivariate(90,567)
  assert all(torch.equal(a,b) for a,b in zip(m.W,m2.W)) and all(torch.equal(a,b) for a,b in zip(m.I,m2.I))
  toy=torch.from_numpy(np.random.default_rng(42).normal(size=(8,567,90)).astype('float32'))
  z=m(toy);torch.testing.assert_close(z,m.batch(toy,batch_size=3));torch.testing.assert_close(z[:3],m(toy[:3]));assert torch.isfinite(z).all() and z.shape==(8,4096)
  save(o/'compatibility.json',dict(deterministic_seed=True,batch_invariance=True,row_independence=True,finite=True,dimensions=4096))
  with np.load(B/'run01/source_ids.npz') as z:ids=z['ids'];u=z['users'];y=z['labels']
  assert len(ids)==6900 and set(u)==set(range(1,7))
  x=np.load(Path(cfg['data'])/'cache/X_clutterremoved.npy',mmap_mode='r');data=torch.from_numpy(np.ascontiguousarray(np.asarray(x[ids]).transpose(0,1,3,2).reshape(6900,567,90)))
  if mode=='cost':
   t=time.monotonic();m(data[:64]);fs=time.monotonic()-t
   dummy=torch.from_numpy(np.random.default_rng(1).random((4800,4096),dtype=np.float32));t=time.monotonic();classifier(dummy,np.arange(4800)%12,np.arange(4800)%4,np.arange(4600),np.arange(4600,4800));cs=time.monotonic()-t
   estimate=1.7*(fs/64*6900+6*cs);save(o/'COST.json',dict(seconds64=fs,synthetic_readout_seconds=cs,estimated_seconds=estimate,passed=estimate<850,diagnostic_outcomes_observed=False));save(o/'state.json',dict(state='cost_complete',wall_seconds=time.monotonic()-t0));return
  save(o/'state.json',dict(state='transforming',pid=os.getpid()))
  chunks=[]
  for b in range(0,6900,128):chunks.append(m(data[b:b+128]));save(o/'state.json',dict(state='transforming',completed_rows=min(6900,b+128),pid=os.getpid()))
  features=torch.cat(chunks);assert features.shape==(6900,4096) and torch.isfinite(features).all()
  torch.save(dict(W=m.W,I=m.I,dilations=m.dilations,paddings=m.paddings,k=m.k,g=m.g),o/'transform.pt');np.savez_compressed(o/'source_features.npz',features=features.numpy(),ids=ids)
  rows=[]
  for h in cfg['source_users']:
   d=o/f'held{h}';d.mkdir();val=h%6+1;tr=np.flatnonzero((u!=h)&(u!=val));ev=np.flatnonzero((u==h)|(u==val));te=u[ev]==h
   p,scaler,model=classifier(features,y,u,tr,ev)
   with np.load(B/f'run01/held{h}/predictions.npz') as ref:
    np.testing.assert_array_equal(ids[ev],ref['ids']);np.testing.assert_array_equal(y[ev],ref['y']);mini=ref['minirocket'];raw=ref['raw_ridge']
   scores={a:float(recall_score(y[ev][te],pred[te],labels=np.unique(y[ev][te]),average='macro',zero_division=0)) for a,pred in [('hydra',p),('minirocket',mini),('raw_ridge',raw)]}
   save(d/'split.json',dict(train=ids[tr].tolist(),validation=ids[ev[~te]].tolist(),diagnostic=ids[ev[te]].tolist()))
   np.savez_compressed(d/'predictions.npz',ids=ids[ev],users=u[ev],y=y[ev],hydra=p,minirocket=mini,raw_ridge=raw)
   with open(d/'readout.pkl','wb') as f:pickle.dump((scaler,model),f)
   row=dict(held=h,validation=val,scores=scores);rows.append(row);save(d/'result.json',row);print(json.dumps(row),flush=True)
  gains=[r['scores']['hydra']-r['scores']['minirocket'] for r in rows]
  s=dict(state='complete',rows=rows,means={k:float(np.mean([r['scores'][k] for r in rows])) for k in ['hydra','minirocket','raw_ridge']},mean_gain_vs_mini=float(np.mean(gains)),wins_vs_mini=int(sum(g>0 for g in gains)),worst_gain_vs_mini=float(min(gains)),gate_passed=bool(np.mean(gains)>=.01 and sum(g>0 for g in gains)>=4 and min(gains)>=-.02),strong_cnn_comparison_completed=False,users7_8_labels_used=False,independent_validation=False,wall_seconds=time.monotonic()-t0,cpu_seconds=time.process_time())
  save(o/'SUMMARY.json',s);save(o/'state.json',dict(state='complete',pid=None));save(o/'FILES.json',{str(p.relative_to(o)):sha(p) for p in o.rglob('*') if p.is_file()});print(json.dumps(s),flush=True)
 except BaseException as e:save(o/'state.json',dict(state='incomplete',error=repr(e),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-t0));raise
if __name__=='__main__':
 with threadpool_limits(limits=1),torch.no_grad():run(sys.argv[1])
