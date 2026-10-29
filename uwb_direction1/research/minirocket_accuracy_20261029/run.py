import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']: os.environ[k]='1'
import sys,json,time,hashlib,signal,resource,traceback,subprocess,pickle,platform
from pathlib import Path
import numpy as np
from numba import njit,set_num_threads
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import recall_score
from threadpoolctl import threadpool_limits
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'vendor/minirocket/code'))
import minirocket_multivariate as mr
@njit
def seed_numba(seed): np.random.seed(seed)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def read_labels(labels,ids): return np.asarray(labels[ids])
def fit_predict(f,y,u,tr,ev):
 scaler=StandardScaler();a=scaler.fit_transform(f[tr]);b=scaler.transform(f[ev])
 weights=np.zeros(len(tr),dtype=float)
 for user in np.unique(u[tr]):
  for c in np.unique(y[tr][u[tr]==user]):
   m=(u[tr]==user)&(y[tr]==c);weights[m]=1/m.sum()
 weights*=len(tr)/weights.sum()
 model=RidgeClassifier(alpha=1.,solver='lsqr',tol=1e-4,max_iter=200)
 model.fit(a,y[tr],sample_weight=weights)
 return model.predict(b),scaler,model

def run(mode):
 out=BASE/('cost' if mode=='cost' else 'run01');out.mkdir(exist_ok=False)
 start=time.monotonic();write(out/'state.json',dict(state='starting',pid=os.getpid(),mode=mode))
 def limit(signum,frame): raise TimeoutError('process CPU limit reached')
 signal.signal(signal.SIGXCPU,limit);signal.signal(signal.SIGTERM,limit);resource.setrlimit(resource.RLIMIT_CPU,(1780,1790))
 try:
  cfg=json.loads((BASE/'PROTOCOL.json').read_text());set_num_threads(1)
  if mode!='cost': assert json.loads((BASE/'cost/COST.json').read_text())['passed']
  write(out/'identity.json',dict(protocol_sha256=sha(BASE/'PROTOCOL.json'),runner_sha256=sha(__file__),author_code_sha256=sha(BASE/'vendor/minirocket/code/minirocket_multivariate.py'),python=platform.python_version(),numpy=np.__version__,numba=__import__('numba').__version__,sklearn=__import__('sklearn').__version__,pid=os.getpid()))
  # Non-data determinism check: seeded Numba and NumPy RNGs both reset.
  toy=np.random.default_rng(7).normal(size=(8,3,90)).astype('float32')
  np.random.seed(cfg['seed']);seed_numba(cfg['seed']);p=mr.fit(toy)
  np.random.seed(cfg['seed']);seed_numba(cfg['seed']);q=mr.fit(toy)
  assert all(np.array_equal(a,b) for a,b in zip(p,q))
  z=mr.transform(toy,p);assert z.shape==(8,9996) and np.isfinite(z).all()
  write(out/'synthetic_check.json',dict(deterministic_fit=True,shape=list(z.shape),finite=True))
  root=Path(cfg['data']);x=np.load(root/'cache/X_clutterremoved.npy',mmap_mode='r');uall=np.load(root/'cache/subj_clutterremoved.npy',mmap_mode='r');ya=np.load(root/'cache/y_clutterremoved.npy',mmap_mode='r')
  ids0=np.flatnonzero(np.isin(uall,cfg['source_users']));us=np.asarray(uall[ids0]);ys=read_labels(ya,ids0)
  keep=~(((us==2)&np.isin(ys,[10,11]))|((us==6)&(ys==3)));ids=ids0[keep];u=us[keep];y=ys[keep]
  assert len(ids)==6900 and {int(h):int(sum(u==h)) for h in cfg['source_users']}=={1:1200,2:1000,3:1200,4:1200,5:1200,6:1100}
  np.savez_compressed(out/'source_ids.npz',ids=ids,users=u,labels=y)
  # Reorder only; all 189 range bins in all 3 views retained as 567 channels.
  data=np.ascontiguousarray(np.asarray(x[ids]).transpose(0,1,3,2).reshape(len(ids),567,90),dtype='float32');assert np.isfinite(data).all()
  rows=[]
  for held in (cfg['source_users'][:1] if mode=='cost' else cfg['source_users']):
   val=held%6+1;tr=np.flatnonzero((u!=held)&(u!=val));ev=np.flatnonzero((u==held)|(u==val));te=(u[ev]==held)
   np.random.seed(cfg['seed']);seed_numba(cfg['seed']);t=time.monotonic();params=mr.fit(np.ascontiguousarray(data[tr]));fitsec=time.monotonic()-t
   if mode=='cost':
    t=time.monotonic();f=mr.transform(np.ascontiguousarray(data[:64]),params);s=time.monotonic()-t
    # Readout timing uses synthetic features, no real diagnostic outcomes.
    dummy=np.random.default_rng(1).normal(size=(len(tr),9996)).astype('float32');t=time.monotonic();fit_predict(dummy,y[tr],u[tr],np.arange(len(tr)-64),np.arange(len(tr)-64,len(tr)));rs=time.monotonic()-t
    estimate=6*(fitsec+s/64*len(ids)+rs)*1.7
    write(out/'COST.json',dict(transform_seconds64=s,fit_seconds=fitsec,synthetic_readout_seconds=rs,estimated_seconds=estimate,budget_seconds=1800,passed=estimate<1700,diagnostic_scores_observed=False))
    write(out/'state.json',dict(state='complete_cost_only',wall_seconds=time.monotonic()-start));return
   fold=out/f'held{held}';fold.mkdir();write(fold/'split.json',dict(train=ids[tr].tolist(),validation=ids[ev[~te]].tolist(),diagnostic=ids[ev[te]].tolist()))
   np.savez_compressed(fold/'transform.npz',**{f'p{i}':p for i,p in enumerate(params)})
   f=np.empty((len(ids),9996),dtype='float32')
   for b in range(0,len(ids),128):
    f[b:b+128]=mr.transform(np.ascontiguousarray(data[b:b+128]),params)
    if b%1024==0: write(out/'state.json',dict(state='transforming',held=held,completed_rows=min(b+128,len(ids)),pid=os.getpid()))
   pred,scaler,model=fit_predict(f,y,u,tr,ev)
   raw=np.asarray(x[ids]).reshape(len(ids),3,15,6,21,9).mean(axis=(3,5)).reshape(len(ids),-1)
   rp,rscale,rmodel=fit_predict(raw,y,u,tr,ev)
   scores={name:float(recall_score(y[ev[te]],pp[te],labels=np.unique(y[ev[te]]),average='macro',zero_division=0)) for name,pp in [('minirocket',pred),('raw_ridge',rp)]}
   np.savez_compressed(fold/'predictions.npz',ids=ids[ev],users=u[ev],y=y[ev],minirocket=pred,raw_ridge=rp)
   with open(fold/'readouts.pkl','wb') as fobj:pickle.dump(dict(minirocket=(scaler,model),raw=(rscale,rmodel)),fobj)
   row=dict(held=held,validation=val,scores=scores,seconds=time.monotonic()-start);rows.append(row);write(fold/'result.json',row);print(json.dumps(row),flush=True)
   del f,raw
  gains=[r['scores']['minirocket']-r['scores']['raw_ridge'] for r in rows]
  result=dict(state='complete',rows=rows,means={k:float(np.mean([r['scores'][k] for r in rows])) for k in ['minirocket','raw_ridge']},mean_gain=float(np.mean(gains)),wins=int(sum(g>0 for g in gains)),gate_passed=bool(np.mean(gains)>=.01 and sum(g>0 for g in gains)>=4),strong_cnn_comparison_completed=False,independent_validation=False,users7_8_labels_read=False,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time())
  write(out/'SUMMARY.json',result);write(out/'state.json',dict(state='complete',pid=None));write(out/'FILES.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()});print(json.dumps(result),flush=True)
 except BaseException as e:
  write(out/'state.json',dict(state='incomplete',error=repr(e),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()));raise
if __name__=='__main__':
 with threadpool_limits(limits=1): run(sys.argv[1])
