"""One registered source-user pilot. Explicit invocation only after MVGLF audit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,hashlib,signal,resource,pickle,traceback,fcntl
from pathlib import Path
from adapter import FixedMultiRocket,fit_readout,logits
import numpy as np
from threadpoolctl import threadpool_limits
B=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
 return h.hexdigest()
def write(p,x):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def macro(y,p):return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)]))
def main():
 cfg=json.loads((B/'PROTOCOL.json').read_text())
 # Fail before making a run directory or opening arrays if predecessor unfinished.
 previous=json.loads(Path(cfg['previous_state']).read_text())
 assert previous['state']=='complete','wait original CNN queue; do not overlap real experiments'
 audited=json.loads(Path(cfg['previous_complete_audit']).read_text())
 assert audited['state']=='pass' and audited['scope']=='complete'
 out=B/'run01';out.mkdir(exist_ok=False)
 lock=open(out/'worker.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 start=time.monotonic();cpu0=time.process_time()
 state=dict(state='starting',pid=os.getpid(),diagnostic_labels_read=False,users7_8_labels_used=False)
 def record(**kw):
  state.update(kw);state.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu0);write(out/'STATE.json',state)
 def stop(sig,frame):raise TimeoutError('budget signal '+str(sig))
 for s in (signal.SIGTERM,signal.SIGALRM,signal.SIGXCPU):signal.signal(s,stop)
 resource.setrlimit(resource.RLIMIT_CPU,(1780,1790));signal.setitimer(signal.ITIMER_REAL,1800)
 record()
 try:
  hashes=json.loads((B/'FROZEN_HASHES.json').read_text())
  for p,h in hashes.items():assert sha(B/p)==h,(p,'changed')
  write(out/'IDENTITY.json',dict(frozen_hashes=hashes,protocol_sha256=sha(B/'PROTOCOL.json'),pid=os.getpid()))
  assert json.loads((B/'SYNTHETIC_CHECK.json').read_text())['state']=='pass'
  cost=json.loads((B/'COST.json').read_text());assert cost['passed']
  assert cost['adapter_sha256']==sha(B/'adapter.py')
  assert sha(cfg['source_manifest'])==cfg['source_manifest_sha256']
  with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
  assert len(ids)==6900 and set(users)==set(range(1,7))
  tr=np.flatnonzero(np.isin(users,[3,4,5,6]));va=np.flatnonzero(users==2);te=np.flatnonzero(users==1)
  assert (len(tr),len(va),len(te))==(4700,1000,1200)
  assert len(set(ids))==6900
  write(out/'split.json',dict(train=ids[tr].tolist(),validation=ids[va].tolist(),diagnostic=ids[te].tolist()))
  root=Path(cfg['data'])/'cache'
  for n,h in cfg['input_sha256'].items():assert sha(root/n)==h
  labels=np.load(root/'y_clutterremoved.npy',mmap_mode='r');ytr=np.array(labels[ids[tr]],copy=True);yv=np.array(labels[ids[va]],copy=True)
  x=np.load(root/'X_clutterremoved.npy',mmap_mode='r')
  data=np.ascontiguousarray(np.asarray(x[ids]).transpose(0,1,3,2).reshape(6900,567,90),dtype='float32');assert np.isfinite(data).all()
  sys.path.insert(0,str(Path(cfg['minirocket_code']).parent));import minirocket_multivariate as mini
  assert sha(cfg['minirocket_code'])==cfg['minirocket_code_sha256']
  assert sha(cfg['minirocket_parameters'])==cfg['minirocket_parameters_sha256']
  with np.load(cfg['minirocket_parameters']) as z:mp=tuple(z['p'+str(i)] for i in range(5))
  multi=FixedMultiRocket(cfg['seed']).fit(data[tr])
  with open(out/'multirocket_transform.pkl','wb') as f:pickle.dump(multi,f)
  with open(out/'minirocket_parameters.pkl','wb') as f:pickle.dump(mp,f)
  receipts={}
  for arm,nf in [('multirocket',49728),('minirocket',9996)]:
   d=out/arm;d.mkdir();record(state='transforming',arm=arm)
   features=np.lib.format.open_memmap(d/'features.npy',mode='w+',dtype='float32',shape=(6900,nf))
   for b in range(0,6900,64):
    v=multi.transform(data[b:b+64]) if arm=='multirocket' else mini.transform(data[b:b+64],mp)
    assert np.isfinite(v).all();features[b:b+64]=v
    if b%512==0:record(completed_rows=min(b+64,6900))
   features.flush();record(state='fitting_readout',arm=arm)
   readout=fit_readout(features[tr],ytr,users[tr])
   with open(d/'readout.pkl','wb') as f:pickle.dump(readout,f)
   # Both validation and diagnostic outputs freeze before any diagnostic labels.
   predictions=np.concatenate([logits(readout,features[ii]) for ii in [va,te]])
   np.savez_compressed(d/'predictions_unscored.npz',ids=ids[np.r_[va,te]],logits=predictions,classes=readout['classes'])
   receipts[arm]=dict(prediction_sha256=sha(d/'predictions_unscored.npz'),readout_sha256=sha(d/'readout.pkl'),feature_sha256=sha(d/'features.npy'),normal_equation_relative_residual=readout['normal_equation_relative_residual'])
   write(d/'FROZEN.json',receipts[arm]);del features,readout
  write(out/'ALL_PREDICTIONS_FROZEN.json',receipts)
  record(state='predictions_frozen',diagnostic_labels_read=False)
  yt=np.array(labels[ids[te]],copy=True);scores={};validation={}
  for arm in ['multirocket','minirocket']:
   with np.load(out/arm/'predictions_unscored.npz') as z:
    p=z['classes'][z['logits'].argmax(1)];scores[arm]=macro(yt,p[len(va):]);validation[arm]=macro(yv,p[:len(va)])
  cnn=json.loads(Path(cfg['previous_summary']).read_text())['scores'];strong=max(cnn['full'],cnn['local'])
  result=dict(state='complete_exploratory_source_pilot',scores=scores,validation_scores=validation,cnn_reference=cnn,gate_passed=bool(scores['multirocket']-scores['minirocket']>=.005 and scores['multirocket']>=strong-.005),independent_validation=False,novel_contribution_validated=False,users7_8_labels_used=False,diagnostic_user=1,validation_user=2,training_users=[3,4,5,6],wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu0)
  write(out/'SUMMARY.json',result);record(state='complete',pid=None,diagnostic_labels_read=True)
  write(out/'FILES.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix!='.tmp'})
  print(json.dumps(result),flush=True)
 except BaseException as e:
  record(state='incomplete',pid=None,error=repr(e),traceback=traceback.format_exc(),scientific_negative=False);raise
if __name__=='__main__':
 assert sys.argv[1:]==['--run'],'explicit --run after completed predecessor audit only'
 with threadpool_limits(limits=1):main()
