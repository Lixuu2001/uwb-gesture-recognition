"""Read-only verification of a completed fixed MultiRocket pilot; no refitting."""
from adapter import FixedMultiRocket
import argparse,hashlib,json,pickle,resource,time,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import recall_score
from threadpoolctl import threadpool_limits
B=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def check_readout(features,y,users,readout):
 """Independent moments, weights, intercept and ridge normal-equation checks."""
 scaler=readout['scaler'];classes=readout['classes'];n,d=features.shape
 np.testing.assert_array_equal(classes,np.unique(y));assert int(scaler.n_samples_seen_)==n
 w=np.zeros(n,dtype='float64')
 for u in np.unique(users):
  for c in np.unique(y[users==u]):
   mask=(users==u)&(y==c);w[mask]=1/mask.sum()
 w*=n/w.sum();np.testing.assert_allclose(w,readout['weights'],rtol=0,atol=1e-14)
 sums=np.zeros(d);squares=np.zeros(d);gradient=np.zeros_like(readout['coef']);rhs=np.zeros_like(gradient);intercept_gradient=np.zeros(len(classes))
 for b in range(0,n,128):
  f=np.asarray(features[b:b+128],dtype='float64');sums+=f.sum(0);squares+=(f*f).sum(0)
  z=(f-scaler.mean_)/scaler.scale_;target=np.where(y[b:b+128,None]==classes[None,:],1.,-1.)
  residual=(z@readout['coef']+readout['intercept']-target)*w[b:b+128,None]
  gradient+=z.T@residual;rhs+=z.T@(target*w[b:b+128,None]);intercept_gradient+=residual.sum(0)
 mean=sums/n;variance=np.maximum(squares/n-mean*mean,0)
 np.testing.assert_allclose(scaler.mean_,mean,rtol=1e-10,atol=1e-10)
 np.testing.assert_allclose(scaler.var_,variance,rtol=1e-8,atol=1e-8)
 active=variance>1e-8
 np.testing.assert_allclose(scaler.scale_[active],np.sqrt(variance[active]),rtol=1e-8,atol=1e-8)
 assert readout['alpha']==1.
 relative=float(np.linalg.norm(gradient+readout['coef'])/max(np.linalg.norm(rhs),1))
 assert relative<1e-7 and np.linalg.norm(intercept_gradient)/n<1e-7
 return relative
def audit(output):
 start=time.monotonic();r=B/'run01'
 # Before any array or label access, require the complete-state and both freezes.
 assert json.loads((r/'STATE.json').read_text())['state']=='complete'
 summary=json.loads((r/'SUMMARY.json').read_text());cfg=json.loads((B/'PROTOCOL.json').read_text())
 assert not output.exists()
 frozen=json.loads((B/'FROZEN_HASHES.json').read_text())
 for p,h in frozen.items():assert sha(B/p)==h
 ident=json.loads((r/'IDENTITY.json').read_text());assert ident['frozen_hashes']==frozen and ident['protocol_sha256']==sha(B/'PROTOCOL.json')
 artifacts=json.loads((r/'FILES.json').read_text())
 for p,h in artifacts.items():assert sha(r/p)==h
 freeze=json.loads((r/'ALL_PREDICTIONS_FROZEN.json').read_text())
 for a in ['multirocket','minirocket']:
  assert freeze[a]==json.loads((r/a/'FROZEN.json').read_text())
  for n,k in [('predictions_unscored.npz','prediction_sha256'),('features.npy','feature_sha256'),('readout.pkl','readout_sha256')]:assert sha(r/a/n)==freeze[a][k]
 assert sha(cfg['source_manifest'])==cfg['source_manifest_sha256']
 with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
 assert len(ids)==len(np.unique(ids))==6900 and set(users)==set(range(1,7))
 split=json.loads((r/'split.json').read_text());tr=np.flatnonzero(np.isin(users,[3,4,5,6]));va=np.flatnonzero(users==2);te=np.flatnonzero(users==1)
 for name,ix,n in [('train',tr,4700),('validation',va,1000),('diagnostic',te,1200)]:assert len(ix)==n;np.testing.assert_array_equal(split[name],ids[ix])
 root=Path(cfg['data'])/'cache'
 for n,h in cfg['input_sha256'].items():assert sha(root/n)==h
 labels=np.load(root/'y_clutterremoved.npy',mmap_mode='r');ytr=np.array(labels[ids[tr]],copy=True)
 raw=np.load(root/'X_clutterremoved.npy',mmap_mode='r')
 with open(r/'multirocket_transform.pkl','rb') as f:multi=pickle.load(f)
 assert multi.seed==cfg['seed'] and multi.model.n_kernels==6250 and multi.model.normalise is False
 with open(r/'minirocket_parameters.pkl','rb') as f:mp=pickle.load(f)
 sys.path.insert(0,str(Path(cfg['minirocket_code']).parent));import minirocket_multivariate as mini
 assert sha(cfg['minirocket_code'])==cfg['minirocket_code_sha256']
 with np.load(cfg['minirocket_parameters']) as z:
  for i,v in enumerate(mp):np.testing.assert_array_equal(v,z['p'+str(i)])
 scores={};validation={};residuals={};ev=np.r_[va,te]
 # Fixed first two records per role, independent of labels/results, for extraction replay.
 sample=np.r_[tr[:2],va[:2],te[:2]]
 data=np.ascontiguousarray(np.asarray(raw[ids[sample]]).transpose(0,1,3,2).reshape(6,567,90),dtype='float32')
 for arm,nf in [('multirocket',49728),('minirocket',9996)]:
  features=np.load(r/arm/'features.npy',mmap_mode='r');assert features.shape==(6900,nf)
  replay=multi.transform(data) if arm=='multirocket' else mini.transform(data,mp)
  np.testing.assert_array_equal(replay,features[sample])
  with open(r/arm/'readout.pkl','rb') as f:readout=pickle.load(f)
  residuals[arm]=check_readout(features[tr],ytr,users[tr],readout)
  prediction=[]
  for b in range(0,len(ev),64):
   f=np.asarray(features[ev[b:b+64]],dtype='float64');prediction.append(((f-readout['scaler'].mean_)/readout['scaler'].scale_)@readout['coef']+readout['intercept'])
  replay=np.concatenate(prediction)
  with np.load(r/arm/'predictions_unscored.npz') as z:
   np.testing.assert_array_equal(z['ids'],ids[ev]);np.testing.assert_array_equal(z['classes'],readout['classes'])
   np.testing.assert_allclose(replay,z['logits'],rtol=1e-7,atol=1e-7);np.testing.assert_array_equal(replay.argmax(1),z['logits'].argmax(1))
   # Both arm freezes already verified before any diagnostic labels are indexed.
   pred=readout['classes'][replay.argmax(1)];yv=np.array(labels[ids[va]],copy=True);yt=np.array(labels[ids[te]],copy=True)
   scores[arm]=float(recall_score(yt,pred[len(va):],labels=np.unique(yt),average='macro',zero_division=0))
   validation[arm]=float(recall_score(yv,pred[:len(va)],labels=np.unique(yv),average='macro',zero_division=0))
   assert abs(scores[arm]-summary['scores'][arm])<1e-12 and abs(validation[arm]-summary['validation_scores'][arm])<1e-12
 cnn=json.loads(Path(cfg['previous_summary']).read_text());ca=json.loads(Path(cfg['previous_complete_audit']).read_text())
 assert ca['state']=='pass' and ca['scope']=='complete' and ca['scores']==cnn['scores']==summary['cnn_reference']
 gate=bool(scores['multirocket']-scores['minirocket']>=.005 and scores['multirocket']>=max(cnn['scores']['full'],cnn['scores']['local'])-.005)
 assert gate==summary['gate_passed']
 result=dict(state='pass',scope='completed_pilot_artifacts_moments_objective_predictions_metrics',scores=scores,validation_scores=validation,gate_passed=gate,
  verified_artifact_hashes=len(artifacts),extraction_replay_records_per_arm=6,extraction_replay_scope='first_two_per_role_not_all6900',prediction_replay_records_per_arm=len(ev),ridge_normal_equation_relative_residuals=residuals,
  independent_validation=False,novel_contribution_validated=False,users7_8_labels_used=False,refit_or_new_sampling=False,auditor_sha256=sha(Path(__file__)),audit_wall_seconds=time.monotonic()-start,audit_cpu_seconds=time.process_time())
 output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();resource.setrlimit(resource.RLIMIT_CPU,(870,890))
 with threadpool_limits(limits=1):audit(a.out)
