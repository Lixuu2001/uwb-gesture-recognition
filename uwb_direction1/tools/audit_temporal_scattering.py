"""Independent SciPy Hilbert/time-domain circular convolution and augmented-intercept ridge replay."""
import argparse,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump
from test_temporal_scattering import reference
from temporal_scattering import features,package_hashes
from run_temporal_scattering import array_sha


def audit(root):
 start=time.perf_counter();c=json.loads((root/'PROTOCOL.json').read_text());assert package_hashes()==c['package_sha256'];s=json.loads((root/'SUMMARY.json').read_text());manifest=json.loads((root/'FILES.json').read_text());ident=json.loads((root/'IDENTITY.json').read_text())
 for n,sha in manifest.items():assert digest(root/n)==sha
 for n,sha in ident['code_sha256'].items():assert digest(Path(__file__).with_name(n))==sha
 assert digest(root/'PROTOCOL.json')==ident['protocol_sha256']
 rawdir=Path('/workspace/uwb-restored-20260930/cache');assert digest(rawdir/'X_clutterremoved.npy')==c['input_sha256']['raw'];raw=np.load(rawdir/'X_clutterremoved.npy',mmap_mode='r');y=np.load(rawdir/'y_clutterremoved.npy');u=np.load(rawdir/'subj_clutterremoved.npy');assert digest(rawdir/'y_clutterremoved.npy')=='d77b10f0fbda231246e71dfcf0b2f852863af48f80f159aed90f5f9b7c61a8bd';assert digest(rawdir/'subj_clutterremoved.npy')=='5ab32ab59e5b41c05d301b76949f159ddb1e9423a749deb760585a5f19cf9480'
 ids=np.load(root/'descriptor_ids.npz')['ids'];eligible=np.isin(u,[3,4,5,6,7,8])&~((u==6)&(y==3));np.testing.assert_array_equal(ids,np.flatnonzero(eligible));refs={k:[] for k in c['arms'][1:]};actuals={k:[] for k in refs};maxdesc=0.
 for st in range(0,len(ids),16):
  ii=ids[st:st+16];a=features(raw[ii],ii,c['seed']);b=reference(raw[ii],ii,c['seed'])
  for k in refs:maxdesc=max(maxdesc,float(abs(a[k]-b[k]).max()));np.testing.assert_allclose(a[k],b[k],atol=2e-12,rtol=2e-12);refs[k].append(b[k]);actuals[k].append(a[k])
 refs={k:np.concatenate(v) for k,v in refs.items()};actuals={k:np.concatenate(v) for k,v in actuals.items()};assert {k:array_sha(v) for k,v in actuals.items()}==json.loads((root/'FEATURE_HASHES.json').read_text());del actuals
 vals={k:[] for k in c['arms']};maxscore=0.;fits=0;lookup={int(v):j for j,v in enumerate(ids)};froot=Path(c['feature_reference']);rroot=Path(c['readout_reference']);assert digest(froot/'FILES.json')==c['feature_manifest_sha256'];fm=json.loads((froot/'FILES.json').read_text());assert digest(rroot/'FILES.json')==c['readout_manifest_sha256'];rm=json.loads((rroot/'FILES.json').read_text())
 for h in c['users']:
  d=root/f'held{h}';m=np.load(d/'models.npz');p=np.load(d/'predictions.npz');ev=np.load(d/'evaluation.npz');split=json.loads((d/'split.json').read_text());sel=json.loads((d/'selection.json').read_text());assert digest(froot/f'held{h}/features.npz')==fm[f'held{h}/features.npz'];f=np.load(froot/f'held{h}/features.npz');np.testing.assert_array_equal(ids,f['ids']);v=h+1 if h<8 else 3;roles={'train':np.flatnonzero(eligible&(u!=h)&(u!=v)),'val':np.flatnonzero(eligible&(u==v)),'diagnostic':np.flatnonzero(eligible&(u==h))}
  for role,ii in roles.items():np.testing.assert_array_equal(ii,split[role])
  tr=roles['train'];users=np.unique(u[tr]);w=np.array([1/(len(users)*len(np.unique(y[tr][u[tr]==uu]))*np.sum((u[tr]==uu)&(y[tr]==yy))) for yy,uu in zip(y[tr],u[tr])]);ix={role:[lookup[int(i)] for i in ii] for role,ii in roles.items()};target=np.eye(12)[y[tr]]
  for arm,desc in refs.items():
   x=desc if arm=='standalone' else np.concatenate([f['hidden'],desc],1);xt=x[ix['train']];mu=np.sum(w[:,None]*xt,0);sd=np.sqrt(np.sum(w[:,None]*(xt-mu)**2,0));sd=np.where(sd>1e-12,sd,1.);z=np.column_stack([np.ones(len(tr)),(xt-mu)/sd]);gram=z.T@(w[:,None]*z);rhs=z.T@(w[:,None]*target);candidates=[];fitted=[]
   for j,lam in enumerate(c['lambdas']):
    penalty=np.diag([0.]+[lam]*x.shape[1]);coef=np.linalg.solve(gram+penalty,rhs);fitted.append(coef);vp=np.column_stack([np.ones(len(ix['val'])),(x[ix['val']]-mu)/sd])@coef;stored=p[arm+'_val_'+str(lam)];err=float(abs(vp-stored).max());maxscore=max(maxscore,err);np.testing.assert_allclose(vp,stored,atol=1e-8,rtol=1e-8);np.testing.assert_array_equal(vp.argmax(1),stored.argmax(1));truth=y[roles['val']];pred=stored.argmax(1);candidate=dict(lam=lam,macro=float(np.mean([(pred[truth==k]==k).mean() for k in np.unique(truth)])),mse=float(np.mean((stored-np.eye(12)[truth])**2)));candidates.append(candidate);np.testing.assert_allclose(coef[1:],m[arm+'_coefs'][j],atol=1e-8,rtol=1e-8);fits+=1
   assert candidates==sel[arm]['candidates'];j=min(range(3),key=lambda j:(-candidates[j]['macro'],candidates[j]['mse'],-candidates[j]['lam']));assert j==sel[arm]['selected']
   for key,ref in [('mean',mu),('scale',sd),('bias',fitted[j][0])]:np.testing.assert_allclose(ref,m[arm+'_'+key],atol=1e-8,rtol=1e-8)
   for role in ['val','diagnostic']:
    ref=np.column_stack([np.ones(len(ix[role])),(x[ix[role]]-mu)/sd])@fitted[j];actual=p[role+'_'+arm];maxscore=max(maxscore,float(abs(ref-actual).max()));np.testing.assert_allclose(ref,actual,atol=1e-8,rtol=1e-8);np.testing.assert_array_equal(ref.argmax(1),actual.argmax(1))
  for role in ['val','diagnostic']:np.testing.assert_array_equal(p[role+'_ids'],roles[role]);np.testing.assert_array_equal(ev[role+'_y'],y[roles[role]])
  rp=rroot/f'held{h}/predictions.npz';assert digest(rp)==rm[f'held{h}/predictions.npz'];old=np.load(rp);np.testing.assert_allclose(p['diagnostic_ridge'],old['onehot']+.5,atol=1e-12,rtol=1e-12);np.testing.assert_array_equal(old['ids'],roles['diagnostic'])
  for arm in vals:
   truth=ev['diagnostic_y'];pred=p['diagnostic_'+arm].argmax(1);vals[arm].append(float(np.mean([(pred[truth==k]==k).mean() for k in np.unique(truth)])))
  print(json.dumps(dict(audited_held=h)),flush=True)
 assert vals==s['scores'] and {k:float(np.mean(v)) for k,v in vals.items()}==s['means'];gates={}
 for arm,gain in c['gate']['minimum_mean_gain'].items():
  delta=np.array(vals[c['primary']])-vals[arm];gates[arm]=dict(mean_gain=float(delta.mean()),wins=int((delta>1e-12).sum()),minimum_gain=float(delta.min()),passed=bool(delta.mean()>=gain and (delta>1e-12).sum()>=c['gate']['minimum_wins'] and delta.min()>=-c['gate']['maximum_paired_decline']))
 assert gates==s['comparisons'] and all(g['passed'] for g in gates.values())==s['gate_passed']
 out=dict(state='passed',artifact_hashes=len(manifest),independent_readout_fits=fits,independent_descriptor_clips=len(ids),independent_descriptor_clip_arms=len(ids)*4,max_descriptor_error=maxdesc,max_score_error=maxscore,all_argmax_equal=True,source_ids_weights_featurehashes_all_coefficients_validation_selection_truth_gate_verified=True,gate_passed=s['gate_passed'],seconds=time.perf_counter()-start);dump(root/'AUDIT.json',out);print(json.dumps(out))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
 with threadpool_limits(limits=1):audit(a.root)
