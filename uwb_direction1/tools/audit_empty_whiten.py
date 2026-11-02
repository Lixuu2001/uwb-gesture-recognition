"""Post-run full-complex FFT/periodogram and independent regression replay."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.signal import periodogram
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump
from empty_whiten import features


def reference(raw,gain):
 x=np.asarray(raw,dtype=np.float64);x/=np.maximum(np.sqrt(np.mean(x*x,axis=(-2,-1),keepdims=True)),1e-12)
 if gain is not None:
  full=np.concatenate([gain,gain[:,-2:0:-1]],axis=-1);x=np.fft.ifft(np.fft.fft(x,n=512,axis=-1)*full[None,:,None,:],axis=-1).real[...,:189]
 blocks=[]
 for ti in range(9):
  blocks.append(np.stack([np.log1p(np.sqrt(np.mean(x[:,:,ti*10:(ti+1)*10,ri*9:(ri+1)*9]**2,axis=(-2,-1)))) for ri in range(21)],axis=-1))
 return np.stack(blocks,axis=2).reshape(len(x),567)


def audit(root):
 c=json.loads((root/'PROTOCOL.json').read_text());s=json.loads((root/'SUMMARY.json').read_text());manifest=json.loads((root/'FILES.json').read_text());ident=json.loads((root/'IDENTITY.json').read_text())
 for n,sha in manifest.items():assert digest(root/n)==sha
 for n,sha in ident['code_sha256'].items():assert digest(Path(__file__).with_name(n))==sha
 assert digest(root/'PROTOCOL.json')==ident['protocol_sha256']
 rawdir=Path('/workspace/uwb-restored-20260930/cache');raw=np.load(rawdir/'X_clutterremoved.npy',mmap_mode='r');assert digest(rawdir/'X_clutterremoved.npy')==c['input_sha256']['raw'];y=np.load(rawdir/'y_clutterremoved.npy');u=np.load(rawdir/'subj_clutterremoved.npy');assert digest(rawdir/'y_clutterremoved.npy')=='d77b10f0fbda231246e71dfcf0b2f852863af48f80f159aed90f5f9b7c61a8bd';assert digest(rawdir/'subj_clutterremoved.npy')=='5ab32ab59e5b41c05d301b76949f159ddb1e9423a749deb760585a5f19cf9480'
 vals={k:[] for k in c['arms']};maxdesc=0.;maxscore=0.;models=0;clips=0
 for h in c['users']:
  d=root/f'held{h}';m=np.load(d/'models.npz');p=np.load(d/'predictions.npz');ev=np.load(d/'evaluation.npz');tmp=np.load(d/'template.npz');ids=np.load(d/'descriptor_ids.npz')['ids'];split=json.loads((d/'split.json').read_text());sel=json.loads((d/'selection.json').read_text());hashes=json.loads((d/'FEATURE_HASHES.json').read_text());f=np.load(Path(c['feature_reference'])/f'held{h}/features.npz');np.testing.assert_array_equal(ids,f['ids']);lookup={int(v):j for j,v in enumerate(ids)}
  eligible=np.isin(u,[3,4,5,6,7,8])&~((u==6)&(y==3));v=h+1 if h<8 else 3;roles={'train':np.flatnonzero(eligible&(u!=h)&(u!=v)),'val':np.flatnonzero(eligible&(u==v)),'diagnostic':np.flatnonzero(eligible&(u==h))}
  for role,ii in roles.items():np.testing.assert_array_equal(ii,split[role])
  tr=roles['train'];empty=tr[y[tr]==11];np.testing.assert_array_equal(empty,tmp['empty_ids']);psds=[]
  for uu in sorted(set(u[empty])):
   ii=empty[u[empty]==uu];total=np.zeros((3,257))
   for st in range(0,len(ii),c['batch_size']):
    xx=np.asarray(raw[ii[st:st+c['batch_size']]],dtype=np.float64);xx/=np.maximum(np.sqrt(np.mean(xx*xx,axis=(-2,-1),keepdims=True)),1e-12);_,pp=periodogram(xx,fs=1,window='boxcar',detrend=False,nfft=512,return_onesided=False,scaling='density',axis=-1);total+=pp[...,:257].mean(-2).sum(0)
   psds.append(total/len(ii))
  np.testing.assert_allclose(psds,tmp['user_psd'],atol=1e-11,rtol=1e-11);psd=np.mean(psds,axis=0);np.testing.assert_allclose(psd,tmp['psd'],atol=1e-11,rtol=1e-11);fullpsd=np.concatenate([psd,psd[:,-2:0:-1]],-1);gain=1/np.sqrt(np.maximum(psd+.1*fullpsd.mean(-1)[:,None],1e-30));gain/=np.sqrt(np.mean(np.concatenate([gain,gain[:,-2:0:-1]],-1)**2,axis=-1))[:,None];np.testing.assert_allclose(gain,tmp['gain'],atol=1e-11,rtol=1e-11)
  for vv in range(3):
   perm=np.random.default_rng(np.random.SeedSequence([c['seed'],h,vv])).permutation(np.arange(1,256));np.testing.assert_array_equal(tmp['shuffled_gain'][vv,1:-1],tmp['gain'][vv,perm]);np.testing.assert_array_equal(tmp['shuffled_gain'][vv,[0,-1]],tmp['gain'][vv,[0,-1]]);np.testing.assert_array_equal(np.sort(tmp['gain'][vv]),np.sort(tmp['shuffled_gain'][vv]))
  users=np.unique(u[tr]);w=np.array([1/(len(users)*len(np.unique(y[tr][u[tr]==uu]))*np.sum((u[tr]==uu)&(y[tr]==yy))) for yy,uu in zip(y[tr],u[tr])]);ix={role:[lookup[int(i)] for i in ii] for role,ii in roles.items()};target=np.eye(12)[y[tr]]
  for arm,g in [('raw_blocks',None),('empty_whiten',tmp['gain']),('shuffled',tmp['shuffled_gain'])]:
   refs=[];sha=hashlib.sha256()
   for st in range(0,len(ids),c['batch_size']):
    batch=raw[ids[st:st+c['batch_size']]];actual=features(batch,g,c);sha.update(np.ascontiguousarray(actual,dtype='<f8').tobytes());ref=reference(batch,g);maxdesc=max(maxdesc,float(abs(ref-actual).max()));np.testing.assert_allclose(ref,actual,atol=2e-12,rtol=2e-12);refs.append(ref);clips+=len(batch)
   assert sha.hexdigest()==hashes[arm];desc=np.concatenate(refs);x=np.concatenate([f['hidden'],desc],1);xt=x[ix['train']];mu=np.sum(w[:,None]*xt,0);sd=np.sqrt(np.sum(w[:,None]*(xt-mu)**2,0));sd=np.where(sd>1e-12,sd,1.);bias=np.sum(w[:,None]*target,0);z=(xt-mu)/sd;gram=z.T@(w[:,None]*z);rhs=z.T@(w[:,None]*(target-bias));candidates=[];fitted=[]
   for lam in c['lambdas']:
    coef=np.linalg.solve(gram+lam*np.eye(695),rhs);fitted.append(coef);vp=(x[ix['val']]-mu)/sd@coef+bias;stored=p[arm+'_val_'+str(lam)];err=float(abs(vp-stored).max());maxscore=max(maxscore,err);np.testing.assert_allclose(vp,stored,atol=1e-8,rtol=1e-8);np.testing.assert_array_equal(vp.argmax(1),stored.argmax(1));truth=y[roles['val']];pred=stored.argmax(1);candidates.append(dict(lam=lam,macro=float(np.mean([(pred[truth==k]==k).mean() for k in np.unique(truth)])),mse=float(np.mean((stored-np.eye(12)[truth])**2))));models+=1
   assert candidates==sel[arm]['candidates'];j=min(range(3),key=lambda j:(-candidates[j]['macro'],candidates[j]['mse'],-candidates[j]['lam']));assert j==sel[arm]['selected']
   for key,ref in [('mean',mu),('scale',sd),('bias',bias),('coef',fitted[j])]:np.testing.assert_allclose(ref,m[arm+'_'+key],atol=1e-8,rtol=1e-8)
   for role in ['val','diagnostic']:
    ref=(x[ix[role]]-mu)/sd@fitted[j]+bias;actual=p[role+'_'+arm];maxscore=max(maxscore,float(abs(ref-actual).max()));np.testing.assert_allclose(ref,actual,atol=1e-8,rtol=1e-8);np.testing.assert_array_equal(ref.argmax(1),actual.argmax(1))
  for role in ['val','diagnostic']:np.testing.assert_array_equal(p[role+'_ids'],roles[role]);np.testing.assert_array_equal(ev[role+'_y'],y[roles[role]])
  for arm in vals:
   truth=ev['diagnostic_y'];pred=p['diagnostic_'+arm].argmax(1);vals[arm].append(float(np.mean([(pred[truth==k]==k).mean() for k in np.unique(truth)])))
  print(json.dumps(dict(audited_held=h)),flush=True)
 assert vals==s['scores'] and {k:float(np.mean(v)) for k,v in vals.items()}==s['means'];gates={}
 for arm,gain in c['gate']['minimum_mean_gain'].items():
  delta=np.array(vals[c['primary']])-vals[arm];gates[arm]=dict(mean_gain=float(delta.mean()),wins=int((delta>1e-12).sum()),minimum_gain=float(delta.min()),passed=bool(delta.mean()>=gain and (delta>1e-12).sum()>=c['gate']['minimum_wins'] and delta.min()>=-c['gate']['maximum_paired_decline']))
 assert gates==s['comparisons'] and all(g['passed'] for g in gates.values())==s['gate_passed']
 out=dict(state='passed',artifact_hashes=len(manifest),independent_readout_fits=models,independent_descriptor_clip_appearances=clips,max_descriptor_error=maxdesc,max_score_error=maxscore,all_argmax_equal=True,roles_source_empty_ids_psd_gain_permutation_featurehashes_validation_selection_truth_gate_verified=True,gate_passed=s['gate_passed']);dump(root/'AUDIT.json',out);print(json.dumps(out))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
 with threadpool_limits(limits=1):audit(a.root)
