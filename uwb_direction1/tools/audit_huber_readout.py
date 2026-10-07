"""Independent post-run IRLS and provenance replay; no model selection."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import huber
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump


def audit(root):
 c=json.loads((root/'PROTOCOL.json').read_text());s=json.loads((root/'SUMMARY.json').read_text());manifest=json.loads((root/'FILES.json').read_text());ident=json.loads((root/'IDENTITY.json').read_text())
 for n,sha in manifest.items():assert digest(root/n)==sha
 for n,sha in ident['code_sha256'].items():assert digest(Path(__file__).with_name(n))==sha
 assert digest(root/'PROTOCOL.json')==ident['protocol_sha256']
 raw=Path('/workspace/uwb-restored-20260930/cache');y=np.load(raw/'y_clutterremoved.npy');u=np.load(raw/'subj_clutterremoved.npy');assert digest(raw/'y_clutterremoved.npy')=='d77b10f0fbda231246e71dfcf0b2f852863af48f80f159aed90f5f9b7c61a8bd';assert digest(raw/'subj_clutterremoved.npy')=='5ab32ab59e5b41c05d301b76949f159ddb1e9423a749deb760585a5f19cf9480'
 values={k:[] for k in c['arms']};maxerr=0.;maxgap=0.;maxgrad=0.;refits=0;iterations=[]
 for h in c['users']:
  d=root/f'held{h}';m=np.load(d/'models.npz');t=np.load(d/'training.npz');p=np.load(d/'predictions.npz');ev=np.load(d/'evaluation.npz');split=json.loads((d/'split.json').read_text());sel=json.loads((d/'selection.json').read_text());old=Path(c['readout_reference'])/f'held{h}';orig=json.loads((old/'selection.json').read_text())['onehot'];oldm=np.load(old/'models.npz');assert sel['inherited_onehot']==orig and sel['new_selection'] is False;lam=orig['candidates'][orig['selected']]['lam'];assert sel['lambda_used']==lam
  for key in ['mean','scale']:np.testing.assert_array_equal(m[key],oldm[key])
  np.testing.assert_array_equal(m['ridge'],np.vstack([oldm['coefs'][orig['selected']],oldm['bias']]))
  eligible=np.isin(u,[3,4,5,6,7,8])&~((u==6)&(y==3));v=h+1 if h<8 else 3;roles={'train':np.flatnonzero(eligible&(u!=h)&(u!=v)),'val':np.flatnonzero(eligible&(u==v)),'diagnostic':np.flatnonzero(eligible&(u==h))}
  for role,ids in roles.items():np.testing.assert_array_equal(ids,split[role])
  tr=roles['train'];np.testing.assert_array_equal(tr,t['ids']);f=np.load(Path(c['feature_reference'])/f'held{h}/features.npz');lookup={int(i):j for j,i in enumerate(f['ids'])}
  def x(ids):return np.c_[(f['hidden'][[lookup[int(i)] for i in ids]]-m['mean'])/m['scale'],np.ones(len(ids))]
  xx=x(tr);target=np.eye(12)[y[tr]];users=np.unique(u[tr]);w=np.array([1/(len(users)*len(np.unique(y[tr][u[tr]==uu]))*np.sum((u[tr]==uu)&(y[tr]==yy))) for yy,uu in zip(y[tr],u[tr])]);np.testing.assert_array_equal(w,t['source_weights']);res=xx@m['ridge']-target;mad=np.median(abs(res-np.median(res,axis=0)),axis=0)/c['normal_mad'];delta=c['epsilon']*np.maximum(mad,c['scale_floor']);np.testing.assert_array_equal(delta,t['delta']);np.testing.assert_array_equal(mad,t['mad']);pen=np.diag(np.r_[np.repeat(lam,128),0.]);reference=np.zeros((129,12))
  for k in range(12):
   theta=np.zeros(129)
   for j in range(1000):
    r=xx@theta-target[:,k];q=np.minimum(1,delta[k]/np.maximum(abs(r),1e-300));ww=w*q;theta=np.linalg.solve(xx.T@(ww[:,None]*xx)+pen,xx.T@(ww*target[:,k]));r=xx@theta-target[:,k];g=xx.T@(w*np.clip(r,-delta[k],delta[k]))+pen@theta
    if abs(g).max()<1e-10:break
   assert j<999;iterations.append(j+1);reference[:,k]=theta;refits+=1;actual=m['huber'][:,k];r=xx@actual-target[:,k];q=np.minimum(1,delta[k]/np.maximum(abs(r),1e-300));np.testing.assert_array_equal(q,t['influence'][:,k]);g=xx.T@(w*np.clip(r,-delta[k],delta[k]))+pen@actual;maxgrad=max(maxgrad,float(abs(g).max()));assert abs(g).max()<=c['gradient_tolerance']
   gap=float(abs((w@huber(delta[k],r)+.5*actual@pen@actual)-(w@huber(delta[k],xx@theta-target[:,k])+.5*theta@pen@theta)));maxgap=max(maxgap,gap);assert gap<1e-10
   qq=q.copy()
   for uu in users:
    for yy in np.unique(y[tr][u[tr]==uu]):
     ix=np.flatnonzero((u[tr]==uu)&(y[tr]==yy));rng=np.random.default_rng(np.random.SeedSequence([c['seed'],h,k,int(uu),int(yy)]));qq[ix]=q[rng.permutation(ix)];np.testing.assert_array_equal(np.sort(qq[ix]),np.sort(q[ix]))
   np.testing.assert_array_equal(qq,t['shuffled_influence'][:,k])
   for arm,influence in [('one_step',np.minimum(1,delta[k]/np.maximum(abs(res[:,k]),1e-300))),('shuffled',qq)]:
    ww=w*influence;bb=np.linalg.solve(xx.T@(ww[:,None]*xx)+pen,xx.T@(ww*target[:,k]));np.testing.assert_allclose(bb,m[arm][:,k],atol=1e-9,rtol=1e-9);refits+=1
  for role in ['val','diagnostic']:
   ids=roles[role];np.testing.assert_array_equal(ids,p[role+'_ids']);np.testing.assert_array_equal(y[ids],ev[role+'_y']);zz=x(ids)
   for arm in values:np.testing.assert_array_equal(zz@m[arm],p[role+'_'+arm])
   ref=zz@reference;actual=p[role+'_huber'];err=float(abs(ref-actual).max());maxerr=max(maxerr,err);assert err<5e-4;np.testing.assert_array_equal(ref.argmax(1),actual.argmax(1))
  for arm in values:
   truth=ev['diagnostic_y'];pred=p['diagnostic_'+arm].argmax(1);values[arm].append(float(np.mean([(pred[truth==k]==k).mean() for k in np.unique(truth)])))
 assert values==s['scores'] and {k:float(np.mean(v)) for k,v in values.items()}==s['means'];gates={}
 for arm,gain in c['gate']['minimum_mean_gain'].items():
  delta=np.array(values[c['primary']])-values[arm];gates[arm]=dict(mean_gain=float(delta.mean()),wins=int((delta>1e-12).sum()),minimum_gain=float(delta.min()),passed=bool(delta.mean()>=gain and (delta>1e-12).sum()>=c['gate']['minimum_wins'] and delta.min()>=-c['gate']['maximum_paired_decline']))
 assert gates==s['comparisons'] and all(g['passed'] for g in gates.values())==s['gate_passed']
 out=dict(state='passed',artifact_hashes=len(manifest),independent_output_refits=refits,independent_huber_refits=72,independent_huber_role_score_systems=144,max_score_error=maxerr,max_objective_gap=maxgap,max_gradient_inf=maxgrad,irls_max_iterations=max(iterations),all_independent_huber_argmax_equal=True,all_saved_model_predictions_bitwise_equal=True,roles_ids_scale_delta_weights_permutations_selection_truth_gate_verified=True,gate_passed=s['gate_passed']);dump(root/'AUDIT.json',out);print(json.dumps(out))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
 with threadpool_limits(limits=1):audit(a.root)
