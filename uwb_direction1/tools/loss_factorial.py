"""Known convex linear readouts; a baseline audit, never an innovation claim."""
import argparse,json,time,platform
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump
from second_order_core import weights,macro


def objective(theta,z,y,w,lam):
 b=theta.reshape(z.shape[1]+1,12);v=z@b[1:]+b[0];p=softmax(v,axis=1);r=p-np.eye(12)[y];r*=w[:,None]
 value=float(w@(logsumexp(v,axis=1)-v[np.arange(len(y)),y])+.5*lam*np.square(b[1:]).sum())
 return value,np.concatenate([r.sum(0)[None],z.T@r+lam*b[1:]],axis=0).ravel()


def fit(z,y,w,lam,loss):
 if loss=='ridge':
  a=np.column_stack([np.ones(len(z)),z]);b=np.linalg.solve(a.T@(w[:,None]*a)+np.diag([0.]+[lam]*z.shape[1]),a.T@(w[:,None]*np.eye(12)[y]));return b,dict(solver='normal_equations')
 r=minimize(objective,np.zeros((z.shape[1]+1)*12),args=(z,y,w,lam),jac=True,method='L-BFGS-B',options=dict(gtol=1e-9,ftol=1e-14,maxiter=3000,maxls=40));g=float(abs(r.jac).max());assert r.success and g<2e-7,(r.message,g)
 return r.x.reshape(z.shape[1]+1,12),dict(solver='L-BFGS-B',iterations=int(r.nit),gradient_max=g,objective=float(r.fun),success=bool(r.success))


def output(z,b,loss):
 v=z@b[1:]+b[0];return softmax(v,axis=1) if loss=='logit' else v


def choose(c):return min(range(len(c)),key=lambda j:(-c[j]['macro'],c[j]['mse'],-c[j]['lam']))


def inputs(c):
 f=Path(c['feature_reference']);old=Path(c['ridge_reference']);assert digest(f/'FILES.json')==c['feature_manifest_sha256'];assert digest(old/'FILES.json')==c['ridge_manifest_sha256']
 for root in [f,old]:
  for n,sha in json.loads((root/'FILES.json').read_text()).items():assert digest(root/n)==sha
 data=Path('/workspace/uwb-restored-20260930/cache');a=[]
 for n,k in [('y_clutterremoved.npy','labels'),('subj_clutterremoved.npy','users')]:assert digest(data/n)==c['input_sha256'][k];a.append(np.load(data/n))
 return f,old,*a


def main(a):
 c=json.loads(a.protocol.read_text());assert not a.out.exists();
 for n,sha in c['code_sha256'].items():assert digest(Path(__file__).with_name(n))==sha
 a.out.mkdir();dump(a.out/'PROTOCOL.json',c);dump(a.out/'IDENTITY.json',dict(protocol_sha256=digest(a.protocol),code_sha256={n:digest(Path(__file__).with_name(n)) for n in ['loss_factorial.py','second_order_core.py','audit_restored.py']},python=platform.python_version(),numpy=np.__version__))
 if a.cost_only:
  rng=np.random.default_rng(20261008);z=rng.normal(size=(7200,128));y=np.arange(7200)%12;z[:,:12]+=np.eye(12)[y]*2;w=np.ones(len(y))/len(y);t=time.perf_counter();_,info=fit(z,y,w,.001,'logit');elapsed=time.perf_counter()-t;v=dict(seconds=elapsed,estimate_seconds=elapsed*48*2,passed=elapsed*48*2<c['budget_seconds'],real_data_read=False,protocol_sha256=digest(a.protocol),optimization=info);dump(a.out/'COST.json',v);print(json.dumps(v));return
 cost=json.loads(a.cost.read_text());assert cost['passed'] and cost['protocol_sha256']==digest(a.protocol);f,old,y,u=inputs(c);ids=np.flatnonzero(~(((u==2)&np.isin(y,[10,11]))|((u==6)&(y==3))));assert len(ids)==9300;lookup=np.full(len(y),-1);lookup[ids]=np.arange(len(ids));start=time.perf_counter()
 for h in range(1,9):
  d=a.out/f'held{h}';d.mkdir();part=json.loads((f/f'held{h}/split.json').read_text());tr,va,te=[np.array(part[k]) for k in ['train','val','diagnostic']];assert set(u[tr])==set(range(1,9))-{h,h%8+1};assert set(u[va])=={h%8+1} and set(u[te])=={h};assert not(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
  features=np.load(f/f'held{h}/features.npz');np.testing.assert_array_equal(features['ids'],ids);x=features['hidden'];w=weights(y[tr],u[tr]);mu=w@x[lookup[tr]];sd=np.sqrt(w@((x[lookup[tr]]-mu)**2));sd=np.where(sd>1e-12,sd,1.);z=(x-mu)/sd;models=dict(mean=mu,scale=sd);pred=dict(diagnostic_ids=te,val_ids=va,cnn=features['cnn'][lookup[te]]);selection={}
  for arm in c['arms']:
   loss,weight=arm.split('_');ww=w if weight=='balanced' else np.ones(len(tr))/len(tr);coefs=[];candidates=[];optim=[]
   for j,lam in enumerate(c['lambdas']):
    b,info=fit(z[lookup[tr]],y[tr],ww,lam,loss);coefs.append(b);optim.append(info);vp=output(z[lookup[va]],b,loss);pred[arm+'_val_'+str(j)]=vp;candidates.append(dict(lam=lam,macro=macro(y[va],vp.argmax(1)),mse=float(np.square(vp-np.eye(12)[y[va]]).mean())))
   j=choose(candidates);selection[arm]=dict(selected=j,candidates=candidates,optimization=optim);models[arm]=np.stack(coefs);pred[arm]=output(z[lookup[te]],coefs[j],loss)
  prior=np.load(old/f'held{h}/predictions.npz');priorchoice=json.loads((old/f'held{h}/selection.json').read_text())['onehot']['selected'];assert selection['ridge_balanced']['selected']==priorchoice;np.testing.assert_allclose(pred['ridge_balanced'],prior['onehot']+.5,atol=1e-11,rtol=1e-11);np.testing.assert_array_equal(pred['ridge_balanced'].argmax(1),prior['onehot'].argmax(1))
  np.savez_compressed(d/'models.npz',**models);np.savez_compressed(d/'predictions.npz',**pred);dump(d/'split.json',part);dump(d/'selection.json',selection);print(json.dumps(dict(held=h,predictions_saved=True)),flush=True)
 # Evaluate only after every fold/candidate/choice/prediction is persisted.
 scores={arm:[] for arm in c['arms']+['cnn']}
 for h in range(1,9):
  d=a.out/f'held{h}';p=np.load(d/'predictions.npz');truth=y[p['diagnostic_ids']];np.savez_compressed(d/'evaluation.npz',y=truth)
  for arm in scores:scores[arm].append(macro(truth,p[arm].argmax(1)))
 pairs={}
 for name,a1,a0 in [('loss_balanced','logit_balanced','ridge_balanced'),('loss_uniform','logit_uniform','ridge_uniform'),('weight_ridge','ridge_balanced','ridge_uniform'),('weight_logit','logit_balanced','logit_uniform')]:
  delta=np.array(scores[a1])-scores[a0];pairs[name]=dict(per_user=delta.tolist(),mean=float(delta.mean()),wins=int((delta>1e-12).sum()),minimum=float(delta.min()))
 q=pairs['loss_balanced'];gate=q['mean']>=c['gate']['mean_gain'] and q['wins']>=6 and q['minimum']>=-.01
 s=dict(state='complete',scores=scores,means={k:float(np.mean(v)) for k,v in scores.items()},paired_effects=pairs,interaction_per_user=(np.array(pairs['loss_balanced']['per_user'])-pairs['loss_uniform']['per_user']).tolist(),primary_effect_gate_passed=bool(gate),novel_contribution_validated=False,fits=96,selected_readouts=32,encoder_training=False,seconds=time.perf_counter()-start);dump(a.out/'SUMMARY.json',s);dump(a.out/'FILES.json',{str(p.relative_to(a.out)):digest(p) for p in sorted(a.out.rglob('*')) if p.is_file()});print(json.dumps(s))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['protocol','out']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--cost-only',action='store_true');p.add_argument('--cost',type=Path);a=p.parse_args()
 with threadpool_limits(limits=1):main(a)
