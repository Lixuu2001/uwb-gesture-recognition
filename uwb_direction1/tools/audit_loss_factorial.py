"""Independent sklearn replay of the preregistered frozen-readout comparison."""
import argparse,json,time,warnings
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.linear_model import LogisticRegression,Ridge
from audit_restored import digest,dump
from second_order_core import macro


def sample_weights(y,u,balanced):
 if not balanced:return np.full(len(y),1/len(y))
 w=np.empty(len(y),dtype=float);us=np.unique(u)
 for subject in us:
  present=np.unique(y[u==subject])
  for label in present:
   mask=(u==subject)&(y==label);w[mask]=1/(len(us)*len(present)*int(mask.sum()))
 assert abs(w.sum()-1)<1e-12
 return w


def main(a):
 root=a.run;cfg=json.loads((root/'PROTOCOL.json').read_text());mfest=json.loads((root/'FILES.json').read_text());ident=json.loads((root/'IDENTITY.json').read_text());t=time.perf_counter()
 for n,sha in mfest.items():assert digest(root/n)==sha
 for n,sha in cfg['code_sha256'].items():assert digest(Path(__file__).with_name(n))==sha
 assert digest(root/'PROTOCOL.json')==ident['protocol_sha256']
 f=Path(cfg['feature_reference']);old=Path(cfg['ridge_reference']);fm=json.loads((f/'FILES.json').read_text());rm=json.loads((old/'FILES.json').read_text());assert digest(f/'FILES.json')==cfg['feature_manifest_sha256'] and digest(old/'FILES.json')==cfg['ridge_manifest_sha256']
 for base,manifest in [(f,fm),(old,rm)]:
  for n,sha in manifest.items():assert digest(base/n)==sha
 data=Path('/workspace/uwb-restored-20260930/cache');labels=np.load(data/'y_clutterremoved.npy');users=np.load(data/'subj_clutterremoved.npy');assert digest(data/'y_clutterremoved.npy')==cfg['input_sha256']['labels'] and digest(data/'subj_clutterremoved.npy')==cfg['input_sha256']['users'];ids=np.flatnonzero(~(((users==2)&np.isin(labels,[10,11]))|((users==6)&(labels==3))));assert len(ids)==9300;lookup=np.full(len(labels),-1);lookup[ids]=np.arange(len(ids));records={arm:[] for arm in cfg['arms']+['cnn']};details=[]
 # First replay every fit and validation choice, using no held diagnostic labels.
 for h in cfg['users']:
  d=root/f'held{h}';split=json.loads((d/'split.json').read_text());selection=json.loads((d/'selection.json').read_text());stored=np.load(d/'models.npz');pred=np.load(d/'predictions.npz');part=json.loads((f/f'held{h}/split.json').read_text());np.testing.assert_array_equal(np.array(split['train']),part['train']);np.testing.assert_array_equal(np.array(split['val']),part['val']);np.testing.assert_array_equal(np.array(split['diagnostic']),part['diagnostic'])
  tr,va,te=[np.asarray(split[k],dtype=int) for k in ['train','val','diagnostic']];assert not(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te));assert set(users[tr])==set(range(1,9))-{h,h%8+1};assert set(users[va])=={h%8+1} and set(users[te])=={h};np.testing.assert_array_equal(pred['diagnostic_ids'],te);np.testing.assert_array_equal(pred['val_ids'],va)
  z=np.load(f/f'held{h}/features.npz');np.testing.assert_array_equal(z['ids'],ids);x=z['hidden'].astype(float);mean=sample_weights(labels[tr],users[tr],True)@x[lookup[tr]];scale=np.sqrt(sample_weights(labels[tr],users[tr],True)@((x[lookup[tr]]-mean)**2));scale=np.where(scale>1e-12,scale,1.);np.testing.assert_allclose(mean,stored['mean'],atol=1e-11,rtol=1e-11);np.testing.assert_allclose(scale,stored['scale'],atol=1e-11,rtol=1e-11);x=(x-mean)/scale
  for arm in cfg['arms']:
   loss,weight=arm.split('_');sw=sample_weights(labels[tr],users[tr],weight=='balanced');coefs=[];candidates=[]
   for j,lam in enumerate(cfg['lambdas']):
    if loss=='ridge':
     model=Ridge(alpha=lam,fit_intercept=True,solver='svd').fit(x[lookup[tr]],np.eye(12)[labels[tr]],sample_weight=sw);b=np.concatenate([model.intercept_[None,:],model.coef_.T],axis=0)
    else:
     with warnings.catch_warnings():
      warnings.simplefilter('error');model=LogisticRegression(C=1/lam,solver='newton-cg',tol=1e-9,max_iter=3000,random_state=0).fit(x[lookup[tr]],labels[tr],sample_weight=sw)
     assert np.array_equal(model.classes_,np.arange(12));b=np.concatenate([model.intercept_[None,:],model.coef_.T],axis=0)
    coefs.append(b);np.testing.assert_allclose(b,stored[arm][j],atol=1e-4,rtol=1e-4)
    v=x[lookup[va]]@b[1:]+b[0]
    if loss=='logit':
     e=np.exp(v-v.max(axis=1,keepdims=True));v=e/e.sum(axis=1,keepdims=True)
    stored_val=pred[arm+'_val_'+str(j)];np.testing.assert_allclose(v,stored_val,atol=1e-4,rtol=1e-4);np.testing.assert_array_equal(v.argmax(1),stored_val.argmax(1));candidates.append(dict(lam=lam,macro=macro(labels[va],stored_val.argmax(1)),mse=float(np.square(stored_val-np.eye(12)[labels[va]]).mean())))
   j=min(range(len(candidates)),key=lambda j:(-candidates[j]['macro'],candidates[j]['mse'],-candidates[j]['lam']));assert j==selection[arm]['selected'] and candidates==selection[arm]['candidates'];b=coefs[j];v=x[lookup[te]]@b[1:]+b[0]
   if loss=='logit':
    e=np.exp(v-v.max(axis=1,keepdims=True));v=e/e.sum(axis=1,keepdims=True)
   np.testing.assert_allclose(v,pred[arm],atol=1e-4,rtol=1e-4);np.testing.assert_array_equal(v.argmax(1),pred[arm].argmax(1))
  np.testing.assert_array_equal(pred['cnn'].argmax(1),z['cnn'][lookup[te]].argmax(1));print(json.dumps({'held':h,'refits_verified':12}),flush=True)
 # Diagnostic labels are used only for the final preregistered scoring, after all 96 fits/predictions and validation choices were saved.
 scores={arm:[] for arm in cfg['arms']+['cnn']}
 for h in cfg['users']:
  d=root/f'held{h}';p=np.load(d/'predictions.npz');te=p['diagnostic_ids'];yy=labels[te];np.savez_compressed(d/'evaluation.npz',y=yy)
  for arm in scores:scores[arm].append(macro(yy,p[arm].argmax(1)))
 paired={}
 for key,arm1,arm0 in [('loss_balanced','logit_balanced','ridge_balanced'),('loss_uniform','logit_uniform','ridge_uniform'),('weight_ridge','ridge_balanced','ridge_uniform'),('weight_logit','logit_balanced','logit_uniform'),('versus_frozen_cnn','logit_balanced','cnn')]:
  delta=np.asarray(scores[arm1])-scores[arm0];paired[key]=dict(per_user=delta.tolist(),mean=float(delta.mean()),wins=int((delta>1e-12).sum()),minimum=float(delta.min()))
 a0=cfg['gate'];passed=paired['loss_balanced']['mean']>=a0['minimum_mean_gain_vs_ridge_balanced'] and paired['loss_balanced']['wins']>=a0['minimum_wins'] and paired['loss_balanced']['minimum']>=a0['minimum_paired_gain'] and paired['versus_frozen_cnn']['mean']>=a0['minimum_mean_gain_vs_frozen_cnn']
 summary=dict(state='complete_audited',scores=scores,means={k:float(np.mean(v)) for k,v in scores.items()},paired_effects=paired,interaction_per_user=(np.asarray(paired['loss_balanced']['per_user'])-paired['loss_uniform']['per_user']).tolist(),primary_effect_gate_passed=bool(passed),novel_contribution_validated=False,independent_sklearn_refits=96,artifact_files=len(mfest),seconds=time.perf_counter()-t,encoder_training=False);dump(a.out/'AUDIT.json',summary);print(json.dumps(summary))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists();a.out.mkdir()
 with threadpool_limits(limits=1):main(a)
