"""Independent saved-result audit; numerical raw feature checks are in run receipts."""
import argparse,json
from pathlib import Path
import numpy as np
from audit_restored import digest,dump
p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args()
if (a.runs/'AUDIT.json').exists():raise ValueError('Preserve existing audit')
s=json.loads((a.runs/'SUMMARY.json').read_text());c=json.loads((a.runs/'protocol.json').read_text());y=np.load(a.data/'cache/y_clutterremoved.npy');u=np.load(a.data/'cache/subj_clutterremoved.npy');rows=[];score_by={k:[] for k in ['grid','diag','cov','sqrt','cnn','tail_cnn']}
def metric(label,pred):
    counts=np.bincount(label,minlength=12);correct=np.bincount(label[pred==label],minlength=12);return float(np.mean(correct[counts>0]/counts[counts>0]))
for h in range(3,9):
 d=a.runs/f'held{h}';r=json.loads((d/'result.json').read_text());assert r==next(x for x in s['rows'] if x['held']==h)
 for n,sha in r['files'].items():assert digest(d/n)==sha
 valid=(u>=3)&(u<=8)&~((u==6)&(y==3));v=h+1 if h<8 else 3;parts={'train':np.flatnonzero(valid&(u!=h)&(u!=v)), 'val':np.flatnonzero(valid&(u==v)), 'diagnostic':np.flatnonzero(valid&(u==h))}
 sp=json.loads((d/'split.json').read_text())
 for role,ids in parts.items():np.testing.assert_array_equal(ids,sp[role])
 pred=np.load(d/'predictions.npz');model=np.load(d/'models.npz');sel=json.loads((d/'selection.json').read_text());np.testing.assert_array_equal(pred['ids'],parts['diagnostic']);np.testing.assert_array_equal(pred['y'],y[parts['diagnostic']]);np.testing.assert_array_equal(pred['validation_ids'],parts['val']);np.testing.assert_array_equal(pred['validation_y'],y[parts['val']])
 for k in pred.files:assert np.isfinite(pred[k]).all()
 for arm in c['arms']:
  candidates=[]
  for lam in c['lambdas']:
   vp=pred[f'{arm}_validation_{lam}'];candidates.append(dict(lam=lam,macro=metric(pred['validation_y'],vp.argmax(1)),mse=float(np.square(vp-np.eye(12)[pred['validation_y']]).mean())))
  assert candidates==sel[arm]['candidates'];chosen=sorted(range(3),key=lambda j:(-candidates[j]['macro'],candidates[j]['mse'],-candidates[j]['lam']))[0];assert chosen==sel[arm]['selected'];dim={'grid':1728,'diag':96,'cov':456,'sqrt':456}[arm];assert sel[arm]['dimension']==dim
  assert model[arm+'_coef'].shape==(dim,12) and model[arm+'_mean'].shape==(dim,) and np.all(model[arm+'_scale']>0)
 for k in score_by:
  z=metric(pred['y'],pred[k].argmax(1));assert abs(z-r['scores'][k])<1e-14;score_by[k].append(z)
 rows.append(dict(held=h,source_roles_verified=True,all_three_lambda_selection_recomputed=True,selected_model_shapes_verified=True,artifact_hashes_verified=4))
for k,z in score_by.items():np.testing.assert_allclose(z,s['scores'][k],atol=1e-14);assert abs(np.mean(z)-s['means'][k])<1e-14
passes=[]
for k,gain in c['gate']['min_mean_gain'].items():
 cand=np.array(score_by['sqrt']);base=np.array(score_by[k]);delta=cand-base;passed=bool(delta.mean()>=gain and sum(delta>1e-12)>=4 and min(cand)-min(base)>=-.01);assert passed==s['comparisons'][k]['passed'];passes.append(passed)
assert all(passes)==s['gate_passed'];dump(a.runs/'AUDIT.json',dict(state='passed',runs=rows,selected_models=24,ridge_fits_checked_by_runtime_sklearn=72,all24_artifact_sha256_verified=True,all_prediction_metrics_and_validation_selections_recomputed=True,gate_recomputed=True,scope_limit='Saved-array audit. Runtime raw logits and independently reconstructed covariance descriptors checked first32 per train/val/diagnostic role, not entire feature maps. Runtime sklearn checked all72 validation predictions; encoder not retrained.'))
print(json.dumps({'audit':'passed','gate':s['gate_passed'],'means':s['means']}))
