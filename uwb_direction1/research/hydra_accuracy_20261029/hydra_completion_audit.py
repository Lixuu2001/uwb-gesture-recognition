from pathlib import Path
import sys,json,hashlib,pickle
import numpy as np
import torch
from sklearn.metrics import recall_score
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'vendor/hydra/code'));r=B/'hydra_readout_completion'
s=json.loads((r/'SUMMARY.json').read_text());n=0
for p,h in json.loads((r/'FILES.json').read_text()).items():assert hashlib.sha256((r/p).read_bytes()).hexdigest()==h;n+=1
with np.load(B/'hydra_run01/source_features.npz') as z:f=torch.from_numpy(z['features']);ids=z['ids'];assert f.shape==(6900,4096);n+=1
with np.load(B/'run01/source_ids.npz') as z:u=z['users'];y=z['labels'];np.testing.assert_array_equal(ids,z['ids']);n+=1
for row in s['rows']:
 h=row['held'];d=r/f'held{h}';sp=json.loads((B/f'hydra_run01/held{h}/split.json').read_text());tr=np.flatnonzero((u!=h)&(u!=h%6+1));assert set(ids[tr])==set(sp['train']);assert not(set(sp['train'])&set(sp['diagnostic']));n+=2
 with open(d/'readout.pkl','rb') as z:sc,m=pickle.load(z)
 tf=f[tr].clamp(0).sqrt();torch.testing.assert_close(sc.mu,tf.mean(0));torch.testing.assert_close(sc.sigma,tf.std(0)+(tf==0).float().mean(0)**4+1e-8);assert m.coef_.shape==(12,4096);n+=3
 with np.load(d/'predictions.npz') as z:
  ii=z['users']==h;assert set(z['ids'][ii])==set(sp['diagnostic']);n+=1
  for arm in ['hydra','minirocket','raw_ridge']:
   got=recall_score(z['y'][ii],z[arm][ii],labels=np.unique(z['y'][ii]),average='macro',zero_division=0);assert abs(got-row['scores'][arm])<1e-12;n+=1
  with np.load(B/f'run01/held{h}/predictions.npz') as old:np.testing.assert_array_equal(z['ids'],old['ids']);np.testing.assert_array_equal(z['minirocket'],old['minirocket']);n+=2
assert s['cumulative_wall_seconds']<900 and s['cumulative_cpu_seconds']<895;assert all(row['normal_equation_relative_residual']<1e-6 for row in s['rows']);n+=1
out=dict(state='pass',checks_passed=n,source_records=6900,folds=6,readouts=6,scaler_train_statistics_recomputed=True,means=s['means'],strong_cnn_comparison_completed=False,independent_validation=False)
(r/'AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
