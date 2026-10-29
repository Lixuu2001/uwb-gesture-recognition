import json,pickle,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import recall_score
b=Path(__file__).resolve().parent;r=b/'run01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((r/'SUMMARY.json').read_text());checks=0
for name,h in json.loads((r/'FILES.json').read_text()).items():assert sha(r/name)==h;checks+=1
identity=json.loads((r/'identity.json').read_text());assert identity['runner_sha256']==sha(b/'run.py');assert identity['protocol_sha256']==sha(b/'PROTOCOL.json');checks+=2
seen=[]
for row in s['rows']:
 h=row['held'];d=r/f'held{h}';sp=json.loads((d/'split.json').read_text());tr=set(sp['train']);va=set(sp['validation']);te=set(sp['diagnostic']);assert not(tr&va or tr&te or va&te);assert len(tr|va|te)==6900;checks+=2
 with np.load(d/'predictions.npz') as z:
  ii=z['users']==h;assert set(z['ids'][ii])==te;assert set(z['ids'][~ii])==va;checks+=2
  for arm in ['minirocket','raw_ridge']:
   v=float(recall_score(z['y'][ii],z[arm][ii],labels=np.unique(z['y'][ii]),average='macro',zero_division=0));assert abs(v-row['scores'][arm])<1e-12;checks+=1
 with np.load(d/'transform.npz') as z:assert len(z['p4'])==9996;checks+=1
 with open(d/'readouts.pkl','rb') as f:m=pickle.load(f)
 assert m['minirocket'][1].coef_.shape==(12,9996);assert m['raw'][1].coef_.shape==(12,945);checks+=2;seen.append(h)
assert sorted(seen)==[1,2,3,4,5,6];assert s['cpu_seconds']<1800 and s['wall_seconds']<1800;checks+=2
out=dict(state='pass',checks_passed=checks,models=12,transform_checkpoints=6,source_records=6900,mean_macro_recall=s['means'],users7_8_in_prediction_or_split=False,original_contribution_validated=False,strong_cnn_comparison_completed=False)
(r/'AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
