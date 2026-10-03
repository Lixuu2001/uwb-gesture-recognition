"""Read-only persisted result audit; exact fractions distinguish numerical ties."""
import argparse,json,hashlib
from pathlib import Path
from fractions import Fraction
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args()
 s=json.loads((a.out/'SUMMARY.json').read_text());f=np.load(a.out/'features.npz');y=np.load(a.data/'cache/y_clutterremoved.npy');u=np.load(a.data/'cache/subj_clutterremoved.npy')
 valid=~(((u==2)&np.isin(y,[10,11]))|((u==6)&(y==3)))
 ids=np.flatnonzero(valid&np.isin(u,[3,4,5,6,7,8]));np.testing.assert_array_equal(f['ids'],ids);np.testing.assert_array_equal(f['y'],y[ids]);np.testing.assert_array_equal(f['u'],u[ids])
 assert sha(a.out/'features.npz')==s['features_sha256'] and sha(a.out/'AUDIT.json')==s['audit_sha256']
 count=0;exact={};maxerr=0.
 for row in s['rows']:
  d=a.out/row['view']/row['arm']/f"user{row['held']}"
  for name,h in row['sha256'].items():assert sha(d/name)==h;count+=1
  tr=ids[u[ids]!=row['held']];te=ids[u[ids]==row['held']];split=json.loads((d/'split.json').read_text())
  assert split=={'train':tr.tolist(),'diagnostic':te.tolist()}
  q=np.load(d/'predictions.npz');m=np.load(d/'model.npz');x=f[row['arm']][:,['Left','Top','Right'].index(row['view'])]
  a_tr=x[u[ids]!=row['held']];a_te=x[u[ids]==row['held']]
  np.testing.assert_array_equal(m['mean'],a_tr.mean(0));np.testing.assert_array_equal(m['scale'],np.maximum(a_tr.std(0),1e-8)*16)
  ref=((a_te-m['mean'])/m['scale'])@m['coef'][:-1]+m['coef'][-1]
  maxerr=max(maxerr,float(np.max(np.abs(ref-q['logits']))));np.testing.assert_allclose(ref,q['logits'],atol=1e-12,rtol=1e-12)
  np.testing.assert_array_equal(q['ids'],te);np.testing.assert_array_equal(q['y'],y[te]);np.testing.assert_array_equal(q['pred'],ref.argmax(1))
  classes=np.unique(y[te]);fr=sum((Fraction(int(((q['pred']==q['y'])&(q['y']==c)).sum()),int((q['y']==c).sum())) for c in classes),Fraction(0))/len(classes)
  assert abs(float(fr)-row['macro_accuracy'])<1e-14
  exact[row['view'],row['arm'],row['held']]=fr
 views={}
 for v in ['Left','Top','Right']:
  delta=[exact[v,'mixed',h]-exact[v,'low',h] for h in range(3,9)]
  views[v]={'mixed_minus_low_exact_mean':float(sum(delta)/6),'wins':sum(z>0 for z in delta),'ties':sum(z==0 for z in delta),'losses':sum(z<0 for z in delta)}
 result={'state':'passed','persisted_hashes_verified':count,'independent_source_ids':len(ids),'source_normalization_recomputed_models':54,'prediction_max_error':maxerr,'exact_fraction_comparison':views,'note':'Top user4 is an exact tie; original floating subtraction ~1e-16 must not be described as a win. All frozen continuation gate outcomes remain false.'}
 (a.out/'POST_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
