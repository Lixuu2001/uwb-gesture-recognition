"""Descriptive paired prediction accounting; no threshold or model fitting."""
import json
import numpy as np
from e2e_reconstruction import OUT
from pipeline import dump

def counts(y,a,b):
    ca=a==y;cb=b==y
    return {'n':len(y),'baseline_correct':int(ca.sum()),'joint_correct':int(cb.sum()),'fixed_errors':int((~ca&cb).sum()),'introduced_errors':int((ca&~cb).sum())}

def main():
    rows=[]
    for h in range(1,9):
        b=OUT/f'fixed_mask_L_s2026_h{h}_predictions.npz';j=OUT/f'joint_L_s2026_h{h}_predictions.npz'
        if not (b.exists() and j.exists()):continue
        a=np.load(b);c=np.load(j)
        np.testing.assert_array_equal(a['ids'],c['ids']);np.testing.assert_array_equal(a['y'],c['y'])
        row={'fold':h,**counts(a['y'],a['pred'],c['pred'])}
        row['by_class']={str(k):counts(a['y'][a['y']==k],a['pred'][a['y']==k],c['pred'][a['y']==k]) for k in np.unique(a['y'])}
        rows.append(row)
    dump(OUT/'error_accounting.json',{'note':'Post-hoc evaluation only; neither thresholds nor labels used to fit a new model.','rows':rows})
    print('ACCOUNTING',len(rows),'folds')
    if len(rows)==8:
        print({k:sum(r[k] for r in rows) for k in ['n','baseline_correct','joint_correct','fixed_errors','introduced_errors']})

if __name__=='__main__':main()
