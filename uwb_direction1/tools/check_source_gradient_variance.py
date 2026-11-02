"""Diagnostic role-isolation and conditional-null tests on synthetic data."""
import json,argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from source_gradient_variance import sample_source,permute_domains,variance_matrix,diagnose

def run():
    u=np.repeat(np.arange(3,9),12*60);y=np.tile(np.repeat(np.arange(12),60),6)
    for h in range(3,9):
        role,ids,half,classes=sample_source(y,u,h,48,20261004)
        altered=y.copy();altered[(u==h)|(u==(h+1 if h<8 else 3))]=999
        rr,ii,hh,cc=sample_source(altered,u,h,48,20261004)
        for a,b in [(role,rr),(ids,ii),(half,hh)]:np.testing.assert_array_equal(a,b)
        assert classes==cc
        assert len(np.unique(ids))==len(ids) and set(ids)<=set(role)
        for d in np.unique(u[ids]):
            for k in classes:
                for q in [0,1]:assert np.sum((u[ids]==d)&(y[ids]==k)&(half==q))==24
    rng=np.random.default_rng(11);g=rng.normal(size=(160,7));d=np.repeat(np.arange(4),40);labels=np.tile(np.repeat([0,1],20),4)
    shuffled=permute_domains(labels,d,rng)
    for k in [0,1]:np.testing.assert_array_equal(np.bincount(shuffled[labels==k]),np.bincount(d[labels==k]))
    np.testing.assert_allclose(variance_matrix(g,d),np.stack([g[d==x].var(0) for x in range(4)]),rtol=1e-12,atol=1e-12)
    # Balanced within each class: identical distributions across users yield no effect.
    z=np.tile(np.arange(20),8).astype(float);same=np.stack([z,z*z],1)
    negative,_,_=diagnose(same,labels,d,22,99);assert negative['permutation_exceedance_fraction']==1.0
    signs=np.tile(np.tile([-1.,1.],10),8);positive=np.stack([signs*(d+1),signs*(d+1)*2],1)
    signal,_,_=diagnose(positive,labels,d,22,99)
    assert signal['permutation_exceedance_fraction']<=.05 and signal['ratio_to_null_median']>=2
    return {'state':'passed','scope':'synthetic diagnostic assembly only','held_and_validation_labels_do_not_change_source_selection':True,'source_only_nonduplicated_stratified_halves':True,'class_domain_counts_preserved_by_null':True,'fast_variance_matches_centered_reference':True,'identical_domain_negative_control':True,'variance_shift_positive_control':True}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 with threadpool_limits(limits=1):r=run()
 a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
