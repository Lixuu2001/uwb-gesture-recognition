"""Aggregation checks with unequal class support and absent classes."""
import json
import numpy as np
from source_group_gap import statistics
z=np.array([[2.,0],[2,0],[2,0],[2,0]]);y=np.array([0,0,0,1]);u=np.ones(4,dtype=int)*6
r=statistics(z,y,u);assert r['6']['macro_error']==.5;assert r['6']['classes']['0']['n']==3
z2=np.vstack([z,[-1,1]]);r2=statistics(z2,np.r_[y,1],np.r_[u,7]);assert r2['6']==r['6'];assert r2['7']['macro_error']==0
r3=statistics(z+1e4,y,u);assert abs(r3['6']['macro_ce']-r['6']['macro_ce'])<1e-10
print(json.dumps(dict(passed=True,checks=['class_macro_not_sample_mean','groups_separate','absent_classes_not_imputed','large_common_logit_offset'])))
