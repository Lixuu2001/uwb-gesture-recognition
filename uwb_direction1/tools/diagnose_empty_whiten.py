"""Descriptive training-template flatness only; never model selection."""
from pathlib import Path
import json,sys
import numpy as np
from audit_restored import dump,digest
import argparse
p=argparse.ArgumentParser();p.add_argument("root",type=Path);args=p.parse_args()
r=args.root;rows=[]
for h in range(3,9):
 f=r/f'held{h}/template.npz';z=np.load(f);p0=z['psd'];full=lambda a:np.concatenate([a,a[:,-2:0:-1]],axis=-1)
 row=dict(held=h,template_sha256=digest(f),flatness={})
 for n,g in [('raw',np.ones_like(p0)),('empty_whiten',z['gain']),('shuffled',z['shuffled_gain'])]:
  pp=full(p0*g*g);row['flatness'][n]=(np.exp(np.log(np.maximum(pp,1e-30)).mean(-1))/pp.mean(-1)).tolist()
 row['gain_mean_square']=np.mean(full(z['gain'])**2,axis=-1).tolist();rows.append(row)
out=dict(scope='Post-hoc descriptive check on the SAME source empty periodogram used to fit whitening. No target data or labels, no selection or gate change. Increased flatness is an algebraic in-sample effect, not evidence of independent noise suppression or accuracy.',rows=rows)
dump(r/'TEMPLATE_DIAGNOSTIC.json',out);print(json.dumps({k:float(np.mean([row['flatness'][k] for row in rows])) for k in ['raw','empty_whiten','shuffled']}))
