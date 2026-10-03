"""Post-hoc execution/response diagnostic, not a hyperparameter selector."""
import json
from pathlib import Path
import numpy as np
import torch
from audit_restored import dump
r=Path('/workspace/uwb-adaptation-source-20261003');c=json.loads((r/'protocol.json').read_text());rows=[]
for h in c['users']:
 d=r/f'held{h}';pred=np.load(d/'predictions.npz');initial=torch.load(Path(c['models'])/f'held{h}/selected.pt',weights_only=False,map_location='cpu')['states']['tail']
 for o in c['orders']:
  base=pred[o+'_frozen']
  for arm in c['arms']:
   states=torch.load(d/f'{o}_{arm}_states.pt',weights_only=False,map_location='cpu');delta=[]
   for s in states:delta.append(float(torch.sqrt(sum(((v-initial[k])**2).sum() for k,v in s['model'].items()))))
   z=pred[o+'_'+arm];rows.append(dict(held=h,order=o,arm=arm,max_abs_logit_change=float(abs(z-base).max()),changed_argmax=int((z.argmax(1)!=base.argmax(1)).sum()),max_post_update_GN_l2=max(delta),nonzero_GN_state_batches=sum(x>0 for x in delta)))
dump(r/'EFFECT_DIAGNOSTIC.json',dict(posthoc=True,labels_used=False,rows=rows,interpretation='Observed update/logit changes are execution checks, not evidence of useful adaptation. Resets correlated with small changes do not prove causation. No parameter search follows.'))
print(json.dumps({arm:dict(max_logit_change=max(x['max_abs_logit_change'] for x in rows if x['arm']==arm),max_GN_l2=max(x['max_post_update_GN_l2'] for x in rows if x['arm']==arm),changed_argmax=sum(x['changed_argmax'] for x in rows if x['arm']==arm)) for arm in c['arms']}))
