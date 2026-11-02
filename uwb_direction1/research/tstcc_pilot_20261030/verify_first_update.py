"""Read-only actual first-step model/optimizer/ID proof, no raw values or labels."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
from pathlib import Path
import json,time
import numpy as np
import torch
from ops import make,indices,equal
from run_pilot import sha,read
B=Path(__file__).resolve().parent
if __name__=='__main__':
 torch.set_num_threads(1);torch.set_num_interop_threads(1);cfg=read(B/'PROTOCOL.json');out=B/'run01';frozen=read(B/'FROZEN_HASHES.json')
 for n,h in frozen.items():assert sha(B/n)==h,n
 a=torch.load(out/'ssl/initial.pt',map_location='cpu',weights_only=False);b=torch.load(out/'ssl/first_update.pt',map_location='cpu',weights_only=False);c,m,t,o,to,s=make(cfg['seed'],'ssl')
 equal(a['model'],m.state_dict());equal(a['temporal'],t.state_dict());assert a['optimizer']['state']==a['temporal_optimizer']['state']=={} and b['total_steps']==1 and b['epoch']==1 and b['batch']==0
 changed={key:sum(not torch.equal(v,a[key][n]) for n,v in b[key].items()) for key in ['model','temporal']};assert changed['model']==18 and changed['temporal']>=65
 for key in ('model','temporal'):
  for v in b[key].values():assert torch.isfinite(v).all()
 for key,count,opt in [('optimizer',9,o),('temporal_optimizer',65,to)]:
  saved=b[key];assert len(saved['state'])==count and {int(v['step']) for v in saved['state'].values()}=={1}
  for v in saved['state'].values():assert torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all()
  opt.load_state_dict(saved);equal(opt.state_dict(),saved)
 assert torch.equal(b['model']['logits.weight'],a['model']['logits.weight']) and torch.equal(b['model']['logits.bias'],a['model']['logits.bias'])
 with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
 tr=np.flatnonzero(np.isin(users,[3,4,5,6]));np.testing.assert_array_equal(b['training_ids'],ids[tr[indices(1,0,cfg['seed'])]])
 assert a['identity']==b['identity']==read(out/'IDENTITY.json')
 result=dict(state='pass',initial_both_models_exact_author_seed=True,changed_state_tensors=changed,optimizer_steps=1,encoder_optimizer_states=9,temporal_optimizer_states=65,both_optimizer_moments_finite_and_reload_exact=True,unused_SSL_logits_unchanged=True,exact_seeded_training_ID_count=128,frozen_entries_verified=len(frozen),first_update_sha256=sha(out/'ssl/first_update.pt'),verifier_sha256=sha(__file__),real_raw_values_or_labels_read=False,real_forward_or_fit=False,users7_8_labels_used=False,observed_unix=time.time())
 (B/'FIRST_UPDATE_VERIFIED.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
