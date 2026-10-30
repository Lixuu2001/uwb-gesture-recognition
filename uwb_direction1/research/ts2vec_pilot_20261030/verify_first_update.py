"""Read-only first-update/model/optimizer/ID proof; no raw inputs or labels."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import json,time
from pathlib import Path
import numpy as np
import torch
from run_pilot import sha
from training_ops import make_model,optimizer,batch_indices
B=Path(__file__).resolve().parent
def main():
 out=B/'run01';cfg=json.loads((B/'PROTOCOL.json').read_text())
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 frozen=json.loads((B/'FROZEN_HASHES.json').read_text())
 for n,h in frozen.items():assert sha(B/n)==h,n
 initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
 first=torch.load(out/'first_update.pt',map_location='cpu',weights_only=False)
 assert initial['step']==0 and first['step']==first['n_iters']==1 and first['n_epochs']==0
 model=make_model(cfg['seed']);opt=optimizer(model)
 for n,v in model._net.state_dict().items():assert torch.equal(v,initial['raw_model'][n])
 for n,v in model.net.state_dict().items():assert torch.equal(v,initial['averaged_model'][n])
 assert initial['optimizer']['state']=={} and int(first['averaged_model']['n_averaged'])==2
 changed=0
 for n,v in first['raw_model'].items():
  assert torch.isfinite(v).all();changed+=int(not torch.equal(v,initial['raw_model'][n]))
  expected=initial['raw_model'][n]+(v-initial['raw_model'][n])/2
  assert torch.equal(expected,first['averaged_model']['module.'+n]),n
 assert changed==48
 states=first['optimizer']['state'];assert len(states)==48 and {int(v['step']) for v in states.values()}=={1}
 for v in states.values():assert torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all()
 opt.load_state_dict(first['optimizer'])
 for n,v in opt.state_dict()['state'].items():
  for k,t in v.items():assert torch.equal(t,states[n][k])
 with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
 tr=np.flatnonzero(np.isin(users,[3,4,5,6]));ix,sampler=batch_indices(1,cfg['seed'])
 np.testing.assert_array_equal(first['training_ids'],ids[tr[ix]]);assert first['sampler']==sampler
 identity=json.loads((out/'IDENTITY.json').read_text());assert first['identity']==initial['identity']==identity
 result=dict(state='pass',observed_unix=time.time(),initial_exact_author_default_seed=True,changed_model_tensors=changed,initial_inclusive_average_exact=True,optimizer_parameter_states=48,optimizer_steps=1,optimizer_reload_exact=True,training_ID_count=16,exact_seeded_batch_and_sampler=True,frozen_entries_verified=len(frozen),first_update_sha256=sha(out/'first_update.pt'),verifier_sha256=sha(__file__),real_raw_values_or_labels_read=False,real_forward_or_fit=False,users7_8_labels_used=False)
 (B/'FIRST_UPDATE_VERIFIED.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
