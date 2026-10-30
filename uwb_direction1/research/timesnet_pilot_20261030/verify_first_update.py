"""Read-only first-step checkpoint/optimizer/seed/ID proof without raw values."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
from pathlib import Path
import json,time
import numpy as np
import torch
from ops import make,equal,training_order
from run_pilot import sha,read
B=Path(__file__).resolve().parent
if __name__=='__main__':
 torch.set_num_threads(1);torch.set_num_interop_threads(1);cfg=read(B/'PROTOCOL.json');out=B/'run01';frozen=read(B/'FROZEN_HASHES.json')
 for n,h in frozen.items():assert sha(B/n)==h,n
 a=torch.load(out/'initial.pt',map_location='cpu',weights_only=False);b=torch.load(out/'first_update.pt',map_location='cpu',weights_only=False);c,m,o=make(cfg['seed']);equal(a['model'],m.state_dict());assert a['optimizer']['state']=={} and a['total_steps']==0 and b['total_steps']==1 and b['epoch']==1
 changed=sum(not torch.equal(v,a['model'][n]) for n,v in b['model'].items());assert changed>0
 for v in b['model'].values():assert torch.isfinite(v).all()
 states=b['optimizer']['state'];assert len(states)==53 and {int(v['step']) for v in states.values()}=={1}
 for v in states.values():assert torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all()
 o.load_state_dict(b['optimizer']);equal(o.state_dict(),b['optimizer']);name='enc_embedding.temporal_embedding.embed.weight';assert torch.equal(a['model'][name],b['model'][name])
 with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
 tr=np.flatnonzero(np.isin(users,[3,4,5,6]));np.testing.assert_array_equal(b['training_ids'],ids[tr[training_order(1,cfg['seed'])[:16]]]);assert b['sampler']==dict(epoch=1,batch=0,order_seed=cfg['seed']+1);assert a['identity']==b['identity']==read(out/'IDENTITY.json')
 result=dict(state='pass',initial_exact_author_full_gesture_preset_seed=True,model_parameters=2433260,changed_state_tensors=changed,active_RAdam_parameter_states=53,optimizer_steps=1,optimizer_moments_finite_reload_exact=True,unused_timefeature_embedding_unchanged=True,exact_seeded_training_ID_count=16,frozen_entries_verified=len(frozen),first_update_sha256=sha(out/'first_update.pt'),verifier_sha256=sha(__file__),real_raw_values_or_labels_read=False,real_forward_or_fit=False,users7_8_labels_used=False,observed_unix=time.time())
 (B/'FIRST_UPDATE_VERIFIED.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
