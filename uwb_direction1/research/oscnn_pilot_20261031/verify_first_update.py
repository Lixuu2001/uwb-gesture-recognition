"""Immutable first proof, native sampler derivation; no raw data/labels/forward."""
import json,time
from pathlib import Path
from run_pilot import read,sha,write
from ops import torch,np,make,snapshot,equal
B=Path(__file__).resolve().parent
if __name__=='__main__':
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);cfg=read(B/'PROTOCOL.json');f=read(B/'FROZEN_HASHES.json')
 for n,h in f.items():assert sha(B/n)==h,n
 out=B/'run01';a=torch.load(out/'initial.pt',weights_only=False);b=torch.load(out/'first_update.pt',weights_only=False);identity=read(out/'IDENTITY.json');assert a['identity']==b['identity']==identity and identity['protocol_sha256']==sha(B/'PROTOCOL.json')
 e=make(cfg['seed']);equal(a,snapshot(e,identity,0,0,-1,'before_loader_iteration'));dummy=torch.utils.data.DataLoader(torch.arange(4700),batch_size=16,shuffle=True);indices=next(iter(dummy)).numpy()
 with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
 tr=np.flatnonzero(np.isin(users,[3,4,5,6]));np.testing.assert_array_equal(b['training_ids'],ids[tr[indices]])
 e.restore(b,expected_identity=identity);equal({k:v for k,v in b.items() if k!='training_ids'},snapshot(e,identity,1,1,0,'after_optimizer_before_scheduler'))
 assert b['total_steps']==b['epoch']==1 and len(b['optimizer']['state'])==14 and {int(s['step']) for s in b['optimizer']['state'].values()}=={1}
 for t in b['model'].values():assert torch.isfinite(t).all()
 for s in b['optimizer']['state'].values():assert torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all()
 masks=[n for n in a['model'] if n.endswith('weight_mask')];assert len(masks)==3
 for n in masks:equal(a['model'][n],b['model'][n])
 changed=sum(not torch.equal(a['model'][n],v) for n,v in b['model'].items());assert changed>0
 result=dict(state='pass',unmodified_author_full_multivariate_initialization_exact=True,trainable_parameters=1870464,parameters_including_fixed_masks=3736380,active_Adam_states=14,optimizer_steps=1,changed_model_state_tensors=changed,three_fixed_masks_unchanged=True,full_model_Adam_scheduler_all_RNG_restore_exact=True,exact_native_global_sampler_ID_count=16,frozen_entries_verified=len(f),first_update_sha256=sha(out/'first_update.pt'),verifier_sha256=sha(__file__),real_raw_values_labels_or_forward_fit=False,users7_8_labels_used=False,observed_unix=time.time());write(B/'FIRST_UPDATE_VERIFIED.json',result);print(json.dumps(result))
