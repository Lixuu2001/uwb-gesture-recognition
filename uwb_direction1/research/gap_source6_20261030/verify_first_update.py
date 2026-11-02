"""Read-only first-update verification, no map/label values or model forward."""
import json,time
from pathlib import Path
import numpy as np
import torch
import train

def main():
 b=Path(__file__).resolve().parent;run=b/'run01';folder=run/'held2';torch.set_num_threads(1)
 assert not (b/'FIRST_UPDATE_VERIFIED.json').exists()
 cfg=json.loads((b/'PROTOCOL.json').read_text());frozen=json.loads((b/'FROZEN_HASHES.json').read_text())
 for n,h in frozen.items():assert train.sha(b/n)==h,n
 identity=json.loads((run/'IDENTITY.json').read_text());assert identity['protocol_sha256']==train.sha(b/'PROTOCOL.json') and identity['frozen_hashes']==frozen
 initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False);first=torch.load(folder/'first_update.pt',map_location='cpu',weights_only=False)
 assert initial['identity']==first['identity']==identity and initial['held']==first['held']==2
 model=train.make_model('gap',cfg['seed']);assert sum(p.numel() for p in model.parameters())==617612
 assert all(torch.equal(v,initial['model'][k]) for k,v in model.state_dict().items())
 old=torch.load(Path(cfg['reused_held1']['directory'])/'initial.pt',map_location='cpu',weights_only=False)
 assert all(torch.equal(v,old['model'][k]) for k,v in initial['model'].items())
 changed=sum(not torch.equal(v,first['model'][k]) for k,v in initial['model'].items());assert changed>0
 assert all(torch.isfinite(v).all() for v in first['model'].values() if v.is_floating_point())
 opt=first['optimizer'];assert opt['count']==1 and opt['slow']=={} and opt['alpha']==.5 and opt['k']==6
 assert len(opt['fast']['state'])==len(list(model.parameters()))
 assert {int(v['step']) for v in opt['fast']['state'].values()}=={1}
 assert all(torch.isfinite(v[k]).all() for v in opt['fast']['state'].values() for k in ('exp_avg','exp_avg_sq'))
 restored=train.StableLookahead(model.parameters());restored.load_state_dict(opt);assert restored.count==1 and restored.slow=={}
 assert train.sha(cfg['source_manifest'])==cfg['source_manifest_sha256']
 with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
 val,tr,va,te=train.fold_indices(users,2);split=json.loads((folder/'split.json').read_text())
 for name,index in [('train',tr),('validation',va),('diagnostic',te)]:np.testing.assert_array_equal(split[name],ids[index])
 order=torch.randperm(len(tr),generator=torch.Generator().manual_seed(cfg['seed']+1)).numpy()[:64]
 np.testing.assert_array_equal(first['training_ids'],ids[tr[order]])
 assert len(set(first['training_ids']))==64 and not (set(first['training_ids'])&set(ids[va])) and not (set(first['training_ids'])&set(ids[te]))
 result=dict(state='pass',observed_unix=time.time(),held=2,validation_user=3,changed_model_tensors=changed,
             parameters=617612,initial_matches_author_and_reused_held1=True,all_model_and_optimizer_values_finite=True,
             optimizer_steps=1,slow_buffer_empty=True,training_ids_verified=64,exact_seeded_training_batch=True,
             split_roles_verified=True,frozen_hashes_verified=len(frozen),first_update_sha256=train.sha(folder/'first_update.pt'),
             real_sample_values_or_labels_read_by_verifier=False,real_forward_or_optimization_by_verifier=False,
             users7_8_labels_used=False,preregistration_commit='0f50505a62ad5d572cdd6d928974150047cce10a',
             verifier_sha256=train.sha(Path(__file__)))
 (b/'FIRST_UPDATE_VERIFIED.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
