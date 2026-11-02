"""First actual SSL update proof; no UWB map/label values or forward/fit."""
import json,time
from pathlib import Path
import numpy as np
import torch
import run_pilot as run
B=Path(__file__).resolve().parent
def main():
 torch.set_num_threads(1);cfg=json.loads((B/'PROTOCOL.json').read_text());frozen=json.loads((B/'FROZEN_HASHES.json').read_text())
 assert not (B/'FIRST_UPDATE_VERIFIED.json').exists()
 for n,h in frozen.items():assert run.sha(B/n)==h,n
 out=B/'run01';folder=out/'pretrained/pretrain';identity=json.loads((out/'IDENTITY.json').read_text())
 assert identity['frozen_hashes']==frozen and identity['protocol_sha256']==run.sha(B/'PROTOCOL.json')
 initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False);first=torch.load(folder/'first_update.pt',map_location='cpu',weights_only=False)
 assert initial['identity']==first['identity']==identity and initial['epoch']==0 and initial['phase']==first['phase']=='pretrain' and initial['arm']==first['arm']=='pretrained'
 model,_=run.make_model(cfg['seed']);assert sum(p.numel() for p in model.parameters())==4132879
 assert all(torch.equal(v,initial['model'][k]) for k,v in model.state_dict().items())
 changed=sum(not torch.equal(v,first['model'][k]) for k,v in initial['model'].items());assert changed>0
 assert all(torch.isfinite(v).all() for v in first['model'].values() if v.is_floating_point() or v.is_complex())
 unused=set(cfg['unused_parameters']['pretrain']);assert all(torch.equal(first['model'][n],initial['model'][n]) for n in unused)
 named=list(model.named_parameters());active={i for i,(n,p) in enumerate(named) if n not in unused};opt=first['optimizer']
 assert len(opt['param_groups'])==1 and opt['param_groups'][0]['params']==list(range(len(named)))
 assert opt['param_groups'][0]['lr']==.001 and opt['param_groups'][0]['weight_decay']==.0001
 assert set(opt['state'])==active and {int(v['step']) for v in opt['state'].values()}=={1}
 assert all(torch.isfinite(v[k]).all() for v in opt['state'].values() for k in ('exp_avg','exp_avg_sq'))
 restored=run.stage_optimizer(model);restored.load_state_dict(opt)
 for i in active:
  p=named[i][1]
  for key in ('exp_avg','exp_avg_sq'):assert torch.equal(restored.state[p][key],opt['state'][i][key])
 assert run.sha(cfg['source_manifest'])==cfg['source_manifest_sha256']
 with np.load(cfg['source_manifest']) as z:ids=z['ids'];users=z['users']
 ix=run.roles(users);split=json.loads((out/'split.json').read_text())
 for role,index in ix.items():np.testing.assert_array_equal(split[role],ids[index])
 order=torch.randperm(len(ix['train']),generator=torch.Generator().manual_seed(cfg['seed']+1)).numpy()[:16]
 np.testing.assert_array_equal(first['training_ids'],ids[ix['train'][order]])
 assert len(set(first['training_ids']))==16 and not(set(first['training_ids'])&set(ids[ix['validation']])) and not(set(first['training_ids'])&set(ids[ix['diagnostic']]))
 result=dict(state='pass',observed_unix=time.time(),phase='pretrained/pretrain',parameters=4132879,changed_model_tensors=changed,initial_exact_author_UEA_seed=True,model_and_optimizer_finite=True,optimizer_steps=1,active_optimizer_parameter_tensors=len(active),unused_head_and_input_layer_unchanged=True,optimizer_reload_exact_moments=True,training_ids_verified=16,exact_seeded_training_batch=True,roles_verified=True,frozen_hashes_verified=len(frozen),first_update_sha256=run.sha(folder/'first_update.pt'),verifier_sha256=run.sha(Path(__file__)),real_sample_or_label_values_read_by_verifier=False,real_forward_or_fit_by_verifier=False,users7_8_labels_used=False)
 (B/'FIRST_UPDATE_VERIFIED.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
