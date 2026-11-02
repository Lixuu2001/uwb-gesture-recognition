"""Read-only first actual update proof; no sample/label array indexing or fit."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import hashlib
import json
import time
from pathlib import Path
import torch

BASE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(1)
    cfg=json.loads((BASE/'PROTOCOL.json').read_text());assert cfg['run_directory']=='run02'
    run=BASE/'run02';folder=run/'timemil'
    assert (folder/'first_update.pt').exists(),'first update not yet ready; inspect live state'
    output=BASE/'FIRST_UPDATE_VERIFIED_RUN02.json';assert not output.exists()
    state=json.loads((run/'STATE.json').read_text());launch=json.loads((BASE/'LAUNCH_RUN02.json').read_text())
    assert state['state']=='training' and state['arm']=='timemil'
    assert not state['diagnostic_labels_read'] and not state['users7_8_labels_used']
    assert Path(f"/proc/{state['pid']}").exists()
    initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False)
    first=torch.load(folder/'first_update.pt',map_location='cpu',weights_only=False)
    assert first['identity']==initial['identity']
    assert first['identity']['protocol_sha256']==sha(BASE/'PROTOCOL.json')==launch['protocol_sha256']
    frozen=json.loads((BASE/'FROZEN_HASHES.json').read_text())
    assert first['identity']['frozen_hashes']==frozen
    assert all(sha(BASE/name)==expected for name,expected in frozen.items())
    changed=[name for name,value in initial['model'].items() if not torch.equal(value,first['model'][name])]
    assert changed and all(torch.isfinite(value).all() for value in first['model'].values() if value.is_floating_point())
    opt=first['optimizer'];assert opt['count']==1 and opt['alpha']==.5 and opt['k']==6 and not opt['slow']
    assert {int(value['step']) for value in opt['fast']['state'].values()}=={1}
    assert all(torch.isfinite(value[key]).all() for value in opt['fast']['state'].values() for key in ('exp_avg','exp_avg_sq'))
    split=json.loads((run/'split.json').read_text());ids=first['training_ids']
    assert len(ids)==len(set(ids))==64
    assert set(ids)<=set(split['train']) and not set(ids)&(set(split['validation'])|set(split['diagnostic']))
    order=torch.randperm(len(split['train']),generator=torch.Generator().manual_seed(cfg['seed']+1)).numpy()[:64]
    assert ids==[split['train'][int(i)] for i in order]
    result=dict(state='pass_first_real_update',observed_unix=time.time(),worker_pid=state['pid'],supervisor_pid=launch['supervisor_pid'],
                run_directory='run02',supervisor_directory='supervisor02',arm='timemil',changed_model_state_tensors=len(changed),
                all_floating_model_and_optimizer_tensors_finite=True,lookahead_count=1,fast_optimizer_steps=1,slow_buffers_empty_before_sync=True,
                batch_size=64,all_ids_unique_and_training_only=True,validation_and_diagnostic_ids_excluded=True,seeded_first_batch_matches=True,
                source_sample_or_label_values_indexed_by_verifier=False,diagnostic_labels_read=False,users7_8_labels_used=False,
                protocol_sha256=launch['protocol_sha256'],repair_registration_commit=launch['repair_registration_commit'],
                first_update_checkpoint_sha256=sha(folder/'first_update.pt'),verifier_sha256=sha(Path(__file__)),
                prior_predata_attempt_budget=cfg['prior_predata_attempt_budget'],state_observed={k:state[k] for k in ['state','pid','arm','epoch','batch','wall_seconds','cpu_seconds','observed_unix']})
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
