"""Observe the actual registered real pilot and verify native first/committed state, never fit/score."""
from pathlib import Path
import sys,time,io,json,hashlib
S=Path('/workspace/uwb-sd-gap64-20261105');M=Path(__file__).resolve().parent
sys.path.insert(0,str(S))
from common import ready,read,write,sha
from controller import identity
def main():
    ready(True);start=time.monotonic()
    while True:
        if (S/'STATE.json').exists():assert read(S/'STATE.json')['state']!='execution_incomplete_requires_evidence_preserving_recovery'
        p=read(S/'run/train/PROGRESS.json') if (S/'run/train/PROGRESS.json').exists() else None
        if p and p['epoch']>=1 and (S/'run/train/FIRST_RELEASE.json').exists() and (S/'run/execution/train/FIRST_AUDIT_TERMINATION.json').exists():break
        assert time.monotonic()-start<600,'No verified first/committed state within bounded observation; inspect originals, no new launch'
        time.sleep(1)
    ctrl=read(S/'CONTROLLER_IDENTITY.json');worker=read(S/'run/execution/train/IDENTITY.json')
    for old in [ctrl,worker]:
        live=identity(old['pid']);assert live and live['state']!='Z' and all(live[k]==old[k] for k in ['pid','startticks','boot_id'])
    for attempt in range(4):
        p=read(S/'run/train/PROGRESS.json');raw=(S/'run/train'/p['slot']).read_bytes()
        if hashlib.sha256(raw).hexdigest()==p['checkpoint_sha256']:break
    else:raise AssertionError('No coherent current checkpoint; inspect/preserve live state')
    import engine as e
    st=e.torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);record=read(S/f'run/train/epoch{p["epoch"]:03}.json');assert e.semantic(st)==record['state_sha256'] and st['updates']==p['updates']
    m,o,g=e.make();e.restore(st,m,o,g);assert e.semantic(e.capture(m,o,g,st['epoch'],st['updates'],st['best_key'],st['best_model'],st['best_epoch']))==e.semantic(st)
    prefix=(S/'run/train/steps.jsonl').read_bytes()[:p['prefix_bytes']];assert hashlib.sha256(prefix).hexdigest()==p['prefix_sha256'] and len(prefix.splitlines())==p['updates']
    first=read(S/'run/train/FIRST_RELEASE.json');term=read(S/'run/execution/train/FIRST_AUDIT_TERMINATION.json');assert term['exit_code']==0 and first['native_first_update_exact'] and first['first_sha256']==sha(S/'run/train/first_update.pt')
    assert read(S/'run/train/FIRST_READY.json')['step']['forward_batch']==64
    ledger=read(S/'RESOURCE_LEDGER.json');assert ledger['stage']=='train' and ledger['wall']<43200 and ledger['cpu']<43200 and ledger['RSS_KiB']<12582912
    q=dict(state='actual_preregistered_SD_training_started_first_update_and_native_committed_state_verified_unscored',observed_unix=time.time(),controller=identity(ctrl['pid']),worker=identity(worker['pid']),progress=p,first_actual_update_file_sha256=first['first_sha256'],first_actual_update_native_state_sha256=first['state_sha256'],first_actual_update_native_exact=True,first_audit_actual_exit_code=0,committed_native_model_optimizer_all_RNG_restore_recapture_exact=True,committed_step_prefix_verified=True,scientific20_unchanged=True,frozen_sha256=sha(S/'FROZEN_HASHES.json'),resource_ledger=ledger,full64_complete=False,own_fresh64_replay_complete=False,diagnostic_labels_read=False,users7_8_labels_used=False,new_accuracy_result=False,prior_EFDMix_exact_failed_config_closed=True)
    write(S/'LAUNCH_OBSERVATION.json',q);write(M/'LAUNCH_OBSERVATION.json',q);print(json.dumps(q))
if __name__=='__main__':main()
