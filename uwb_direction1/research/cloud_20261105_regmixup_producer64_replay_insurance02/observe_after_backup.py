"""Read-only live observation after private complete-producer preservation."""
from pathlib import Path
import sys,time,json,hashlib
S=Path('/workspace/uwb-regmixup-gap64-20261105');D=Path(__file__).resolve().parent
sys.path.insert(0,str(S))
from common import ready,read,write
from controller import identity
import engine as e
def main():
    ready(True)
    assert read(D/'private_backup/READBACK_VERIFIED.json')['archive_sha256']==read(D/'private_backup/MANIFEST.json')['archive_sha256']
    oldc=read(S/'CONTROLLER_IDENTITY.json');oldw=read(S/'run/execution/replay/IDENTITY.json')
    for old in [oldc,oldw]:
        live=identity(old['pid']);assert live and live['state']!='Z' and all(old[k]==live[k] for k in ['pid','startticks','boot_id'])
    assert read(S/'run/execution/train/TERMINATION.json')['exit_code']==0
    for base in [S,D/'capture01/study']:
        done=read(base/'run/train/COMPLETE.json')
        for n,k in [('selected.pt','selected_sha256'),('validation_unscored.npz','validation_outputs_sha256'),('diagnostic_unscored.npz','diagnostic_outputs_sha256')]:
            assert hashlib.sha256((base/'run/train'/n).read_bytes()).hexdigest()==done[k]
    assert not (S/'SUMMARY.json').exists() and not (S/'FINALIZE_ONCE.lock').exists()
    assert not (S/'run/replay/COMPLETE.json').exists()
    for attempt in range(4):
        p=read(S/'run/replay/PROGRESS.json');raw=(S/'run/replay'/p['slot']).read_bytes()
        if hashlib.sha256(raw).hexdigest()==p['checkpoint_sha256']:break
    else:raise AssertionError('No coherent current replay checkpoint; preserve current state and inspect')
    import io
    st=e.torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    rec=read(S/f'run/replay/epoch{p["epoch"]:03}.json');producer=read(S/f'run/train/epoch{p["epoch"]:03}.json')
    assert rec==producer and e.semantic(st)==rec['state_sha256']
    m,o,g=e.make();e.restore(st,m,o,g)
    assert e.semantic(e.capture(m,o,g,st['epoch'],st['updates'],st['best_key'],st['best_model'],st['best_epoch']))==e.semantic(st)
    for ep in range(p['epoch']+1):assert read(S/f'run/replay/epoch{ep:03}.json')==read(S/f'run/train/epoch{ep:03}.json')
    rawlog=(S/'run/replay/steps.jsonl').read_bytes()[:p['prefix_bytes']]
    assert hashlib.sha256(rawlog).hexdigest()==p['prefix_sha256'] and len(rawlog.splitlines())==p['updates']
    assert rawlog==(S/'run/train/steps.jsonl').read_bytes()[:p['prefix_bytes']]
    ledger=read(S/'RESOURCE_LEDGER.json');assert ledger['stage']=='replay' and ledger['wall']<43200 and ledger['cpu']<43200 and ledger['RSS_KiB']<12582912
    q=dict(state='same_original_controller_OWN_fresh_replay_healthy_after_complete_producer64_private_backup',observed_unix=time.time(),controller=identity(oldc['pid']),replay_worker=identity(oldw['pid']),producer_actual_termination=read(S/'run/execution/train/TERMINATION.json'),producer_complete=read(S/'run/train/COMPLETE.json'),replay_progress=p,resource_ledger=ledger,all_committed_replay_records_and_steps_equal_producer=True,native_model_optimizer_all_RNG_restore_recapture_exact=True,scientific16source_verified_unchanged=True,full64_producer_complete=True,own_fresh64_replay_complete=False,full_final_pilot_private_backup_complete=False,diagnostic_labels_read=False,users7_8_labels_used=False,duplicate_fit_or_new_configuration_started=False,new_accuracy_result=False,budget_reset=False)
    write(D/'POST_BACKUP_LIVE_OBSERVATION.json',q);print(json.dumps(q))
if __name__=='__main__':main()
