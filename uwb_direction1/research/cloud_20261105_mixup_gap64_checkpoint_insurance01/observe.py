"""Read-only actual stage/complete-producer/own-replay evidence, no held truth."""
from pathlib import Path
import sys,json,time,hashlib
S=Path('/workspace/uwb-mixup-gap64-20261105');D=Path(__file__).resolve().parent
sys.path.insert(0,str(S))
from common import read,write,sha,ready
from controller import identity
def main():
    ready(True);old=read(S/'CONTROLLER_IDENTITY.json');live=identity(old['pid'])
    assert live and live['state']!='Z' and live['boot_id']==old['boot_id'] and live['startticks']==old['startticks']
    ledger=read(S/'RESOURCE_LEDGER.json');assert ledger['wall']<43200 and ledger['cpu']<43200 and ledger['RSS_KiB']<12*1024*1024
    stage=ledger['stage'];worker=read(S/f'run/execution/{stage}/IDENTITY.json');current=identity(worker['pid'])
    assert current is None or (current['boot_id']==worker['boot_id'] and current['startticks']==worker['startticks'])
    q=dict(state='same_original_registered_controller_healthy_no_scoring',observed_unix=time.time(),controller=live,current_stage=stage,worker=current,resource_ledger=ledger,diagnostic_labels_read=False,users7_8_labels_used=False,new_fitting_started_by_observation=False,scientific_configuration_changed=False,budget_reset=False,producer_complete=False,full64_replay_complete=False)
    import engine as e
    from journal import load
    def native(path):
        st,p=load(path);m,o,g=e.make();e.restore(st,m,o,g)
        actual=e.capture(m,o,g,st['epoch'],st['updates'],st['best_key'],st['best_model'],st['best_epoch'])
        assert e.semantic(actual)==e.semantic(st)
        raw=(path/'steps.jsonl').read_bytes();prefix=raw[:p['prefix_bytes']]
        assert len(prefix.splitlines())==p['updates'] and hashlib.sha256(prefix).hexdigest()==p['prefix_sha256']
        return st,p
    train=S/'run/train'
    if (train/'COMPLETE.json').exists():
        c=read(train/'COMPLETE.json');st,p=native(train)
        assert c['epochs']==64 and c['updates']==st['updates']==4736 and c['final_state_sha256']==e.semantic(st)
        assert c['selected_sha256']==sha(train/'selected.pt') and c['validation_outputs_sha256']==sha(train/'validation_unscored.npz') and c['diagnostic_outputs_sha256']==sha(train/'diagnostic_unscored.npz')
        selected=e.torch.load(train/'selected.pt',map_location='cpu',weights_only=True)
        assert selected['selected_epoch']==st['best_epoch'] and e.semantic(selected['model'])==e.semantic(st['best_model'])
        term=read(S/'run/execution/train/TERMINATION.json');assert term['exit_code']==0 and term['resource_stop_reason'] is None
        q.update(producer_complete=True,producer_summary=c,producer_actual_termination=term,producer_final_native_model_optimizer_rng_verified=True)
    else:
        st,p=native(train);q['producer_progress']=p
    own=S/'run/replay'
    if (own/'PROGRESS.json').exists():
        st,p=native(own)
        for ep in range(st['epoch']+1):assert read(own/f'epoch{ep:03}.json')==read(train/f'epoch{ep:03}.json')
        ownraw=(own/'steps.jsonl').read_bytes()[:p['prefix_bytes']];original=(train/'steps.jsonl').read_bytes();assert original.startswith(ownraw)
        q.update(own_replay_progress=p,own_native_model_optimizer_rng_verified=True,own_complete_records_equal_producer=st['epoch']+1,own_update_prefix_equal_producer=True)
        q['full64_replay_complete']=(own/'COMPLETE.json').exists()
    assert not (S/'FINALIZE_ONCE.lock').exists() and not (S/'SUMMARY.json').exists()
    write(D/'POST_BACKUP_LIVE_OBSERVATION.json',q);print(json.dumps(q))
if __name__=='__main__':main()
