"""Read-only assurance of retained producer certificates and own audit state."""
from pathlib import Path
import sys,json,hashlib
from delta import read,write,sha
ROOT=Path('/workspace/uwb-softvote-source6-real-20261103')
def main():
    out=Path(sys.argv[1]);sys.path.insert(0,str(ROOT));import engine as e;e.setup('shuffle');from journal import load
    held=read(out/'SNAPSHOT.json')['held'];d=out/'folds'/f'held{held}'/'shuffle';state,p=load(d/'replay')
    with e.np.load(d/'split.npz') as ids:ntrain=len(ids['train'])
    assert state['updates']==p['epoch']*((ntrain+19)//20)
    for epoch in range(p['epoch']+1):
        assert read(d/'replay'/f'epoch{epoch:03}.json')==read(d/f'epoch{epoch:03}.json')
        assert read(d/'replay'/f'journal{epoch:03}.json')['producer_record_sha256']==sha(d/f'epoch{epoch:03}.json')
    prefix=(d/'steps.jsonl').read_bytes()[:p['prefix_bytes']];assert len(prefix.splitlines())==p['updates'] and hashlib.sha256(prefix).hexdigest()==p['prefix_sha256']
    t=out/'folds'/f'held{held}'/'timemil/inference_audit';cert=read(t/'CERTIFICATE.json');term=read(t/'TERMINATION.json');assert cert['state']=='pass' and term['exit_code']==0 and term['resource_stop_reason'] is None and term['cgroup_OOM_kill_delta']==0
    for name,h in cert['artifact_hashes'].items():assert sha(out/Path(name).relative_to(ROOT))==h
    result=dict(state='captured_own_replay_commits_equal_original_producer_and_retained_TimeMIL_output_audit_bound',own_replay_epoch=p['epoch'],own_replay_updates=p['updates'],TimeMIL_selected_state_output_audit_preserved=True,TimeMIL_full_training_replayed=False,full100Shuffle_replay_complete=p['epoch']==100,diagnostic_labels_read=False,new_model_fit=False)
    write(out/'AUGMENTED_VERIFIED.json',result);print(json.dumps(result))
if __name__=='__main__':main()
