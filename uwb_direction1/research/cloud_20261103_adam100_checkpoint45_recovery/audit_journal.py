"""Crash-consistent rolling state for independently replayed audit epochs.

Two alternating slots preserve the previous committed state until progress is
atomically published. This never imports a producer checkpoint as audit state.
"""
from pathlib import Path
from trial_common import read,write,sha
from adam_engine import torch,semantic,atomic_checkpoint,record_epoch

def commit(directory,out,record,state):
    directory=Path(directory);directory.mkdir(exist_ok=True)
    epoch=record['epoch'];old=read(directory/'PROGRESS.json') if (directory/'PROGRESS.json').exists() else None
    assert epoch==(old['epoch']+1 if old else 1)
    assert record==read(out/f'epoch{epoch:03d}.json')
    slot=directory/f'replay_slot{epoch%2}.pt';atomic_checkpoint(slot,state)
    journal=dict(epoch=epoch,updates=235*epoch,checkpoint_sha256=sha(slot),
        checkpoint_state_sha256=semantic(state),producer_epoch_record_sha256=sha(out/f'epoch{epoch:03d}.json'),
        previous_journal_sha256=old['journal_sha256'] if old else None,
        initial_audit_started_from_fresh_seed=True,diagnostic_labels_read=False)
    path=directory/f'epoch{epoch:03d}.json';write(path,journal)
    write(directory/'PROGRESS.json',dict(epoch=epoch,updates=235*epoch,
        slot=slot.name,journal_sha256=sha(path),checkpoint_sha256=sha(slot)))

def load(directory,out):
    directory=Path(directory)
    if not (directory/'PROGRESS.json').exists():return None
    p=read(directory/'PROGRESS.json');epoch=p['epoch'];assert 1<=epoch<=100
    previous=None
    for i in range(1,epoch+1):
        path=directory/f'epoch{i:03d}.json';j=read(path)
        assert j['epoch']==i and j['updates']==235*i
        assert j['initial_audit_started_from_fresh_seed'] is True and j['diagnostic_labels_read'] is False
        assert j['previous_journal_sha256']==previous
        assert j['producer_epoch_record_sha256']==sha(out/f'epoch{i:03d}.json')
        previous=sha(path)
    assert previous==p['journal_sha256']
    assert p['slot']==f'replay_slot{epoch%2}.pt'
    slot=directory/p['slot'];assert sha(slot)==p['checkpoint_sha256']==j['checkpoint_sha256']
    state=torch.load(slot,map_location='cpu',weights_only=True)
    assert state['epoch']==epoch and state['updates']==235*epoch
    assert semantic(state)==j['checkpoint_state_sha256']
    # Replayed model/moments/RNG state was checked against the producer record
    # before publication; verify the restored semantics against that record too.
    r=read(out/f'epoch{epoch:03d}.json')
    for key,name in [('model','model_sha256'),('optimizer','optimizer_sha256'),
                     ('order_rng','order_rng_sha256'),('torch_rng','torch_rng_sha256')]:
        assert semantic(state[key])==r[name]
    return state
