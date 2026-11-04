"""Read-only verification of captured native tensor/optimizer/RNG states.

Separate process per family; no model update, inference or held truth access.
"""
from pathlib import Path
import sys,math,hashlib,json,time,resource
from snapshot import ROOT,read,write,sha
def main():
    assert len(sys.argv)==3 and sys.argv[1] in ['timemil','shuffle'];arm=sys.argv[1];out=Path(sys.argv[2]);start=time.monotonic()
    manifest=read(out/'PAYLOAD_HASHES.json')
    for n,h in manifest.items():assert sha(out/n)==h,n
    sys.path.insert(0,str(ROOT));import engine as e
    e.setup(arm);from journal import load,StepLog
    held=read(out/'SNAPSHOT.json')['held'];d=out/'folds'/f'held{held}'/arm;state,p=load(d)
    assert state is not None
    with e.np.load(d/'split.npz') as z:n=len(z['train'])
    assert state['updates']==p['epoch']*math.ceil(n/(64 if arm=='timemil' else 20))
    log=StepLog(d/'steps.jsonl',state['updates'],p['prefix_bytes'],p['prefix_sha256']);rows=log.original;log.close()
    first=e.torch.load(d/'first_update.pt',map_location='cpu',weights_only=True)
    assert e.semantic(first)==read(d/'FIRST_READY.json')['state_sha256']
    assert sha(d/'first_update.pt')==read(d/'first_audit/CERTIFICATE.json')['first_checkpoint_sha256']
    if arm=='timemil' and p['epoch']:
        expected=max((read(d/f'epoch{epoch:03}.json')['validation_macro'],-read(d/f'epoch{epoch:03}.json')['validation_loss'],-epoch) for epoch in range(1,p['epoch']+1))
        assert tuple(state['best_key'])==expected and state['best_epoch']==-expected[2] and state['best_model'] is not None
        assert state['model']['wave1'].data_ptr()==state['model']['wave1_'].data_ptr()
    elif arm=='timemil':assert state['best_key'] is None and state['best_epoch']==0
    replay=None
    if (d/'replay/PROGRESS.json').exists():
        r,rp=load(d/'replay');assert r is not None
        for epoch in range(rp['epoch']+1):assert read(d/'replay'/f'journal{epoch:03}.json')['producer_record_sha256']==sha(d/f'epoch{epoch:03}.json')
        replay=rp['epoch']
    for n,h in manifest.items():assert sha(out/n)==h,n
    usage=resource.getrusage(resource.RUSAGE_SELF)
    result=dict(state='pass_captured_native_committed_model_optimizer_RNG_journal_and_log_prefix',arm=arm,committed_epoch=p['epoch'],committed_updates=state['updates'],captured_complete_step_rows=rows,uncommitted_suffix_rows=rows-state['updates'],state_sha256=e.semantic(state),checkpoint_sha256=sha(d/p['slot']),own_replay_epoch=replay,wall_seconds=time.monotonic()-start,CPU_seconds=usage.ru_utime+usage.ru_stime,peak_RSS_KiB=usage.ru_maxrss,diagnostic_labels_read=False,users7_8_labels_used=False,new_model_fits_started=False,full_training_replayed=False)
    write(out/(arm+'_SNAPSHOT_VERIFIED.json'),result);print(json.dumps(result))
if __name__=='__main__':main()
