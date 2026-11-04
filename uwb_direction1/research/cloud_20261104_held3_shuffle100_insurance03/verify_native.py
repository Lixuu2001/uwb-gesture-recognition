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
    # Recreate through the original family model/optimizer, then restore every
    # native sampler and RNG state. Read-only state equality, no forward/update.
    m,opt,g=e.make(arm);e.restore(state,m,opt,g)
    restored=e.capture(m,opt,g,state['epoch'],state['updates'],state['best_key'],state['best_model'],state['best_epoch'])
    assert e.semantic(restored)==e.semantic(state)
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
    complete=None
    if (d/'ARM_COMPLETE.json').exists():
        complete=read(d/'ARM_COMPLETE.json')
        assert complete['state']=='complete_member_unscored' and complete['arm']==arm and complete['held']==held
        assert not complete['diagnostic_labels_read'] and not complete['users7_8_labels_used']
        assert complete['epochs']==(64 if arm=='timemil' else 100)
        assert complete['updates']==state['updates'] and complete['selected_epoch']==state['best_epoch']
        for n,h in complete['files'].items():assert sha(d/n)==h,n
        selected=e.torch.load(d/'selected.pt',map_location='cpu',weights_only=True)
        assert selected['selected_epoch']==complete['selected_epoch'] and selected['arm']==arm and selected['held']==held
        assert e.semantic(selected['model'])==e.semantic(state['best_model'] if arm=='timemil' else state['model'])
    replay=None
    if (d/'replay/PROGRESS.json').exists():
        r,rp=load(d/'replay');assert r is not None
        for epoch in range(rp['epoch']+1):assert read(d/'replay'/f'journal{epoch:03}.json')['producer_record_sha256']==sha(d/f'epoch{epoch:03}.json')
        replay=rp['epoch']
    for n,h in manifest.items():assert sha(out/n)==h,n
    usage=resource.getrusage(resource.RUSAGE_SELF)
    result=dict(state='pass_captured_native_committed_model_optimizer_RNG_journal_and_log_prefix',arm=arm,committed_epoch=p['epoch'],committed_updates=state['updates'],captured_complete_step_rows=rows,uncommitted_suffix_rows=rows-state['updates'],state_sha256=e.semantic(state),checkpoint_sha256=sha(d/p['slot']),native_instantiation_restore_recapture_exact=True,complete_member_manifest_verified=complete is not None,selected_model_matches_native_committed_state=complete is not None,own_replay_epoch=replay,wall_seconds=time.monotonic()-start,CPU_seconds=usage.ru_utime+usage.ru_stime,peak_RSS_KiB=usage.ru_maxrss,diagnostic_labels_read=False,users7_8_labels_used=False,new_model_fits_started=False,full_training_replayed=False)
    write(out/(arm+'_SNAPSHOT_VERIFIED.json'),result);print(json.dumps(result))
if __name__=='__main__':main()
