"""One fixed member/fold, labels restricted to its inner train/validation."""
from pathlib import Path
import sys,time,hashlib,copy,fcntl,shutil
from common import B,require_initial,read,write,sha
from data import load
import engine as e
from journal import StepLog,commit,load as load_journal,put_json_match
def main():
    assert len(sys.argv) in [3,4];held=int(sys.argv[1]);arm=sys.argv[2];resume=len(sys.argv)==4 and sys.argv[3]=='--resume'
    c,p=require_initial();assert held in [2,3,4,5,6] and arm in ['timemil','shuffle']
    out=B/'folds'/f'held{held}'/arm
    if resume:assert out.exists()
    else:out.mkdir(parents=True,exist_ok=False)
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'ARM_COMPLETE.json').exists(),'Completed member must be verified/reused, never rerun'
    m,opt,g=e.make(arm);ids,users,loc,maps,y,v=load(held,labels=True);np=e.np
    split={k:ids[x] for k,x in loc.items()}
    if (out/'split.npz').exists():
        with np.load(out/'split.npz') as z:
            for k,x in split.items():np.testing.assert_array_equal(z[k],x)
    else:np.savez_compressed(out/'split.npz',**split)
    state,progress=load_journal(out)
    if state is None:
        state=e.capture(m,opt,g,0,0);seed=read(B/'SEED_FILES.json')[arm]
        assert e.semantic(state)==seed['state_sha256']
        e.atomic_save(out/'initial.pt',state)
        with (out/'steps.jsonl').open('ab'):pass
        log=StepLog(out/'steps.jsonl');commit(out,state,e.record(state,0,0),log.prefix())
    else:
        assert resume;e.restore(state,m,opt,g)
        log=StepLog(out/'steps.jsonl',state['updates'],progress['prefix_bytes'],progress['prefix_sha256'])
        write(out/f'RESUME_{time.time_ns()}.json',dict(committed_epoch=state['epoch'],committed_updates=state['updates'],original_log_rows=log.original,diagnostic_labels_read=False))
    old_updates=state['updates'];old_suffix=log.original-old_updates
    epochs=64 if arm=='timemil' else 100;batch=64 if arm=='timemil' else 20
    updates=state['updates'];best_key=state['best_key'];best_model=state['best_model'];best_epoch=state['best_epoch']
    try:
        for epoch in range(state['epoch']+1,epochs+1):
            m.train();order=e.order(arm,len(y),epoch,g)
            for offset in range(0,len(y),batch):
                positions=order[offset:offset+batch];loss=e.step(arm,m,opt,e.raw(arm,maps,loc['train'][positions]),y[positions],epoch);updates+=1
                row=dict(epoch=epoch,offset=offset,update=updates,loss=loss,positions_sha256=hashlib.sha256(positions.tobytes()).hexdigest(),training_ids_sha256=hashlib.sha256(ids[loc['train'][positions]].tobytes()).hexdigest())
                log.append(row)
                if updates==1:
                    first=e.capture(m,opt,g,0,1)
                    if (out/'first_update.pt').exists():assert e.semantic(e.torch.load(out/'first_update.pt',map_location='cpu',weights_only=True))==e.semantic(first)
                    else:e.atomic_save(out/'first_update.pt',first)
                    write(out/'FIRST_READY.json',dict(checkpoint_sha256=sha(out/'first_update.pt'),state_sha256=e.semantic(first),step=row))
                    start=time.monotonic()
                    while not (out/'FIRST_RELEASE.json').exists():
                        if time.monotonic()-start>c['first_update_release_timeout_seconds']:raise TimeoutError('First update audit release absent')
                        time.sleep(.2)
                    release=read(out/'FIRST_RELEASE.json');assert release['first_sha256']==sha(out/'first_update.pt')
                    assert release['certificate_sha256']==sha(out/'first_audit/CERTIFICATE.json')
                if old_suffix and updates==old_updates+old_suffix:
                    write(out/'SUFFIX_REPLAY_VERIFIED.json',dict(original_committed_updates=old_updates,uncommitted_updates_reproduced_exactly=old_suffix,original_raw_log_bytes_preserved=True))
            scores,vm,vl=e.evaluate(arm,m,maps,loc['validation'],v)
            key=(vm,-vl,-epoch)
            if arm=='timemil' and (best_key is None or key>tuple(best_key)):
                best_key=key;best_epoch=epoch;best_model=copy.deepcopy(m.state_dict())
            if arm=='shuffle':best_epoch=epoch
            state=e.capture(m,opt,g,epoch,updates,best_key,best_model,best_epoch)
            rec=e.record(state,epoch,updates,vm,vl,scores)
            path=out/f'validation_epoch{epoch:03}.npz'
            if path.exists():
                with np.load(path) as z:np.testing.assert_array_equal(z['scores'],scores);np.testing.assert_array_equal(z['ids'],ids[loc['validation']])
            else:np.savez_compressed(path,ids=ids[loc['validation']],scores=scores)
            commit(out,state,rec,log.prefix())
            write(out/'STATE.json',dict(state='training',held=held,arm=arm,epoch=epoch,updates=updates,diagnostic_labels_read=False));print(__import__('json').dumps(dict(held=held,arm=arm,epoch=epoch,updates=updates,validation_macro=vm)),flush=True)
        assert log.index>=log.original and updates==epochs*((len(y)+batch-1)//batch)
        final=B/'folds'/f'held{held}'/arm/'final_full.pt';shutil.copyfile(out/f'slot{epochs%2}.pt',final)
        selected=state['best_model'] if arm=='timemil' else state['model'];selected_epoch=state['best_epoch']
        e.atomic_save(out/'selected.pt',dict(model=selected,selected_epoch=selected_epoch,arm=arm,held=held))
        m.load_state_dict(selected,strict=True)
        for role in ['validation','diagnostic']:
            scores=e.evaluate(arm,m,maps,loc[role]);np.savez_compressed(out/(role+'_unscored.npz'),ids=ids[loc[role]],scores=scores)
        write(out/'STATE.json',dict(state='complete_unscored',held=held,arm=arm,epoch=epochs,updates=updates,selected_epoch=selected_epoch,diagnostic_labels_read=False))
        files={str(x.relative_to(out)):sha(x) for x in out.rglob('*') if x.is_file() and x.name not in ['worker.lock','ARM_COMPLETE.json'] and '.tmp' not in x.name}
        write(out/'ARM_COMPLETE.json',dict(state='complete_member_unscored',held=held,arm=arm,epochs=epochs,updates=updates,selected_epoch=selected_epoch,files=files,diagnostic_labels_read=False,users7_8_labels_used=False))
    finally:log.close();lock.close()
if __name__=='__main__':main()
