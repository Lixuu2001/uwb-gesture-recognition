"""Fixed pilot training, own-state replay and selected-output audit; no held truth."""
from pathlib import Path
import copy,fcntl,hashlib,json,sys,time,shutil
from common import B,ready,read,write,sha
import engine as e
from data import load
from journal import StepLog,commit,load as journal_load
def row_for(ids,loc,pos,epoch,offset,updates,step):
    return dict(epoch=epoch,offset=offset,update=updates,positions_sha256=hashlib.sha256(pos.tobytes()).hexdigest(),training_ids_sha256=hashlib.sha256(ids[loc['train'][pos]].tobytes()).hexdigest(),**step)
def first():
    c=ready(True);ids,u,loc,maps,y,v=load(True);m,o,g=e.make();state=e.torch.load(B/'run/train/initial.pt',map_location='cpu',weights_only=True)
    e.restore(state,m,o,g);m.train();pos=e.order(len(y),1)[:64]
    result=e.step(m,o,e.np.array(maps[loc['train'][pos]],copy=True),y[pos],1,u[loc['train'][pos]])
    actual=e.capture(m,o,g,0,1);stored=e.torch.load(B/'run/train/first_update.pt',map_location='cpu',weights_only=True)
    assert e.semantic(actual)==e.semantic(stored)
    r=read(B/'run/train/FIRST_READY.json');assert r['step']==row_for(ids,loc,pos,1,0,1,result)
    write(B/'run/train/FIRST_RELEASE.json',dict(first_sha256=sha(B/'run/train/first_update.pt'),native_first_update_exact=True,state_sha256=e.semantic(actual),diagnostic_labels_read=False))
def fit(replay=False,resume=False):
    c=ready(True);root=B/'run';out=root/('replay' if replay else 'train')
    if resume:assert out.exists()
    else:out.mkdir(parents=True,exist_ok=False)
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'COMPLETE.json').exists()
    m,o,g=e.make();ids,u,loc,maps,y,v=load(True)
    if not (out/'split.npz').exists():e.np.savez_compressed(out/'split.npz',**{k:ids[p] for k,p in loc.items()})
    with e.np.load(out/'split.npz') as z:
        for k,p in loc.items():e.np.testing.assert_array_equal(z[k],ids[p])
    state,p=journal_load(out)
    if state is None:
        state=e.capture(m,o,g,0,0);initial=e.torch.load(B/'initial.pt',map_location='cpu',weights_only=True)
        assert e.semantic(state)==e.semantic(initial)
        e.atomic_save(out/'initial.pt',state);log=StepLog(out/'steps.jsonl');rec=e.record(state,0,0)
        if replay:assert rec==read(root/'train/epoch000.json')
        commit(out,state,rec,log.prefix())
    else:
        assert resume;e.restore(state,m,o,g);log=StepLog(out/'steps.jsonl',state['updates'],p['prefix_bytes'],p['prefix_sha256'])
    reference=(root/'train/steps.jsonl').open() if replay else None
    if replay:
        for _ in range(state['updates']):assert reference.readline()
    updates=state['updates'];best_key=state['best_key'];best_model=state['best_model'];best_epoch=state['best_epoch']
    try:
        for epoch in range(state['epoch']+1,65):
            m.train();order=e.order(len(y),epoch)
            for offset in range(0,len(y),64):
                pos=order[offset:offset+64];step=e.step(m,o,e.np.array(maps[loc['train'][pos]],copy=True),y[pos],epoch,u[loc['train'][pos]]);updates+=1
                row=row_for(ids,loc,pos,epoch,offset,updates,step)
                if replay:assert json.loads(reference.readline())==row,'Full fresh replay update mismatch'
                log.append(row)
                if updates==1 and not replay:
                    e.atomic_save(out/'first_update.pt',e.capture(m,o,g,0,1));write(out/'FIRST_READY.json',dict(step=row,checkpoint_sha256=sha(out/'first_update.pt')))
                    start=time.monotonic()
                    while not (out/'FIRST_RELEASE.json').exists():
                        assert time.monotonic()-start<600,'First update audit release timeout';time.sleep(.2)
                    release=read(out/'FIRST_RELEASE.json');assert release['first_sha256']==sha(out/'first_update.pt') and release['native_first_update_exact']
            scores,vm,vl=e.evaluate(m,maps,loc['validation'],v);key=(vm,-vl,-epoch)
            if best_key is None or key>tuple(best_key):best_key=key;best_epoch=epoch;best_model=copy.deepcopy(m.state_dict())
            state=e.capture(m,o,g,epoch,updates,best_key,best_model,best_epoch);rec=e.record(state,epoch,updates,vm,vl,scores)
            if replay:assert rec==read(root/f'train/epoch{epoch:03}.json'),'Full fresh replay epoch/state mismatch'
            commit(out,state,rec,log.prefix())
            write(out/'STATE.json',dict(state='full_replay' if replay else 'training',epoch=epoch,updates=updates,selected_epoch=best_epoch,diagnostic_labels_read=False,users7_8_labels_used=False))
            print(json.dumps(dict(stage='replay' if replay else 'train',epoch=epoch,updates=updates,validation_macro=vm)),flush=True)
        assert updates==4736 and log.index>=log.original
        if replay:assert reference.read()==''
        shutil.copyfile(out/'slot0.pt',out/'final_full.pt')
        terminal_native_digest=e.semantic(state)
        e.atomic_save(out/'selected.pt',dict(model=best_model,selected_epoch=best_epoch));m.load_state_dict(best_model,strict=True)
        for role in ['validation','diagnostic']:
            z=e.evaluate(m,maps,loc[role]);e.np.savez_compressed(out/(role+'_unscored.npz'),ids=ids[loc[role]],scores=z)
            if replay:
                with e.np.load(root/'train'/(role+'_unscored.npz')) as prior:
                    e.np.testing.assert_array_equal(prior['ids'],ids[loc[role]]);e.np.testing.assert_array_equal(prior['scores'],z)
        write(out/'COMPLETE.json',dict(state='full64_fresh_training_replay_exact_unscored' if replay else 'full64_training_complete_unscored',epochs=64,updates=updates,selected_epoch=best_epoch,final_state_sha256=terminal_native_digest,selected_sha256=sha(out/'selected.pt'),validation_outputs_sha256=sha(out/'validation_unscored.npz'),diagnostic_outputs_sha256=sha(out/'diagnostic_unscored.npz'),diagnostic_labels_read=False,users7_8_labels_used=False))
    finally:
        log.close();lock.close()
        if reference:reference.close()
def output_audit():
    ready(True);root=B/'run';ids,u,loc,maps,y,v=load(True);m,o,g=e.make()
    selected=e.torch.load(root/'train/selected.pt',map_location='cpu',weights_only=True);m.load_state_dict(selected['model'],strict=True)
    final,_=journal_load(root/'train');own,_=journal_load(root/'replay');assert e.semantic(final)==e.semantic(own)
    assert selected['selected_epoch']==final['best_epoch']
    assert e.semantic(selected['model'])==e.semantic(final['best_model'])
    for role in ['validation','diagnostic']:
        actual=e.evaluate(m,maps,loc[role])
        with e.np.load(root/'train'/(role+'_unscored.npz')) as z:
            e.np.testing.assert_array_equal(z['ids'],ids[loc[role]]);e.np.testing.assert_array_equal(z['scores'],actual)
    frozen=dict(state='full64_training_replay_and_selected_outputs_verified_frozen_unscored',epochs=64,updates=4736,selected_epoch=selected['selected_epoch'],full_training_replayed=True,diagnostic_labels_read=False,users7_8_labels_used=False,files={str(p.relative_to(root)):sha(p) for p in [root/'train/selected.pt',root/'train/final_full.pt',root/'train/validation_unscored.npz',root/'train/diagnostic_unscored.npz',root/'replay/COMPLETE.json']})
    write(root/'PREDICTIONS_FROZEN.json',frozen);print(json.dumps(frozen))
if __name__=='__main__':
    assert sys.argv[1] in ['train','replay','first','output']
    if sys.argv[1] in ['train','replay']:fit(sys.argv[1]=='replay','--resume' in sys.argv[2:])
    elif sys.argv[1]=='first':first()
    else:output_audit()
