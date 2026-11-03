"""Full generic-fold Shuffle100 independent replay; own resumable state only."""
from pathlib import Path
import sys,hashlib,json
from common import require_initial,read,write,sha
from checks import verify_member
from data import load
import engine as e
from journal import commit,load as load_journal
def main():
    held=int(sys.argv[1]);c,p=require_initial();out,complete=verify_member(held,'shuffle');dest=out/'replay';dest.mkdir(exist_ok=True)
    assert not (dest/'COMPLETE_AUDIT.json').exists()
    m,opt,g=e.make('shuffle');ids,users,loc,maps,y,v=load(held,labels=True);np=e.np
    state,progress=load_journal(dest)
    if state is None:
        state=e.capture(m,opt,g,0,0);rec=e.record(state,0,0)
        assert rec==read(out/'epoch000.json')
        prefix=dict(updates=0,prefix_bytes=0,prefix_sha256=hashlib.sha256(b'').hexdigest())
        commit(dest,state,rec,prefix,sha(out/'epoch000.json'));progress=read(dest/'PROGRESS.json')
    else:
        for epoch in range(progress['epoch']+1):assert read(dest/f'journal{epoch:03}.json')['producer_record_sha256']==sha(out/f'epoch{epoch:03}.json')
        e.restore(state,m,opt,g)
    lines=(out/'steps.jsonl').read_bytes().splitlines(keepends=True);assert len(lines)==complete['updates']
    prefix=b''.join(lines[:state['updates']]);assert len(prefix)==progress['prefix_bytes'] and hashlib.sha256(prefix).hexdigest()==progress['prefix_sha256']
    h=hashlib.sha256(prefix);size=len(prefix);updates=state['updates']
    for epoch in range(state['epoch']+1,101):
        m.train();order=e.order('shuffle',len(y),epoch,g)
        for offset in range(0,len(y),20):
            pos=order[offset:offset+20];loss=e.step('shuffle',m,opt,e.raw('shuffle',maps,loc['train'][pos]),y[pos],epoch);updates+=1
            row=dict(epoch=epoch,offset=offset,update=updates,loss=loss,positions_sha256=hashlib.sha256(pos.tobytes()).hexdigest(),training_ids_sha256=hashlib.sha256(ids[loc['train'][pos]].tobytes()).hexdigest())
            line=lines[updates-1];assert json.loads(line)==row,'Independent replay update differs'
            h.update(line);size+=len(line)
        scores,vm,vl=e.evaluate('shuffle',m,maps,loc['validation'],v)
        state=e.capture(m,opt,g,epoch,updates,None,None,epoch);rec=e.record(state,epoch,updates,vm,vl,scores)
        assert rec==read(out/f'epoch{epoch:03}.json')
        with np.load(out/f'validation_epoch{epoch:03}.npz') as z:np.testing.assert_array_equal(z['scores'],scores)
        commit(dest,state,rec,dict(updates=updates,prefix_bytes=size,prefix_sha256=h.hexdigest()),sha(out/f'epoch{epoch:03}.json'))
        write(dest/'STATE.json',dict(state='independent_replay',epoch=epoch,updates=updates,diagnostic_labels_read=False));print(json.dumps(dict(held=held,replay_epoch=epoch,updates=updates)),flush=True)
    assert updates==complete['updates']
    final=e.torch.load(out/'final_full.pt',map_location='cpu',weights_only=True);assert e.semantic(final)==e.semantic(state)
    for role in ['validation','diagnostic']:
        scores=e.evaluate('shuffle',m,maps,loc[role])
        with np.load(out/(role+'_unscored.npz')) as z:np.testing.assert_array_equal(z['ids'],ids[loc[role]]);np.testing.assert_array_equal(z['scores'],scores)
    write(dest/'COMPLETE_AUDIT.json',dict(state='pass',held=held,all100epochs_all_updates_replayed_exactly=True,initial_audit_started_from_fresh_seed=True,own_checkpoint_only_recovery=True,replay_progress_sha256=sha(dest/'PROGRESS.json'),updates=updates,diagnostic_labels_read=False,users7_8_labels_used=False))
if __name__=='__main__':main()
