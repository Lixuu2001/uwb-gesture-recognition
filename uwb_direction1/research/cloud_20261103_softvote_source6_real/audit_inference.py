"""Selected-state/optimizer/first-update/role and full output replay, no held truth."""
import sys,hashlib
from common import B,require_initial,read,write,sha
from checks import verify_member
from data import load,macro
import engine as e
from journal import load as load_journal
def main():
    held=int(sys.argv[1]);arm=sys.argv[2];c,p=require_initial();out,complete=verify_member(held,arm);dest=out/'inference_audit';dest.mkdir(exist_ok=False)
    m,o,g=e.make(arm);ids,users,loc,maps,y,v=load(held,labels=True);np=e.np
    state,progress=load_journal(out);assert state['epoch']==complete['epochs'] and state['updates']==complete['updates']
    assert e.semantic(e.torch.load(out/'final_full.pt',map_location='cpu',weights_only=True))==e.semantic(state)
    n=(len(y)+(63 if arm=='timemil' else 19))//(64 if arm=='timemil' else 20)
    assert state['updates']==n*complete['epochs']
    rows=[read(out/f'epoch{i:03}.json') for i in range(1,complete['epochs']+1)]
    assert all(r['updates']==n*(i+1) for i,r in enumerate(rows))
    if arm=='timemil':
        chosen=max(rows,key=lambda r:(r['validation_macro'],-r['validation_loss'],-r['epoch']))
        assert chosen['epoch']==state['best_epoch']==complete['selected_epoch']
        assert tuple(state['best_key'])==(chosen['validation_macro'],-chosen['validation_loss'],-chosen['epoch'])
        selected=state['best_model'];assert state['optimizer']['count']==state['updates']
        fast=state['optimizer']['fast'];assert all(int(s['step'])==state['updates'] for s in fast['state'].values())
    else:
        assert state['best_epoch']==100==complete['selected_epoch'];selected=state['model']
        assert all(int(s['step'])==state['updates'] for s in state['optimizer']['state'].values())
    actual=e.torch.load(out/'selected.pt',map_location='cpu',weights_only=True)
    assert e.semantic(actual['model'])==e.semantic(selected) and actual['selected_epoch']==complete['selected_epoch']
    with np.load(out/'split.npz') as z:
        for role in loc:np.testing.assert_array_equal(z[role],ids[loc[role]])
    m.load_state_dict(selected,strict=True)
    for role in ['validation','diagnostic']:
        scores=e.evaluate(arm,m,maps,loc[role])
        with np.load(out/(role+'_unscored.npz')) as z:
            np.testing.assert_array_equal(z['ids'],ids[loc[role]]);np.testing.assert_array_equal(z['scores'],scores)
        if role=='validation':
            rec=rows[complete['selected_epoch']-1]
            assert abs(macro(v,scores.argmax(1))-rec['validation_macro'])<=1e-12
            assert hashlib.sha256(scores.tobytes()).hexdigest()==rec['validation_logits_sha256']
    first=read(out/'first_audit/CERTIFICATE.json');assert first['state']=='pass' and first['first_checkpoint_sha256']==sha(out/'first_update.pt')
    targets=['selected.pt','final_full.pt','split.npz','ARM_COMPLETE.json','validation_unscored.npz','diagnostic_unscored.npz','PROGRESS.json','steps.jsonl','first_update.pt','first_audit/CERTIFICATE.json']
    write(dest/'CERTIFICATE.json',dict(state='pass',held=held,arm=arm,complete_model_optimizer_rng_journal_and_selection_checked=True,all_validation_and_unscored_diagnostic_logits_replayed_exactly=True,full_training_replayed=False,artifact_hashes={str(out/n):sha(out/n) for n in targets},diagnostic_labels_read=False,users7_8_labels_used=False))
if __name__=='__main__':main()
