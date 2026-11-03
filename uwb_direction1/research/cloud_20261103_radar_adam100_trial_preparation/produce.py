"""UNEXECUTED producer: all100 epochs, fixed final model, held labels sealed."""
import time,json
from trial_common import B,ready,write,sha,load_inputs
def main():
    c=ready();out=B/'run01';out.mkdir(exist_ok=False);start=time.monotonic()
    from adam_engine import np,torch,make,run,record_epoch,atomic_checkpoint,evaluate,semantic
    ids,users,loc,maps,y,v=load_inputs(c)
    np.savez(out/'split.npz',ids=ids,users=users,**loc)
    model=make();torch.save(model.fc.state_dict(),out/'initial12_head.pt')
    initial=semantic(model.state_dict());del model
    write(out/'IDENTITY.json',dict(protocol_sha256=sha(B/'TRIAL_PROTOCOL.json'),
        source_ids_sha256=c['source_manifest_sha256'],seed=20261029,
        training_users=[3,4,5,6],validation_user=2,diagnostic_user=1,
        checkpoint_selection='fixed_epoch100',
        validation_labels_for_checkpoint_selection=False,
        diagnostic_labels_read=False,users7_8_labels_used=False))
    with (out/'steps.jsonl').open('x',buffering=1) as log:
        def on_step(r):log.write(json.dumps(r,separators=(',',':'))+'\n')
        def on_epoch(r,scores,state):
            epoch=r['epoch'];assert epoch in range(1,101)
            assert state['epoch']==epoch and state['updates']==235*epoch
            row=record_epoch(r,state)
            np.savez_compressed(out/('validation_epoch%03d.npz'%epoch),
                ids=ids[loc['validation']],scores=scores,predictions=scores.argmax(1))
            # One rolling full state, overwritten atomically at all100 boundaries.
            atomic_checkpoint(out/'final_full.pt',state)
            write(out/('epoch%03d.json'%epoch),row)
            write(out/'STATE.json',dict(state='full100_training',epoch=epoch,
                updates=state['updates'],wall_seconds=time.monotonic()-start,
                diagnostic_labels_read=False))
            print(json.dumps(row),flush=True)
        state=run(maps,loc,y,v,users,on_step,on_epoch)
    assert state['epoch']==100 and state['updates']==23500
    atomic_checkpoint(out/'final_model.pt',dict(model=state['model'],epoch=100))
    model=make();model.load_state_dict(state['model'],strict=True);model.eval()
    for role in ['validation','diagnostic']:
        scores=evaluate(model,maps,loc[role])
        np.savez_compressed(out/(role+'_unscored.npz'),
            ids=ids[loc[role]],scores=scores,predictions=scores.argmax(1))
    write(out/'TRAIN_ENDPOINT.json',dict(epochs=100,updates=23500,
        endpoint='fixed_epoch100',best_validation_selection_used=False,
        initial_full_model_sha256=initial,final_full_state_sha256=semantic(state),
        all_layers_trainable=True,diagnostic_labels_read=False,users7_8_labels_used=False))
    write(out/'STATE.json',dict(state='producer_complete_diagnostic_unscored',
        wall_seconds=time.monotonic()-start,diagnostic_labels_read=False))
    write(out/'FILES.json',{p.name:sha(p) for p in sorted(out.iterdir())
        if p.is_file() and p.name!='FILES.json'})
if __name__=='__main__':main()
