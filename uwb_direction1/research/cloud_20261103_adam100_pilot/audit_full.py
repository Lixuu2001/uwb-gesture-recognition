"""UNEXECUTED separate-process all100 replay and raw preprocessing audit.

Call only after the separately registered supervisor has verified producer exit.
Diagnostic labels remain sealed; completion/private/once gates are separate.
"""
import time,json
from trial_common import B,ready,read,write,sha,load_inputs
def main():
    c=ready();start=time.monotonic();out=B/'run01'
    assert not (B/'COMPLETE_AUDIT.json').exists()
    term=read(B/'supervisor01/TERMINATION.json')
    assert type(term['exit_code']) is int and term['exit_code']==0
    assert term['wall_timeout'] is False and term['resource_stop_reason'] is None
    assert type(term['cgroup_oom_kill_delta']) is int and term['cgroup_oom_kill_delta']==0
    files=read(out/'FILES.json')
    for n,h in files.items():assert sha(out/n)==h,n
    from adam_engine import np,torch,make,run,record_epoch,evaluate,semantic
    ids,users,loc,maps,y,v=load_inputs(c)
    with np.load(out/'split.npz') as z:
        for key,a in dict(ids=ids,users=users,**loc).items():
            np.testing.assert_array_equal(z[key],a)
    model=make();initial=semantic(model.state_dict())
    head=torch.load(out/'initial12_head.pt',map_location='cpu',weights_only=True)
    assert semantic(head)==semantic(model.fc.state_dict());del model,head
    with (out/'steps.jsonl').open() as steps:
        def on_step(r):assert json.loads(next(steps))==r,'step/order/loss mismatch'
        def on_epoch(r,scores,state):
            epoch=r['epoch']
            assert read(out/('epoch%03d.json'%epoch))==record_epoch(r,state)
            with np.load(out/('validation_epoch%03d.npz'%epoch)) as z:
                np.testing.assert_array_equal(z['ids'],ids[loc['validation']])
                np.testing.assert_array_equal(z['scores'],scores)
                np.testing.assert_array_equal(z['predictions'],scores.argmax(1))
            print(json.dumps(dict(stage='full100_replay',epoch=epoch)),flush=True)
        state=run(maps,loc,y,v,users,on_step,on_epoch)
        assert next(steps,None) is None
    assert state['epoch']==100 and state['updates']==23500
    saved=torch.load(out/'final_full.pt',map_location='cpu',weights_only=True)
    assert semantic(saved)==semantic(state)
    endpoint=read(out/'TRAIN_ENDPOINT.json')
    assert endpoint['endpoint']=='fixed_epoch100'
    assert endpoint['best_validation_selection_used'] is False
    assert endpoint['initial_full_model_sha256']==initial
    assert endpoint['final_full_state_sha256']==semantic(state)
    final=torch.load(out/'final_model.pt',map_location='cpu',weights_only=True)
    assert final['epoch']==100 and semantic(final['model'])==semantic(state['model'])
    model=make();model.load_state_dict(state['model'],strict=True);model.eval()
    for role in ['validation','diagnostic']:
        scores=evaluate(model,maps,loc[role])
        with np.load(out/(role+'_unscored.npz')) as z:
            np.testing.assert_array_equal(z['ids'],ids[loc[role]])
            np.testing.assert_array_equal(z['scores'],scores)
            np.testing.assert_array_equal(z['predictions'],scores.argmax(1))
    from audit_pixels import verify
    pixels=verify(c,maps,ids);assert pixels['pixel_count']==1038643200
    for n,h in files.items():assert sha(out/n)==h,n
    write(B/'COMPLETE_AUDIT.json',dict(state='pass',
        full100_epoch23500_update_replay_exact=True,
        all100_validation_outputs_exact=True,fixed_final100_endpoint_exact=True,
        full_model_BN_Adam_two_moments_steps_order_and_Torch_RNG_exact=True,
        all2200_final_scores_predictions_exact=True,diagnostic_labels_read=False,
        users7_8_labels_used=False,independent_validation=False,
        novel_contribution_validated=False,auditor_sha256=sha(__file__),
        artifact_hashes={**{str(out/n):h for n,h in files.items()},
            str(out/'FILES.json'):sha(out/'FILES.json')},
        wall_seconds=time.monotonic()-start,**pixels))
if __name__=='__main__':main()
