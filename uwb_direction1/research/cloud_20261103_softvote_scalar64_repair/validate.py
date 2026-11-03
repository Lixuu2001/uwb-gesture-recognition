"""Only validation user2 labels decoded; no diagnostic-output arrays loaded."""
from common import B,ready,read,write,sha,ids_maps,macro,validation_pass
def main():
    p=ready();out=B/'validation01';out.mkdir(exist_ok=False)
    import sys,time
    sys.path.insert(0,p['timemil_root'])
    from adapter import make_model,torch
    from vote import np,combine
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    started=time.monotonic();ids,users,va,_,maps=ids_maps(p)
    m=make_model('timemil',20261029)
    state=torch.load(p['timemil_root']+'/run02/timemil/best.pt',map_location='cpu',weights_only=False)
    assert state['epoch']==41
    m.load_state_dict(state['model'],strict=True);m.eval();parts=[]
    with torch.inference_mode():
        for start in range(0,len(va),64):
            x=torch.from_numpy(np.array(maps[va[start:start+64]],copy=True))
            parts.append(m(x).numpy())
    logits=np.concatenate(parts);assert logits.shape==(1000,12) and np.isfinite(logits).all()
    with np.load(p['shuffle_root']+'/run01/validation_unscored.npz') as z:
        other_ids=z['ids'];other=z['scores']
    np.testing.assert_array_equal(ids[va],other_ids)
    q,pred=combine(ids[va],logits,other_ids,other)
    truth=np.load(p['data']+'/cache/y_clutterremoved.npy',mmap_mode='r')
    y=np.array(truth[ids[va]],copy=True);assert set(np.unique(y))==set(range(10))
    scores=dict(timemil=macro(y,logits.argmax(1)),shuffle=macro(y,other.argmax(1)),mixture=macro(y,pred))
    assert abs(scores['timemil']-state['validation_macro'])<=1e-12,'Recovered model validation metric changed'
    assert abs(scores['shuffle']-.995)<=1e-12
    np.savez_compressed(out/'OUTPUTS.npz',ids=ids[va],timemil_logits=logits,shuffle_logits=other,probabilities=q,predictions=pred)
    result=dict(state='validation_only_complete',scores=scores,validation_gate_passed=validation_pass(scores,p),
        gain_over_best_member=scores['mixture']-max(scores['timemil'],scores['shuffle']),
        outputs_sha256=sha(out/'OUTPUTS.npz'),wall_seconds=time.monotonic()-started,
        diagnostic_outputs_decoded=False,diagnostic_labels_read=False,users7_8_labels_used=False,
        route='freeze_unscored_diagnostic_and_independently_audit_then_private_backup' if validation_pass(scores,p) else p['fail_route'])
    write(out/'RESULT.json',result);print(__import__('json').dumps(result))
if __name__=='__main__':main()
