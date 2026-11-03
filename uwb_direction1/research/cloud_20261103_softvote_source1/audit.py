"""Independent inference replay and scalar-link verification; no held truth."""
from common import B,ready,read,write,sha,ids_maps,macro,validation_pass
import math
def independent_mixture(a,b):
    import numpy as np
    result=[]
    for za,zb in zip(a,b):
        # log(sigmoid) independently with scalar log1p, stable on both tails.
        la=[-math.log1p(math.exp(-float(z))) if z>=0 else float(z)-math.log1p(math.exp(float(z))) for z in za]
        aa=[math.exp(x-max(la)) for x in la];bb=[math.exp(float(x)-max(zb)) for x in zb]
        result.append([.5*x/math.fsum(aa)+.5*y/math.fsum(bb) for x,y in zip(aa,bb)])
    return np.array(result,dtype=np.float64)
def main():
    p=ready();out=B/'audit01';out.mkdir(exist_ok=False)
    import sys
    sys.path.insert(0,p['timemil_root'])
    from adapter import make_model,torch
    from vote import np,combine
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    ids,users,va,held,maps=ids_maps(p);v=read(B/'validation01/RESULT.json')
    assert v['outputs_sha256']==sha(B/'validation01/OUTPUTS.npz')
    with np.load(B/'validation01/OUTPUTS.npz') as z:
        saved={k:z[k] for k in z.files}
    np.testing.assert_array_equal(saved['ids'],ids[va])
    m=make_model('timemil',p['seed']);state=torch.load(p['timemil_root']+'/run02/timemil/best.pt',map_location='cpu',weights_only=False)
    m.load_state_dict(state['model'],strict=True);m.eval();logits=[]
    with torch.inference_mode():
        for start in range(0,1000,64):logits.append(m(torch.from_numpy(np.array(maps[va[start:start+64]],copy=True))).numpy())
    fresh=np.concatenate(logits);np.testing.assert_array_equal(fresh,saved['timemil_logits'])
    with np.load(p['shuffle_root']+'/run01/validation_unscored.npz') as z:
        np.testing.assert_array_equal(z['ids'],ids[va]);np.testing.assert_array_equal(z['scores'],saved['shuffle_logits'])
    q=independent_mixture(fresh,saved['shuffle_logits'])
    np.testing.assert_allclose(q,saved['probabilities'],rtol=0,atol=1e-14)
    np.testing.assert_array_equal(q.argmax(1),saved['predictions'])
    truth=np.load(p['data']+'/cache/y_clutterremoved.npy',mmap_mode='r');y=np.array(truth[ids[va]],copy=True)
    scores=dict(timemil=macro(y,fresh.argmax(1)),shuffle=macro(y,saved['shuffle_logits'].argmax(1)),mixture=macro(y,q.argmax(1)))
    assert scores==v['scores'] and validation_pass(scores,p)==v['validation_gate_passed']
    hashes={str(B/'validation01'/n):sha(B/'validation01'/n) for n in ['RESULT.json','OUTPUTS.npz']}
    if v['validation_gate_passed']:
        with np.load(p['timemil_root']+'/run02/timemil/predictions_unscored.npz') as z:ai=z['ids'];a=z['logits']
        with np.load(p['shuffle_root']+'/run01/diagnostic_unscored.npz') as z:bi=z['ids'];b=z['scores']
        np.testing.assert_array_equal(ai,ids[held]);np.testing.assert_array_equal(bi,ids[held])
        mix,pred=combine(ai,a,bi,b);ind=independent_mixture(a,b)
        np.testing.assert_allclose(mix,ind,rtol=0,atol=1e-14);np.testing.assert_array_equal(pred,ind.argmax(1))
        np.savez_compressed(out/'DIAGNOSTIC_UNSCORED.npz',ids=ai,probabilities=mix,predictions=pred)
        hashes[str(out/'DIAGNOSTIC_UNSCORED.npz')]=sha(out/'DIAGNOSTIC_UNSCORED.npz')
    result=dict(state='pass',validation_fresh_TimeMIL_replay_exact=True,all1000_probabilities_independent_scalar_verified=True,
        all1200_unscored_diagnostic_probabilities_independent_scalar_verified=v['validation_gate_passed'],
        validation_gate_passed=v['validation_gate_passed'],artifact_hashes=hashes,
        diagnostic_labels_read=False,users7_8_labels_used=False,new_model_training_audited=False,
        limitation='Existing member training audits reused; this verifies inference, links, ID order and fixed-mixture outputs, not new training or independent validation.')
    write(out/'CERTIFICATE.json',result);print(__import__('json').dumps(result))
if __name__=='__main__':main()
