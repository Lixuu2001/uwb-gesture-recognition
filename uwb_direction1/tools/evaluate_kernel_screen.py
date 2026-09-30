"""One-shot outer evaluation after the primary validation gate, including repeat."""
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from audit_restored import digest, dump, metrics
from kernel_screen import features, kernel

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--runs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Use a new output directory')
    primary=json.loads((a.runs[0]/'RESULTS.json').read_text())
    assert primary['gate_passed'] and len(primary['folds'])==8
    a.out.mkdir()
    report={'state':'running','exploratory':True,'encoder_retrained':False,'runs':[],'script_sha256':digest(__file__)}
    for run in a.runs:
        protocol=json.loads((run/'protocol.json').read_text()); selected=json.loads((run/'RESULTS.json').read_text())
        assert len(selected['folds'])==8 and digest(run/'protocol.json')==selected['protocol_sha256']
        rows=[]
        for h in range(1,9):
            base=a.data/f'experiments/p1_20260928/multiview_s2026/fold{h}'
            with np.load(base/'predictions.npz') as c,np.load(run/f'fold{h}/mapping.npz') as mp:
                tr,va,te=(c['role']==i for i in range(3)); mapping=dict(mp)
                np.testing.assert_array_equal(mapping['source_ids'],c['ids'][tr])
                assert not np.isin(mapping['landmark_ids'],c['ids'][te|va]).any()
                x=c['features'][te].astype(float);y=c['y'][te];ids=c['ids'][te]
                result={'held_user':h,'metrics':{'original_LTR':metrics(y,c['logits'][te,6].argmax(-1))},'hashes':{}}
                scores={'original_LTR':c['logits'][te,6]}
                for kind in protocol['kernels']:
                    path=run/f'fold{h}/{kind}.npz'
                    with np.load(path) as m:
                        score=features(x,mapping,kind,m['whitening'])@m['w']+m['b']
                        # Algebraically independent order of multiplications verifies saved readout replay.
                        z=(x-mapping['mean'])/mapping['scale']
                        replay=kernel(z,mapping['landmarks'],mapping['median'],kind)@(m['whitening']@m['w'])+m['b']
                        np.testing.assert_allclose(score,replay,atol=1e-10,rtol=1e-10)
                        scores[kind]=score
                        result['metrics'][kind]=metrics(y,score.argmax(-1));result['hashes'][kind]=digest(path)
                dest=a.out/f'seed{protocol["seed"]}';dest.mkdir(exist_ok=True)
                np.savez_compressed(dest/f'fold{h}_test.npz',ids=ids,y=y,**scores)
                result['prediction_sha256']=digest(dest/f'fold{h}_test.npz');rows.append(result)
        arrays={k:np.array([r['metrics'][k]['macro_accuracy'] for r in rows]) for k in ['original_LTR','additive','joint']}
        summary={k:dict(mean_user_macro_accuracy=float(v.mean()),worst_user_macro_accuracy=float(v.min()),per_user=v.tolist()) for k,v in arrays.items()}
        diffs={}
        rng=np.random.default_rng(20260930);draw=rng.integers(0,8,(10000,8))
        for k in ['original_LTR','additive']:
            d=100*(arrays['joint']-arrays[k]);ci=np.quantile(d[draw].mean(1),[.025,.975])
            diffs[k]=dict(mean_pp=float(d.mean()),positive_users=int((d>0).sum()),per_user_pp=d.tolist(),descriptive_user_bootstrap_95_pp=ci.tolist())
        report['runs'].append(dict(seed=protocol['seed'],validation_gate=selected['gate_passed'],selection_sha256=digest(run/'RESULTS.json'),folds=rows,summary=summary,paired=diffs))
        dump(a.out/'RESULTS.json',report)
    report['state']='complete';dump(a.out/'RESULTS.json',report)
    print(json.dumps([{k:v for k,v in r.items() if k!='folds'} for r in report['runs']]))
if __name__=='__main__':
    with threadpool_limits(limits=3):main()
