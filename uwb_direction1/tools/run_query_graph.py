import argparse,json,time
from pathlib import Path
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump
from query_graph import graphs,smooth,probabilities
from second_order_core import score,macro


def main(a):
    cfg=json.loads(a.protocol.read_text());assert not a.out.exists();a.out.mkdir();dump(a.out/'PROTOCOL.json',cfg)
    identity={'protocol_sha256':digest(a.protocol),'code_sha256':{n:digest(Path(__file__).with_name(n)) for n in ['query_graph.py','run_query_graph.py','second_order_core.py','audit_restored.py']}};dump(a.out/'IDENTITY.json',identity)
    if a.cost_only:
        rng=np.random.default_rng(12);x=rng.normal(size=(1200,128));p=probabilities(rng.random((1200,12)));t=time.perf_counter()
        for w in graphs(x,np.arange(1200),cfg['k'],cfg['seed']).values():smooth(p,w,cfg['mu'])
        elapsed=time.perf_counter()-t;estimate=elapsed*12*3;cost=dict(synthetic_seconds=elapsed,estimated_seconds=estimate,budget=cfg['budget_seconds'],passed=estimate<cfg['budget_seconds'],protocol_sha256=identity['protocol_sha256'],real_data_read=False);dump(a.out/'COST.json',cost);print(json.dumps(cost));return
    cost=json.loads(a.cost.read_text());assert cost['passed'] and cost['protocol_sha256']==identity['protocol_sha256']
    old=Path(cfg['readout_reference']);feat=Path(cfg['feature_reference']);assert digest(old/'FILES.json')==cfg['readout_manifest_sha256'];assert digest(feat/'FILES.json')==cfg['feature_manifest_sha256'];rm=json.loads((old/'FILES.json').read_text());fm=json.loads((feat/'FILES.json').read_text())
    scores={k:[] for k in cfg['arms']};rows=[];t=time.perf_counter()
    for h in cfg['users']:
        d=a.out/f'held{h}';d.mkdir();od=old/f'held{h}';fd=feat/f'held{h}'
        for n in ['models.npz','predictions.npz','split.json','selection.json']:assert digest(od/n)==rm[f'held{h}/{n}']
        assert digest(fd/'features.npz')==fm[f'held{h}/features.npz'];split=json.loads((od/'split.json').read_text());selection=json.loads((od/'selection.json').read_text());checkpoint=np.load(od/'models.npz');features=np.load(fd/'features.npz');ids=features['ids'];hidden=features['hidden'];lookup={int(v):i for i,v in enumerate(ids)};j=selection['onehot']['selected'];dump(d/'selection.json',{'inherited_onehot':selection['onehot'],'new_selection':False});dump(d/'split.json',split)
        pred={};stats={}
        for role in ['val','diagnostic']:
            ii=np.array(split[role]);x=hidden[[lookup[int(v)] for v in ii]];z=(x-checkpoint['mean'])/checkpoint['scale'];s=score(x,checkpoint['mean'],checkpoint['scale'],checkpoint['bias'],checkpoint['coefs'][j]);p=probabilities(s);np.testing.assert_array_equal(p.argmax(1),s.argmax(1));pred[role+'_ids']=ii;pred[role+'_ridge']=p;stats[role]={}
            for arm,w in graphs(z,ii,cfg['k'],cfg['seed']).items():
                f,st=smooth(p,w,cfg['mu']);pred[role+'_'+arm]=f;stats[role][arm]=st;sparse.save_npz(d/f'{role}_{arm}_graph.npz',w)
        np.savez_compressed(d/'predictions.npz',**pred) # Persist all unlabeled inference before opening evaluation labels.
        with np.load(od/'predictions.npz') as truth:
            for role,ik,yk in [('val','validation_ids','validation_y'),('diagnostic','ids','y')]:np.testing.assert_array_equal(pred[role+'_ids'],truth[ik])
            y=truth['y'];ss={arm:macro(y,pred['diagnostic_'+arm].argmax(1)) for arm in cfg['arms']};np.testing.assert_array_equal(pred['diagnostic_ridge'].argmax(1),truth['onehot'].argmax(1));np.savez_compressed(d/'evaluation.npz',diagnostic_y=y,val_y=truth['validation_y'])
        for arm,v in ss.items():scores[arm].append(v)
        row=dict(held=h,scores=ss,solver=stats);dump(d/'result.json',row);rows.append(row);print(json.dumps({'held':h,'scores':ss}),flush=True)
    comparisons={}
    for arm,gain in cfg['gate']['minimum_mean_gain'].items():
        delta=np.array(scores[cfg['primary']])-scores[arm];comparisons[arm]=dict(mean_gain=float(delta.mean()),wins=int((delta>1e-12).sum()),minimum_gain=float(delta.min()),passed=bool(delta.mean()>=gain and (delta>1e-12).sum()>=cfg['gate']['minimum_wins'] and delta.min()>=-cfg['gate']['maximum_paired_decline']))
    summary=dict(state='complete',means={k:float(np.mean(v)) for k,v in scores.items()},scores=scores,comparisons=comparisons,gate_passed=all(v['passed'] for v in comparisons.values()),seconds=time.perf_counter()-t,rows=rows,query_labels_used_for_prediction=False,encoder_training=False);dump(a.out/'SUMMARY.json',summary);dump(a.out/'FILES.json',{str(p.relative_to(a.out)):digest(p) for p in sorted(a.out.rglob('*')) if p.is_file()});print(json.dumps({k:summary[k] for k in ['means','gate_passed','seconds']}))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['protocol','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cost-only',action='store_true');p.add_argument('--cost',type=Path);a=p.parse_args()
    with threadpool_limits(limits=1):main(a)
