"""Preregistered fixed source radar-exchange inference screen."""
import argparse,json,time,platform
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump
from radar_exchange import Q,S,arms,extract,decision
from run_energy_area import load_net
from second_order_core import score,macro


def main(a):
    c=json.loads(a.protocol.read_text());assert c['label_permutation']==Q.tolist() and c['view_permutation']==S
    assert not a.out.exists();a.out.mkdir();dump(a.out/'PROTOCOL.json',c);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    paths={k:Path(c['data'])/v for k,v in [('raw','X_clutterremoved.npy'),('labels','y_clutterremoved.npy'),('users','subj_clutterremoved.npy')]};paths['prepared']=Path(c['prepared'])
    hashes={k:digest(v) for k,v in paths.items()};assert hashes==c['input_sha256']
    identity=dict(protocol_sha256=digest(a.protocol),input_sha256=hashes,torch=torch.__version__,python=platform.python_version(),code_sha256={n:digest(Path(__file__).with_name(n)) for n in ['radar_exchange.py','run_radar_exchange.py','run_energy_area.py','circular_signal.py','second_order_core.py','audit_restored.py']});dump(a.out/'IDENTITY.json',identity)
    cache=np.load(c['prepared'],mmap_mode='r');old=Path(c['readout_reference']);feat=Path(c['feature_reference']);assert digest(old/'FILES.json')==c['readout_manifest_sha256'];assert digest(feat/'FILES.json')==c['feature_manifest_sha256'];rm=json.loads((old/'FILES.json').read_text());fm=json.loads((feat/'FILES.json').read_text())
    if a.cost_only:
        split=json.loads((old/'held3/split.json').read_text());net,_,_=load_net(Path(c['models'])/'held3');t=time.perf_counter();extract(net,cache,np.array(split['train'][:64]));elapsed=time.perf_counter()-t
        estimate=elapsed*14400/64*c['cost_margin'];r=dict(train64_seconds=elapsed,estimated_seconds=estimate,budget_seconds=c['budget_seconds'],passed=estimate<c['budget_seconds'],protocol_sha256=identity['protocol_sha256'],no_validation_or_diagnostic_inference=True);dump(a.out/'COST.json',r);print(json.dumps(r));return
    cost=json.loads(a.cost.read_text());assert cost['passed'] and cost['protocol_sha256']==identity['protocol_sha256'];t=time.perf_counter();rows=[];scores={k:[] for k in c['arms']}
    for h in c['users']:
        d=a.out/f'held{h}';d.mkdir();od=old/f'held{h}';fd=feat/f'held{h}'
        for n in ['models.npz','predictions.npz','split.json','selection.json']:assert digest(od/n)==rm[f'held{h}/{n}']
        assert digest(fd/'features.npz')==fm[f'held{h}/features.npz'];split=json.loads((od/'split.json').read_text());sel=json.loads((od/'selection.json').read_text())['onehot'];m=np.load(od/'models.npz');f=np.load(fd/'features.npz');lookup={int(v):i for i,v in enumerate(f['ids'])};net,sha,_=load_net(Path(c['models'])/f'held{h}');pred={};ff={};maxerr=0.
        for role in ['val','diagnostic']:
            ids=np.array(split[role]);x,sx=extract(net,cache,ids);ref=f['hidden'][[lookup[int(v)] for v in ids]];maxerr=max(maxerr,float(abs(x-ref).max()));np.testing.assert_allclose(x,ref,atol=2e-5,rtol=2e-5)
            o=score(x,m['mean'],m['scale'],m['bias'],m['coefs'][sel['selected']]);s=score(sx,m['mean'],m['scale'],m['bias'],m['coefs'][sel['selected']]);refscore=score(ref,m['mean'],m['scale'],m['bias'],m['coefs'][sel['selected']]);np.testing.assert_array_equal(o.argmax(1),refscore.argmax(1))
            ff.update({role+'_ids':ids,role+'_hidden':x,role+'_swapped_hidden':sx});pred[role+'_ids']=ids
            for k,v in arms(o,s).items():pred[role+'_'+k]=v
        np.savez_compressed(d/'features.npz',**ff);np.savez_compressed(d/'predictions.npz',**pred);dump(d/'split.json',split);dump(d/'selection.json',dict(inherited_onehot=sel,new_selection=False))
        # No evaluation labels used above. Complete both role predictions before opening them.
        with np.load(od/'predictions.npz') as z:
            np.testing.assert_array_equal(pred['val_ids'],z['validation_ids']);np.testing.assert_array_equal(pred['diagnostic_ids'],z['ids']);np.testing.assert_array_equal(pred['diagnostic_ridge'].argmax(1),z['onehot'].argmax(1));ss={k:macro(z['y'],pred['diagnostic_'+k].argmax(1)) for k in c['arms']};np.savez_compressed(d/'evaluation.npz',val_y=z['validation_y'],diagnostic_y=z['y'])
        for k,v in ss.items():scores[k].append(v)
        row=dict(held=h,scores=ss,checkpoint_sha256=sha,original_hidden_max_error=maxerr);dump(d/'result.json',row);rows.append(row);print(json.dumps(row),flush=True)
    comparisons=decision(scores,c);summary=dict(state='complete',scores=scores,means={k:float(np.mean(v)) for k,v in scores.items()},comparisons=comparisons,gate_passed=all(v['passed'] for v in comparisons.values()),rows=rows,seconds=time.perf_counter()-t,encoder_training=False,new_fit=False);dump(a.out/'SUMMARY.json',summary);dump(a.out/'FILES.json',{str(p.relative_to(a.out)):digest(p) for p in sorted(a.out.rglob('*')) if p.is_file()});print(json.dumps({k:summary[k] for k in ['means','gate_passed','seconds']}))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['protocol','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--cost-only',action='store_true');p.add_argument('--cost',type=Path);a=p.parse_args()
    with threadpool_limits(limits=1):main(a)
