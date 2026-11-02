"""Fixed coherence-gated orientation readout source screen."""
import argparse,json,time,platform
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump
from energy_area import source_roles
from orientation_features import original_tensors,describe
from second_order_core import fit_family,score,macro


def main(a):
    c=json.loads(a.protocol.read_text());assert not a.out.exists();a.out.mkdir()
    paths={k:a.data/'cache'/n for k,n in [('raw','X_clutterremoved.npy'),('labels','y_clutterremoved.npy'),('users','subj_clutterremoved.npy')]};paths['prepared']=a.prepared/'signals.npy';hashes={k:digest(v) for k,v in paths.items()};assert hashes==c['input_sha256']
    raw=np.load(paths['raw'],mmap_mode='r');y=np.load(paths['labels']);u=np.load(paths['users']);ids=np.flatnonzero(np.isin(u,range(3,9))&~((u==6)&(y==3)));assert len(ids)==7100
    identity={'protocol_sha256':digest(a.protocol),'inputs':hashes,'python':platform.python_version(),'numpy':np.__version__,'code_sha256':{n:digest(Path(__file__).with_name(n)) for n in ['orientation_features.py','orientation_screen.py','run_orientation_readout.py','second_order_core.py','energy_area.py']}}
    dump(a.out/'IDENTITY.json',identity);dump(a.out/'PROTOCOL.json',c)
    assert digest(a.screen/'SUMMARY.json')==c['diagnostic_summary_sha256'];assert digest(a.screen/'AUDIT.json')==c['diagnostic_audit_sha256'];assert json.loads((a.screen/'SUMMARY.json').read_text())['gate_passed']
    if a.cost_only:
        tr=source_roles(y,u,3)['train'][:16];t=time.perf_counter()
        for sid in tr:describe(original_tensors(raw[int(sid)]),int(sid),c['seed'])
        ds=time.perf_counter()-t;rng=np.random.default_rng(24);x=rng.normal(size=(4800,290));t=time.perf_counter();fit_family(x,np.arange(4800)%12,np.arange(4800)%4,c['lambdas']);rs=time.perf_counter()-t
        estimate=2*(7100/16*ds+24*rs);out=dict(descriptor_seconds16=ds,synthetic_fit_seconds=rs,estimate_with_2x_margin_seconds=estimate,budget_seconds=c['budget_seconds'],passed=estimate<c['budget_seconds'],no_validation_or_diagnostic_scores=True);dump(a.out/'COST.json',out);print(json.dumps(out),flush=True);return
    assert json.loads(a.cost.read_text())['passed'];assert json.loads((a.cost.parent/'IDENTITY.json').read_text())['protocol_sha256']==identity['protocol_sha256']
    assert digest(a.reference/'FILES.json')==c['reference_manifest_sha256'];ref=json.loads((a.reference/'FILES.json').read_text());start=time.perf_counter();tensors=[];desc={k:[] for k in c['arms']}
    for j,sid in enumerate(ids):
        z=original_tensors(raw[int(sid)]);tensors.append(z);features=describe(z,int(sid),c['seed'])
        for k in desc:desc[k].append(features[k])
        if (j+1)%1000==0:print(json.dumps({'extracted':j+1}),flush=True)
    desc={k:np.array(v) for k,v in desc.items()};np.savez_compressed(a.out/'descriptors.npz',ids=ids,tensors=np.array(tensors),**desc);del tensors
    lookup=np.full(len(y),-1);lookup[ids]=np.arange(len(ids));scores={k:[] for k in c['arms']+['cnn_readout','tail_cnn']};rows=[]
    for h in c['source_users']:
        d=a.out/f'held{h}';d.mkdir();old=a.reference/f'held{h}';parts=source_roles(y,u,h);split=json.loads((old/'split.json').read_text())
        for n in ['features.npz','split.json','predictions.npz']:assert digest(old/n)==ref[f'held{h}/{n}']
        for k,ii in parts.items():np.testing.assert_array_equal(ii,split[k])
        with np.load(old/'features.npz') as z:hidden=z['hidden'];np.testing.assert_array_equal(z['ids'],ids)
        tr,va,te=[parts[k] for k in ['train','val','diagnostic']];it,iv,ie=[lookup[ii] for ii in [tr,va,te]];selection={};models={};pred={'ids':te,'y':y[te],'validation_ids':va,'validation_y':y[va]}
        for arm in c['arms']:
            x=np.concatenate([hidden,desc[arm]],1);assert x.shape==(7100,290);mean,scale,bias,coefs=fit_family(x[it],y[tr],u[tr],c['lambdas']);candidates=[]
            for lam,coef in zip(c['lambdas'],coefs):
                vp=score(x[iv],mean,scale,bias,coef);pred[f'{arm}_validation_{lam}']=vp;candidates.append(dict(lam=lam,macro=macro(y[va],vp.argmax(1)),mse=float(np.square(vp-np.eye(12)[y[va]]).mean())))
            j=min(range(3),key=lambda j:(-candidates[j]['macro'],candidates[j]['mse'],-candidates[j]['lam']));selection[arm]={'selected':j,'candidates':candidates}
            for k,v in [('mean',mean),('scale',scale),('bias',bias),('coef',coefs[j])]:models[arm+'_'+k]=v
            pred[arm]=score(x[ie],mean,scale,bias,coefs[j])
        with np.load(old/'predictions.npz') as z:
            np.testing.assert_array_equal(z['ids'],te);np.testing.assert_array_equal(z['y'],y[te])
            for k in ['cnn_readout','tail_cnn']:pred[k]=z[k]
        ss={k:macro(y[te],pred[k].argmax(1)) for k in scores}
        for k,v in ss.items():scores[k].append(v)
        dump(d/'split.json',{k:ii.tolist() for k,ii in parts.items()});dump(d/'selection.json',selection);np.savez_compressed(d/'models.npz',**models);np.savez_compressed(d/'predictions.npz',**pred)
        row=dict(held=h,scores=ss,hidden_sha256=ref[f'held{h}/features.npz'],reference_predictions_sha256=ref[f'held{h}/predictions.npz'],files={p.name:digest(p) for p in sorted(d.iterdir())});dump(d/'result.json',row);rows.append(row);print(json.dumps({'completed_held':h}),flush=True)
    comparisons={}
    for k,threshold in c['gate']['minimum_mean_gain'].items():
        delta=np.array(scores['coherent'])-scores[k];comparisons[k]=dict(mean_gain=float(delta.mean()),wins=int((delta>1e-12).sum()),minimum_gain=float(delta.min()),passed=bool(delta.mean()>=threshold and (delta>1e-12).sum()>=4 and delta.min()>=-.01))
    summary=dict(state='complete',scores=scores,means={k:float(np.mean(v)) for k,v in scores.items()},comparisons=comparisons,gate_passed=all(v['passed'] for v in comparisons.values()),rows=rows,seconds=time.perf_counter()-start,selected_readouts=24,ridge_fits=72,encoder_training=False)
    dump(a.out/'SUMMARY.json',summary);dump(a.out/'FILES.json',{str(p.relative_to(a.out)):digest(p) for p in sorted(a.out.rglob('*')) if p.is_file()});print(json.dumps({k:summary[k] for k in ['means','gate_passed','seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['data','prepared','reference','screen','protocol','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cost-only',action='store_true');p.add_argument('--cost',type=Path);a=p.parse_args()
    with threadpool_limits(limits=1):main(a)
