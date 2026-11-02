"""Preregistered six-source small temporal response pooling screen."""
import argparse,json,time,platform
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump
from energy_area import source_roles
from response_pool import bank,calibrate,extract,decision
from second_order_core import fit_family,score,macro


def main(a):
    c=json.loads(a.protocol.read_text());assert not a.out.exists();a.out.mkdir()
    y=np.load(a.data/'cache/y_clutterremoved.npy');u=np.load(a.data/'cache/subj_clutterremoved.npy');cache=np.load(a.prepared/'signals.npy',mmap_mode='r')
    paths={'raw':a.data/'cache/X_clutterremoved.npy','labels':a.data/'cache/y_clutterremoved.npy','users':a.data/'cache/subj_clutterremoved.npy','prepared':a.prepared/'signals.npy'}
    hashes={k:digest(v) for k,v in paths.items()};assert hashes==c['input_sha256']
    identity={'protocol_sha256':digest(a.protocol),'inputs':hashes,'python':platform.python_version(),'numpy':np.__version__,'code_sha256':{n:digest(Path(__file__).with_name(n)) for n in ['response_pool.py','run_response_pool.py','second_order_core.py','energy_area.py','audit_restored.py']}}
    dump(a.out/'IDENTITY.json',identity);dump(a.out/'protocol.json',c)
    kernels,channels=bank(c['bank_seed']);ids=np.flatnonzero(np.isin(u,range(3,9))&~((u==6)&(y==3)));assert len(ids)==7100
    if a.cost_only:
        tr=source_roles(y,u,3)['train'];t=time.perf_counter();bias,chosen=calibrate(cache,tr,kernels,channels);bs=time.perf_counter()-t
        t=time.perf_counter();extract(cache[tr[:64]],tr[:64],kernels,channels,bias,c['shuffle_seed']);fs=time.perf_counter()-t
        rng=np.random.default_rng(75);s=rng.normal(size=(4800,2016));t=time.perf_counter();fit_family(s,np.arange(4800)%12,np.arange(4800)%4,c['lambdas']);rs=time.perf_counter()-t
        estimate=c['cost_margin']*(6*bs+6*7100/64*fs+6*4*rs)
        out=dict(calibration_seconds=bs,features_seconds64=fs,synthetic_ridge_seconds=rs,estimated_seconds=estimate,budget_seconds=c['budget_seconds'],passed=estimate<c['budget_seconds'],no_validation_or_diagnostic_scores=True)
        dump(a.out/'COST.json',out);print(json.dumps(out),flush=True);return
    cost=json.loads(a.cost.read_text());assert cost['passed'];assert json.loads((a.cost.parent/'IDENTITY.json').read_text())['protocol_sha256']==identity['protocol_sha256']
    lookup=np.full(len(y),-1);lookup[ids]=np.arange(len(ids));scores={k:[] for k in c['arms']+['cnn_readout','tail_cnn']};rows=[];started=time.perf_counter()
    assert digest(a.reference/'FILES.json')==c['reference_manifest_sha256']
    reference_manifest=json.loads((a.reference/'FILES.json').read_text())
    for held in c['source_users']:
        d=a.out/f'held{held}';d.mkdir();parts=source_roles(y,u,held);tr=parts['train'];va=parts['val'];te=parts['diagnostic'];it=lookup[tr];iv=lookup[va];ie=lookup[te]
        old=a.reference/f'held{held}'
        for n in ['split.json','predictions.npz']:
            assert digest(old/n)==reference_manifest[f'held{held}/{n}']
        split=json.loads((old/'split.json').read_text())
        for key,ii in parts.items():np.testing.assert_array_equal(ii,split[key])
        biases,chosen=calibrate(cache,tr,kernels,channels,c['bank_seed']);assert set(chosen.flatten())<=set(tr)
        np.savez_compressed(d/'bank.npz',kernels=kernels,channels=channels,biases=biases,calibration_ids=chosen)
        features={k:[] for k in c['arms']}
        for start in range(0,len(ids),128):
            ii=ids[start:start+128];batch=extract(cache[ii],ii,kernels,channels,biases,c['shuffle_seed'])
            for name in features:features[name].append(batch[name].astype(np.float32))
        features={k:np.concatenate(v).astype(np.float64) for k,v in features.items()}
        np.savez_compressed(d/'features.npz',ids=ids,**{k:v.astype(np.float32) for k,v in features.items()})
        sel={};models={};pred={'ids':te,'y':y[te],'validation_ids':va,'validation_y':y[va]}
        for arm,x in features.items():
            assert x.shape==(7100,c['dimensions'][arm]);mean,scale,bias,coefs=fit_family(x[it],y[tr],u[tr],c['lambdas']);candidates=[]
            for lam,coef in zip(c['lambdas'],coefs):
                vp=score(x[iv],mean,scale,bias,coef);pred[f'{arm}_validation_{lam}']=vp
                candidates.append(dict(lam=lam,macro=macro(y[va],vp.argmax(1)),mse=float(np.square(vp-np.eye(12)[y[va]]).mean())))
            choice=min(range(len(candidates)),key=lambda j:(-candidates[j]['macro'],candidates[j]['mse'],-candidates[j]['lam']))
            sel[arm]={'selected':choice,'candidates':candidates,'dimension':x.shape[1]}
            for key,value in [('mean',mean),('scale',scale),('bias',bias),('coef',coefs[choice])]:models[arm+'_'+key]=value
            pred[arm]=score(x[ie],mean,scale,bias,coefs[choice])
        with np.load(old/'predictions.npz') as z:
            np.testing.assert_array_equal(z['ids'],te);np.testing.assert_array_equal(z['y'],y[te])
            for key in ['cnn_readout','tail_cnn']:pred[key]=z[key]
        ss={k:macro(y[te],pred[k].argmax(1)) for k in scores}
        for k,v in ss.items():scores[k].append(v)
        dump(d/'split.json',{k:v.tolist() for k,v in parts.items()});dump(d/'selection.json',sel);np.savez_compressed(d/'models.npz',**models);np.savez_compressed(d/'predictions.npz',**pred)
        row=dict(held=held,scores=ss,reference_predictions_sha256=digest(old/'predictions.npz'),files={p.name:digest(p) for p in sorted(d.iterdir())})
        dump(d/'result.json',row);rows.append(row);print(json.dumps({'completed':held,'seconds':time.perf_counter()-started}),flush=True)
    comparisons=decision(scores,c);summary=dict(state='complete',scores=scores,means={k:float(np.mean(v)) for k,v in scores.items()},comparisons=comparisons,gate_passed=all(z['passed'] for z in comparisons.values()),rows=rows,seconds=time.perf_counter()-started,selected_readouts=24,ridge_fits=72,encoder_training=False)
    dump(a.out/'SUMMARY.json',summary);dump(a.out/'FILES.json',{str(p.relative_to(a.out)):digest(p) for p in sorted(a.out.rglob('*')) if p.is_file()})
    print(json.dumps({k:summary[k] for k in ['state','means','gate_passed','seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['data','prepared','reference','protocol','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cost-only',action='store_true');p.add_argument('--cost',type=Path);a=p.parse_args()
    with threadpool_limits(limits=1):main(a)
