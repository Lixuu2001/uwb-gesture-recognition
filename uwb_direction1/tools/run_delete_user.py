import argparse,json,time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump
from delete_user import subsets,aggregate
from second_order_core import fit_family,score,weights,macro
from radar_exchange import decision


def main(a):
    c=json.loads(a.protocol.read_text());assert not a.out.exists();a.out.mkdir();dump(a.out/'PROTOCOL.json',c);identity=dict(protocol_sha256=digest(a.protocol),code_sha256={n:digest(Path(__file__).with_name(n)) for n in ['delete_user.py','run_delete_user.py','second_order_core.py','radar_exchange.py','audit_restored.py']});dump(a.out/'IDENTITY.json',identity)
    if a.cost_only:
        rng=np.random.default_rng(17);x=rng.normal(size=(4800,128));y=np.tile(np.arange(12).repeat(100),4);u=np.repeat(np.arange(3,7),1200);ids=np.arange(4800);t=time.perf_counter()
        for ii in subsets(ids,y,u,8,c['seed']).values():fit_family(x[ii],y[ii],u[ii],[.01])
        elapsed=time.perf_counter()-t;r=dict(synthetic_seconds=elapsed,estimate_seconds=elapsed*6*3,passed=elapsed*6*3<c['budget_seconds'],protocol_sha256=identity['protocol_sha256'],real_data_read=False);dump(a.out/'COST.json',r);print(json.dumps(r));return
    cost=json.loads(a.cost.read_text());assert cost['passed'] and cost['protocol_sha256']==identity['protocol_sha256'];old=Path(c['readout_reference']);feat=Path(c['feature_reference']);assert digest(old/'FILES.json')==c['readout_manifest_sha256'];assert digest(feat/'FILES.json')==c['feature_manifest_sha256'];rm=json.loads((old/'FILES.json').read_text());fm=json.loads((feat/'FILES.json').read_text())
    raw=Path('/workspace/uwb-restored-20260930/cache');yp=raw/'y_clutterremoved.npy';up=raw/'subj_clutterremoved.npy';assert digest(yp)=='d77b10f0fbda231246e71dfcf0b2f852863af48f80f159aed90f5f9b7c61a8bd';assert digest(up)=='5ab32ab59e5b41c05d301b76949f159ddb1e9423a749deb760585a5f19cf9480';y=np.load(yp);u=np.load(up);scores={k:[] for k in c['arms']};rows=[];t=time.perf_counter();maxerr=0.
    for h in c['users']:
        d=a.out/f'held{h}';d.mkdir();od=old/f'held{h}';fd=feat/f'held{h}'
        for n in ['models.npz','predictions.npz','selection.json','split.json']:assert digest(od/n)==rm[f'held{h}/{n}']
        assert digest(fd/'features.npz')==fm[f'held{h}/features.npz'];f=np.load(fd/'features.npz');m=np.load(od/'models.npz');sel=json.loads((od/'selection.json').read_text())['onehot'];split=json.loads((od/'split.json').read_text());tr=np.array(split['train']);va=np.array(split['val']);te=np.array(split['diagnostic']);lookup={int(v):i for i,v in enumerate(f['ids'])}
        def hidden(ids):return f['hidden'][[lookup[int(v)] for v in ids]]
        masks=subsets(tr,y[tr],u[tr],h,c['seed']);lam=sel['candidates'][sel['selected']]['lambda'] if 'lambda' in sel['candidates'][sel['selected']] else sel['candidates'][sel['selected']]['lam'];pred={};saved={};params={};errors=[]
        for role,ids in [('val',va),('diagnostic',te)]:pred[role+'_ids']=ids;pred[role+'_ridge']=score(hidden(ids),m['mean'],m['scale'],m['bias'],m['coefs'][sel['selected']])
        per_role={r:{} for r in ['val','diagnostic']}
        for name,ii in masks.items():
            assert set(ii)<=set(tr) and not(set(ii)&set(va) or set(ii)&set(te));x=hidden(ii);mean,scale,bias,coefs=fit_family(x,y[ii],u[ii],[lam]);coef=coefs[0];ref=Ridge(alpha=lam*len(ii),solver='cholesky').fit((x-mean)/scale,np.eye(12)[y[ii]],sample_weight=weights(y[ii],u[ii])*len(ii))
            saved[name]=ii
            for key,v in [('mean',mean),('scale',scale),('bias',bias),('coef',coef)]:params[name+'_'+key]=v
            for role,ids in [('val',va),('diagnostic',te)]:
                hh=hidden(ids);v=score(hh,mean,scale,bias,coef);rr=ref.predict((hh-mean)/scale);err=float(abs(rr-v).max());maxerr=max(maxerr,err);np.testing.assert_allclose(rr,v,atol=1e-9,rtol=1e-9);np.testing.assert_array_equal(rr.argmax(1),v.argmax(1));per_role[role][name]=v;pred[role+'_'+name]=v;errors.append(dict(model=name,role=role,max_error=err))
        for role in per_role:
            for k,v in aggregate(per_role[role]).items():pred[role+'_'+k]=v
        np.savez_compressed(d/'models.npz',**params);np.savez_compressed(d/'subsets.npz',**saved);np.savez_compressed(d/'predictions.npz',**pred);dump(d/'split.json',split);dump(d/'selection.json',dict(new_selection=False,inherited_onehot=sel,lambda_used=lam));dump(d/'NUMERICAL_AUDIT.json',errors)
        with np.load(od/'predictions.npz') as truth:
            np.testing.assert_array_equal(te,truth['ids']);np.testing.assert_array_equal(va,truth['validation_ids']);np.testing.assert_allclose(pred['diagnostic_ridge'],truth['onehot']+.5,atol=1e-12,rtol=1e-12);np.testing.assert_array_equal(pred['diagnostic_ridge'].argmax(1),truth['onehot'].argmax(1));ss={k:macro(truth['y'],pred['diagnostic_'+k].argmax(1)) for k in c['arms']};np.savez_compressed(d/'evaluation.npz',diagnostic_y=truth['y'],val_y=truth['validation_y'])
        for k,v in ss.items():scores[k].append(v)
        row=dict(held=h,scores=ss);rows.append(row);dump(d/'result.json',row);print(json.dumps(row),flush=True)
    comp=decision(scores,c);summary=dict(state='complete',means={k:float(np.mean(v)) for k,v in scores.items()},scores=scores,comparisons=comp,gate_passed=all(v['passed'] for v in comp.values()),rows=rows,seconds=time.perf_counter()-t,readout_fits=48,independent_sklearn_role_predictions=96,max_sklearn_error=maxerr,encoder_training=False);dump(a.out/'SUMMARY.json',summary);dump(a.out/'FILES.json',{str(p.relative_to(a.out)):digest(p) for p in sorted(a.out.rglob('*')) if p.is_file()});print(json.dumps({k:summary[k] for k in ['means','gate_passed','seconds']}))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['protocol','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cost-only',action='store_true');p.add_argument('--cost',type=Path);a=p.parse_args()
    with threadpool_limits(limits=1):main(a)
