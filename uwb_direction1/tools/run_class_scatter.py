import argparse,json,time
from pathlib import Path
import numpy as np
import sklearn
from scipy.stats import multivariate_normal
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump
from class_scatter import fit,predict
from second_order_core import score,macro
from radar_exchange import decision


def independent(x,y,means,covs,info,c):
    # Independently rederive installed OAS approximation from raw centered rows.
    vv=[]
    for k in range(12):
        z=x[y==k];mu=z.mean(0);z=z-mu;n,p=z.shape;emp=z.T@z/n;t=np.trace(emp);t2=np.sum(emp*emp);den=(n+1)*(t2-t*t/p);sh=1. if den==0 else min((t2+t*t)/den,1.);v=(1-sh)*emp+sh*(t/p)*np.eye(p)+c['numeric_jitter']*np.eye(p)
        np.testing.assert_allclose(mu,means[k],atol=1e-12);np.testing.assert_allclose(sh,info['shrinkage'][k],atol=1e-12);np.testing.assert_allclose(v,covs['qda'][k],atol=1e-12);vv.append(v)
    vv=np.array(vv);np.testing.assert_allclose(covs['pooled'],np.repeat(vv.mean(0)[None],12,axis=0),atol=1e-12);np.testing.assert_allclose(covs['diagonal'],np.array([np.diag(np.diag(v)) for v in vv]),atol=1e-12);np.testing.assert_allclose(covs['permuted'],vv[np.random.default_rng(c['seed']).permutation(12)],atol=1e-12)


def check_scores(x,means,cov,s):
    ref=np.array([multivariate_normal.logpdf(x,mean=m,cov=v)+x.shape[1]/2*np.log(2*np.pi)-np.log(12) for m,v in zip(means,cov)]).T
    err=float(abs(ref-s).max());np.testing.assert_allclose(ref,s,atol=1e-6,rtol=1e-8);np.testing.assert_array_equal(ref.argmax(1),s.argmax(1));return err


def main(a):
    c=json.loads(a.protocol.read_text());assert sklearn.__version__=='1.8.0';assert not a.out.exists();a.out.mkdir();dump(a.out/'PROTOCOL.json',c)
    identity=dict(protocol_sha256=digest(a.protocol),numpy=np.__version__,sklearn=sklearn.__version__,code_sha256={n:digest(Path(__file__).with_name(n)) for n in ['class_scatter.py','run_class_scatter.py','second_order_core.py','radar_exchange.py','audit_restored.py']});dump(a.out/'IDENTITY.json',identity)
    if a.cost_only:
        rng=np.random.default_rng(4);x=rng.normal(size=(4800,128));y=np.arange(4800)%12;t=time.perf_counter();means,covs,info=fit(x,y,c['seed'],c['numeric_jitter']);independent(x,y,means,covs,info,c)
        for v in covs.values():s=predict(x[:1200],means,v);check_scores(x[:1200],means,v,s)
        elapsed=time.perf_counter()-t;estimate=elapsed*6*2*3;r=dict(synthetic_seconds=elapsed,estimate_seconds=estimate,passed=estimate<c['budget_seconds'],protocol_sha256=identity['protocol_sha256'],real_data_read=False);dump(a.out/'COST.json',r);print(json.dumps(r));return
    cost=json.loads(a.cost.read_text());assert cost['passed'] and cost['protocol_sha256']==identity['protocol_sha256'];old=Path(c['readout_reference']);feat=Path(c['feature_reference']);assert digest(old/'FILES.json')==c['readout_manifest_sha256'];assert digest(feat/'FILES.json')==c['feature_manifest_sha256'];rm=json.loads((old/'FILES.json').read_text());fm=json.loads((feat/'FILES.json').read_text())
    # Global labels are opened only for source training indexing; no validation/held values used to fit.
    raw=Path('/workspace/uwb-restored-20260930/cache');yp=raw/'y_clutterremoved.npy';up=raw/'subj_clutterremoved.npy';assert digest(yp)=='d77b10f0fbda231246e71dfcf0b2f852863af48f80f159aed90f5f9b7c61a8bd';assert digest(up)=='5ab32ab59e5b41c05d301b76949f159ddb1e9423a749deb760585a5f19cf9480';y=np.load(yp);u=np.load(up)
    scores={k:[] for k in c['arms']};rows=[];t=time.perf_counter();maxerr=0.
    for h in c['users']:
        d=a.out/f'held{h}';d.mkdir();od=old/f'held{h}';fd=feat/f'held{h}'
        for n in ['models.npz','split.json','selection.json','predictions.npz']:assert digest(od/n)==rm[f'held{h}/{n}']
        assert digest(fd/'features.npz')==fm[f'held{h}/features.npz'];f=np.load(fd/'features.npz');m=np.load(od/'models.npz');sel=json.loads((od/'selection.json').read_text())['onehot'];split=json.loads((od/'split.json').read_text());lookup={int(v):i for i,v in enumerate(f['ids'])};tr=np.array(split['train']);va=np.array(split['val']);te=np.array(split['diagnostic']);assert not(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te));assert len(np.unique(u[tr]))==4 and h not in u[tr]
        def hidden(ids):return f['hidden'][[lookup[int(v)] for v in ids]]
        x=(hidden(tr)-m['mean'])/m['scale'];means,covs,info=fit(x,y[tr],c['seed'],c['numeric_jitter']);independent(x,y[tr],means,covs,info,c);info['class_user_counts']={str(k):{str(v):int(((y[tr]==k)&(u[tr]==v)).sum()) for v in np.unique(u[tr])} for k in range(12)}
        np.savez_compressed(d/'models.npz',mean=m['mean'],scale=m['scale'],class_means=means,**covs);dump(d/'fit.json',info);dump(d/'split.json',split);dump(d/'selection.json',dict(new_selection=False,inherited_ridge=sel));pred={}
        for role,ids in [('val',va),('diagnostic',te)]:
            hh=hidden(ids);z=(hh-m['mean'])/m['scale'];pred[role+'_ids']=ids;pred[role+'_ridge']=score(hh,m['mean'],m['scale'],m['bias'],m['coefs'][sel['selected']])
            for k,cov in covs.items():s=predict(z,means,cov);maxerr=max(maxerr,check_scores(z,means,cov,s));pred[role+'_'+k]=s
        np.savez_compressed(d/'predictions.npz',**pred)
        with np.load(od/'predictions.npz') as truth:
            np.testing.assert_array_equal(va,truth['validation_ids']);np.testing.assert_array_equal(te,truth['ids']);np.testing.assert_array_equal(y[te],truth['y']);np.testing.assert_allclose(pred['diagnostic_ridge'],truth['onehot']+.5,atol=1e-12,rtol=1e-12);np.testing.assert_array_equal(pred['diagnostic_ridge'].argmax(1),truth['onehot'].argmax(1));ss={k:macro(truth['y'],pred['diagnostic_'+k].argmax(1)) for k in c['arms']};np.savez_compressed(d/'evaluation.npz',val_y=truth['validation_y'],diagnostic_y=truth['y'])
        for k,v in ss.items():scores[k].append(v)
        row=dict(held=h,scores=ss);rows.append(row);dump(d/'result.json',row);print(json.dumps(row),flush=True)
    comp=decision(scores,c);summary=dict(state='complete',means={k:float(np.mean(v)) for k,v in scores.items()},scores=scores,comparisons=comp,gate_passed=all(v['passed'] for v in comp.values()),rows=rows,seconds=time.perf_counter()-t,new_encoder_training=False,gaussian_fits=72,independent_gaussian_score_systems=576,max_scipy_logpdf_difference=maxerr);dump(a.out/'SUMMARY.json',summary);dump(a.out/'FILES.json',{str(p.relative_to(a.out)):digest(p) for p in sorted(a.out.rglob('*')) if p.is_file()});print(json.dumps({k:summary[k] for k in ['means','gate_passed','seconds']}))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['protocol','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cost-only',action='store_true');p.add_argument('--cost',type=Path);a=p.parse_args()
    with threadpool_limits(limits=1):main(a)
