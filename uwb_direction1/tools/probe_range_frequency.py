"""Source-only fixed range-frequency probes. No target fitting or novelty claim."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.fft import dct
from scipy.linalg import solve
from sklearn.linear_model import Ridge
from audit_restored import digest, dump, quality_mask, metrics

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'research/RANGE_FREQUENCY_PROTOCOL.json'


def projections(seed):
    rng=np.random.default_rng(seed)
    return (rng.normal(size=(90*21,256))/16, rng.normal(size=(90*42,256))/16)


def features(x, matrices):
    assert x.ndim==4 and x.shape[1:]==(3,90,63)
    z=dct(x,type=2,norm='ortho',axis=-1)
    lo=z[...,:21].reshape(len(x),3,-1)@matrices[0]
    hi=z[...,21:].reshape(len(x),3,-1)@matrices[1]
    return {'low':lo,'high':hi,'mixed':np.concatenate([lo[...,:128],hi[...,128:]],axis=-1)}


def weights(y,u):
    cells=np.stack([u,y],axis=1)
    _,inv,count=np.unique(cells,axis=0,return_inverse=True,return_counts=True)
    w=1./count[inv]
    # Equal user contribution, then equal present-class contribution within user.
    for h in np.unique(u):w[u==h]/=len(np.unique(y[u==h]))
    return w*(len(w)/w.sum())


def fit(x,y,u,alpha):
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-8)*np.sqrt(x.shape[1])
    a=(x-mean)/scale;w=weights(y,u);a=np.column_stack([a,np.ones(len(a))]);target=np.eye(12)[y]
    gram=a.T@(w[:,None]*a);penalty=np.eye(a.shape[1])*alpha;penalty[-1,-1]=0
    coef=solve(gram+penalty,a.T@(w[:,None]*target),assume_a='pos')
    return {'mean':mean,'scale':scale,'coef':coef}


def logits(x,m):
    return ((x-m['mean'])/m['scale'])@m['coef'][:-1]+m['coef'][-1]


def checks():
    rng=np.random.default_rng(419);x=rng.normal(size=(4,3,90,63));p=projections(20261003)
    basis=np.sqrt(2/63)*np.cos(np.pi*(np.arange(63)+.5)*np.arange(63)[:,None]/63);basis[0]/=np.sqrt(2)
    got=features(x,p);z=x@basis.T
    refs={'low':z[...,:21].reshape(4,3,-1)@p[0], 'high':z[...,21:].reshape(4,3,-1)@p[1]}
    refs['mixed']=np.concatenate([refs['low'][...,:128],refs['high'][...,128:]],axis=-1)
    error=max(float(np.max(np.abs(got[k]-refs[k]))) for k in refs)
    for k in refs:np.testing.assert_allclose(got[k],refs[k],atol=2e-12,rtol=1e-11)
    a=rng.normal(size=(100,20));y=np.arange(100)%12;u=np.arange(100)%5
    m=fit(a,y,u,1.)
    ref=Ridge(alpha=1.,fit_intercept=True,solver='cholesky').fit((a-m['mean'])/m['scale'],np.eye(12)[y],sample_weight=weights(y,u))
    err=float(np.max(np.abs(logits(a,m)-ref.predict((a-m['mean'])/m['scale']))))
    assert err<1e-11
    return {'independent_cosine_projection_max_error':error,'sklearn_ridge_max_error':err}


def load_source(data,prepared):
    paths=[data/'cache/y_clutterremoved.npy',data/'cache/subj_clutterremoved.npy',prepared/'signals.npy']
    expected=['d77b10f0fbda231246e71dfcf0b2f852863af48f80f159aed90f5f9b7c61a8bd','5ab32ab59e5b41c05d301b76949f159ddb1e9423a749deb760585a5f19cf9480','1f689bba7f10bb37faf07a557c52653843e81b5907f04d17fa92dbf4dbae7c5f']
    for p,h in zip(paths,expected):assert digest(p)==h
    y,u=np.load(paths[0]),np.load(paths[1]);ids=np.flatnonzero(quality_mask(y,u)&np.isin(u,[3,4,5,6,7,8]))
    assert len(ids)==7100 and not np.isin(u[ids],[1,2]).any()
    return np.load(paths[2],mmap_mode='r'),ids,y[ids],u[ids]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['check','cost','run'])
    ap.add_argument('--data',type=Path);ap.add_argument('--prepared',type=Path);ap.add_argument('--out',type=Path)
    ap.add_argument('--cost',type=Path)
    a=ap.parse_args();c=json.loads(PROTOCOL.read_text())
    if a.mode=='check':print(json.dumps(checks()));return
    if a.out.exists():raise ValueError('Output exists; preserve it and use a new path')
    cache,ids,y,u=load_source(a.data,a.prepared);p=projections(c['projection_seed'])
    if a.mode=='cost':
        x=np.asarray(cache[ids[:64],:,0],dtype=np.float64);features(x[:2],p)
        tick=time.perf_counter();features(x,p);feature_seconds=time.perf_counter()-tick
        rng=np.random.default_rng(21);z=rng.normal(size=(6000,256));sy=np.arange(6000)%12;su=np.repeat(np.arange(5),1200)
        tick=time.perf_counter();fit(z,sy,su,c['alpha']);fit_seconds=time.perf_counter()-tick
        estimate=feature_seconds/64*len(ids)+fit_seconds*54
        dump(a.out,{'source_script_sha256':digest(__file__),'protocol_sha256':digest(PROTOCOL),'feature64_seconds':feature_seconds,'synthetic_fit6000_seconds':fit_seconds,'estimated_seconds':estimate,'budget_seconds':c['cost_gate_seconds'],'passed':estimate<=c['cost_gate_seconds'],'real_label_scoring':False})
        print(a.out.read_text());return
    cost=json.loads(a.cost.read_text());assert cost['passed'] and cost['source_script_sha256']==digest(__file__) and cost['protocol_sha256']==digest(PROTOCOL)
    a.out.mkdir(parents=True);dump(a.out/'protocol.json',c);dump(a.out/'cost.json',cost)
    started=time.perf_counter();fs={k:[] for k in c['arms']}
    for j in range(0,len(ids),32):
        batch=features(np.asarray(cache[ids[j:j+32],:,0],dtype=np.float64),p)
        for k in fs:fs[k].append(batch[k])
    fs={k:np.concatenate(v) for k,v in fs.items()}
    np.savez_compressed(a.out/'features.npz',ids=ids,y=y,u=u,**fs)
    dump(a.out/'SOURCE_IDS.json',ids.tolist());rows=[];audits=[]
    for v,name in enumerate(c['views']):
        for arm in c['arms']:
            x=fs[arm][:,v]
            for h in c['source_users']:
                train=np.flatnonzero(u!=h);test=np.flatnonzero(u==h)
                tick=time.perf_counter();m=fit(x[train],y[train],u[train],c['alpha'])
                predlog=logits(x[test],m);pred=predlog.argmax(1)
                d=a.out/name/arm/f'user{h}';d.mkdir(parents=True)
                np.savez_compressed(d/'model.npz',**m)
                np.savez_compressed(d/'predictions.npz',ids=ids[test],y=y[test],logits=predlog,pred=pred)
                dump(d/'split.json',{'train':ids[train].tolist(),'diagnostic':ids[test].tolist()})
                score=metrics(y[test],pred)
                # Independent sklearn centered fit checks unpenalized intercept and weighting.
                ref=Ridge(alpha=c['alpha'],fit_intercept=True,solver='cholesky').fit((x[train]-m['mean'])/m['scale'],np.eye(12)[y[train]],sample_weight=weights(y[train],u[train]))
                ref_log=ref.predict((x[test]-m['mean'])/m['scale']);err=float(np.max(np.abs(ref_log-predlog)))
                np.testing.assert_allclose(ref_log,predlog,rtol=1e-9,atol=1e-10)
                assert np.array_equal(ref_log.argmax(1),pred)
                # Reload every model, prediction, split, and recompute predictions/IDs.
                saved=dict(np.load(d/'model.npz'));np.testing.assert_array_equal(logits(x[test],saved),predlog)
                q=np.load(d/'predictions.npz');assert np.array_equal(q['ids'],ids[test]) and np.array_equal(q['y'],y[test])
                assert json.loads((d/'split.json').read_text())=={'train':ids[train].tolist(),'diagnostic':ids[test].tolist()}
                hashes={f.name:digest(f) for f in sorted(d.iterdir())}
                row={'view':name,'arm':arm,'held':h,'macro_accuracy':score['macro_accuracy'],'metrics':score,'seconds_including_sklearn_audit':time.perf_counter()-tick,'sha256':hashes}
                dump(d/'result.json',row);rows.append(row);audits.append({'view':name,'arm':arm,'held':h,'max_logit_error':err})
    summary={}
    for name in c['views']:
        s={k:np.array([r['macro_accuracy'] for r in rows if r['view']==name and r['arm']==k]) for k in c['arms']}
        delta=s['mixed']-s['low'];g=c['information_gate'];cg=c['complementarity_gate']
        info=bool(s['high'].mean()>=g['high_mean_min'] and (s['high']>=g['high_at_least_four_users_min']).sum()>=4)
        complement=bool(delta.mean()>=cg['mixed_minus_low_mean_min'] and (delta>0).sum()>=cg['mixed_beats_low_users_min'] and s['mixed'].min()-s['low'].min()>=cg['mixed_worst_minus_low_worst_min'])
        summary[name]={'scores':{k:z.tolist() for k,z in s.items()},'means':{k:float(z.mean()) for k,z in s.items()},'mixed_minus_low':delta.tolist(),'information_gate_passed':info,'complementarity_gate_passed':complement,'continue_gate_passed':info and complement}
    # Independent explicit cosine feature audit on real source first16.
    xx=np.asarray(cache[ids[:16],:,0],dtype=np.float64)
    basis=np.sqrt(2/63)*np.cos(np.pi*(np.arange(63)+.5)*np.arange(63)[:,None]/63);basis[0]/=np.sqrt(2)
    zz=xx@basis.T;refs={'low':zz[...,:21].reshape(16,3,-1)@p[0],'high':zz[...,21:].reshape(16,3,-1)@p[1]}
    refs['mixed']=np.concatenate([refs['low'][...,:128],refs['high'][...,128:]],axis=-1)
    realerr=max(float(np.max(np.abs(refs[k]-fs[k][:16]))) for k in refs)
    for k in refs:np.testing.assert_allclose(refs[k],fs[k][:16],rtol=1e-10,atol=1e-10)
    dump(a.out/'AUDIT.json',{'state':'passed','independent_sklearn_models':len(audits),'models_reloaded_and_predictions_replayed':54,'model_prediction_split_hashes':162,'source_ids_exclude_users12':True,'real_first16_independent_feature_max_error':realerr,'synthetic_checks':checks(),'rows':audits,'limitation':'Feature formula independently checked on16 real clips, not all7100; ridge fits all54 independently checked'})
    dump(a.out/'SUMMARY.json',{'state':'complete_source_only_diagnostic','source_users':c['source_users'],'records':len(ids),'views':summary,'models':54,'seconds':time.perf_counter()-started,'source_script_sha256':digest(__file__),'protocol_sha256':digest(PROTOCOL),'features_sha256':digest(a.out/'features.npz'),'audit_sha256':digest(a.out/'AUDIT.json'),'rows':rows,'novelty_or_full8_success':False})
    print(json.dumps({'state':'complete','views':summary,'seconds':time.perf_counter()-started}),flush=True)

if __name__=='__main__':main()
