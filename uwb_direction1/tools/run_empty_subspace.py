import argparse,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump
from empty_subspace import moments,projectors,fractions,evaluate


def main(a):
    c=json.loads(a.protocol.read_text());assert not a.out.exists();a.out.mkdir();dump(a.out/'PROTOCOL.json',c)
    identity=dict(protocol_sha256=digest(a.protocol),code_sha256={n:digest(Path(__file__).with_name(n)) for n in ['empty_subspace.py','run_empty_subspace.py','audit_restored.py']});dump(a.out/'IDENTITY.json',identity)
    if a.cost_only:
        rng=np.random.default_rng(19);x=rng.normal(size=(20,3,90,189));ids=np.arange(20);t=time.perf_counter();cov=moments(x,ids,c['batch_size']);model=projectors(cov,3,c);fractions(x,ids,model,c['batch_size']);elapsed=time.perf_counter()-t
        estimate=elapsed*(600+6*240)/20*3;out=dict(synthetic20_seconds=elapsed,estimate_seconds=estimate,passed=estimate<c['budget_seconds'],protocol_sha256=identity['protocol_sha256'],real_data_read=False);dump(a.out/'COST.json',out);print(json.dumps(out));return
    cost=json.loads(a.cost.read_text());assert cost['passed'] and cost['protocol_sha256']==identity['protocol_sha256']
    data=Path(c['data']);paths={k:data/n for k,n in [('raw','X_clutterremoved.npy'),('labels','y_clutterremoved.npy'),('users','subj_clutterremoved.npy')]}
    assert {k:digest(p) for k,p in paths.items()}==c['input_sha256']
    raw=np.load(paths['raw'],mmap_mode='r');y=np.load(paths['labels']);u=np.load(paths['users']);valid=np.isin(u,c['users'])&~((u==6)&(y==3));source=np.flatnonzero(valid);assert len(source)==7100
    empty={h:source[(u[source]==h)&(y[source]==11)] for h in c['users']};chosen={}
    for h in c['users']:
        ids=source[u[source]==h];chosen[h]=np.concatenate([ids[y[ids]==k][:c['examples_per_class']] for k in np.unique(y[ids])])
        assert len(chosen[h])==c['examples_per_class']*len(np.unique(y[ids]))
    t=time.perf_counter();user_cov={h:moments(raw,empty[h],c['batch_size']) for h in c['users']};np.savez_compressed(a.out/'source_moments.npz',**{f'user{h}_cov':v for h,v in user_cov.items()},**{f'user{h}_empty_ids':v for h,v in empty.items()})
    rows=[]
    for h in c['users']:
        d=a.out/f'source{h}';d.mkdir();fit_users=[k for k in c['users'] if k!=h];fit_ids=np.concatenate([empty[k] for k in fit_users]);ids=chosen[h]
        assert not set(fit_ids)&set(ids) and not np.isin(u[fit_ids],[1,2,h]).any() and not np.isin(u[ids],[1,2]).any()
        cov=np.mean([user_cov[k] for k in fit_users],axis=0);model=projectors(cov,h,c);np.savez_compressed(d/'model.npz',cov=cov,**model,fit_ids=fit_ids,fit_users=fit_users)
        actual,random=fractions(raw,ids,model,c['batch_size']);np.savez_compressed(d/'energies.npz',ids=ids,retained=actual,random_retained=random)
        # Labels summarize this source-only diagnostic after energy computation.
        out=evaluate(y[ids],actual,random,model,c);dump(d/'result.json',out);rows.extend([dict(source_user=h,**r) for r in out]);print(json.dumps(dict(source_user=h,ranks=[r['rank'] for r in out],passes=[r['passed'] for r in out])),flush=True)
    views=[]
    for v in range(3):
        rr=[r for r in rows if r['view']==v];success=sum(r['passed'] for r in rr);views.append(dict(view=v,successful_users=success,passed=success>=c['minimum_successful_users'],mean_empty_retention=float(np.mean([r['empty_retention'] for r in rr])),mean_active_retention=float(np.mean([r['active_retention'] for r in rr])),ranks=[r['rank'] for r in rr]))
    summary=dict(state='complete',rows=rows,views=views,gate_passed=any(v['passed'] for v in views),seconds=time.perf_counter()-t,source_empty_clips=600,unique_diagnostic_clips=sum(len(v) for v in chosen.values()),classification_run=False,outer_users_1_2_used=False)
    dump(a.out/'SUMMARY.json',summary);dump(a.out/'FILES.json',{str(p.relative_to(a.out)):digest(p) for p in sorted(a.out.rglob('*')) if p.is_file()});print(json.dumps({k:summary[k] for k in ['views','gate_passed','seconds']}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['protocol','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cost-only',action='store_true');p.add_argument('--cost',type=Path)
    with threadpool_limits(limits=1):main(p.parse_args())
