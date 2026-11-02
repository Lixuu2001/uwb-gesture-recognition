"""Rebuild uncentered moments and direct residuals; no production projection helper."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump


def main(root):
    t=time.perf_counter();c=json.loads((root/'PROTOCOL.json').read_text());identity=json.loads((root/'IDENTITY.json').read_text());manifest=json.loads((root/'FILES.json').read_text());assert digest(root/'PROTOCOL.json')==identity['protocol_sha256']
    for n,h in manifest.items():assert digest(root/n)==h
    for n,h in identity['code_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    data=Path(c['data']);paths={k:data/n for k,n in [('raw','X_clutterremoved.npy'),('labels','y_clutterremoved.npy'),('users','subj_clutterremoved.npy')]}
    for k,p in paths.items():assert digest(p)==c['input_sha256'][k]
    raw=np.load(paths['raw'],mmap_mode='r');y=np.load(paths['labels']);u=np.load(paths['users']);saved=np.load(root/'source_moments.npz');covs={};max_cov=0.;max_energy=0.;max_projector=0.;flags=np.zeros((6,3),bool);s=json.loads((root/'SUMMARY.json').read_text());count=0
    for user in c['users']:
        ids=np.flatnonzero((u==user)&(y==11));np.testing.assert_array_equal(ids,saved[f'user{user}_empty_ids']);cov=np.zeros((3,189,189))
        for i in ids:
            x=np.asarray(raw[i],dtype=np.float64);norm=np.sqrt(np.sum(x*x,axis=(1,2))/(90*189));x=x/np.maximum(norm[:,None,None],1e-12)
            cov+=np.einsum('vti,vtj->vij',x,x)/(90*len(ids))
        max_cov=max(max_cov,float(abs(cov-saved[f'user{user}_cov']).max()));np.testing.assert_allclose(cov,saved[f'user{user}_cov'],atol=1e-11,rtol=1e-11);covs[user]=cov
    for hi,h in enumerate(c['users']):
        d=root/f'source{h}';m=np.load(d/'model.npz');e=np.load(d/'energies.npz');rows=json.loads((d/'result.json').read_text());fit_users=[k for k in c['users'] if k!=h];fit_ids=np.concatenate([np.flatnonzero((u==k)&(y==11)) for k in fit_users]);np.testing.assert_array_equal(m['fit_users'],fit_users);np.testing.assert_array_equal(m['fit_ids'],fit_ids)
        ids=np.concatenate([np.flatnonzero((u==h)&(y==k))[:c['examples_per_class']] for k in sorted(set(y[u==h])) if not(h==6 and k==3)]);np.testing.assert_array_equal(ids,e['ids']);assert not set(ids)&set(fit_ids);assert not np.isin(u[np.r_[ids,fit_ids]],[1,2]).any();cov=np.mean([covs[k] for k in fit_users],axis=0);np.testing.assert_allclose(cov,m['cov'],atol=1e-11,rtol=1e-11)
        for v in range(3):
            eig,vec=eigh(cov[v],driver='evd');order=np.argsort(eig)[::-1];eig=eig[order];vec=vec[:,order];rank=np.flatnonzero(np.cumsum(np.maximum(eig,0))/np.maximum(eig,0).sum()>=c['energy_fraction'])[0]+1;assert rank==m[f'v{v}_basis'].shape[1]
            basis=vec[:,:rank];p=basis@basis.T;stored=m[f'v{v}_basis']@m[f'v{v}_basis'].T;max_projector=max(max_projector,float(abs(p-stored).max()));np.testing.assert_allclose(p,stored,atol=1e-9,rtol=1e-9)
            rb,_=np.linalg.qr(np.random.default_rng(np.random.SeedSequence([c['seed'],h,v])).normal(size=(189,rank)));np.testing.assert_array_equal(rb,m[f'v{v}_random_basis']);rp=rb@rb.T
            for projector in [p,rp]:np.testing.assert_allclose(projector@projector,projector,atol=1e-12,rtol=1e-12)
            xx=np.asarray(raw[ids,v],dtype=np.float64);den=np.sum(xx*xx,axis=(1,2));actual=np.sum((xx-xx@p)**2,axis=(1,2))/den;random=np.sum((xx-xx@rp)**2,axis=(1,2))/den
            for value,key in [(actual,'retained'),(random,'random_retained')]:max_energy=max(max_energy,float(abs(value-e[key][:,v]).max()));np.testing.assert_allclose(value,e[key][:,v],atol=1e-11,rtol=1e-11)
            means={str(int(k)):float(actual[y[ids]==k].mean()) for k in np.unique(y[ids])};active=float(np.mean([val for key,val in means.items() if key!='11']));empty=means['11'];ra=float(np.mean([random[y[ids]==k].mean() for k in np.unique(y[ids]) if k!=11]));re=float(random[y[ids]==11].mean())
            for key,value in [('active_retention',active),('empty_retention',empty),('random_active_retention',ra),('random_empty_retention',re)]:assert abs(value-rows[v][key])<1e-11
            flags[hi,v]=rank<=c['maximum_rank'] and empty<=c['maximum_empty_retention'] and active>=c['minimum_active_retention'] and active-empty>=c['minimum_selectivity'] and active-empty-ra+re>=c['minimum_selectivity_over_random'];assert bool(flags[hi,v])==rows[v]['passed'];count+=len(ids)
    for v in range(3):assert int(flags[:,v].sum())==s['views'][v]['successful_users'];assert bool(flags[:,v].sum()>=c['minimum_successful_users'])==s['views'][v]['passed']
    assert bool((flags.sum(0)>=c['minimum_successful_users']).any())==s['gate_passed']
    out=dict(state='passed',artifact_hashes=len(manifest),independent_source_moments=18,independent_eigensystems=18,direct_residual_clip_view_checks=count,max_covariance_error=max_cov,max_projector_error=max_projector,max_retention_error=max_energy,outer_users_1_2_used=False,seconds=time.perf_counter()-t);dump(root/'AUDIT.json',out);print(json.dumps(out))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    with threadpool_limits(limits=1):main(p.parse_args().root)
