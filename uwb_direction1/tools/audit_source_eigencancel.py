"""Independent raw preprocessing/full CNN, source moments and ridge equations."""
import argparse,json,time,math
from pathlib import Path
import numpy as np
import torch
from scipy.signal import hilbert
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from semantic_code_eightfold_prepare import functional_hidden
from energy_area import source_roles
from audit_restored import digest,dump


def input_tensor(raw):
    z=hilbert(np.asarray(raw,dtype=np.float64),axis=-1);z-=z.mean(-2,keepdims=True);d=z[...,1:,:]*z[...,:-1,:].conj();a=(np.sqrt(abs(d))*np.angle(d)).astype(np.float32);a=np.concatenate([a[...,:1,:],a],axis=-2)
    x=np.stack([abs(raw).astype(np.float32),a],axis=2);x=(x-x.mean((-2,-1),keepdims=True))/(x.std((-2,-1),keepdims=True)+1e-6);x=x.astype(np.float16).astype(np.float32).reshape(len(x),3,2,90,63,3).mean(-1).astype(np.float16).astype(np.float32)
    return np.concatenate([x,np.zeros_like(x)],axis=2)


def basis(cov,c,h,v,seed):
    eig,vec=eigh(cov,driver='evd');eig=eig[::-1];vec=vec[:,::-1];rank=int(np.flatnonzero(np.cumsum(np.maximum(eig,0))/np.maximum(eig,0).sum()>=c['energy_fraction'])[0]+1)
    rb,_=np.linalg.qr(np.random.default_rng(np.random.SeedSequence([seed,h,v])).normal(size=(189,rank)))
    return vec[:,:rank],rb


def wait_file(path):
    deadline=time.monotonic()+7200
    while not path.exists():
        if time.monotonic()>deadline:raise TimeoutError(str(path))
        time.sleep(5)


def main(root):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);t=time.perf_counter();c=json.loads((root/'PROTOCOL.json').read_text());checked={};ident=json.loads((root/'IDENTITY.json').read_text());assert digest(root/'PROTOCOL.json')==ident['protocol_sha256']
    for n,h in ident['code_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    data=Path(c['data']);paths={'raw':data/'X_clutterremoved.npy','labels':data/'y_clutterremoved.npy','users':data/'subj_clutterremoved.npy','prepared':Path(c['prepared'])}
    for k,p in paths.items():assert digest(p)==c['input_sha256'][k]
    raw=np.load(paths['raw'],mmap_mode='r');cache=np.load(paths['prepared'],mmap_mode='r');y=np.load(paths['labels']);u=np.load(paths['users']);cfg=c['diagnostic'];covs={}
    for h in c['users']:
        ids=np.flatnonzero((u==h)&(y==11));cov=np.zeros((3,189,189))
        for i in ids:
            x=np.asarray(raw[i],dtype=np.float64);x/=np.maximum(np.sqrt(np.mean(x*x,axis=(1,2),keepdims=True)),1e-12);cov+=np.einsum('vti,vtj->vij',x,x)/(90*len(ids))
        covs[h]=cov
    old=Path(c['readout_reference']);feat=Path(c['feature_reference']);assert digest(old/'FILES.json')==c['readout_manifest_sha256'] and digest(feat/'FILES.json')==c['feature_manifest_sha256'];rm=json.loads((old/'FILES.json').read_text());fm=json.loads((feat/'FILES.json').read_text());scores={k:[] for k in c['arms']};max_hidden=0.;max_score=0.;max_projector=0.;count=0;fits=0;energy_checks=0
    for h in c['users']:
        d=root/f'held{h}';wait_file(d/'result.json');checked.update({str(f.relative_to(root)):digest(f) for f in d.iterdir() if f.is_file()});parts=source_roles(y,u,h);saved=json.loads((d/'split.json').read_text())
        for role,ids in parts.items():np.testing.assert_array_equal(ids,saved[role])
        tr=parts['train'];va=parts['val'];te=parts['diagnostic'];tu=sorted(set(u[tr]));m=np.load(d/'subspace.npz');g=json.loads((d/'SOURCE_GATE.json').read_text());np.testing.assert_array_equal(tu,m['training_users']);np.testing.assert_array_equal(m['fit_ids'],np.concatenate([tr[(u[tr]==k)&(y[tr]==11)] for k in tu]));flags=np.zeros((len(tu),3),bool);seed=c['seed']+1000*h
        for j,k in enumerate(tu):
            cov=np.mean([covs[v] for v in tu if v!=k],0);ii=tr[u[tr]==k];ii=np.concatenate([ii[y[ii]==v][:cfg['examples_per_class']] for v in np.unique(y[ii])]);np.testing.assert_array_equal(ii,g['inner_diagnostic_ids'][str(int(k))]);assert not set(ii)&set(np.r_[va,te])
            for v in range(3):
                b,rb=basis(cov[v],cfg,int(k),v,seed);x=np.asarray(raw[ii,v],dtype=np.float64);den=np.sum(x*x,(1,2));aa=np.sum((x-x@(b@b.T))**2,(1,2))/den;rr=np.sum((x-x@(rb@rb.T))**2,(1,2))/den;active=np.mean([aa[y[ii]==cl].mean() for cl in np.unique(y[ii]) if cl!=11]);empty=aa[y[ii]==11].mean();ra=np.mean([rr[y[ii]==cl].mean() for cl in np.unique(y[ii]) if cl!=11]);re=rr[y[ii]==11].mean();row=next(r for r in g['rows'] if r['source_user']==k and r['view']==v)
                for key,val in [('active_retention',active),('empty_retention',empty),('random_active_retention',ra),('random_empty_retention',re)]:assert abs(row[key]-val)<1e-10
                flags[j,v]=b.shape[1]<=cfg['maximum_rank'] and empty<=cfg['maximum_empty_retention'] and active>=cfg['minimum_active_retention'] and active-empty>=cfg['minimum_selectivity'] and active-empty-ra+re>=cfg['minimum_selectivity_over_random'];assert bool(flags[j,v])==row['passed'];energy_checks+=len(ii)
        mask=flags.sum(0)>=math.ceil(2*len(tu)/3);np.testing.assert_array_equal(mask,m['selected_views']);np.testing.assert_array_equal(mask,g['selected_views']);np.testing.assert_array_equal(flags.sum(0),g['successful_users']);assert g['minimum_successful_users']==math.ceil(2*len(tu)/3);cov=np.mean([covs[k] for k in tu],0);np.testing.assert_allclose(cov,m['cov'],atol=1e-10,rtol=1e-10)
        for v in range(3):
            b,rb=basis(cov[v],cfg,0,v,seed);sb=m[f'v{v}_basis'];assert b.shape==sb.shape;err=float(abs(b@b.T-sb@sb.T).max());max_projector=max(max_projector,err);assert err<1e-9;np.testing.assert_array_equal(rb,m[f'v{v}_random_basis'])
        f=np.load(feat/f'held{h}/features.npz');assert digest(feat/f'held{h}/features.npz')==fm[f'held{h}/features.npz'];ff=np.load(d/'features.npz');np.testing.assert_array_equal(f['ids'],ff['ids']);lookup={int(i):j for j,i in enumerate(ff['ids'])};ix={k:np.array([lookup[int(i)] for i in ids]) for k,ids in parts.items()}
        for n in ['models.npz','predictions.npz','selection.json','split.json']:assert digest(old/f'held{h}'/n)==rm[f'held{h}/{n}']
        ck=Path(c['models'])/f'held{h}/segment_24.pt';assert digest(ck)==json.loads((d/'result.json').read_text())['checkpoint_sha256'];rec=torch.load(ck,map_location='cpu',weights_only=False);state={k.removeprefix('module.'):v for k,v in rec['average']['weights'].items() if k.startswith('module.')};xx={k:[] for k in c['arms'][1:]}
        for st in range(0,len(ff['ids']),c['batch_size']):
            ids=ff['ids'][st:st+c['batch_size']];r=np.asarray(raw[ids],dtype=np.float64);a=r.copy();rr=r.copy()
            for v in range(3):
                b=m[f'v{v}_basis'];a[:,v]-=(a[:,v]@b)@b.T;b=m[f'v{v}_random_basis'];rr[:,v]-=(rr[:,v]@b)@b.T
            allx=input_tensor(a.astype(np.float32));randx=input_tensor(rr.astype(np.float32));orig=np.asarray(cache[ids],dtype=np.float32);orig=np.stack([orig[:,:,0],orig[:,:,2],np.zeros_like(orig[:,:,0]),np.zeros_like(orig[:,:,0])],axis=2);chosen=allx.copy();chosen[:,~mask]=orig[:,~mask];randx[:,~mask]=orig[:,~mask]
            with torch.no_grad():
                for arm,x in [('selective',chosen),('all_views',allx),('random_selected',randx)]:xx[arm].append(functional_hidden(torch.from_numpy(x),state).numpy())
        xx={k:np.concatenate(v).astype(np.float64) for k,v in xx.items()};count+=3*len(ff['ids'])
        for arm,x in xx.items():max_hidden=max(max_hidden,float(abs(x-ff[arm]).max()));np.testing.assert_allclose(x,ff[arm],atol=2e-5,rtol=2e-5)
        model=np.load(d/'models.npz');pred=np.load(d/'predictions.npz');truth=np.load(d/'evaluation.npz');sel=json.loads((d/'selection.json').read_text())
        for role,ids in [('val',va),('diagnostic',te)]:np.testing.assert_array_equal(pred[role+'_ids'],ids);np.testing.assert_array_equal(truth[role+'_y'],y[ids])
        for arm,x in xx.items():
            xt=x[ix['train']];yy=y[tr];uu=u[tr];w=np.array([1/(len(tu)*len(np.unique(yy[uu==k]))*np.sum((uu==k)&(yy==cl))) for k,cl in zip(uu,yy)]);mean=np.average(xt,axis=0,weights=w);scale=np.sqrt(np.average((xt-mean)**2,axis=0,weights=w));scale[scale<=1e-12]=1.;design=np.column_stack([np.ones(len(xt)),(xt-mean)/scale]);pp=[];candidates=[]
            np.testing.assert_allclose(mean,model[arm+'_mean'],atol=2e-5,rtol=2e-5);np.testing.assert_allclose(scale,model[arm+'_scale'],atol=2e-5,rtol=2e-5)
            for j,lam in enumerate(c['lambdas']):
                coef=np.linalg.solve(design.T@(w[:,None]*design)+np.diag(np.r_[0.,np.repeat(lam,128)]),design.T@(w[:,None]*np.eye(12)[yy]));fits+=1;out={role:np.column_stack([np.ones(len(ix[role])),(x[ix[role]]-mean)/scale])@coef for role in ['val','diagnostic']};pp.append(out);v=out['val'];ma=float(np.mean([np.mean(v.argmax(1)[y[va]==k]==k) for k in np.unique(y[va])]));candidates.append(dict(lam=lam,macro=ma,mse=float(np.mean((v-np.eye(12)[y[va]])**2))));prod=pred[arm+'_val_'+str(lam)];max_score=max(max_score,float(abs(v-prod).max()));np.testing.assert_allclose(v,prod,atol=2e-5,rtol=2e-5);np.testing.assert_array_equal(v.argmax(1),prod.argmax(1));stored=((ff[arm][ix['val']].astype(np.float64)-model[arm+'_mean'])/model[arm+'_scale'])@model[arm+'_coefs'][j]+model[arm+'_bias'];np.testing.assert_allclose(stored,prod,atol=1e-12,rtol=1e-12)
            chosen=min(range(3),key=lambda j:(-candidates[j]['macro'],candidates[j]['mse'],-candidates[j]['lam']));assert chosen==sel[arm]['selected']
            for j,can in enumerate(candidates):assert can['macro']==sel[arm]['candidates'][j]['macro'];assert abs(can['mse']-sel[arm]['candidates'][j]['mse'])<1e-7
            for role in ['val','diagnostic']:
                v=pp[chosen][role];prod=pred[role+'_'+arm];max_score=max(max_score,float(abs(v-prod).max()));np.testing.assert_allclose(v,prod,atol=2e-5,rtol=2e-5);np.testing.assert_array_equal(v.argmax(1),prod.argmax(1))
                if role=='diagnostic':scores[arm].append(float(np.mean([np.mean(v.argmax(1)[y[te]==k]==k) for k in np.unique(y[te])])))
        om=np.load(old/f'held{h}/models.npz');os=json.loads((old/f'held{h}/selection.json').read_text())['onehot']['selected'];ot=np.load(old/f'held{h}/predictions.npz');base=((f['hidden'][ix['diagnostic']]-om['mean'])/om['scale'])@om['coefs'][os]+om['bias'];np.testing.assert_allclose(base,ot['onehot']+.5,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(base,pred['diagnostic_ridge'],atol=1e-12,rtol=1e-12);scores['ridge'].append(float(np.mean([np.mean(base.argmax(1)[y[te]==k]==k) for k in np.unique(y[te])])));print(json.dumps(dict(held=h,checked_full_forwards=count,max_hidden=max_hidden,max_score=max_score)),flush=True)
    wait_file(root/'FILES.json');manifest=json.loads((root/'FILES.json').read_text())
    for n,sha in manifest.items():assert digest(root/n)==sha
    for n,sha in checked.items():assert manifest[n]==sha
    s=json.loads((root/'SUMMARY.json').read_text());passed=[]
    for k,v in scores.items():np.testing.assert_allclose(v,s['scores'][k],atol=1e-15,rtol=0);assert abs(np.mean(v)-s['means'][k])<1e-15
    for k,gain in c['gate']['minimum_mean_gain'].items():
        delta=np.array(scores[c['primary']])-scores[k];yes=bool(delta.mean()>=gain and (delta>1e-12).sum()>=c['gate']['minimum_wins'] and delta.min()>=-c['gate']['maximum_paired_decline']);assert yes==s['comparisons'][k]['passed'];passed.append(yes)
    assert all(passed)==s['gate_passed'];out=dict(state='passed',artifact_hashes=len(manifest),source_moments=18,independent_inner_eigensystems=72,independent_final_eigensystems=18,direct_inner_residual_clip_views=energy_checks,functional_forward_clips=count,independent_ridge_fits=fits,max_projector_error=max_projector,max_hidden_error=max_hidden,max_score_error=max_score,all_argmax_equal=True,audit_code_sha256=digest(Path(__file__)),seconds=time.perf_counter()-t);dump(root/'AUDIT.json',out);print(json.dumps(out))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    with threadpool_limits(limits=1):main(p.parse_args().root)
