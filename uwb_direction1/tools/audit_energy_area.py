"""Independent raw Chen recursion, functional CNN spot checks, selection and fit audit."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump


def chen(p):
    delta=np.zeros(p.shape[:-2]+(7,));s=np.zeros(p.shape[:-2]+(7,7))
    for t in range(89):
        dx=p[...,t+1,:]-p[...,t,:]
        s+=delta[..., :,None]*dx[...,None,:]+.5*dx[..., :,None]*dx[...,None,:];delta+=dx
    l=s-.5*delta[..., :,None]*delta[...,None,:]
    return l[...,np.triu_indices(7,1)[0],np.triu_indices(7,1)[1]].reshape(len(p),63)


def metric(y,z):
    count=np.bincount(y,minlength=12);hit=np.bincount(y[z.argmax(1)==y],minlength=12)
    return float((hit[count>0]/count[count>0]).mean())


@torch.no_grad()
def functional_hidden(prepared,ids,state):
    arr=np.asarray(prepared[ids],dtype=np.float32);x=torch.from_numpy(np.stack([arr[:,:,0],arr[:,:,2],np.zeros_like(arr[:,:,0]),np.zeros_like(arr[:,:,0])],2)).reshape(-1,4,90,63)
    for stage in range(3):
        b=f'encoder.{stage}.layers.'
        for conv,gn,stride in [('0','1',2),('3','4',1)]:
            x=F.gelu(F.group_norm(F.conv2d(x,state[b+conv+'.weight'],stride=stride,padding=1),4,state[b+gn+'.weight'],state[b+gn+'.bias']))
    x=F.adaptive_avg_pool2d(x,(3,4)).reshape(len(ids),-1)
    z=F.gelu(F.linear(x,state['head.1.weight'],state['head.1.bias']))
    return z.numpy(),F.linear(z,state['head.4.weight'],state['head.4.bias']).numpy()


def main(a):
    start=time.perf_counter();assert not (a.runs/'AUDIT.json').exists();torch.set_num_threads(1)
    c=json.loads((a.runs/'protocol.json').read_text());summary=json.loads((a.runs/'SUMMARY.json').read_text());files=json.loads((a.runs/'FILES.json').read_text())
    for p,sha in files.items():assert digest(a.runs/p)==sha
    data=a.data/'cache';raw=np.load(data/'X_clutterremoved.npy',mmap_mode='r');y=np.load(data/'y_clutterremoved.npy');u=np.load(data/'subj_clutterremoved.npy');cache=np.load(a.prepared/'signals.npy',mmap_mode='r')
    with np.load(a.runs/'descriptors.npz') as z:desc={k:z[k] for k in z.files}
    ids=desc['ids'];valid=np.isin(u,range(3,9))&~((u==6)&(y==3));np.testing.assert_array_equal(ids,np.flatnonzero(valid))
    maxraw=0.
    for startrow in range(0,len(ids),64):
        ii=ids[startrow:startrow+64];x=np.asarray(raw[ii],dtype=np.float64)
        p=np.stack([np.sqrt((x[...,27*j:27*(j+1)]**2).sum(-1)/27) for j in range(7)],-1);scale=np.max(p,axis=(-2,-1),keepdims=True);p=np.divide(p,scale,out=np.zeros_like(p),where=scale>0)
        ref={'base':np.concatenate([np.mean(p,axis=2),np.std(p,axis=2),p[:,:,-1]-p[:,:,0]],-1).reshape(len(ii),63),'signed':chen(p)}
        ref['unsigned']=np.abs(ref['signed']);q=p.copy()
        for n,identity in enumerate(ii):
            rng=np.random.default_rng(np.random.SeedSequence([c['shuffle_seed'],int(identity)]));order=np.r_[0,rng.permutation(np.arange(1,89)),89];q[n]=p[n][:,order]
        ref['shuffled']=chen(q)
        for k,r in ref.items():
            target=desc[k][startrow:startrow+len(ii)];err=float(abs(r-target).max());maxraw=max(maxraw,err);np.testing.assert_allclose(r,target,atol=1e-11,rtol=1e-11)
    lookup=np.full(len(y),-1);lookup[ids]=np.arange(len(ids));rows=[];scores={k:[] for k in summary['scores']}
    for held in range(3,9):
        d=a.runs/f'held{held}';row=json.loads((d/'result.json').read_text());v=held+1 if held<8 else 3
        parts={'train':np.flatnonzero(valid&(u!=held)&(u!=v)),'val':np.flatnonzero(valid&(u==v)),'diagnostic':np.flatnonzero(valid&(u==held))}
        sp=json.loads((d/'split.json').read_text());sel=json.loads((d/'selection.json').read_text())
        for k,ii in parts.items():np.testing.assert_array_equal(ii,sp[k])
        with np.load(d/'features.npz') as z:f={k:z[k] for k in z.files}
        with np.load(d/'predictions.npz') as z:pred={k:z[k] for k in z.files}
        with np.load(d/'models.npz') as z:model={k:z[k] for k in z.files}
        np.testing.assert_array_equal(ids,f['ids']);np.testing.assert_array_equal(parts['diagnostic'],pred['ids']);np.testing.assert_array_equal(parts['val'],pred['validation_ids']);np.testing.assert_array_equal(y[parts['diagnostic']],pred['y']);np.testing.assert_array_equal(y[parts['val']],pred['validation_y'])
        assert digest(row['checkpoint'])==row['checkpoint_sha256'];ck=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
        st={k[7:]:z for k,z in ck['average']['weights'].items() if k.startswith('module.')};spots=np.concatenate([parts[k][:2] for k in parts]);h,logits=functional_hidden(cache,spots,st)
        np.testing.assert_allclose(h,f['hidden'][lookup[spots]],atol=3e-6,rtol=3e-6);np.testing.assert_allclose(logits,f['cnn'][lookup[spots]],atol=3e-6,rtol=3e-6)
        feats={'cnn_readout':f['hidden'],'base':np.c_[f['hidden'],desc['base']]}
        for k in ['signed','unsigned','shuffled']:feats[k]=np.c_[feats['base'],desc[k]]
        tr=parts['train'];va=parts['val'];te=parts['diagnostic'];w=np.zeros(len(tr))
        for user in sorted(set(u[tr])):
            labels=sorted(set(y[tr][u[tr]==user]))
            for lab in labels:
                m=(u[tr]==user)&(y[tr]==lab);w[m]=1/(4*len(labels)*sum(m))
        errors=[]
        for arm in c['arms']:
            xx=feats[arm];mean=np.average(xx[lookup[tr]],axis=0,weights=w);sd=np.sqrt(np.average((xx[lookup[tr]]-mean)**2,axis=0,weights=w));sd[sd<=1e-12]=1
            np.testing.assert_allclose(mean,model[arm+'_mean'],atol=1e-12);np.testing.assert_allclose(sd,model[arm+'_scale'],atol=1e-12)
            candidates=[dict(lam=lam,macro=metric(y[va],pred[f'{arm}_validation_{lam}']),mse=float(np.square(pred[f'{arm}_validation_{lam}']-np.eye(12)[y[va]]).mean())) for lam in c['lambdas']]
            assert candidates==sel[arm]['candidates'];choice=min(range(3),key=lambda j:(-candidates[j]['macro'],candidates[j]['mse'],-candidates[j]['lam']));assert choice==sel[arm]['selected']
            ref=Ridge(alpha=c['lambdas'][choice]*len(tr),solver='cholesky').fit((xx[lookup[tr]]-mean)/sd,np.eye(12)[y[tr]],sample_weight=w*len(tr))
            rp=ref.predict((xx[lookup[te]]-mean)/sd);err=float(abs(rp-pred[arm]).max());assert err<1e-7;errors.append(err)
            saved=((xx[lookup[te]]-model[arm+'_mean'])/model[arm+'_scale'])@model[arm+'_coef']+model[arm+'_bias'];np.testing.assert_allclose(saved,pred[arm],atol=1e-13)
        for k in scores:
            val=metric(y[te],pred[k]);assert abs(val-row['scores'][k])<1e-14;scores[k].append(val)
        rows.append(dict(held=held,independent_sklearn_held_prediction_max_error=max(errors),functional_cnn_spot_checks=len(spots),train_val_diagnostic_roles_verified=True))
    passes=[]
    for k,gain in c['gate']['minimum_mean_gain'].items():
        diff=np.array(scores['signed'])-scores[k];ok=bool(diff.mean()>=gain and (diff>1e-12).sum()>=4 and diff.min()>=-.01);assert ok==summary['comparisons'][k]['passed'];passes.append(ok)
    for k,v in scores.items():np.testing.assert_allclose(v,summary['scores'][k],atol=1e-14);assert abs(np.mean(v)-summary['means'][k])<1e-14
    assert all(passes)==summary['gate_passed']
    audit=dict(state='passed',raw_records_independently_recomputed=len(ids),raw_descriptor_max_error=maxraw,verified_artifact_hashes=len(files),selected_model_refits=30,rows=rows,gate_passed=all(passes),seconds=time.perf_counter()-start,limits='CNN raw functional checks first2 per role per fold (36 total), not all hidden vectors; full descriptor recomputation and selected ridge refits; no encoder retraining')
    dump(a.runs/'AUDIT.json',audit);print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['runs','data','prepared']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1):main(a)
