"""Compare fixed equal-width range projections on source clips only.

Pooling and DCT are prior-art controls, not new methods. No model is trained.
Prototype contrast is in-sample descriptive evidence, not held-out accuracy.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.fft import dct,idct
from audit_restored import digest,dump,quality_mask


def reconstruct(x,width,kind):
    if x.shape[-1]%width:raise ValueError('Width must divide range size')
    if kind=='block_mean':
        stride=x.shape[-1]//width
        return np.repeat(x.reshape(*x.shape[:-1],width,stride).mean(-1),stride,axis=-1)
    if kind=='dct_lowpass':
        z=dct(x,type=2,norm='ortho',axis=-1);z[...,width:]=0
        return idct(z,type=2,norm='ortho',axis=-1)
    raise ValueError(kind)


def ratio(x,p):
    den=np.square(x).sum(axis=(-2,-1));num=np.square(p).sum(axis=(-2,-1))
    # Zero-energy signals defined as retained; count separately in report.
    return np.divide(num,den,out=np.ones_like(num),where=den>1e-20)


def checks():
    rng=np.random.default_rng(20261003);x=rng.normal(size=(4,3,2,8,63))
    n=63;r=np.arange(n);k=np.arange(n)[:,None]
    basis=np.sqrt(2/n)*np.cos(np.pi*(r+.5)*k/n);basis[0]/=np.sqrt(2)
    error=0.
    for width in [21,7]:
        for kind in ['block_mean','dct_lowpass']:
            got=reconstruct(x,width,kind)
            if kind=='dct_lowpass':ref=(x@basis[:width].T)@basis[:width]
            else:
                ref=np.empty_like(x);stride=n//width
                for i in range(width):ref[...,i*stride:(i+1)*stride]=x[...,i*stride:(i+1)*stride].sum(-1,keepdims=True)/stride
            np.testing.assert_allclose(got,ref,atol=2e-14,rtol=1e-13)
            np.testing.assert_allclose((x*x).sum(-1),(got*got).sum(-1)+((x-got)**2).sum(-1),rtol=1e-12)
            np.testing.assert_allclose(reconstruct(got,width,kind),got,atol=2e-14,rtol=1e-13)
            error=max(error,float(np.max(np.abs(got-ref))))
    return {'independent_cosine_matrix_and_loop_projection_passed':True,'orthogonal_energy_decomposition_passed':True,'idempotence_passed':True,'max_absolute_error':error}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['data','prepared','out']:ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--protocol',type=Path,default=Path(__file__).resolve().parents[1]/'research/RANGE_COMPRESSION_PROTOCOL.json')
    a=ap.parse_args();c=json.loads(a.protocol.read_text())
    if a.out.exists():raise ValueError('Use a new output directory')
    test=checks();started=time.perf_counter()
    cache_path=a.prepared/'signals.npy';cache_sha=digest(cache_path)
    assert cache_sha=='1f689bba7f10bb37faf07a557c52653843e81b5907f04d17fa92dbf4dbae7c5f'
    yp=a.data/'cache/y_clutterremoved.npy';up=a.data/'cache/subj_clutterremoved.npy'
    assert digest(yp)=='d77b10f0fbda231246e71dfcf0b2f852863af48f80f159aed90f5f9b7c61a8bd'
    assert digest(up)=='5ab32ab59e5b41c05d301b76949f159ddb1e9423a749deb760585a5f19cf9480'
    y=np.load(yp);u=np.load(up);ids=np.flatnonzero(quality_mask(y,u)&np.isin(u,c['source_users']))
    assert len(ids)==c['expected_records'] and not np.isin(u[ids],c['excluded_users']).any()
    cache=np.load(cache_path,mmap_mode='r');assert cache.shape==(9593,3,6,90,63)
    cells=np.zeros((6,12,3,2,90,63),dtype=np.float64);counts=np.zeros((6,12),dtype=int)
    arms=[(kind,w) for w in c['widths'] for kind in c['transforms']]
    retained={f'{k}_{w}':[] for k,w in arms};dtretained={k:[] for k in retained};zero=0
    independent_real_errors=[]
    for start in range(0,len(ids),16):
        batch=ids[start:start+16];x=np.asarray(cache[batch][:,:,c['channels']],dtype=np.float64)
        assert np.isfinite(x).all();zero+=int((np.square(x).sum((-2,-1))<=1e-20).sum())
        for i,sid in enumerate(batch):cells[u[sid]-3,y[sid]]+=x[i];counts[u[sid]-3,y[sid]]+=1
        dx=np.diff(x,axis=-2)
        for kind,width in arms:
            key=f'{kind}_{width}';p=reconstruct(x,width,kind)
            retained[key].append(ratio(x,p));dtretained[key].append(ratio(dx,np.diff(p,axis=-2)))
            if start==0:
                n=63;r=np.arange(n);k=np.arange(width)[:,None];basis=np.sqrt(2/n)*np.cos(np.pi*(r+.5)*k/n);basis[0]/=np.sqrt(2)
                if kind=='dct_lowpass':ref=(x@basis.T)@basis
                else:
                    stride=n//width;ref=np.stack([x[...,i*stride:(i+1)*stride].sum(-1)/stride for i in range(width)],axis=-1).repeat(stride,axis=-1)
                np.testing.assert_allclose(ref,p,atol=2e-12,rtol=1e-12)
                independent_real_errors.append(float(np.max(np.abs(ref-p))))
    common=np.flatnonzero((counts>0).all(0));means=cells[:,common]/counts[:,common,None,None,None,None]
    prototypes=means.mean(0);contrast=prototypes-prototypes.mean(0,keepdims=True)
    den=np.square(contrast).sum((0,3,4));result={};payload={'ids':ids,'labels':y[ids],'users':u[ids],'cell_counts':counts,'common_classes':common}
    for kind,width in arms:
        key=f'{kind}_{width}';v=np.concatenate(retained[key]);dv=np.concatenate(dtretained[key]);payload[key]=v;payload[key+'_dt']=dv
        projected=reconstruct(contrast,width,kind);cr=np.square(projected).sum((0,3,4))/den
        med=np.median(v,axis=0);dmed=np.median(dv,axis=0);g=c['eligibility_gate']
        passed=bool(np.all(med>=g['median_clip_energy_each_view_channel']) and np.all(dmed>=g['median_temporal_difference_each_view_channel']) and np.all(cr>=g['class_contrast_each_view_channel']))
        result[key]={'range_width':width,'input_elements_ratio':width/63,'clip_energy_mean':v.mean(0).tolist(),'clip_energy_median':med.tolist(),'clip_energy_q05':np.quantile(v,.05,axis=0).tolist(),'temporal_difference_median':dmed.tolist(),'prototype_contrast_retention':cr.tolist(),'eligible_for_future_accuracy_test':passed,
            'by_source_user':{str(h):{'clip_energy_median':np.median(v[u[ids]==h],axis=0).tolist(),'temporal_difference_median':np.median(dv[u[ids]==h],axis=0).tolist()} for h in c['source_users']}}
        assert np.max(v)<=1+1e-10 and np.min(v)>=0
    a.out.mkdir(parents=True);np.savez_compressed(a.out/'per_clip.npz',**payload)
    summary={'state':'completed_source_only_diagnostic','records':len(ids),'source_users':c['source_users'],'common_classes':common.tolist(),'axis_order':['Left/Top/Right','amplitude/weighted_wrapped_phase'],'zero_energy_cases':zero,'arms':result,'checks':dict(test,real_first16_max_error=max(independent_real_errors)),
       'seconds':time.perf_counter()-started,'quality_or_novelty_proven':False,'trained_models':0,'source_script_sha256':digest(__file__),'protocol_sha256':digest(a.protocol),'input_sha256':cache_sha,'per_clip_sha256':digest(a.out/'per_clip.npz'),
       'limitations':['not classification accuracy','prototype contrast is in-sample source descriptive statistic','energy discarded may be noise or discriminative; cannot decide from energy alone','input is already factor3 range-pooled, not raw189','no measured inference speedup','pooling and DCT are existing transforms']}
    dump(a.out/'SUMMARY.json',summary);dump(a.out/'protocol.json',c)
    print(json.dumps({'state':summary['state'],'seconds':summary['seconds'],'eligible':{k:r['eligible_for_future_accuracy_test'] for k,r in result.items()}}),flush=True)


if __name__=='__main__':main()
