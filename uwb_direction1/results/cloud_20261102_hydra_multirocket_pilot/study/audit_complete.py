"""Separate full native replay, every raw-core finite check, independent scaler/RMS reference."""
from pathlib import Path
import sys,time,pickle,hashlib
from common import B,ready,read,sha,write,load_inputs
def arrays(m,np):
    h=m._transform_hydra._hydra
    for value in [h.W,h.dilations,h.paddings,*h.idx,m._scale_hydra.mu,m._scale_hydra.sigma,m._scale_hydra.epsilon]:yield value.detach().cpu().numpy()
    for value in [*m._transform_multirocket.parameter,*m._transform_multirocket.parameter1,m._scale_multirocket.mean_,m._scale_multirocket.var_,m._scale_multirocket.scale_,m.classifier.coef_,m.classifier.intercept_,m.classifier.classes_]:yield np.asarray(value)
def verify_scaler_statistics(np,replay,hsum,hsq,zeros,msum,msq,n):
    assert n>1
    hm=hsum/n;hv=np.maximum((hsq-hsum*hm)/(n-1),0);heps=(zeros/n)**4+1e-8
    np.testing.assert_allclose(replay._scale_hydra.mu.numpy(),hm,rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose(replay._scale_hydra.epsilon.numpy(),heps,rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose(replay._scale_hydra.sigma.numpy(),np.sqrt(hv)+heps,rtol=2e-5,atol=2e-6)
    mm=msum/n;mv=np.maximum(msq/n-mm*mm,0)
    np.testing.assert_allclose(replay._scale_multirocket.mean_,mm,rtol=1e-10,atol=1e-9)
    np.testing.assert_allclose(replay._scale_multirocket.var_,mv,rtol=2e-5,atol=1e-9)
def main():
    c=ready();start=time.monotonic();out=B/'run01';term=read(B/'supervisor01/TERMINATION.json')
    assert term['exit_code']==0 and not term['wall_timeout'] and not term['resource_stop_reason'] and read(out/'STATE.json')['state']=='producer_complete_unscored'
    for n,d in read(out/'FILES.json').items():assert sha(out/n)==d,n
    sys.path.insert(0,c['source_preparation'])
    from native_hydramr import np,torch,make,assert_native,_transform_multi
    ids,users,loc,maps,y=load_inputs(c,training_labels=True)
    with np.load(out/'split.npz') as z:
        for k,v in dict(ids=ids,users=users,**loc).items():np.testing.assert_array_equal(z[k],v)
    with (out/'initial_unfitted.pkl').open('rb') as f:init=pickle.load(f)
    assert_native(init);assert not init.is_fitted
    replay=make();x=np.array(maps[loc['train']].transpose(0,2,1),copy=True,order='C');replay.fit(x,y);del x
    assert_native(replay,True);assert replay._scale_multirocket.n_samples_seen_==4700
    for name in ['native_fitted.pkl','final_fitted.pkl']:
        with (out/name).open('rb') as f:ref=pickle.load(f)
        assert_native(ref,True)
        ra=list(arrays(replay,np));rb=list(arrays(ref,np));assert len(ra)==len(rb)
        for a,b in zip(ra,rb):np.testing.assert_array_equal(a,b)
        assert replay.classifier.alpha_==ref.classifier.alpha_ and replay.classifier.best_score_==ref.classifier.best_score_
        assert replay._scale_multirocket.n_samples_seen_==ref._scale_multirocket.n_samples_seen_==4700
        del ref
    # Separate explicit transform driver; every6900 raw MR row checked before fallback.
    hsum=np.zeros(4096,np.float64);hsq=hsum.copy();zeros=np.zeros(4096,np.int64)
    msum=np.zeros(49728,np.float64);msq=msum.copy();raw_checked=0;held_checked=0
    mr=replay._transform_multirocket
    for role in ['train','validation','diagnostic']:
        decisions=[];predictions=[];hashes=[]
        for start_batch in range(0,len(loc[role]),64):
            ix=loc[role][start_batch:start_batch+64];x=np.array(maps[ix].transpose(0,2,1),copy=True,order='C')
            hydra=replay._transform_hydra.transform(x);assert np.isfinite(hydra.numpy()).all()
            raw=_transform_multi(x,np.diff(x,axis=2),mr.parameter,mr.parameter1,4,mr._indices,mr.random_state_)
            assert raw.shape==(len(ix),49728) and np.isfinite(raw).all(),'Nonfinite native core cannot be imputed into a scientific negative'
            np.testing.assert_array_equal(raw,mr.transform(x));raw_checked+=len(ix)
            if role=='train':
                h=np.sqrt(np.maximum(hydra.numpy().astype(np.float64),0));hsum+=h.sum(0);hsq+=(h*h).sum(0);zeros+=(h==0).sum(0)
                d=raw.astype(np.float64);msum+=d.sum(0);msq+=(d*d).sum(0)
            else:
                a=replay._scale_hydra.transform(hydra).numpy();b=replay._scale_multirocket.transform(raw)
                features=np.concatenate([a,b],axis=1);assert np.isfinite(features).all()
                decisions.append(replay.classifier.decision_function(features));predictions.append(replay.classifier.predict(features))
                hashes.append(hashlib.sha256(features.tobytes()).hexdigest());held_checked+=len(ix)
        if role!='train':
            with np.load(out/(role+'_unscored.npz')) as z:
                np.testing.assert_array_equal(z['ids'],ids[loc[role]])
                np.testing.assert_array_equal(z['decisions'],np.concatenate(decisions))
                np.testing.assert_array_equal(z['predictions'],np.concatenate(predictions))
                np.testing.assert_array_equal(z['feature_batch_digests'],hashes)
    assert raw_checked==6900 and held_checked==2200
    # Independent double-precision streaming statistics, not native refit code.
    verify_scaler_statistics(np,replay,hsum,hsq,zeros,msum,msq,4700)
    raw=np.load(Path(c['data'])/'cache/X_clutterremoved.npy',mmap_mode='r')
    for offset in range(0,6900,64):
        a=np.array(raw[ids[offset:offset+64]],copy=True);r=np.sqrt(np.mean(a*a,axis=(2,3),keepdims=True,dtype=np.float32))
        ref=(a/np.maximum(r,np.float32(1e-8))).transpose(0,2,1,3).reshape(len(a),90,567)
        np.testing.assert_allclose(ref,maps[offset:offset+64],rtol=2e-5,atol=2e-6)
        d=a.astype(np.float64);r64=np.sqrt(np.mean(d*d,axis=(2,3),keepdims=True))
        fp64=(d/np.maximum(r64,1e-8)).transpose(0,2,1,3).reshape(len(a),90,567)
        np.testing.assert_allclose(ref,fp64,rtol=2e-5,atol=2e-6)
    for name,digest in read(out/'FILES.json').items():assert sha(out/name)==digest,name
    hashes={str(out/name):d for name,d in read(out/'FILES.json').items()};hashes[str(out/'FILES.json')]=sha(out/'FILES.json')
    r=dict(state='pass',full_native_4700_fit_replayed=True,Hydra_weights_indices_MR_parameters_scalers_coefficients_alpha_LOO_score_exact=True,
       all2200_predictions_decisions_and_feature_batch_hashes_exact=True,all6900raw_MultiRocket_features_finite_before_native_fallback=True,
       raw_MultiRocket_core_public_API_exact=True,independent_training_SparseScaler_and_standard_scaler_FP64_statistics_verified=True,
       all6900source_raw_RMS_cache_FP64_verified=True,initial_unfitted_and_native_final_models_exact=True,diagnostic_labels_read=False,
       users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,auditor_sha256=sha(__file__),
       artifact_hashes=hashes,audit_wall_seconds=time.monotonic()-start)
    write(B/'COMPLETE_AUDIT.json',r);print(__import__('json').dumps({k:v for k,v in r.items() if k!='artifact_hashes'}))
if __name__=='__main__':main()
