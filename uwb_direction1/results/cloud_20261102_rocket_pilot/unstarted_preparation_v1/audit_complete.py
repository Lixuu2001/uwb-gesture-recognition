"""Independent full native fit/replay plus raw-RMS reference; no held labels."""
from pathlib import Path
import sys,time,pickle,hashlib
from common import B,ready,read,sha,write,load_inputs
def main():
    c=ready();start=time.monotonic();out=B/'run01';term=read(B/'supervisor01/TERMINATION.json')
    assert term['exit_code']==0 and not term['wall_timeout'] and read(out/'STATE.json')['state']=='producer_complete_unscored'
    for n,d in read(out/'FILES.json').items():assert sha(out/n)==d,n
    sys.path.insert(0,c['source_preparation'])
    from native_rocket import np,make,assert_native
    ids,users,loc,maps,y=load_inputs(c,training_labels=True)
    with np.load(out/'split.npz') as z:
        for k,v in dict(ids=ids,users=users,**loc).items():np.testing.assert_array_equal(z[k],v)
    with (out/'initial_unfitted.pkl').open('rb') as f:init=pickle.load(f)
    assert_native(init);assert not init.is_fitted
    # Independent driver: full native fit, never import/call producer fit/predict.
    replay=make();X=np.array(maps[loc['train']].transpose(0,2,1),copy=True,order='C');replay.fit(X,y);del X
    assert_native(replay,True);assert replay._scaler.n_samples_seen_==4700
    with (out/'native_fitted.pkl').open('rb') as f:original=pickle.load(f)
    with (out/'final_fitted.pkl').open('rb') as f:final=pickle.load(f)
    for ref in [original,final]:
        assert_native(ref,True)
        for a,b in zip(replay._transformer.kernels,ref._transformer.kernels):np.testing.assert_array_equal(a,b)
        for name in ['mean_','var_','scale_']:np.testing.assert_array_equal(getattr(replay._scaler,name),getattr(ref._scaler,name))
        for name in ['coef_','intercept_','classes_']:np.testing.assert_array_equal(getattr(replay.estimator_,name),getattr(ref.estimator_,name))
        assert float(replay.estimator_.alpha_)==float(ref.estimator_.alpha_)
        assert float(replay.estimator_.best_score_)==float(ref.estimator_.best_score_)
    total=0
    for role in ['validation','diagnostic']:
        ds=[];ps=[];feature_hash=[]
        for start_batch in range(0,len(loc[role]),64):
            ix=loc[role][start_batch:start_batch+64];x=np.array(maps[ix].transpose(0,2,1),copy=True,order='C')
            f=replay.pipeline_[:-1].transform(x);ds.append(replay.estimator_.decision_function(f));ps.append(replay.estimator_.predict(f))
            feature_hash.append(hashlib.sha256(f.tobytes()).hexdigest());total+=len(ix)
        with np.load(out/(role+'_unscored.npz')) as z:
            np.testing.assert_array_equal(z['ids'],ids[loc[role]])
            np.testing.assert_array_equal(z['feature_batch_digests'],feature_hash)
            np.testing.assert_array_equal(z['decisions'],np.concatenate(ds))
            np.testing.assert_array_equal(z['predictions'],np.concatenate(ps))
    assert total==2200
    # Independent reference for signed 3-view RMS maps, every canonical source record.
    raw=np.load(Path(c['data'])/'cache/X_clutterremoved.npy',mmap_mode='r')
    for start_batch in range(0,6900,64):
        a=np.array(raw[ids[start_batch:start_batch+64]],copy=True)
        rms=np.sqrt(np.mean(a*a,axis=(2,3),keepdims=True,dtype=np.float32))
        ref=(a/np.maximum(rms,np.float32(1e-8))).transpose(0,2,1,3).reshape(len(a),90,567)
        np.testing.assert_allclose(ref,maps[start_batch:start_batch+64],rtol=2e-5,atol=2e-6)
        d=a.astype(np.float64);r=np.sqrt(np.mean(d*d,axis=(2,3),keepdims=True))
        fp64=(d/np.maximum(r,1e-8)).transpose(0,2,1,3).reshape(len(a),90,567)
        np.testing.assert_allclose(ref,fp64,rtol=2e-5,atol=2e-6)
    for n,d in read(out/'FILES.json').items():assert sha(out/n)==d,n
    hashes={str(out/n):d for n,d in read(out/'FILES.json').items()};hashes[str(out/'FILES.json')]=sha(out/'FILES.json')
    r=dict(state='pass',full_native_4700_fit_replayed=True,kernels_scaler_coefficients_alpha_LOO_score_exact=True,
       all2200_predictions_decisions_and_feature_batch_hashes_exact=True,all6900source_raw_RMS_cache_FP64_verified=True,
       initial_unfitted_and_native_final_models_exact=True,diagnostic_labels_read=False,users7_8_labels_used=False,
       independent_validation=False,novel_contribution_validated=False,auditor_sha256=sha(__file__),
       artifact_hashes=hashes,audit_wall_seconds=time.monotonic()-start)
    write(B/'COMPLETE_AUDIT.json',r);print(__import__('json').dumps({k:v for k,v in r.items() if k!='artifact_hashes'}))
if __name__=='__main__':main()
