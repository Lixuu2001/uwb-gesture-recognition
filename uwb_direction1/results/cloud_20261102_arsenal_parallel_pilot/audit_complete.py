"""Separate full25 native fit,allkernel/scaler/readout/vote replay,FP64stats/RMS."""
from pathlib import Path
import sys,time,pickle,hashlib
from common import B,ready,read,sha,write,load_inputs
def arrays(m,np):
    for pipe in m.estimators_:
        transform,scaler,ridge=(part[1] for part in pipe.steps)
        for a in transform.kernels:yield np.asarray(a)
        for a in [scaler.mean_,scaler.var_,scaler.scale_,ridge.coef_,ridge.intercept_,ridge.classes_,ridge.alpha_,ridge.best_score_]:yield np.asarray(a)
    for a in [m.weights_,m._weight_sum,m.classes_]:yield np.asarray(a)
def verify_statistics(np,scalers,sums,squares,n):
    for i,s in enumerate(scalers):
        mean=sums[i]/n;variance=np.maximum(squares[i]/n-mean*mean,0)
        np.testing.assert_allclose(s.mean_,mean,rtol=1e-10,atol=1e-9)
        np.testing.assert_allclose(s.var_,variance,rtol=2e-5,atol=1e-9)
def main():
    c=ready();start=time.monotonic();out=B/'run01';term=read(B/'supervisor01/TERMINATION.json');assert term['exit_code']==0 and not term['wall_timeout'] and not term['resource_stop_reason']
    assert read(out/'STATE.json')['state']=='producer_complete_unscored'
    for name,d in read(out/'FILES.json').items():assert sha(out/name)==d,name
    sys.path.insert(0,c['source_preparation']);from native_arsenal import np,make,assert_native
    from aeon.transformations.collection import Normalizer
    ids,users,loc,maps,y=load_inputs(c,True)
    with np.load(out/'split.npz') as z:
        for key,value in dict(ids=ids,users=users,**loc).items():np.testing.assert_array_equal(z[key],value)
    with (out/'initial_unfitted.pkl').open('rb') as f:initial=pickle.load(f)
    assert_native(initial);assert not initial.is_fitted
    replay=make();x=np.array(maps[loc['train']].transpose(0,2,1),copy=True,order='C');replay.fit(x,y);del x;assert_native(replay,True)
    for name in ['native_fitted.pkl','final_fitted.pkl']:
        with (out/name).open('rb') as f:reference=pickle.load(f)
        assert_native(reference,True);a=list(arrays(replay,np));b=list(arrays(reference,np));assert len(a)==len(b)
        for left,right in zip(a,b):np.testing.assert_array_equal(left,right)
        for p in reference.estimators_:assert p.steps[1][1].n_samples_seen_==4700
        del reference
    sums=np.zeros((25,4000),np.float64);squares=sums.copy();raw_count=0;held_count=0
    for role in ['train','validation','diagnostic']:
        probabilities=[];member_preds=[];hashes=[]
        for offset in range(0,len(loc[role]),64):
            ix=loc[role][offset:offset+64];x=np.array(maps[ix].transpose(0,2,1),copy=True,order='C');norm=Normalizer().fit_transform(x).astype(np.float32,copy=False)
            probability=np.zeros((len(ix),12));preds=[];h=hashlib.sha256()
            for member,(pipe,weight) in enumerate(zip(replay.estimators_,replay.weights_)):
                t,s,r=[part[1] for part in pipe.steps];raw=t._transform_kernels(norm);assert raw.shape==(len(ix),4000) and np.isfinite(raw).all()
                if role=='train':d=raw.astype(np.float64);sums[member]+=d.sum(0);squares[member]+=(d*d).sum(0)
                else:
                    f=s.transform(raw);assert np.isfinite(f).all();h.update(np.ascontiguousarray(f).tobytes());p=r.predict(f);preds.append(p);probability[np.arange(len(ix)),np.searchsorted(replay.classes_,p)]+=weight
            raw_count+=len(ix)
            if role!='train':probabilities.append(probability/replay._weight_sum);member_preds.append(np.stack(preds,axis=1));hashes.append(h.hexdigest());held_count+=len(ix)
        if role!='train':
            prob=np.concatenate(probabilities)
            # Separate native tie-RNG reconstruction over the whole canonical role.
            tie_rng=np.random.RandomState(20261029);prediction=np.array([replay.classes_[tie_rng.choice(np.flatnonzero(row==row.max()))] for row in prob])
            with np.load(out/(role+'_unscored.npz')) as z:
                for key,value in dict(ids=ids[loc[role]],probabilities=prob,predictions=prediction,member_predictions=np.concatenate(member_preds),feature_batch_digests=np.asarray(hashes)).items():np.testing.assert_array_equal(z[key],value)
    assert raw_count==6900 and held_count==2200;verify_statistics(np,[p.steps[1][1] for p in replay.estimators_],sums,squares,4700)
    raw=np.load(Path(c['data'])/'cache/X_clutterremoved.npy',mmap_mode='r')
    for offset in range(0,6900,64):
        a=np.array(raw[ids[offset:offset+64]],copy=True);r=np.sqrt(np.mean(a*a,axis=(2,3),keepdims=True,dtype=np.float32));ref=(a/np.maximum(r,np.float32(1e-8))).transpose(0,2,1,3).reshape(len(a),90,567)
        np.testing.assert_allclose(ref,maps[offset:offset+64],rtol=2e-5,atol=2e-6)
        d=a.astype(np.float64);r64=np.sqrt(np.mean(d*d,axis=(2,3),keepdims=True));double=(d/np.maximum(r64,1e-8)).transpose(0,2,1,3).reshape(len(a),90,567);np.testing.assert_allclose(ref,double,rtol=2e-5,atol=2e-6)
    for name,digest in read(out/'FILES.json').items():assert sha(out/name)==digest,name
    hashes={str(out/name):value for name,value in read(out/'FILES.json').items()};hashes[str(out/'FILES.json')]=sha(out/'FILES.json')
    result=dict(state='pass',full_native4700_25member_fit_replayed=True,all50000kernels_25scalers_readouts_alphas_LOOaccuracy_votes_exact=True,all2200_probabilities_member_votes_predictions_featurehashes_exact=True,all6900_all25_raw_feature_sets_finite=True,independent_training_all25_FP64_scaler_statistics_verified=True,all6900_raw_RMS_cache_FP64_verified=True,initial_unfitted_native_and_final_models_exact=True,diagnostic_labels_read=False,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,auditor_sha256=sha(__file__),artifact_hashes=hashes,audit_wall_seconds=time.monotonic()-start)
    write(B/'COMPLETE_AUDIT.json',result);print(__import__('json').dumps({k:v for k,v in result.items() if k!='artifact_hashes'}))
if __name__=='__main__':main()
