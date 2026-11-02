from pathlib import Path
import sys,time,json,resource,pickle,shutil
from guard import B,ready,sha
from cost_math import select
def main():
    p=ready();P=B/'profile01';P.mkdir(exist_ok=False)
    (P/'DECLARATION_BEFORE_TIMING.json').write_text(json.dumps(dict(policy_sha256=sha(B/'POLICY.json'),cost_hashes_sha256=sha(B/'COST_HASHES.json'),synthetic_only=True,actual_UWB_values_labels_fit=False,full_shape=[4700,567,90]),indent=2)+'\n')
    sys.path.insert(0,'/workspace/uwb-arsenal-source-probe-20261102')
    from native_arsenal import np,make,assert_native,canonical_predictions
    def state(s,**kw):(P/'STATE.json').write_text(json.dumps(dict(state=s,actual_UWB_values_labels_fit=False,**kw),indent=2)+'\n')
    # One full native end-to-end4700 fit,not linear/cubic tinyN extrapolation.
    rng=np.random.default_rng(20261029);state('generating_full4700_synthetic_fixture')
    x=rng.standard_normal((4700,567,90),dtype=np.float32);y=np.arange(4700)%12
    m=make();state('full4700_native_25x2000_Arsenal_fit');t=time.monotonic();m.fit(x,y);fit=time.monotonic()-t;assert_native(m,True)
    ev=[];saves=[];loads=[];sizes=[]
    for i in range(3):
        state('timing_full25_member_predict_proba64',index=i);t=time.monotonic();pr=m.predict_proba(x[i*64:(i+1)*64]);canonical_predictions(m,pr);ev.append(time.monotonic()-t);assert np.isfinite(pr).all();np.testing.assert_allclose(pr.sum(1),1,rtol=1e-12,atol=1e-12)
    for i in range(3):
        state('timing_complete_native_model_save_reload',index=i);path=P/f'full25_synthetic_{i}.pkl';t=time.monotonic()
        with path.open('wb') as f:pickle.dump(m,f,protocol=5)
        saves.append(time.monotonic()-t);sizes.append(path.stat().st_size);t=time.monotonic()
        with path.open('rb') as f:q=pickle.load(f)
        loads.append(time.monotonic()-t);assert_native(q,True);np.testing.assert_array_equal(q.predict_proba(x[128:192]),pr)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;free=shutil.disk_usage('/workspace').free
    r=select(p,fit,ev,saves,loads,rss,max(sizes),free);r.update(state='complete_full4700_native_Arsenal25x2000_synthetic_cost',full_native_fit_seconds=fit,evaluation64_seconds=ev,complete_save_seconds=saves,complete_reload_seconds=loads,peak_RSS_KiB=rss,checkpoint_sizes=sizes,full_native_members=25,kernels_per_member=2000,features_per_member=4000,loaded_probabilities_exact=True,actual_UWB_values_labels_fit=False)
    (P/'COST_RESULT.json').write_text(json.dumps(r,indent=2)+'\n');state('complete',resource_gate_passed=r['resource_gate_passed']);print(json.dumps(r))
if __name__=='__main__':main()
