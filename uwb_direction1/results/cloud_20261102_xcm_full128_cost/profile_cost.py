"""One full native warm update,3updates/evals/full checkpoint saves/loads."""
from pathlib import Path
import sys,time,json,resource,shutil,gc
from guard import B,ready
def main():
    policy=ready();P=B/'profile01';P.mkdir(exist_ok=False)
    (P/'DECLARATION_BEFORE_TIMING.json').write_text(json.dumps(policy,indent=2)+'\n')
    sys.path.insert(0,'/workspace/uwb-xcm-source-probe-20261102')
    from native_xcm import make,np,torch,keras
    def state(s,**kw):(P/'STATE.json').write_text(json.dumps(dict(state=s,actual_UWB_values_labels_fit=False,**kw),indent=2)+'\n')
    rng=np.random.default_rng(20261029);x=rng.standard_normal((32,90,567,1),dtype=np.float32);y=np.eye(12,dtype=np.float32)[np.arange(32)%12]
    state('building_full_128_filter_author_AST');m=make();assert m.count_params()==policy['parameters']
    state('warming_one_full_native_Keras_Adam_update');m.train_on_batch(x,y)
    train=[];ev=[];sv=[];ld=[];checkpoint_bytes=[]
    for i in range(3):
        state('timing_full_native_training_update',index=i);m.reset_metrics();s=time.monotonic();r=m.train_on_batch(x,y);train.append(time.monotonic()-s);assert np.isfinite(r).all()
    for i in range(3):
        state('timing_full_native_evaluation',index=i);s=time.monotonic();pred=m.predict_on_batch(x);ev.append(time.monotonic()-s);assert np.isfinite(pred).all()
    for i in range(3):
        state('timing_full_native_checkpoint_save_and_reload',index=i);path=P/f'synthetic_full_{i}.keras';s=time.monotonic();m.save(path);sv.append(time.monotonic()-s);checkpoint_bytes.append(path.stat().st_size)
        s=time.monotonic();reloaded=keras.models.load_model(path);ld.append(time.monotonic()-s)
        np.testing.assert_array_equal(pred,reloaded.predict_on_batch(x))
        for a,b in zip(m.optimizer.variables,reloaded.optimizer.variables):np.testing.assert_array_equal(a.numpy(),b.numpy())
        del reloaded;gc.collect()
    # All three frozen synthetic checkpoints retained; no actual UWB values.
    from cost_math import decision
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    free=shutil.disk_usage('/workspace').free
    result=decision(train,ev,sv,ld,peak,max(checkpoint_bytes),free)
    result.update(state='complete_full_author_XCM128_native_Keras3_Torch_synthetic_cost',training_update_seconds=train,evaluation_batch_seconds=ev,checkpoint_save_seconds=sv,checkpoint_reload_seconds=ld,checkpoint_sizes=checkpoint_bytes,peak_RSS_KiB=peak,parameters=m.count_params(),input_shape=list(x.shape),threads=torch.get_num_threads(),actual_UWB_values_labels_fit=False,accuracy_measured=False,scientific_negative=False,checkpoint_predictions_and_Adam_state_exact=True)
    (P/'COST_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');state('complete',resource_gate_passed=result['resource_gate_passed']);print(json.dumps(result))
if __name__=='__main__':main()
