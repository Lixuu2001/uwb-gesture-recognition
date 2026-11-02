from pathlib import Path
import sys,time,json,resource,shutil,pickle,hashlib,gc,copy
from guard import B,ready,sha
from cost_math import select
class Counter:
 def __init__(self):self.size=0;self.h=hashlib.sha256()
 def write(self,b):self.size+=len(b);self.h.update(b);return len(b)
def main():
 p=ready();P=B/'profile01';P.mkdir(exist_ok=False)
 (P/'DECLARATION_BEFORE_TIMING.json').write_text(json.dumps(dict(policy_sha256=sha(B/'POLICY.json'),cost_hashes_sha256=sha(B/'COST_HASHES.json'),synthetic_only=True,actual_UWB_values_labels_fit=False,full_shape=[4700,567,90]),indent=2)+'\n')
 sys.path.insert(0,'/workspace/uwb-drcif500-batched-spectrum-source-20261102')
 from native_drcif_batched import np,make,fit,predict,assert_native,digest
 def state(s,**kw):(P/'STATE.json').write_text(json.dumps(dict(state=s,actual_UWB_values_labels_fit=False,**kw),indent=2)+'\n')
 def statehash(m):
  h=hashlib.sha256();q=copy.copy(m)
  for t,g in zip(m.estimators_,m.intervals_):q.estimators_=(t,);q.intervals_=(g,);h.update(digest(q).encode())
  return h.hexdigest()
 rng=np.random.default_rng(20261029);state('generating_full4700_synthetic_float32_fixture');x=rng.standard_normal((4700,567,90),dtype=np.float32);y=np.arange(4700)%12;m=make();state('full4700_float64_DrCIF500_CIT20_batched_native_spectrum_fit');t=time.monotonic();fit(m,x,y);elapsed=time.monotonic()-t;assert_native(m,True);assert m._n_intervals==[79,78,67]
 ev=[];saves=[];loads=[];sizes=[];receipts=[];d=statehash(m)
 for i in range(3):
  state('timing_native_DrCIF500_predict64',index=i);xx=x[i*64:(i+1)*64];t=time.monotonic();pr=predict(m,xx);ev.append(time.monotonic()-t)
 state('counting_protocol5_full_model_bytes_before_any_disk_write');counter=Counter();pickle.dump(m,counter,protocol=5);free=shutil.disk_usage('/workspace').free
 preflight=dict(state='full_DrCIF500_model_serialized_count_before_disk_write',model_bytes=counter.size,serialized_sha256=counter.h.hexdigest(),required_workspace_bytes=8*counter.size+67108864,workspace_free_bytes=free,synthetic_only=True,full_fit_seconds=elapsed,full_native_fit_completed=True,actual_UWB_values_labels_fit=False)
 (P/'CHECKPOINT_PREFLIGHT.json').write_text(json.dumps(preflight,indent=2)+'\n')
 if preflight['required_workspace_bytes']>free:
  r=dict(state='full_DrCIF500_storage_policy_closed_before_checkpoint_write',**{k:v for k,v in preflight.items() if k!='state'},resource_gate_passed=False,full_timing_result_available=False,accuracy_measured=False,no_checkpoint_written=True,all500_trees_fitted=True,requested_intervals_per_representation=m._n_intervals)
  (P/'RESOURCE_RESULT.json').write_text(json.dumps(r,indent=2)+'\n');state('closed_storage_resource_policy');print(json.dumps(r));return
 for i in range(3):
  state('timing_full_native_DrCIF500_save_reload',index=i);path=P/f'synthetic_{i}.pkl';t=time.monotonic()
  with path.open('xb') as f:pickle.dump(m,f,protocol=5)
  saves.append(time.monotonic()-t);sizes.append(path.stat().st_size);t=time.monotonic()
  with path.open('rb') as f:q=pickle.load(f)
  loads.append(time.monotonic()-t);assert_native(q,True);assert statehash(q)==d;other=predict(q,xx);np.testing.assert_array_equal(other['probabilities'],pr['probabilities']);np.testing.assert_array_equal(other['predictions'],pr['predictions']);receipts.append(dict(name=path.name,bytes=path.stat().st_size,sha256=sha(path),all500_CIT_interval_feature_subset_state_exact=True,predictions_probabilities_exact=True,disposable_synthetic_removed_after_check=True));(P/'SYNTHETIC_CHECKPOINT_RECEIPTS.json').write_text(json.dumps(receipts,indent=2)+'\n');del q;gc.collect();path.unlink()
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;free=shutil.disk_usage('/workspace').free;r=select(p,elapsed,ev,saves,loads,rss,max(sizes),free);r.update(state='complete_full4700_DrCIF500_CIT20_batched_native_spectrum_synthetic_cost',spectrum_batch_size=64,scientific_configuration_unchanged=True,full_native_fit_seconds=elapsed,evaluation64_seconds=ev,complete_save_seconds=saves,complete_reload_seconds=loads,peak_RSS_KiB=rss,checkpoint_sizes=sizes,full_trees=500,CIT_thresholds=20,requested_intervals_per_representation=m._n_intervals,loaded_learned_state_probabilities_predictions_exact=True,synthetic_checkpoint_receipts_sha256=sha(P/'SYNTHETIC_CHECKPOINT_RECEIPTS.json'),actual_UWB_values_labels_fit=False)
 (P/'COST_RESULT.json').write_text(json.dumps(r,indent=2)+'\n');state('complete',resource_gate_passed=r['resource_gate_passed']);print(json.dumps(r))
if __name__=='__main__':main()
