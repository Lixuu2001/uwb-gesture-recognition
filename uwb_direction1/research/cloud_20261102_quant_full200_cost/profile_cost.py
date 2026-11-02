from pathlib import Path
import sys,time,json,resource,shutil,hashlib,gc
from guard import B,ready,sha
from cost_math import select
def main():
 p=ready();P=B/'profile01';P.mkdir(exist_ok=False)
 (P/'DECLARATION_BEFORE_TIMING.json').write_text(json.dumps(dict(policy_sha256=sha(B/'POLICY.json'),cost_hashes_sha256=sha(B/'COST_HASHES.json'),synthetic_only=True,actual_UWB_values_labels_fit=False,full_shape=[4700,567,90]),indent=2)+'\n')
 sys.path.insert(0,'/workspace/uwb-quant-author-probe-20261102')
 from native_quant import np,make,fit,predict,scientific_arrays,cloudpickle
 class Fixture:
  def __init__(self,n):self.shape=(n,567,90)
  def __len__(self):return self.shape[0]
  def __getitem__(self,key):
   assert isinstance(key,slice);start,stop,step=key.indices(len(self));assert step==1
   return np.stack([np.random.default_rng(20261029+i).standard_normal((567,90),dtype=np.float32) for i in range(start,stop)])
 def state(s,**kw):(P/'STATE.json').write_text(json.dumps(dict(state=s,actual_UWB_values_labels_fit=False,**kw),indent=2)+'\n')
 def digest(m):
  h=hashlib.sha256()
  for a in scientific_arrays(m):h.update(str((a.shape,a.dtype.str)).encode());h.update(np.ascontiguousarray(a).tobytes())
  return h.hexdigest()
 x=Fixture(4700);y=np.arange(4700)%12;m=make();state('full4700_all539217_features_then_full200_ExtraTrees_fit');t=time.monotonic();fit(m,x,y,32);elapsed=time.monotonic()-t
 assert len(m['classifier'].estimators_)==200 and m['feature_count']==539217 and m['training_cases']==4700
 ev=[];saves=[];loads=[];sizes=[];receipts=[];d=digest(m)
 for i in range(3):
  state('timing_full_QUANT_predict64',index=i);xx=x[i*64:(i+1)*64];t=time.monotonic();pr=predict(m,xx);ev.append(time.monotonic()-t)
 for i in range(3):
  state('timing_full_QUANT_cloudpickle_save_reload',index=i);path=P/f'synthetic_{i}.pkl';t=time.monotonic()
  with path.open('xb') as f:cloudpickle.dump(m,f,protocol=5)
  saves.append(time.monotonic()-t);sizes.append(path.stat().st_size);t=time.monotonic()
  with path.open('rb') as f:q=cloudpickle.load(f)
  loads.append(time.monotonic()-t);assert digest(q)==d;other=predict(q,xx);np.testing.assert_array_equal(other['predictions'],pr['predictions']);np.testing.assert_allclose(other['probabilities'],pr['probabilities'],rtol=1e-14,atol=1e-14)
  receipts.append(dict(name=path.name,bytes=path.stat().st_size,sha256=sha(path),all_tree_interval_arrays_exact=True,predictions_exact=True,probabilities_predeclared_tolerance=True,disposable_synthetic_file_removed_after_check=True));(P/'SYNTHETIC_CHECKPOINT_RECEIPTS.json').write_text(json.dumps(receipts,indent=2)+'\n');del q;gc.collect();path.unlink()
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;free=shutil.disk_usage('/workspace').free;r=select(p,elapsed,ev,saves,loads,rss,max(sizes),free);r.update(state='complete_full4700_QUANT539217_ExtraTrees200_synthetic_cost',full_native_fit_seconds=elapsed,evaluation64_seconds=ev,complete_save_seconds=saves,complete_reload_seconds=loads,peak_RSS_KiB=rss,checkpoint_sizes=sizes,full_trees=200,full_features=539217,loaded_tree_interval_arrays_exact=True,loaded_predictions_exact=True,loaded_probabilities_predeclared_tolerance=True,synthetic_checkpoint_receipts_sha256=sha(P/'SYNTHETIC_CHECKPOINT_RECEIPTS.json'),actual_UWB_values_labels_fit=False)
 (P/'COST_RESULT.json').write_text(json.dumps(r,indent=2)+'\n');state('complete',resource_gate_passed=r['resource_gate_passed']);print(json.dumps(r))
if __name__=='__main__':main()
