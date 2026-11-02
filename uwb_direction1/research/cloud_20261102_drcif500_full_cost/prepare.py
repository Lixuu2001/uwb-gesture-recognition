from pathlib import Path
import json,hashlib,subprocess,sys,py_compile
B=Path(__file__).resolve().parent;S=Path('/workspace/uwb-drcif500-source-probe-20261102');A=Path('/workspace/uwb-quant-full200-cost-20261102')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(n,s):(B/n).write_text(s)
assert not (B/'COST_HASHES.json').exists() and not (B/'profile01').exists()
integration=json.loads((S/'INTEGRATION_CHECK.json').read_bytes());assert integration['full_trees']==500 and integration['serial_jobs1_parallel_threading4_all_intervals_feature_subsets_nodes_leaf_probabilities_exact']
closed=json.loads((A/'VERIFIED_EXECUTION_CLOSURE_20261102.json').read_bytes());assert closed['no_old_screen_retry'] and closed['no_accuracy_measured']
p=json.loads((A/'POLICY.json').read_bytes())
for k in ['depth','div','trees','max_features','criterion','features','Torch_threads','feature_batch_size','feature_matrix','probability_reload_rtol','probability_reload_atol','no_feature_tree_depth_seed_precision_view_or_role_rescue']:p.pop(k,None)
p.update(method='paper_scale_DrCIF500_native_CIT20_modern_aeon_transfer',source_commit='b694b1c03c0a753cc844dcb9ee357d77ca6bda6b',n_estimators=500,base_estimator='ContinuousIntervalTree',CIT_thresholds=20,att_subsample_size=10,n_intervals=[4,'sqrt-div'],min_interval_length=3,max_interval_length=.5,representation_lengths=[90,89,64],requested_intervals_per_representation=[79,78,67],features_before_native_duplicate_removal_per_tree=2240,seed=20261029,native_parallel_jobs=4,parallel_backend='threading',BLAS_threads=1,Numba_threads=1,input_dtype='float64_promotion_of_float32_RMS_values',native_catch22_source_unmodified=True,native_nonfinite_policy='replace_inf_to_NaN,CITmissingbranch,notarbitraryzero',contract_time_limit=None,probabilities_reload_exact=True,no_tree_interval_feature_seed_base_estimator_precision_or_view_rescue=True,checkpoint_preflight='Protocol5 countingSHA sink determines fullnative model bytes before write;8models+64MiB>free closes storage policy without touching UWB')
put('POLICY.json',json.dumps(p,indent=2)+'\n')
put('PREREGISTRATION.md','''# DrCIF500/CIT20: full synthetic resource screen

Prior exact QUANT depth6/div4/539217features/200ExtraTrees/seed20261029/jobs4 synthetic screen terminated by supervisorTERM after3600.0768854669295wall/13597.843748CPU,peak10498296KiB,cgroupOOMdelta0,without complete full4700 fit/COST_RESULT. Full12h estimate and accuracy unavailable;exact onehour execution closed,no retry or tree/feature/seed/view/precision rescue. Original source/tests/logs untouched,terminal evidence independently verified. This is not a negative accuracy result or whole family rejection.

Next fixed candidate: DrCIF from Middlehurstetal2021 HIVE-COTE2.0,MachineLearning110:3211–3243,DOI10.1007/s10994-021-06057-9,primary section3.2,experimental structure andTable2 checked. Full paper-scale500trees;CIT information-gain/margin-gain trees with20thresholds,10-of29 interval attributes,three representations raw/difference/periodogram,4+floor(sqrt(repLength)*sqrt(channels)/3) intervals each,half-representation maximum interval length. No added z-normalization. Signed per-record/viewRMS maps if a future real pilot is separately registered. This is a published baseline transfer,not original contribution.

Unchanged aeon1.6.0 commit b694b1c03c0a753cc844dcb9ee357d77ca6bda6b. Six installed classifier/CIT/baseforest/periodogram/randominterval/catch22 files exactly match pinned Gitblobs;originalBSDlicense retained. Disclose differences: current library default200/entropyDecisionTreeNone is explicitly set to500/CIT20;modern native random interval duplicate-removal and independent attribute subsetting per representation retained;periodogram pads90to128 andkeeps64bins,mininterval3,predictprobability average500. Not originalJava execution parity or exact published random draws. Seed20261029,n_jobs4/threading,BLAS1/Numba1 fixedbeforecost. Threadbackend shares complete input representations under16GiB quota and scarce disk;it does not promise4xspeedup or release allPython/NumbaGILs. NativeTimeSeriesTree and all500 retained,not a smaller CART substitute.

Pre-freeze execution fault: nativeCatch22 rejects float32 dueNumba cannotunify float32/64outlier_series. First full500 serial synthetic fit failed before completion. Fix is inputpromotion tofloat64 before unchanged nativefit/predict;allfloat32RMS values exactlyrepresentable. Originalfailedwrapper/test/log retained;no native source patch, no UWB labels/values accessed. Independent FP64 diff/zero-paddedFFT/sevenbasicstats and three-waymissingCITsplit/entropy checks passed. Full500smallN24x3x90 serial1vsparallel4 all learned intervals/feature-subsets/CITnodes/leafdistribution exact;native/batched predictions,probabilities andpickle roundtrip exact;manual unskipped allselectedfeature/500memberprobabilitycomposition exact. These are fixtures,not full567throughput,not independent reimplementation of all22catch22 kernels. Source check is not evidence of new UWB accuracy.

After every changed file exactremote readback andbranchheadverification: full4700x567x90 synthetic original float32fixture then native inputfloat64 promotion, full500CIT trees and requested[79,78,67]intervals. Fullfit includes promotion andtransform,not smallN extrapolation. Threeeval64/fullmodel save/reload measurements;allnative learned-state digests and probabilities/predictions exact afterload. Model digest streams per-tree semantic state to bound operational memory. Protocol5 serialized model size/SHA first counted to a nonstoring sink;if8fullmodels+64MiB exceedsfree,write explicit storage-resource-closure evidence and no checkpointwrites,not fitcost/accuracy rejection. Only disposable synthetic checkpoints can be removed after retained size/SHA/exactreload;UWB checkpoints would require completeprivatepreservation.

One synthetic profile3600wall/CPU14380soft14400hard/RSS12GiB/.2s+cgroupOOM;no auto retry. Success requires3fullfit+2*2200maxeval64/64+4maxsave+2maxload+180<=12h;4*estimate<=48CPUh;maxmeasuredRSS+2GiB<=14GiB;8models+64MiB<=remainingdisk. Resource failure closes exact CPU4/500/CIT20/runtimepolicy,not method family accuracy. Fullsyntheticprofile must finishbefore anyseparatelyregisteredrealfit. No held labels or hyperparameter tuning in cost screen.

Future protocol fixed train3–6/4700,val2/1000,diag1/1200,canonicalIDs;no internal tuning orOOB trainestimate (fit,notfit_predict). Actualmodels/IDs/predictions/logs retained;full independentfit/numericaudit/privatechunk+allmemberrecovery beforeonceheldscore. Supportedmacro classes0..9val2,0..11diag1,referenceTimeMIL.9866666666666667,minGain.005,tol1e-12. Failedaccuracy closes exactconfiguration,norescore/seed/tree/interval/attribute/CIT/voting/role rescue. Passonlyseparatelyregisteredsource6confirmation. Users7–8labels forbidden. Eightuserhistoricaldevelopment exploratory,noindependentvalidation/8foldgain/originalcontribution yet. Goalnotmet;original789/userREADME/currentrepo visibility/Driveprivate/automationenabled preserved.Cloudonly.
''')
s=(A/'cost_math.py').read_text();a=s.index('    assert (');z=s.index('\n    assert len',a);s=s[:a]+"    assert (p['n_estimators'],p['CIT_thresholds'],p['att_subsample_size'],p['seed'],p['threads'],p['synthetic_train_size'])==(500,20,10,20261029,4,4700)"+s[z:];s=s.replace('no_feature_tree_depth_seed_precision_view_or_role_rescue','no_tree_interval_feature_seed_base_estimator_precision_or_view_rescue');put('cost_math.py',s)
s=(A/'test_cost_math.py').read_text();a=s.index('P=dict(');z=s.index('\nclass Test',a);s=s[:a]+"P=dict(n_estimators=500,CIT_thresholds=20,att_subsample_size=10,seed=20261029,threads=4,synthetic_train_size=4700)"+s[z:];s=s.replace('dict(P,trees=2)','dict(P,n_estimators=2)');put('test_cost_math.py',s)
for n in ['guard.py','run_screen.py']:put(n,(A/n).read_text())
put('profile_cost.py','''from pathlib import Path
import sys,time,json,resource,shutil,pickle,hashlib,gc,copy
from guard import B,ready,sha
from cost_math import select
class Counter:
 def __init__(self):self.size=0;self.h=hashlib.sha256()
 def write(self,b):self.size+=len(b);self.h.update(b);return len(b)
def main():
 p=ready();P=B/'profile01';P.mkdir(exist_ok=False)
 (P/'DECLARATION_BEFORE_TIMING.json').write_text(json.dumps(dict(policy_sha256=sha(B/'POLICY.json'),cost_hashes_sha256=sha(B/'COST_HASHES.json'),synthetic_only=True,actual_UWB_values_labels_fit=False,full_shape=[4700,567,90]),indent=2)+'\\n')
 sys.path.insert(0,'/workspace/uwb-drcif500-source-probe-20261102')
 from native_drcif import np,make,fit,predict,assert_native,digest
 def state(s,**kw):(P/'STATE.json').write_text(json.dumps(dict(state=s,actual_UWB_values_labels_fit=False,**kw),indent=2)+'\\n')
 def statehash(m):
  h=hashlib.sha256();q=copy.copy(m)
  for t,g in zip(m.estimators_,m.intervals_):q.estimators_=(t,);q.intervals_=(g,);h.update(digest(q).encode())
  return h.hexdigest()
 rng=np.random.default_rng(20261029);state('generating_full4700_synthetic_float32_fixture');x=rng.standard_normal((4700,567,90),dtype=np.float32);y=np.arange(4700)%12;m=make();state('full4700_float64_promoted_native_DrCIF500_CIT20_fit');t=time.monotonic();fit(m,x,y);elapsed=time.monotonic()-t;assert_native(m,True);assert m._n_intervals==[79,78,67]
 ev=[];saves=[];loads=[];sizes=[];receipts=[];d=statehash(m)
 for i in range(3):
  state('timing_native_DrCIF500_predict64',index=i);xx=x[i*64:(i+1)*64];t=time.monotonic();pr=predict(m,xx);ev.append(time.monotonic()-t)
 state('counting_protocol5_full_model_bytes_before_any_disk_write');counter=Counter();pickle.dump(m,counter,protocol=5);free=shutil.disk_usage('/workspace').free
 preflight=dict(state='full_DrCIF500_model_serialized_count_before_disk_write',model_bytes=counter.size,serialized_sha256=counter.h.hexdigest(),required_workspace_bytes=8*counter.size+67108864,workspace_free_bytes=free,synthetic_only=True,full_fit_seconds=elapsed,full_native_fit_completed=True,actual_UWB_values_labels_fit=False)
 (P/'CHECKPOINT_PREFLIGHT.json').write_text(json.dumps(preflight,indent=2)+'\\n')
 if preflight['required_workspace_bytes']>free:
  r=dict(state='full_DrCIF500_storage_policy_closed_before_checkpoint_write',**{k:v for k,v in preflight.items() if k!='state'},resource_gate_passed=False,full_timing_result_available=False,accuracy_measured=False,no_checkpoint_written=True,all500_trees_fitted=True,requested_intervals_per_representation=m._n_intervals)
  (P/'RESOURCE_RESULT.json').write_text(json.dumps(r,indent=2)+'\\n');state('closed_storage_resource_policy');print(json.dumps(r));return
 for i in range(3):
  state('timing_full_native_DrCIF500_save_reload',index=i);path=P/f'synthetic_{i}.pkl';t=time.monotonic()
  with path.open('xb') as f:pickle.dump(m,f,protocol=5)
  saves.append(time.monotonic()-t);sizes.append(path.stat().st_size);t=time.monotonic()
  with path.open('rb') as f:q=pickle.load(f)
  loads.append(time.monotonic()-t);assert_native(q,True);assert statehash(q)==d;other=predict(q,xx);np.testing.assert_array_equal(other['probabilities'],pr['probabilities']);np.testing.assert_array_equal(other['predictions'],pr['predictions']);receipts.append(dict(name=path.name,bytes=path.stat().st_size,sha256=sha(path),all500_CIT_interval_feature_subset_state_exact=True,predictions_probabilities_exact=True,disposable_synthetic_removed_after_check=True));(P/'SYNTHETIC_CHECKPOINT_RECEIPTS.json').write_text(json.dumps(receipts,indent=2)+'\\n');del q;gc.collect();path.unlink()
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;free=shutil.disk_usage('/workspace').free;r=select(p,elapsed,ev,saves,loads,rss,max(sizes),free);r.update(state='complete_full4700_DrCIF500_CIT20_synthetic_cost',full_native_fit_seconds=elapsed,evaluation64_seconds=ev,complete_save_seconds=saves,complete_reload_seconds=loads,peak_RSS_KiB=rss,checkpoint_sizes=sizes,full_trees=500,CIT_thresholds=20,requested_intervals_per_representation=m._n_intervals,loaded_learned_state_probabilities_predictions_exact=True,synthetic_checkpoint_receipts_sha256=sha(P/'SYNTHETIC_CHECKPOINT_RECEIPTS.json'),actual_UWB_values_labels_fit=False)
 (P/'COST_RESULT.json').write_text(json.dumps(r,indent=2)+'\\n');state('complete',resource_gate_passed=r['resource_gate_passed']);print(json.dumps(r))
if __name__=='__main__':main()
''')
put('verify_result.py','''import json
from guard import B,ready,sha
from cost_math import select
def main():
 p=ready();t=json.loads((B/'supervisor01/TERMINATION.json').read_bytes());d=json.loads((B/'profile01/DECLARATION_BEFORE_TIMING.json').read_bytes())
 assert t['exit_code']==0 and not t['wall_timeout'] and not t['resource_stop_reason'] and t['cgroup_oom_kill_delta']==0
 assert d['policy_sha256']==sha(B/'POLICY.json') and d['cost_hashes_sha256']==sha(B/'COST_HASHES.json') and d['full_shape']==[4700,567,90]
 if (B/'profile01/RESOURCE_RESULT.json').exists():
  assert not (B/'profile01/COST_RESULT.json').exists();r=json.loads((B/'profile01/RESOURCE_RESULT.json').read_bytes());pf=json.loads((B/'profile01/CHECKPOINT_PREFLIGHT.json').read_bytes());assert r['no_checkpoint_written'] and r['all500_trees_fitted'] and r['required_workspace_bytes']==8*r['model_bytes']+67108864>r['workspace_free_bytes'] and not r['actual_UWB_values_labels_fit'] and r['model_bytes']==pf['model_bytes'];v=dict(state='verified_full_DrCIF500_storage_policy_closed',resource_gate_passed=False,full_fit_completed=True,full_timing_result_available=False,model_bytes=r['model_bytes'],required_workspace_bytes=r['required_workspace_bytes'],workspace_free_bytes=r['workspace_free_bytes'],no_checkpoint_written=True,accuracy_measured=False,real_UWB_values_labels_fit=False,real_fit_authorized=False)
 else:
  r=json.loads((B/'profile01/COST_RESULT.json').read_bytes());assert r['full_trees']==500 and r['CIT_thresholds']==20 and r['requested_intervals_per_representation']==[79,78,67] and r['loaded_learned_state_probabilities_predictions_exact'] and not r['actual_UWB_values_labels_fit'];assert r['synthetic_checkpoint_receipts_sha256']==sha(B/'profile01/SYNTHETIC_CHECKPOINT_RECEIPTS.json');receipts=json.loads((B/'profile01/SYNTHETIC_CHECKPOINT_RECEIPTS.json').read_bytes());assert len(receipts)==3 and [v['bytes'] for v in receipts]==r['checkpoint_sizes'] and all(v['disposable_synthetic_removed_after_check'] and v['all500_CIT_interval_feature_subset_state_exact'] and v['predictions_probabilities_exact'] for v in receipts)
  v=dict(state='independently_verified_full4700_DrCIF500_CIT20_cost',**select(p,r['full_native_fit_seconds'],r['evaluation64_seconds'],r['complete_save_seconds'],r['complete_reload_seconds'],max(r['peak_RSS_KiB'],t['peak_observed_RSS_KiB']),max(r['checkpoint_sizes']),r['workspace_free_bytes']),real_fit_authorized=False)
 v.update(registration_commit=json.loads((B/'REMOTE_PREREGISTRATION.json').read_bytes())['commit'],next_action='separately_register_real_runner_full_audit_private_backup' if v['resource_gate_passed'] else 'close_exact_DrCIF500_CIT20_resource_configuration_no_rescue');(B/'VERIFIED_COST_RESULT.json').write_text(json.dumps(v,indent=2)+'\\n');print(json.dumps(v))
if __name__=='__main__':main()
''')
for path in B.glob('*.py'):py_compile.compile(str(path),doraise=True)
q=subprocess.run([sys.executable,str(B/'test_cost_math.py')],capture_output=True,text=True);put('COST_TESTS.log',q.stdout+q.stderr);assert q.returncode==0
deps=json.loads((A/'DEPENDENCIES.json').read_bytes())['files']
for path in S.rglob('*'):
 if path.is_file() and '__pycache__' not in path.parts:deps[str(path)]=sha(path)
for name in ['verify_execution_closure_20261102.py','VERIFIED_EXECUTION_CLOSURE_20261102.json','supervisor01/TERMINATION.json']:
 path=A/name;deps[str(path)]=sha(path)
for path,h in deps.items():assert sha(Path(path))==h,path
put('DEPENDENCIES.json',json.dumps(dict(files=deps,scope='Pinned scientific import closure andsix exactnative sources/full500integration/closedQUANTevidence;notwholeOS'),indent=2)+'\n')
names=['prepare.py','POLICY.json','PREREGISTRATION.md','cost_math.py','test_cost_math.py','guard.py','run_screen.py','profile_cost.py','verify_result.py','COST_TESTS.log','DEPENDENCIES.json'];put('COST_HASHES.json',json.dumps({n:sha(B/n) for n in names},indent=2)+'\n')
from guard import ready
try:ready();raise AssertionError('Missingremote marker mustblock')
except FileNotFoundError:pass
put('PRE_ACCESS_GUARD_CHECK.json',json.dumps(dict(missing_remote_marker_blocks=True,frozen_files=len(names),dependencies=len(deps),actual_UWB_values_labels_fit=False,formal_timing_started=False),indent=2)+'\n');print(json.dumps(dict(frozen_files=len(names),dependencies=len(deps),cost_hashes_sha256=sha(B/'COST_HASHES.json'))))
