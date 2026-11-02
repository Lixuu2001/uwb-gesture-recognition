from pathlib import Path
import json,hashlib,subprocess,sys
B=Path(__file__).resolve().parent;S=Path('/workspace/uwb-quant-author-probe-20261102');A=Path('/workspace/uwb-arsenal-parallel-full25-cost-20261102')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(n,s):(B/n).write_text(s)
assert not (B/'COST_HASHES.json').exists() and not (B/'profile01').exists()
closed=json.loads(Path('/workspace/uwb-arsenal-parallel-pilot-20261102/run01/SUMMARY.json').read_bytes());assert not closed['predeclared_gate_passed'] and closed['complete_private_modelset_verified']
p=dict(method='complete_current_author_QUANT_depth6_div4_ExtraTrees200_multivariate_transfer',author_commit='f3907f7e531b6a304e3c54310869a0023a9e0fd6',depth=6,div=4,trees=200,max_features=.1,criterion='entropy',seed=20261029,threads=4,synthetic_train_size=4700,channels=567,timepoints=90,features=539217,class_weight=None,Torch_threads=1,feature_batch_size=32,feature_matrix='all539217_dense_float32_Fortran_RAM',extra_normalization=None,profile_wall_seconds=3600,profile_CPU_soft_seconds=14380,profile_CPU_hard_seconds=14400,RSS_stop_KiB=12582912,map_margin_KiB=2097152,memory_limit_KiB=14680064,budget_seconds=43200,CPU_budget_seconds=172800,probability_reload_rtol=1e-14,probability_reload_atol=1e-14,full_native_end_to_end_fit=True,formula='3*full4700fit+2*2200*maxeval64/64+4*maxsave+2*maxload+180',checkpoint_storage='8*largest_model+64MiB<=remainingdisk',actual_UWB_values_labels_fit_authorized=False,eventual_source_train_users=[3,4,5,6],eventual_validation_user=2,eventual_diagnostic_user=1,fixed_reference=.9866666666666667,minimum_gain=.005,tolerance=1e-12,users7_8_labels_forbidden=True,all_exploratory=True,no_feature_tree_depth_seed_precision_view_or_role_rescue=True,novelty_claim=False)
put('POLICY.json',json.dumps(p,indent=2)+'\n')
put('PREREGISTRATION.md','''# QUANT complete author configuration: synthetic resource screen

Arsenal25x2000/jobs4 once-score closed at diagnostic .9675000000000001, validation .943 versus fixed .9866666666666667/+ .005. Full numerical audit and private four-chunk/992-member recovery passed before labels. No rescue, re-score or additional held-label query. This next method is chosen from literature/source and existing synthetic integration evidence, not a held-label parameter grid.

Dempster/Schmidt/Webb2024 QUANT, DMKD38:2377–2402, DOI10.1007/s10618-024-01036-9. Primary methods3.1–3.4/evaluation and complete author code/run.py reviewed. Author angus924/quant commit f3907f7e531b6a304e3c54310869a0023a9e0fd6/GPL3 remains unchanged. Current author code explicitly supports multivariate series, whereas paper experiments were univariate: this is a modern multivariate transfer of a published baseline, not author-paper experimental parity or original contribution. Full depth6/div4 four representations and shifted/unshifted intervals yield951 features/channel,539217 total. Complete200 ExtraTrees/max_features.1/entropy. Author driver has no explicit seed/n_jobs−1; fixed20261029/jobs4 is a disclosed project execution choice. No extra normalizer. Use signed per-record/view RMS source maps if future real pilot is separately authorized. No dataset values/labels are accessed by this screen.

Original permute/view suspicion was not reproduced: unchanged author transform passes all full-width567 features against independent FP64 interval/diff/FFT/quantile reference. Feature batching and Fortran RAM storage preserve all features. Synthetic24x3x90 integration kept full200trees and showed exact native-vs-batched features, all tree arrays C-vs-F, probabilities/predictions and cloudpickle roundtrip. These fixtures are not full4700 throughput or all-input proof. Missing standalone cloudpickle import was fixed using installed joblib.externals.cloudpickle before any author transform; original failed source/log retained. Original GPL files and license are local/private; repository holds pinned retrieval hashes and new compatible wrappers.

After every changed file has exact GitHub readback and branch-head verification: one full4700x567x90 synthetic fit, all539217 features and all200trees. Deterministic row-addressable normal fixtures avoid allocating the entire raw input; full dense float32 Fortran feature matrix remains RAM-resident. This is not smallN extrapolation or feature subsampling. Feature generation time is included in full fit. Three eval64 measurements, three full model cloudpickle save/reloads, all fitted interval/tree arrays exact plus class predictions exact and probabilities rtol=atol1e-14 (predeclared summation-order tolerance). Synthetic checkpoint files may be removed immediately after recorded SHA/size and successful exact reload so three disposable copies do not consume scarce disk; they are not UWB checkpoints. Real study, if cost passes, must retain models/IDs/predictions and full independent audit/private recovery before held score.

One profile only,3600wall/14380soft14400hardCPU/12GiB childRSS, .2s monitoring/cgroupOOM. Conservative3*fullfit+2*2200*maxeval64/64+4*maxsave+2*maxload+180<=12h,4*estimate<=48CPUh,observed peak+2GiB<=14GiB,8fullmodels+64MiB<=free. Resource failure/incomplete screen is not accuracy evidence or a family rejection. No feature/tree/depth/seed/scaler/view/precision rescue within exact configuration. No automated retry. Cost pass alone never starts real training.

Future separately preregistered fixed source protocol: users3–6/4700 train,user2/1000 validation,user1/1200 diagnostic,canonical IDs and actual artifacts retained. No test labels for training/tuning. Supported macro only classes0..9 on val2,0..11 on diag1; fixed TimeMIL reference .9866666666666667/min gain.005/tol1e-12. Only after full independent numeric audit/private archive recovery permit once-score. Fail closes exact configuration,pass only separatelyregisteredsource6confirmation. Users7–8 labels forbidden; historical eight-user development is exploratory,not independent validation/8foldgain/originality. Existing data/789source snapshot/userREADME/current visibility/Drive owner-only/automation remain. Cloud only, no Windows.
''')
s=(A/'cost_math.py').read_text().replace("(p['members'],p['kernels_per_member'],p['seed'],p['threads'],p['synthetic_train_size'])==(25,2000,20261029,4,4700)","(p['trees'],p['features'],p['depth'],p['div'],p['max_features'],p['criterion'],p['seed'],p['threads'],p['synthetic_train_size'])==(200,539217,6,4,.1,'entropy',20261029,4,4700)").replace('no_member_kernel_seed_normalization_alpha_weighting_fallback','no_feature_tree_depth_seed_precision_view_or_role_rescue');put('cost_math.py',s)
s=(A/'test_cost_math.py').read_text().replace('P=dict(members=25,kernels_per_member=2000,seed=20261029,threads=4,synthetic_train_size=4700)',"P=dict(trees=200,features=539217,depth=6,div=4,max_features=.1,criterion='entropy',seed=20261029,threads=4,synthetic_train_size=4700)").replace('dict(P,members=2)','dict(P,trees=2)');put('test_cost_math.py',s)
s=(A/'guard.py').read_text();a=s.index('    previous=json.loads(');z=s.index('    return json.loads',a);s=s[:a]+s[z:];put('guard.py',s)
s=(A/'run_screen.py').read_text().replace('raise SystemExit(0 if code==0 else 1)','raise SystemExit(0 if code==0 and reason is None and oom_delta==0 else 1)');put('run_screen.py',s)
put('profile_cost.py','''from pathlib import Path
import sys,time,json,resource,shutil,hashlib,gc
from guard import B,ready,sha
from cost_math import select
def main():
 p=ready();P=B/'profile01';P.mkdir(exist_ok=False)
 (P/'DECLARATION_BEFORE_TIMING.json').write_text(json.dumps(dict(policy_sha256=sha(B/'POLICY.json'),cost_hashes_sha256=sha(B/'COST_HASHES.json'),synthetic_only=True,actual_UWB_values_labels_fit=False,full_shape=[4700,567,90]),indent=2)+'\\n')
 sys.path.insert(0,'/workspace/uwb-quant-author-probe-20261102')
 from native_quant import np,make,fit,predict,scientific_arrays,cloudpickle
 class Fixture:
  def __init__(self,n):self.shape=(n,567,90)
  def __len__(self):return self.shape[0]
  def __getitem__(self,key):
   assert isinstance(key,slice);start,stop,step=key.indices(len(self));assert step==1
   return np.stack([np.random.default_rng(20261029+i).standard_normal((567,90),dtype=np.float32) for i in range(start,stop)])
 def state(s,**kw):(P/'STATE.json').write_text(json.dumps(dict(state=s,actual_UWB_values_labels_fit=False,**kw),indent=2)+'\\n')
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
  receipts.append(dict(name=path.name,bytes=path.stat().st_size,sha256=sha(path),all_tree_interval_arrays_exact=True,predictions_exact=True,probabilities_predeclared_tolerance=True,disposable_synthetic_file_removed_after_check=True));(P/'SYNTHETIC_CHECKPOINT_RECEIPTS.json').write_text(json.dumps(receipts,indent=2)+'\\n');del q;gc.collect();path.unlink()
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;free=shutil.disk_usage('/workspace').free;r=select(p,elapsed,ev,saves,loads,rss,max(sizes),free);r.update(state='complete_full4700_QUANT539217_ExtraTrees200_synthetic_cost',full_native_fit_seconds=elapsed,evaluation64_seconds=ev,complete_save_seconds=saves,complete_reload_seconds=loads,peak_RSS_KiB=rss,checkpoint_sizes=sizes,full_trees=200,full_features=539217,loaded_tree_interval_arrays_exact=True,loaded_predictions_exact=True,loaded_probabilities_predeclared_tolerance=True,synthetic_checkpoint_receipts_sha256=sha(P/'SYNTHETIC_CHECKPOINT_RECEIPTS.json'),actual_UWB_values_labels_fit=False)
 (P/'COST_RESULT.json').write_text(json.dumps(r,indent=2)+'\\n');state('complete',resource_gate_passed=r['resource_gate_passed']);print(json.dumps(r))
if __name__=='__main__':main()
''')
put('verify_result.py','''import json
from guard import B,ready,sha
from cost_math import select
def main():
 p=ready();r=json.loads((B/'profile01/COST_RESULT.json').read_bytes());t=json.loads((B/'supervisor01/TERMINATION.json').read_bytes());d=json.loads((B/'profile01/DECLARATION_BEFORE_TIMING.json').read_bytes())
 assert t['exit_code']==0 and not t['wall_timeout'] and not t['resource_stop_reason'] and t['cgroup_oom_kill_delta']==0
 assert d['policy_sha256']==sha(B/'POLICY.json') and d['cost_hashes_sha256']==sha(B/'COST_HASHES.json') and d['full_shape']==[4700,567,90]
 assert r['full_trees']==200 and r['full_features']==539217 and r['loaded_tree_interval_arrays_exact'] and r['loaded_predictions_exact'] and r['loaded_probabilities_predeclared_tolerance'] and not r['actual_UWB_values_labels_fit']
 assert r['synthetic_checkpoint_receipts_sha256']==sha(B/'profile01/SYNTHETIC_CHECKPOINT_RECEIPTS.json');receipts=json.loads((B/'profile01/SYNTHETIC_CHECKPOINT_RECEIPTS.json').read_bytes());assert len(receipts)==3 and [v['bytes'] for v in receipts]==r['checkpoint_sizes'] and all(v['disposable_synthetic_file_removed_after_check'] and v['all_tree_interval_arrays_exact'] for v in receipts)
 v=select(p,r['full_native_fit_seconds'],r['evaluation64_seconds'],r['complete_save_seconds'],r['complete_reload_seconds'],max(r['peak_RSS_KiB'],t['peak_observed_RSS_KiB']),max(r['checkpoint_sizes']),r['workspace_free_bytes']);result=dict(state='independently_verified_full4700_QUANT539217_ExtraTrees200_cost',**v,registration_commit=json.loads((B/'REMOTE_PREREGISTRATION.json').read_bytes())['commit'],real_fit_authorized=False,next_action='separately_register_real_runner_full_audit_private_backup' if v['resource_gate_passed'] else 'close_exact_full_QUANT_resource_configuration_no_rescue')
 (B/'VERIFIED_COST_RESULT.json').write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result))
if __name__=='__main__':main()
''')
q=subprocess.run([sys.executable,str(B/'test_cost_math.py')],capture_output=True,text=True);put('COST_TESTS.log',q.stdout+q.stderr);assert q.returncode==0
deps=json.loads(Path('/workspace/uwb-arsenal-parallel-pilot-20261102/DEPENDENCIES.json').read_bytes())['files']
for path in S.rglob('*'):
 if path.is_file() and '__pycache__' not in path.parts:deps[str(path)]=sha(path)
for n in ['run01/SUMMARY.json','COMPLETE_AUDIT.json','complete_backup/DRIVE_BACKUP_RECEIPT.json','complete_backup/READBACK_VERIFIED.json']:
 path=Path('/workspace/uwb-arsenal-parallel-pilot-20261102')/n;deps[str(path)]=sha(path)
for path,h in deps.items():assert sha(Path(path))==h,path
put('DEPENDENCIES.json',json.dumps(dict(files=deps,scope='Pinned scientific import closure,full author source/license/integration evidence,previous exact study closure;not whole OS'),indent=2)+'\n')
names=['prepare.py','POLICY.json','PREREGISTRATION.md','cost_math.py','test_cost_math.py','guard.py','run_screen.py','profile_cost.py','verify_result.py','COST_TESTS.log','DEPENDENCIES.json']
put('COST_HASHES.json',json.dumps({n:sha(B/n) for n in names},indent=2)+'\n')
from guard import ready
try:ready();raise AssertionError('Missingremote marker mustblock')
except FileNotFoundError:pass
put('PRE_ACCESS_GUARD_CHECK.json',json.dumps(dict(missing_remote_marker_blocks=True,frozen_files=len(names),dependencies=len(deps),actual_UWB_values_labels_fit=False,formal_timing_started=False),indent=2)+'\n')
print(json.dumps(dict(frozen_files=len(names),dependencies=len(deps),cost_hashes_sha256=sha(B/'COST_HASHES.json'))))
