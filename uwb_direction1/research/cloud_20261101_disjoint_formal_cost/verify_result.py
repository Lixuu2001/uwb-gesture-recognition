"""Verify actual successful synthetic profile, including resource-failure output."""
from pathlib import Path
import hashlib,json,math,sys
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B))
from cost_tools import select
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 freeze=read(B/'COST_HASHES.json')
 for name,digest in freeze.items():assert sha(B/name)==digest
 marker=read(B/'REMOTE_PREREGISTRATION.json');assert marker['state']=='remote_cost_preregistration_exact_readback_verified' and marker['cost_hashes_sha256']==sha(B/'COST_HASHES.json')
 p=read(B/'POLICY.json');r=read(B/'profile01/COST_CHECK.json');d=read(B/'profile01/DECLARATION_BEFORE_TIMING.json');t=read(B/'supervisor01/TERMINATION.json')
 assert (B/'profile01/POLICY.json').read_bytes()==(B/'POLICY.json').read_bytes()
 assert d['policy_sha256']==sha(B/'POLICY.json') and d['utility_sha256']==freeze['profile_cost.py'] and d['policy_fixed_before_timing'] and not d['real_UWB_values_labels_or_fit']
 assert t['exit_code']==0 and not t['wall_timeout'] and 0<t['supervisor_wall_seconds']<=620 and 0<t['child_cpu_seconds']<=600
 assert r['policy']==p and r['full_author_default_DCNN2L_architecture'] and r['total_parameters']==2587468 and r['trainable_parameters']==2586956
 for name,length in [('training_seconds',8),('evaluation_seconds',4),('keras_save_seconds',3),('snapshot_save_seconds',3)]:assert len(r[name])==length
 for name in ['original_training_classweight_formula','validation_unweighted','synthetic_fixture_only','preprocessing_choice_recorded_but_no_UWB_preprocessing_executed']:assert r[name]
 assert not r['real_UWB_values_labels_or_fit'] and not r['real_pilot_started']
 decision=select(p,training_seconds=r['training_seconds'],evaluation_seconds=r['evaluation_seconds'],keras_save_seconds=r['keras_save_seconds'],snapshot_save_seconds=r['snapshot_save_seconds'],max_RSS_KiB=r['max_RSS_KiB'])
 for k,v in decision.items():assert r[k]==v
 assert p['epoch_candidates']==[500] and p['batch_size']==8 and p['seed']==20261029 and decision['chosen_epochs'] is None and decision['state']=='resource_closed_exact_policy'
 report={'state':'verified_actual_synthetic_profile_resource_closed_exact_full500_B8_policy','registration_commit':marker['commit'],'policy_sha256':sha(B/'POLICY.json'),'profiler_sha256':freeze['profile_cost.py'],'result_sha256':sha(B/'profile01/COST_CHECK.json'),'declaration_sha256':sha(B/'profile01/DECLARATION_BEFORE_TIMING.json'),'termination_sha256':sha(B/'supervisor01/TERMINATION.json'),'supervisor_exit_code':0,'timed_profile_completed_successfully':True,'prospective_full_run_resource_gate_passed':False,'full500_conservative_estimate_seconds':decision['estimated_seconds_by_epoch_budget']['500'],'full500_conservative_estimate_hours':decision['estimated_seconds_by_epoch_budget']['500']/3600,'declared_budget_seconds':43200,'RSS_KiB':r['max_RSS_KiB'],'memory_gate_passed':decision['memory_gate_passed'],'configuration_closed_without_smaller_epoch_fallback':True,'accuracy_measured':False,'real_UWB_values_labels_fit':False,'scientific_negative_about_accuracy':False,'automatic_retry':False}
 (B/'VERIFIED_COST_RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
