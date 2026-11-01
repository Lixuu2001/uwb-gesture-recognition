"""Verify preregistered complete TimeCNN2000/B16 timing; no resource-policy rescue."""
from pathlib import Path
import json
from guard import read,sha
from cost_math import select
B=Path(__file__).resolve().parent
def main():
 m=read(B/'REMOTE_PREREGISTRATION.json');hs=read(B/'COST_HASHES.json');assert m['state']=='remote_cost_preregistration_exact_readback_verified' and m['cost_hashes_sha256']==sha(B/'COST_HASHES.json')
 for n,d in hs.items():assert sha(B/n)==d,n
 p=read(B/'POLICY.json');r=read(B/'profile01/COST_CHECK.json');d=read(B/'profile01/DECLARATION_BEFORE_TIMING.json');t=read(B/'supervisor01/TERMINATION.json')
 assert d['policy_fixed_before_timing'] and not d['real_UWB_values_labels_or_fit'] and d['policy_sha256']==sha(B/'POLICY.json') and d['profiler_sha256']==hs['profile_cost.py'] and d['cost_hashes_sha256']==sha(B/'COST_HASHES.json')
 assert (B/'profile01/POLICY.json').read_bytes()==(B/'POLICY.json').read_bytes() and r['policy']==p
 assert t['exit_code']==0 and not t['wall_timeout'] and 0<t['supervisor_wall_seconds']<=620 and 0<t['child_cpu_seconds']<=600
 assert r['full_review_TimeCNN_Conv6_12_kernel7_avgpool3_sigmoid_MSE'] and (r['total_parameters'],r['trainable_parameters'],r['trainable_tensors'],r['Adam_variables'])==(25356,25356,6,14)
 assert r['training_unweighted'] and r['validation_unweighted'] and r['synthetic_fixture_only'] and not r['real_UWB_values_labels_or_fit'] and not r['real_pilot_started']
 v=select(p,r['training_seconds'],r['evaluation_seconds'],r['native_save_seconds'],r['fullstate_save_seconds'],r['max_RSS_KiB'])
 for n,x in v.items():assert r[n]==x,n
 result={'state':'verified_actual_full_TimeCNN2000_B16_synthetic_cost','registration_commit':m['commit'],'policy_sha256':sha(B/'POLICY.json'),'result_sha256':sha(B/'profile01/COST_CHECK.json'),'termination_sha256':sha(B/'supervisor01/TERMINATION.json'),'profile_completed_successfully':True,'resource_gate_passed':v['resource_gate_passed'],'estimated_seconds':v['estimated_seconds'],'estimated_hours':v['estimated_hours'],'memory_gate_passed':v['memory_gate_passed'],'max_RSS_KiB':r['max_RSS_KiB'],'budget_seconds':43200,'accuracy_measured':False,'real_UWB_values_labels_fit':False,'no_smaller_epoch_batch_fallback':True,'next_action':'separately_register_real_UWB_runner_protocol_and_full_auditor' if v['resource_gate_passed'] else 'close_exact_full2000_B16_resource_policy_without_accuracy_negative'}
 (B/'VERIFIED_COST_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
