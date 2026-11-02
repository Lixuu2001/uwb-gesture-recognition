from pathlib import Path
import json
from guard import ready,read,sha
from cost_math import select
B=Path(__file__).resolve().parent
def main():
 p=ready();r=read(B/'profile01/COST_CHECK.json');t=read(B/'supervisor01/TERMINATION.json');d=read(B/'profile01/DECLARATION_BEFORE_TIMING.json');m=read(B/'REMOTE_PREREGISTRATION.json')
 assert t['exit_code']==0 and not t['wall_timeout'] and t['supervisor_wall_seconds']<=620 and t['child_cpu_seconds']<=600
 assert d['policy_sha256']==sha(B/'POLICY.json') and d['profiler_sha256']==sha(B/'profile_cost.py') and d['cost_hashes_sha256']==sha(B/'COST_HASHES.json') and d['policy_fixed_before_timing'] and not d['real_UWB_values_labels_or_fit']
 assert r['policy']==p and r['full_TSLib_DLinear_classification_CLI'] and r['synthetic_fixture_only'] and not r['real_UWB_values_labels_or_fit'] and not r['real_pilot_started']
 v=select(p,r['training_seconds'],r['evaluation_seconds'],r['native_save_seconds'],r['fullstate_save_seconds'],r['max_RSS_KiB'])
 for n,x in v.items():assert r[n]==x,n
 result=dict(state='verified_actual_full_DLinear10_B32_synthetic_cost',registration_commit=m['commit'],result_sha256=sha(B/'profile01/COST_CHECK.json'),termination_sha256=sha(B/'supervisor01/TERMINATION.json'),policy_sha256=sha(B/'POLICY.json'),profile_completed_successfully=True,resource_gate_passed=v['resource_gate_passed'],estimated_seconds=v['estimated_seconds'],estimated_hours=v['estimated_hours'],max_RSS_KiB=r['max_RSS_KiB'],memory_gate_passed=v['memory_gate_passed'],accuracy_measured=False,real_UWB_values_labels_fit=False,no_smaller_epoch_batch_fallback=True)
 (B/'VERIFIED_COST_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
