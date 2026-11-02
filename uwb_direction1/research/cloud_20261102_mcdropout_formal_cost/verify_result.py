from pathlib import Path
import json
from guard import read,sha,ready
from cost_math import select
B=Path(__file__).resolve().parent
def main():
 p=ready();r=read(B/'profile01/COST_CHECK.json');d=read(B/'profile01/DECLARATION_BEFORE_TIMING.json');t=read(B/'supervisor01/TERMINATION.json');m=read(B/'REMOTE_PREREGISTRATION.json')
 assert t['exit_code']==0 and not t['wall_timeout'] and t['supervisor_wall_seconds']<=620 and t['child_cpu_seconds']<=600
 assert d['policy_fixed_before_timing'] and d['policy_sha256']==sha(B/'POLICY.json') and d['profiler_sha256']==sha(B/'profile_cost.py') and d['cost_hashes_sha256']==sha(B/'COST_HASHES.json') and not d['real_UWB_values_labels_or_fit'] and not d['trained_UWB_checkpoint_loaded']
 assert r['policy']==p and r['full_random_TimeMIL850973'] and r['model_BN_buffers_unchanged'] and r['synthetic_fixture_only'] and not r['trained_UWB_checkpoint_loaded'] and not r['real_UWB_values_labels_or_fit'] and not r['real_pilot_started']
 v=select(p,r['forward_seconds'],r['output_save_seconds'],r['state_save_seconds'],r['max_RSS_KiB'])
 for n,x in v.items():assert r[n]==x,n
 result=dict(state='verified_actual_MC100_B64_synthetic_forward_cost',registration_commit=m['commit'],result_sha256=sha(B/'profile01/COST_CHECK.json'),termination_sha256=sha(B/'supervisor01/TERMINATION.json'),policy_sha256=sha(B/'POLICY.json'),profile_completed_successfully=True,resource_gate_passed=v['resource_gate_passed'],estimated_seconds=v['estimated_seconds'],estimated_hours=v['estimated_hours'],max_RSS_KiB=r['max_RSS_KiB'],memory_gate_passed=v['memory_gate_passed'],accuracy_measured=False,real_UWB_values_labels_fit=False,trained_UWB_checkpoint_loaded=False,no_smaller_T_batch_or_rate_fallback=True)
 (B/'VERIFIED_COST_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
