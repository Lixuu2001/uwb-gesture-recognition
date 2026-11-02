import json
from guard import B,ready,sha
from cost_math import select
def main():
    p=ready();r=json.loads((B/'profile01/COST_RESULT.json').read_bytes());t=json.loads((B/'supervisor01/TERMINATION.json').read_bytes());d=json.loads((B/'profile01/DECLARATION_BEFORE_TIMING.json').read_bytes())
    assert t['exit_code']==0 and not t['wall_timeout'] and not t['resource_stop_reason'] and t['cgroup_oom_kill_delta']==0
    assert d['policy_sha256']==sha(B/'POLICY.json') and d['cost_hashes_sha256']==sha(B/'COST_HASHES.json') and d['full_shape']==[4700,567,90]
    assert r['full_native_members']==25 and r['kernels_per_member']==2000 and r['loaded_probabilities_exact'] and not r['actual_UWB_values_labels_fit']
    for i,n in enumerate(r['checkpoint_sizes']):assert (B/f'profile01/full25_synthetic_{i}.pkl').stat().st_size==n
    peak=max(r['peak_RSS_KiB'],t['peak_observed_RSS_KiB']);v=select(p,r['full_native_fit_seconds'],r['evaluation64_seconds'],r['complete_save_seconds'],r['complete_reload_seconds'],peak,max(r['checkpoint_sizes']),r['workspace_free_bytes'])
    result=dict(state='independently_verified_full4700_native_Arsenal25x2000_cost',**v,registration_commit=json.loads((B/'REMOTE_PREREGISTRATION.json').read_bytes())['commit'],profile_peak_RSS_KiB=r['peak_RSS_KiB'],supervisor_peak_RSS_KiB=t['peak_observed_RSS_KiB'],real_fit_authorized=False,next_action='separately_register_fixed_real_pilot_runner_full_auditor_private_preservation' if v['resource_gate_passed'] else 'close_exact_Arsenal25x2000_native_default_configuration_no_rescue')
    (B/'VERIFIED_COST_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
