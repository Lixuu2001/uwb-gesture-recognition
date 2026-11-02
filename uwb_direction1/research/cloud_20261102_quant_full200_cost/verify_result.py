import json
from guard import B,ready,sha
from cost_math import select
def main():
 p=ready();r=json.loads((B/'profile01/COST_RESULT.json').read_bytes());t=json.loads((B/'supervisor01/TERMINATION.json').read_bytes());d=json.loads((B/'profile01/DECLARATION_BEFORE_TIMING.json').read_bytes())
 assert t['exit_code']==0 and not t['wall_timeout'] and not t['resource_stop_reason'] and t['cgroup_oom_kill_delta']==0
 assert d['policy_sha256']==sha(B/'POLICY.json') and d['cost_hashes_sha256']==sha(B/'COST_HASHES.json') and d['full_shape']==[4700,567,90]
 assert r['full_trees']==200 and r['full_features']==539217 and r['loaded_tree_interval_arrays_exact'] and r['loaded_predictions_exact'] and r['loaded_probabilities_predeclared_tolerance'] and not r['actual_UWB_values_labels_fit']
 assert r['synthetic_checkpoint_receipts_sha256']==sha(B/'profile01/SYNTHETIC_CHECKPOINT_RECEIPTS.json');receipts=json.loads((B/'profile01/SYNTHETIC_CHECKPOINT_RECEIPTS.json').read_bytes());assert len(receipts)==3 and [v['bytes'] for v in receipts]==r['checkpoint_sizes'] and all(v['disposable_synthetic_file_removed_after_check'] and v['all_tree_interval_arrays_exact'] for v in receipts)
 v=select(p,r['full_native_fit_seconds'],r['evaluation64_seconds'],r['complete_save_seconds'],r['complete_reload_seconds'],max(r['peak_RSS_KiB'],t['peak_observed_RSS_KiB']),max(r['checkpoint_sizes']),r['workspace_free_bytes']);result=dict(state='independently_verified_full4700_QUANT539217_ExtraTrees200_cost',**v,registration_commit=json.loads((B/'REMOTE_PREREGISTRATION.json').read_bytes())['commit'],real_fit_authorized=False,next_action='separately_register_real_runner_full_audit_private_backup' if v['resource_gate_passed'] else 'close_exact_full_QUANT_resource_configuration_no_rescue')
 (B/'VERIFIED_COST_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
