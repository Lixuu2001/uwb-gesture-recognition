import json
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
 v.update(registration_commit=json.loads((B/'REMOTE_PREREGISTRATION.json').read_bytes())['commit'],next_action='separately_register_real_runner_full_audit_private_backup' if v['resource_gate_passed'] else 'close_exact_DrCIF500_CIT20_resource_configuration_no_rescue');(B/'VERIFIED_COST_RESULT.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
if __name__=='__main__':main()
