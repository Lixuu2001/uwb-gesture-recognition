"""Exactly one whole frozen/private full8 score; no fitting, tuning or rescue."""
from common import *
from metrics import classification_metrics
def main():
 started=time.monotonic();m=verify_manifest();resource_check()
 assert not (ROOT/'SUMMARY.json').exists() and not (ROOT/'FINALIZE_ONCE.lock').exists()
 release=read(HERE/'SCORING_RELEASE.json')
 assert release['branch_head_verified'] and release['force'] is False and release['source_hashes_sha256']==sha(HERE/'SCORING_SOURCE_HASHES.json')
 assert release['all8_frozen_manifest_sha256']==sha(HERE/'ALL_EIGHT_FROZEN.json') and release['inference_cost_sha256']==sha(HERE/'INFERENCE_COST.json')
 inference=read(HERE/'INFERENCE_COST.json');assert not inference['held_labels_read'] and {(r['held'],r['arm']) for r in inference['rows']}=={(h,a) for h in range(1,9) for a in ARMS}
 for r in inference['rows']:assert r['bitwise_saved_output_verified'] and not r['gesture_truth_read']
 assert sha(m['truth_path'])==m['truth_sha256']
 with np.load(m['cohort_ids'],allow_pickle=False) as z:cohort_ids=z['ids'].copy();users=z['users'].copy()
 assert len(cohort_ids)==len(np.unique(cohort_ids))==9300 and set(users.tolist())==set(range(1,9))
 # All proofs, sources, manifest, code publication and blind inference precede first truth access.
 fd=os.open(ROOT/'FINALIZE_ONCE.lock',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
 try:
  truth=np.load(m['truth_path'],mmap_mode='r',allow_pickle=False);assert truth.shape==(9593,) and truth.dtype==np.int64
  rows=[];audit_ids=[];audit_truth=[];audit_users=[];audit_preds={a:[] for a in ARMS}
  for fold in m['folds']:
   resource_check();h=fold['held'];ids=cohort_ids[users==h];assert len(ids)==fold['test_count'];y=np.array(truth[ids],copy=True);metrics={}
   for arm in ARMS:
    with np.load(fold['arms'][arm]['prediction'],allow_pickle=False) as z:np.testing.assert_array_equal(z['ids'],ids);pred=z['scores'].argmax(1)
    metrics[arm]=classification_metrics(y,pred);audit_preds[arm].append(pred)
   rows.append(dict(held=h,**metrics));audit_ids.append(ids);audit_truth.append(y);audit_users.append(np.full(len(ids),h,dtype=np.int64))
  assert sum(r['native']['n'] for r in rows)==9300
  sums={a:aggregate(rows,a) for a in ARMS};pairs={a:paired(rows,a) for a in ['tail','native_sourceBN']};passed=gate(pairs['tail'],m['gate'])
  private=HERE/'SCORED_SAMPLE_AUDIT.npz';assert not private.exists();np.savez_compressed(private,ids=np.concatenate(audit_ids),truth=np.concatenate(audit_truth),users=np.concatenate(audit_users),**{a:np.concatenate(v) for a,v in audit_preds.items()})
  result=dict(state='complete_exploratory_prospective_full8_native64_fixed_dense_tail33_64_sourceBN',primary_arm='tail',baseline_arm='native',descriptive_control='native_sourceBN',rows=rows,aggregates=sums,paired_vs_native=pairs,predeclared_accuracy_gate=m['gate'],cohort_spending_gate_passed=passed,gate_is_not_definition_of_any_positive_effect=True,finalizer_invocations=1,old_singlefold_once_invocations=0,all8_production_fresh64_audits_private_verified_before_scoring=True,score_release=release,all8_frozen_manifest_sha256=sha(HERE/'ALL_EIGHT_FROZEN.json'),private_sample_audit_sha256=sha(private),inference_cost=inference,stage_costs=stage_costs(),cost_observation_at_score=resource_check(),archive_external_charges=read(ROOT/'EXTERNAL_PHASE_CHARGES.json'),prior_closed_methods_unchanged=True,method_config_seed_selected_on_dev6_before_new7_8_method_choice=True,users7_8_labels_used_for_new_method_selection=False,users7_8_truth_used_in_fixed_outer_train_validation_and_final_reporting=True,all8_historical_exposure_exploratory=True,independent_new_user_validation=False,novel_contribution_validated=False,causal_attribution_validated=False,exact_original_SWA_protocol_reproduced=False,paired_baseline_budget='same native617612/three radar/RMS/B64/64/seed20261029 shared passive-averaging trajectory; standalone baseline wall/CPU not separately measured',evaluation_wall_seconds=time.monotonic()-started)
  write(ROOT/'SUMMARY.json',result);write(ROOT/'FINAL_RESULT_STATE.json',dict(state='full8_fixed_tail_accuracy_gate_passed_exploratory' if passed else 'close_exact_prospective_full8_fixed_tail_native64_seed20261029_gate_failed_no_rescue',summary_sha256=sha(ROOT/'SUMMARY.json'),finalizer_consumed=True,no_automatic_test_feedback_or_new_fits=True))
  print(json.dumps(dict(aggregates=sums,primary_pair=pairs['tail'],gate_passed=passed)))
 except BaseException as exc:
  write(HERE/'SCORING_FAILURE.json',dict(exception_type=type(exc).__name__,message=str(exc),once_lock_retained=True,automatic_retry_forbidden=True));raise
if __name__=='__main__':main()
