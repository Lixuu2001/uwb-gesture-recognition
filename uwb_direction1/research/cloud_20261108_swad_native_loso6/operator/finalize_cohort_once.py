"""One complete frozen/private cohort score for primary SWAD and predeclared controls."""
from pathlib import Path
import os,sys,json,time
P=Path('/workspace/uwb-swad-loso6-20261108');sys.path.insert(0,str(P/'shared'))
from common import read,write,sha,ready
from stream_archive import verify
from controller import identity
from loso_roles_metrics import classification_metrics,paired_sixfold

def main():
 ready(True);start=time.monotonic()
 assert not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists()
 release=read(P/'FULL_PRIVATE_RELEASE.json');assert release['branch_head_verified'] and release['force'] is False and release['once_scorer_sha256']==sha(__file__)
 old=read(P/'PHASE_IDENTITY.json');now=identity(old['pid']);assert now is None or now['state']=='Z' or any(now[k]!=old[k] for k in ['startticks','boot_id'])
 assert read(P/'STATE.json')['state']=='all_six_method_folds_frozen_unscored_requires_complete_private_backup_before_cohort_once'
 frozen=read(P/'ALL_SIX_PREDICTIONS_FROZEN.json');assert frozen['all4_arms_frozen']
 for row in frozen['files'].values():assert sha(row['path'])==row['sha256']
 d=P/'full_backup';m=read(d/'MANIFEST.json');v=read(d/'READBACK_VERIFIED.json');assert verify(m,read(d/'READBACK_INPUT.json'))==v and v['complete_modelset_verified']
 assert m['archive_sha256']==release['complete_six_archive_sha256']
 plan=read(P/'DEVELOPMENT_PROTOCOL.json');cfg=read(P/'operator/INFERENCE_PROTOCOL.json')
 timing=read(P/'INFERENCE_COST.json');assert not timing['held_labels_read'] and len(timing['rows'])==24 and {(r['held'],r['arm']) for r in timing['rows']}=={(h,a) for h in range(1,7) for a in cfg['arms']}
 for r in cfg['baseline_artifacts'].values():assert sha(r['model'])==r['model_sha256'] and sha(r['prediction'])==r['prediction_sha256']
 fd=os.open(P/'FINALIZE_ONCE.lock',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
 import numpy as np
 with np.load(plan['source_ids']) as z:ids=z['ids'];users=z['users']
 truth=np.load(Path(plan['data'])/'cache/y_clutterremoved.npy',mmap_mode='r');by_arm={a:[] for a in ['swad','tail','bn_only']};allrows=[]
 original=read('/workspace/uwb-cloud-current/uwb_direction1/results/cloud_20261030_gap_source6/SUMMARY.json')
 for h in range(1,7):
  ix=np.flatnonzero(users==h);y=np.array(truth[ids[ix]],copy=True).astype(int)
  bp=cfg['baseline_artifacts'][str(h)]['prediction']
  with np.load(bp) as z:np.testing.assert_array_equal(z['ids'],ids[ix]);baseline=classification_metrics(y,z['logits'].argmax(1))
  assert abs(baseline['present_class_macro_recall']-original['rows'][h-1]['gap'])<=1e-12
  metrics={}
  for arm in by_arm:
   mp=frozen['files'][f'held{h}/{arm}/diagnostic_unscored.npz']['path']
   with np.load(mp) as z:np.testing.assert_array_equal(z['ids'],ids[ix]);metric=classification_metrics(y,z['scores'].argmax(1))
   metrics[arm]=metric;by_arm[arm].append(dict(held=h,baseline=baseline,method=metric))
  allrows.append(dict(held=h,baseline=baseline,**metrics))
 result=paired_sixfold(by_arm['swad']);gate=plan['bounded_LOSO_continuation_gate'];tol=gate['tolerance']
 passed=result['mean_paired_macro_recall_delta']+tol>=gate['mean_paired_macro_recall_gain_min'] and result['strict_user_wins']>=gate['strict_wins_min'] and result['worst_paired_macro_recall_delta']+tol>=gate['worst_paired_macro_recall_delta_min']
 descriptive={a:paired_sixfold(by_arm[a]) for a in ['tail','bn_only']}
 matched={a:paired_sixfold([dict(held=r['held'],baseline=r[a],method=r['swad']) for r in allrows]) for a in ['tail','bn_only']}
 result.update(state='complete_exploratory_SWAD_LOSO6_native64_dense_default3_6_0p3_sourceBN',rows=by_arm['swad'],all_predeclared_arm_rows=allrows,descriptive_control_vs_original_GAP=descriptive,descriptive_SWAD_vs_matched_controls=matched,primary_arm='swad',cohort_spending_gate_passed=bool(passed),gate_is_not_definition_of_any_positive_effect=True,route='plan_separately_preregistered_same_condition_outer8_after_matched_control_analysis' if passed else 'close_exact_SWAD_LOSO6_default3_6_0p3_sourceBN_nativeGAP64_seed20261029_no_rescue',finalizer_invocations=1,legacy_singlefold_once_invocations=0,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,causal_attribution_validated=False,no_relabeling_control_as_primary=True,remaining_phase_cost=read(P/'STATE.json'),last_resource_observation=read(P/'RESOURCE_LEDGER.json'),evaluation_wall_seconds=time.monotonic()-start)
 result['prior_closed_method_costs']={name:dict(summary_sha256=sha(Path(path)/'SUMMARY.json'),terminal_state=read(Path(path)/'FINAL_RESULT_STATE.json'),main_phase=read(Path(path)/'STATE.json')) for name,path in [('MMD','/workspace/uwb-mmd-loso6-20261106'),('VREx','/workspace/uwb-vrex-loso6-20261106'),('Fishr','/workspace/uwb-fishr-native-loso6-20261107'),('IRM','/workspace/uwb-irm-loso6-20261107'),('GroupDRO','/workspace/uwb-groupdro-loso6-20261108')]}
 result['fold_stage_costs']={str(h):{k:read(P/f'folds/held{h}/run/execution/{k}/TERMINATION.json')['cumulative'] for k in ['train','replay','output']} for h in range(1,7)}
 result['inference_cost']=timing
 write(P/'SUMMARY.json',result);write(P/'FINAL_RESULT_STATE.json',dict(state=result['route'],finalizer_consumed=True,summary_sha256=sha(P/'SUMMARY.json'),no_automatic_test_feedback_or_new_fits=True));print(json.dumps(result))
if __name__=='__main__':main()
