"""Only complete/frozen/private sixfold cohort scoring; no individual entry decision."""
from pathlib import Path
import os,sys,json,time
P=Path('/workspace/uwb-irm-loso6-20261107');sys.path.insert(0,str(P/'shared'))
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
    frozen=read(P/'ALL_SIX_PREDICTIONS_FROZEN.json')
    for row in frozen['files'].values():assert sha(row['path'])==row['sha256']
    d=P/'full_backup';m=read(d/'MANIFEST.json');v=read(d/'READBACK_VERIFIED.json');assert verify(m,read(d/'READBACK_INPUT.json'))==v and v['complete_modelset_verified']
    assert m['archive_sha256']==release['complete_six_archive_sha256']
    baseline=read(P/'BASELINE_ARTIFACTS.json');plan=read(P/'DEVELOPMENT_PROTOCOL.json')
    timing=read(P/'INFERENCE_COST.json');assert not timing['held_labels_read'] and len(timing['rows'])==12
    for r in read(P/'operator/INFERENCE_PROTOCOL.json')['baseline_artifacts'].values():assert sha(r['model'])==r['model_sha256'] and sha(r['prediction'])==r['prediction_sha256']
    fd=os.open(P/'FINALIZE_ONCE.lock',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
    import numpy as np
    with np.load(plan['source_ids']) as z:ids=z['ids'];users=z['users']
    truth=np.load(Path(plan['data'])/'cache/y_clutterremoved.npy',mmap_mode='r');rows=[]
    original=read('/workspace/uwb-cloud-current/uwb_direction1/results/cloud_20261030_gap_source6/SUMMARY.json')
    for h in range(1,7):
        ix=np.flatnonzero(users==h);y=np.array(truth[ids[ix]],copy=True).astype(int)
        mp=P/f'folds/held{h}/run/train/diagnostic_unscored.npz'
        bp=Path(baseline['held1'] if h==1 else baseline['restored_folds'][str(h)])/'predictions_unscored.npz'
        with np.load(mp) as z:np.testing.assert_array_equal(z['ids'],ids[ix]);method=classification_metrics(y,z['scores'].argmax(1))
        with np.load(bp) as z:np.testing.assert_array_equal(z['ids'],ids[ix]);control=classification_metrics(y,z['logits'].argmax(1))
        assert abs(control['present_class_macro_recall']-original['rows'][h-1]['gap'])<=1e-12
        rows.append(dict(held=h,baseline=control,method=method))
    result=paired_sixfold(rows);gate=plan['bounded_LOSO_continuation_gate'];tol=gate['tolerance']
    passed=result['mean_paired_macro_recall_delta']+tol>=gate['mean_paired_macro_recall_gain_min'] and result['strict_user_wins']>=gate['strict_wins_min'] and result['worst_paired_macro_recall_delta']+tol>=gate['worst_paired_macro_recall_delta_min']
    result.update(state='complete_exploratory_IRM_LOSO6_same_native_budget_with_declared_globalBCE_reset500_and_dummy_scale_gradient_product_differences',rows=rows,cohort_spending_gate_passed=bool(passed),gate_is_not_definition_of_any_positive_effect=True,route='execute_preregistered_globalBCE_reset500_no_penalty_control_LOSO6_before_IRM_attribution_or_outer8' if passed else 'close_exact_IRM_LOSO6_lambda100_warmup1_anneal500_nativeGAP64_seed20261029_no_rescue',finalizer_invocations=1,legacy_singlefold_once_invocations=0,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,prior_MMD6_cost=dict(wall=30865.434969089023,cpu=30891.040083422995),prior_VREx6_cost=dict(wall=31157.163366941997,cpu=31189.589142517),prior_Fishr6_cost=dict(wall=29559.94331930898,cpu=29599.386025808),remaining_phase_cost=read(P/'STATE.json'),last_resource_observation=read(P/'RESOURCE_LEDGER.json'),evaluation_wall_seconds=time.monotonic()-start)
    result['fold_stage_costs']={str(h):{k:read(P/f'folds/held{h}/run/execution/{k}/TERMINATION.json')['cumulative'] for k in ['train','replay','output']} for h in range(1,7)}
    write(P/'SUMMARY.json',result);write(P/'FINAL_RESULT_STATE.json',dict(state=result['route'],finalizer_consumed=True,summary_sha256=sha(P/'SUMMARY.json'),no_automatic_test_feedback_or_new_fits=True));print(json.dumps(result))
if __name__=='__main__':main()
