"""Read-only all-six native freeze, schedule intent and actual cost audit. No truth pairing."""
from pathlib import Path
import json, hashlib, sys, time
P = Path('/workspace/uwb-fishr-native-loso6-20261107')
sys.path.insert(0, str(P / 'shared'))
from controller import identity
from common import read, write, sha

def main(completed_only=False):
    started = time.monotonic(); cpu = time.process_time()
    old=read(P/'PHASE_IDENTITY.json');now=identity(old['pid']);state=read(P/'STATE.json')
    held=list(state['completed_folds']) if completed_only else list(range(1,7))
    assert held and held==sorted(set(held)) and all(h in range(1,7) for h in held)
    if completed_only:
        assert state['state'] in ['full_sixfold_in_progress_unscored','all_six_method_folds_frozen_unscored_requires_complete_private_backup_before_cohort_once']
        output=P/('COMPLETED_'+'_'.join(map(str,held))+'_FREEZE_COUNTER_COST_AUDIT.json')
    else:
        output=P/'ALL_SIX_FREEZE_COUNTER_COST_AUDIT.json'
        assert now is None or now['state']=='Z' or any(now[k]!=old[k] for k in ['boot_id','startticks'])
        assert state['state']=='all_six_method_folds_frozen_unscored_requires_complete_private_backup_before_cohort_once'
        assert state['completed_folds']==held and state['diagnostic_labels_read'] is False
        for row in read(P/'ALL_SIX_PREDICTIONS_FROZEN.json')['files'].values():assert sha(row['path'])==row['sha256']
    assert not output.exists()
    for name,value in read(P/'FROZEN_HASHES.json').items():assert sha(P/name)==value,name
    for name,value in read(P/'shared/DEPENDENCIES.json')['files'].items():assert sha(name)==value,name
    role_proof=read(P/('COMPLETED_FOLDS_'+'_'.join(map(str,held))+'_ROLE_SELECTION_AUDIT.json'))
    assert role_proof['scope']==held and role_proof['optimizer_updates']==0 and not role_proof['test_labels_or_predictions_opened']
    folds = []
    for h in held:
        fold = P / f'folds/held{h}'; config = read(fold / 'PROTOCOL.json')
        n = config['expected_updates']
        producer = read(fold / 'run/train/COMPLETE.json'); replay = read(fold / 'run/replay/COMPLETE.json')
        assert producer['state'] == 'full64_training_complete_unscored'
        assert replay['state'] == 'full64_fresh_training_replay_exact_unscored'
        assert {k: v for k, v in producer.items() if k != 'state'} == {k: v for k, v in replay.items() if k != 'state'}
        assert producer['epochs'] == 64 and producer['updates'] == n
        assert read(fold / 'PHASE_TERMINATION.json')['exit_code'] == 0
        for epoch in range(65):
            assert read(fold / f'run/train/epoch{epoch:03}.json') == read(fold / f'run/replay/epoch{epoch:03}.json')
        steps = (fold / 'run/train/steps.jsonl').read_bytes()
        assert steps == (fold / 'run/replay/steps.jsonl').read_bytes()
        rows = [json.loads(x) for x in steps.splitlines()]; assert len(rows) == n
        for i, row in enumerate(rows):
            assert row['update'] == i + 1 and row['completed_updates_before'] == i and row['completed_updates_after'] == i + 1
            assert row['optimizer_reset_this_update'] == (i == 1500)
            assert row['optimizer_reset_events'] == ([] if i < 1500 else [1500])
            assert row['fishr_penalty_weight'] == (0 if i < 1500 else 1000)
            assert row['classification_loss']==row['global_BCE'] and row['classifier_gradient_coordinates']==18060
            assert row['forward_batch'] in [64,config['counts'][0]%64] and row['forward_batch']>0
            expected_loss=row['global_BCE']+row['fishr_penalty_weight']*row['penalty']
            assert abs(row['loss']-expected_loss)<=1e-6*max(1.,abs(expected_loss))
        previous = dict(wall=0, cpu=0); cost = {}
        for stage in ['train', 'replay', 'output']:
            term = read(fold / f'run/execution/{stage}/TERMINATION.json')
            assert term['exit_code'] == 0 and term['resource_stop_reason'] is None
            cumulative = term['cumulative']
            cost[stage] = {key: cumulative[key] - previous[key] for key in previous}; previous = cumulative
        folds.append(dict(held=h, updates_per_trajectory=n, selected_epoch_validation_only=producer['selected_epoch'],
            counts=config['counts'], train_users=config['train_users'], validation_user=config['validation_user'], test_user=h,
            all65_epoch_records_equal=True, all_step_logs_bitwise_equal=True, steps_sha256=hashlib.sha256(steps).hexdigest(),
            counter_reset_weight_intent_validated=True, native_final_state_sha256=producer['final_state_sha256'],
            incremental_stage_seconds=cost, final_fold_stage_cumulative=previous))
    assert not (P / 'SUMMARY.json').exists() and not (P / 'FINALIZE_ONCE.lock').exists()
    result = dict(state='completed_native_folds_independent_Fishr_schedule_and_cost_audited_before_test_pairing' if completed_only else 'all6_native_frozen_and_independent_Fishr_schedule_and_cost_audited_before_test_pairing',
        audited_held_users=held,complete_sixfold_cohort=held==list(range(1,7)),completed_only_mode=completed_only,
        observed_unix=time.time(), original_phase_identity=old, actual_original_phase=now, all_six_completed=state,
        resource_ledger_is_last_periodic_observation_not_final_sixfold_total=True, folds=folds,
        train_wall_sum=sum(x['incremental_stage_seconds']['train']['wall'] for x in folds),
        replay_wall_sum=sum(x['incremental_stage_seconds']['replay']['wall'] for x in folds),
        output_audit_wall_sum=sum(x['incremental_stage_seconds']['output']['wall'] for x in folds),
        train_CPU_sum=sum(x['incremental_stage_seconds']['train']['cpu'] for x in folds),
        replay_CPU_sum=sum(x['incremental_stage_seconds']['replay']['cpu'] for x in folds),
        output_audit_CPU_sum=sum(x['incremental_stage_seconds']['output']['cpu'] for x in folds),
        frozen_scientific_manifest_sha256=sha(P / 'FROZEN_HASHES.json'),
        all_six_predictions_frozen_sha256=None if completed_only else sha(P/'ALL_SIX_PREDICTIONS_FROZEN.json'), source_script_sha256=sha(__file__),
        new_optimizer_updates=False, test_truth_or_predictions_paired=False, users7_8_labels_used=False,
        deterministic_replay_does_not_exclude_all_shared_implementation_errors=True,
        audit_wall=time.monotonic()-started, audit_CPU=time.process_time()-cpu)
    write(output, result)
    print(json.dumps({key: result[key] for key in ['state', 'train_wall_sum', 'replay_wall_sum', 'output_audit_wall_sum']}))

if __name__ == '__main__':
    assert sys.argv[1:] in [[],['--completed']];main(sys.argv[1:]==['--completed'])
