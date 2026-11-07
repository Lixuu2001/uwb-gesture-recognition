"""Read-only all-six native freeze, schedule intent and actual cost audit. No truth pairing."""
from pathlib import Path
import json, hashlib, sys, time
P = Path('/workspace/uwb-vrex-loso6-20261106')
sys.path.insert(0, str(P / 'shared'))
from controller import identity
from common import read, write, sha

def main():
    started = time.monotonic(); cpu = time.process_time()
    output = P / 'ALL_SIX_FREEZE_COUNTER_COST_AUDIT.json'
    assert not output.exists()
    old = read(P / 'PHASE_IDENTITY.json'); now = identity(old['pid'])
    assert now is None or now['state'] == 'Z' or any(now[k] != old[k] for k in ['boot_id', 'startticks'])
    state = read(P / 'STATE.json')
    assert state['state'] == 'all_six_method_folds_frozen_unscored_requires_complete_private_backup_before_cohort_once'
    assert state['completed_folds'] == [1, 2, 3, 4, 5, 6] and state['diagnostic_labels_read'] is False
    for name, value in read(P / 'FROZEN_HASHES.json').items():
        assert sha(P / name) == value, name
    frozen = read(P / 'ALL_SIX_PREDICTIONS_FROZEN.json')
    for row in frozen['files'].values():
        assert sha(row['path']) == row['sha256']
    folds = []
    for held in range(1, 7):
        fold = P / f'folds/held{held}'; config = read(fold / 'PROTOCOL.json')
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
            assert row['optimizer_reset_this_update'] == (i == 500)
            assert row['optimizer_reset_events'] == ([] if i < 500 else [500])
            assert row['vrex_penalty_weight'] == (1 if i < 500 else 10)
        previous = dict(wall=0, cpu=0); cost = {}
        for stage in ['train', 'replay', 'output']:
            term = read(fold / f'run/execution/{stage}/TERMINATION.json')
            assert term['exit_code'] == 0 and term['resource_stop_reason'] is None
            cumulative = term['cumulative']
            cost[stage] = {key: cumulative[key] - previous[key] for key in previous}; previous = cumulative
        folds.append(dict(held=held, updates_per_trajectory=n, selected_epoch_validation_only=producer['selected_epoch'],
            counts=config['counts'], train_users=config['train_users'], validation_user=config['validation_user'], test_user=held,
            all65_epoch_records_equal=True, all_step_logs_bitwise_equal=True, steps_sha256=hashlib.sha256(steps).hexdigest(),
            counter_reset_weight_intent_validated=True, native_final_state_sha256=producer['final_state_sha256'],
            incremental_stage_seconds=cost, final_fold_stage_cumulative=previous))
    assert not (P / 'SUMMARY.json').exists() and not (P / 'FINALIZE_ONCE.lock').exists()
    result = dict(state='all6_native_frozen_and_independent_schedule_intent_cost_audited_before_test_pairing',
        observed_unix=time.time(), original_phase_identity=old, actual_original_phase=now, all_six_completed=state,
        resource_ledger_is_last_periodic_observation_not_final_sixfold_total=True, folds=folds,
        train_wall_sum=sum(x['incremental_stage_seconds']['train']['wall'] for x in folds),
        replay_wall_sum=sum(x['incremental_stage_seconds']['replay']['wall'] for x in folds),
        output_audit_wall_sum=sum(x['incremental_stage_seconds']['output']['wall'] for x in folds),
        train_CPU_sum=sum(x['incremental_stage_seconds']['train']['cpu'] for x in folds),
        replay_CPU_sum=sum(x['incremental_stage_seconds']['replay']['cpu'] for x in folds),
        output_audit_CPU_sum=sum(x['incremental_stage_seconds']['output']['cpu'] for x in folds),
        frozen_scientific_manifest_sha256=sha(P / 'FROZEN_HASHES.json'),
        all_six_predictions_frozen_sha256=sha(P / 'ALL_SIX_PREDICTIONS_FROZEN.json'), source_script_sha256=sha(__file__),
        new_optimizer_updates=False, test_truth_or_predictions_paired=False, users7_8_labels_used=False,
        deterministic_replay_does_not_exclude_all_shared_implementation_errors=True,
        audit_wall=time.monotonic()-started, audit_CPU=time.process_time()-cpu)
    write(output, result)
    print(json.dumps({key: result[key] for key in ['state', 'train_wall_sum', 'replay_wall_sum', 'output_audit_wall_sum']}))

if __name__ == '__main__':
    main()
