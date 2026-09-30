import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location('mild_audit_test_target', Path(__file__).with_name('audit.py'))
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def synthetic_schedule(mode='AUG', smoke_steps=0):
    # Producer helpers are used only for the test oracle, never by the auditor.
    from augment import sampler_seed, epoch_schedule
    train = np.arange(0, 2100, 3, dtype=np.int64)
    rng = np.random.default_rng(2027)
    rows = [{'epoch': 0, 'steps': 0, 'episode_hash': None, 'val_ce': 3.}]
    for epoch in range(1, 4):
        seed = sampler_seed({'rng': rng.bit_generator.state}, 1)
        plan = dict(zip(train, epoch_schedule(train, seed)))
        shuffled = rng.permutation(train)
        steps = smoke_steps or 6
        consumed = shuffled[:steps*128]
        signs = [int(plan[i]) if mode == 'AUG' else 0 for i in consumed]
        trace = hashlib.sha256()
        for sample, sign in zip(consumed, signs):
            trace.update(np.array([sample, sign], dtype=np.int64).tobytes())
        rows.append({'epoch': epoch, 'augmentation_seed': seed, 'augmentation_trace': trace.hexdigest(),
                     'augmentation_counts': {str(s): signs.count(s) for s in [-1, 0, 1]},
                     'steps': steps, 'source_rows': len(consumed), 'episode_hash': hashlib.sha256(consumed.tobytes()).hexdigest()})
    identity = {'fold': 1, 'config': {'seed': 2026, 'warmup_batch': 128, 'smoke_steps': smoke_steps,
                                    'active_mode': mode, 'augmentation_fold': 1},
                'source': {}, 'partitions': {}, 'matrix_sha256': 'same', 'initial_state_sha256': 'same'}
    return {'identity': identity, 'history': rows}, train, {'ids': train, 'rng': rng.bit_generator.state}


@pytest.mark.parametrize('mode', ['CONTROL', 'AUG'])
@pytest.mark.parametrize('smoke_steps', [0, 3])
def test_independent_schedule_matches_producer_full_and_smoke(mode, smoke_steps):
    result, train, sampler = synthetic_schedule(mode, smoke_steps)
    before = torch.get_rng_state().clone()
    assert audit.audit_schedule(result, train, sampler)['independent_schedule_verified']
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('field,value', [('augmentation_seed', 9), ('augmentation_trace', 'bad'),
                                       ('augmentation_counts', {'-1': 0, '0': 700, '1': 0}), ('source_rows', 999)])
def test_audit_rejects_tampered_schedule(field, value):
    result, train, sampler = synthetic_schedule()
    result['history'][2][field] = value
    with pytest.raises(ValueError):
        audit.audit_schedule(result, train, sampler)


def test_audit_rejects_rng_and_partition_tampering():
    result, train, sampler = synthetic_schedule()
    for key, value in [('ids', train[::-1]), ('rng', np.random.default_rng(4).bit_generator.state)]:
        changed = dict(sampler, **{key: value})
        with pytest.raises(ValueError):
            audit.audit_schedule(result, train, changed)


def test_pair_allows_different_stops_but_requires_common_prefix():
    control, _, _ = synthetic_schedule('CONTROL'); aug, _, _ = synthetic_schedule('AUG')
    aug['history'].pop()
    audit.audit_pair({'CONTROL': control, 'AUG': aug})
    aug['history'][1]['episode_hash'] = 'wrong'
    with pytest.raises(ValueError, match='episode_hash'):
        audit.audit_pair({'CONTROL': control, 'AUG': aug})


@pytest.mark.parametrize('field', ['source', 'partitions', 'matrix_sha256', 'initial_state_sha256'])
def test_pair_rejects_mismatched_data_matrix_or_initialization(field):
    control, _, _ = synthetic_schedule('CONTROL'); aug, _, _ = synthetic_schedule('AUG')
    aug['identity'][field] = 'different'
    with pytest.raises(ValueError, match=field):
        audit.audit_pair({'CONTROL': control, 'AUG': aug})


def scores(delta):
    return {mode: {str(h): {'accuracy': .7+(delta[h-1] if mode == 'AUG' else 0),
                           'macro_f1': .65+(delta[h-1] if mode == 'AUG' else 0)} for h in range(1, 9)} for mode in audit.MODES}


def test_gate_requires_complete_pairs_effect_and_six_users():
    good = scores([.03]*8)
    result = audit.summarize_scores(good)
    assert result['gate']['passed'] and result['paired_effect']['accuracy']['mean_pp'] == pytest.approx(3.)
    assert result['paired_effect']['accuracy']['bootstrap_95_percentile_pp'] == pytest.approx([3., 3.])
    assert not audit.summarize_scores(scores([.01]*8))['gate']['passed']
    assert not audit.summarize_scores(scores([.1]*5+[-.001]*3))['gate']['passed']
    del good['CONTROL']['8']
    assert audit.summarize_scores(good)['gate'] is None


def test_actual_updates_require_right_encoder_head_and_frozen_auxiliary():
    initial = {k: torch.zeros(1) for k in ['right.encoder.w', 'right.head.w', 'fuser.w']}
    trained = copy.deepcopy(initial)
    trained['right.encoder.w'] += 1; trained['right.head.w'] += 1
    assert audit.legacy.audit_last_updates(trained, initial, 'A0')['encoder']['changed'] == 1
    with pytest.raises(ValueError, match='did not update'):
        audit.legacy.audit_last_updates(initial, initial, 'A0')
    trained['fuser.w'] += 1
    with pytest.raises(ValueError, match='auxiliary'):
        audit.legacy.audit_last_updates(trained, initial, 'A0')


def test_report_keeps_incomplete_and_no_replay_distinct(tmp_path):
    report = {'status': 'incomplete', 'summary': {'gate': None, 'gate_reason': 'incomplete'},
              'runs': [], 'errors': [], 'replay_requested': False}
    target = tmp_path/'RESULTS.md'
    audit.write_results(report, target)
    value = target.read_text(encoding='utf-8')
    assert '当前不能判定' in value and '尚未运行独立 Right' in value


def test_completed_report_includes_absolute_clean_stress_scores_and_epochs(tmp_path):
    clean = scores([.03]*8); stress = scores([.04]*8)
    report = {'status': 'verified', 'summary': audit.summarize_scores(clean), 'errors': [],
              'runs': [{'mode': mode, 'fold': fold, 'metrics': clean[mode][str(fold)],
                        'best_epoch': 12, 'last_epoch': 20} for mode in audit.MODES for fold in range(1, 9)],
              'replay_requested': True, 'stress_summary': {'minus2': audit.summarize_scores(stress)}}
    target = tmp_path/'RESULTS.md'
    audit.write_results(report, target)
    value = target.read_text(encoding='utf-8')
    assert '| CONTROL | 70.000% | 65.000% |' in value
    assert '| AUG | 73.000% | 68.000% |' in value
    assert '| 1 | 12 / 20 | 12 / 20 |' in value
    assert '| minus2 | 70.000% | 74.000% | +4.000 | 65.000% | 69.000% |' in value
    assert '自然手势速度变化' in value and '三维重建' not in value
