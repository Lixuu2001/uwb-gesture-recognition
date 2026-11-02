import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).with_name('audit.py')
audit = None
if path.exists():
    spec = importlib.util.spec_from_file_location('style_independent_audit_test', path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)


def target():
    assert audit is not None, 'independent style auditor has not been implemented'
    return audit


def scores(deltas):
    return {m: {str(h): {'accuracy': .7+(deltas[h-1] if m == 'STYLE' else 0),
                          'macro_f1': .65+(deltas[h-1] if m == 'STYLE' else 0)}
                for h in range(1, 9)} for m in ('CONTROL', 'STYLE')}


def synthetic_schedule(smoke_steps=0):
    # Independent test oracle with explicit sorted candidate lists.
    labels = np.tile(np.arange(3), 18)
    subjects = np.repeat(np.arange(1, 7), 9)
    train = np.arange(54, dtype=np.int64)
    sampler = np.random.default_rng(2027)
    history = [{'epoch': 0, 'steps': 0, 'episode_hash': None, 'val_ce': 3.}]
    for epoch in range(1, 4):
        payload = json.dumps({'sampler': sampler.bit_generator.state, 'fold': 1,
                              'augmentation_seed': 20260909}, sort_keys=True).encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')
        rng = np.random.Generator(np.random.PCG64(seed))
        masks = rng.random(len(train)) < .25
        donors = []
        for anchor in train:
            same = [int(i) for i in train if labels[i] == labels[anchor] and subjects[i] != subjects[anchor]]
            user = rng.choice(sorted({int(subjects[i]) for i in same}))
            donors.append(rng.choice([i for i in same if subjects[i] == user]))
        consumed = sampler.permutation(train)[:(smoke_steps or 4)*16]
        triples = np.array([[i, donors[i], int(masks[i])] for i in consumed], dtype=np.int64)
        history.append({'epoch': epoch, 'steps': smoke_steps or 4, 'source_rows': len(consumed),
                        'episode_hash': hashlib.sha256(consumed.tobytes()).hexdigest(),
                        'style_seed': seed, 'style_trace': hashlib.sha256(triples.tobytes()).hexdigest(),
                        'style_counts': {'clean': int((~masks[consumed]).sum()), 'style': int(masks[consumed].sum())}})
    identity = {'fold': 1, 'config': {'seed': 2026, 'warmup_batch': 16, 'smoke_steps': smoke_steps,
                                    'style_fold': 1, 'active_mode': 'STYLE'},
                'partitions': {}, 'matrix_sha256': 'same', 'initial_state_sha256': 'same'}
    return {'identity': identity, 'history': history}, train, labels, subjects, {'ids': train, 'rng': sampler.bit_generator.state}


@pytest.mark.parametrize('smoke_steps', [0, 2])
def test_schedule_full_and_truncated(smoke_steps):
    args = synthetic_schedule(smoke_steps)
    assert target().audit_schedule(*args)['independent_schedule_verified']


@pytest.mark.parametrize('field,value', [('style_seed', 0), ('style_trace', 'forged'),
                                       ('style_counts', {'clean': 54, 'style': 0}), ('source_rows', 1)])
def test_schedule_tampering(field, value):
    args = synthetic_schedule()
    args[0]['history'][1][field] = value
    with pytest.raises(ValueError):
        target().audit_schedule(*args)


def test_schedule_rng_and_training_ids_tampering():
    for key, value in [('ids', np.arange(54)[::-1]), ('rng', np.random.default_rng(99).bit_generator.state)]:
        args = list(synthetic_schedule())
        args[-1] = dict(args[-1], **{key: value})
        with pytest.raises(ValueError):
            target().audit_schedule(*args)


def test_gate_full_pairs_threshold_and_smoke_error_suppression():
    a = target()
    good = scores([.03]*8)
    summary = a.summarize_scores(good)
    assert summary['gate']['passed']
    assert summary['paired_effect']['accuracy']['mean_pp'] == pytest.approx(3.)
    assert summary['paired_effect']['accuracy']['bootstrap_95_percentile_pp'] == pytest.approx([3., 3.])
    assert not a.summarize_scores(scores([.01]*8))['gate']['passed']
    assert not a.summarize_scores(scores([.1]*5+[-.001]*3))['gate']['passed']
    for kwargs in ({'smoke': True}, {'errors': ['failure']}, {'missing': ['fold8']}):
        assert a.summarize_scores(good, **kwargs)['gate'] is None
    del good['CONTROL']['8']
    assert a.summarize_scores(good)['gate'] is None


def test_pair_allows_stop_difference_but_rejects_changed_initial_or_sampling():
    a = target()
    style = synthetic_schedule()[0]
    control = copy.deepcopy(style)
    control['identity']['config'].pop('style_fold')
    control['identity']['config']['active_mode'] = 'CONTROL'
    control['history'].pop()
    a.audit_pair({'CONTROL': control, 'STYLE': style})
    for field in ('episode_hash', 'steps'):
        changed = copy.deepcopy(style)
        changed['history'][1][field] = 'wrong'
        with pytest.raises(ValueError):
            a.audit_pair({'CONTROL': control, 'STYLE': changed})
    changed = copy.deepcopy(style)
    changed['history'][0]['val_ce'] += .01
    with pytest.raises(ValueError, match='initial validation'):
        a.audit_pair({'CONTROL': control, 'STYLE': changed})


def test_fixed_config_rejects_extra_or_changed_settings():
    a = target()
    baseline = {'seed': 2026, 'warmup_lr': .0001, 'smoke': False, 'smoke_steps': 0,
                'warmup_epochs': 40, 'warmup_min': 12, 'epochs': 16, 'min_epochs': 6}
    config = dict(baseline, style=a.STYLE_CONFIG)
    a.audit_config(config, baseline)
    for key, value in [('warmup_lr', .1), ('surprise_setting', True), ('smoke_steps', 3)]:
        with pytest.raises(ValueError):
            a.audit_config(dict(config, **{key: value}), baseline)
    altered = copy.deepcopy(config)
    altered['style']['share'] = .2
    with pytest.raises(ValueError):
        a.audit_config(altered, baseline)


def test_pretrain_certificate_requires_verified_source_binding(tmp_path):
    a = target()
    path = tmp_path/'PRETRAIN_CHECK.json'
    source = {'code': {'runner.py': 'frozen'}}
    for value in [{'status': 'prepared', 'source_code': source['code']},
                  {'status': 'verified', 'source_code': {'runner.py': 'different'}}]:
        path.write_text(json.dumps(value), encoding='utf-8')
        with pytest.raises(ValueError):
            a.audit_pretrain(path, {'pretrain_check_sha256': a.legacy.file_hash(path)}, source, a.legacy.HashVerifier())
    path.write_text(json.dumps({'status': 'verified', 'source_code': source['code']}), encoding='utf-8')
    ledger = {'pretrain_check_sha256': a.legacy.file_hash(path)}
    a.audit_pretrain(path, ledger, source, a.legacy.HashVerifier())
    path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        a.audit_pretrain(path, ledger, source, a.legacy.HashVerifier())


def test_report_incomplete_and_completed(tmp_path):
    a = target()
    report = {'status': 'incomplete', 'summary': {'gate': None, 'gate_reason': 'incomplete'},
              'runs': [], 'errors': [], 'replay_requested': False}
    path = tmp_path/'RESULTS.md'
    a.write_results(report, path)
    value = path.read_text(encoding='utf-8')
    assert '当前不能判定' in value and '尚未运行独立 Right' in value
    data = scores([.03]*8)
    report.update(status='verified', summary=a.summarize_scores(data), replay_requested=True,
                  runs=[{'mode': m, 'fold': h, 'metrics': data[m][str(h)], 'best_epoch': 12,
                         'last_epoch': 20} for m in a.MODES for h in range(1, 9)])
    a.write_results(report, path)
    value = path.read_text(encoding='utf-8')
    assert '| STYLE | 73.000% | 68.000% |' in value
    assert '| 1 | 12 / 20 | 12 / 20 |' in value
    assert '描述性' in value and '复用' in value
