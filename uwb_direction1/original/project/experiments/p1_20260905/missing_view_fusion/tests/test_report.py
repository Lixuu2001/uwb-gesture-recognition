"""Integrity and user-level aggregation contracts for the result audit."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score


def module():
    path = Path(__file__).resolve().parents[1] / 'report.py'
    assert path.exists(), 'The experiment report implementation is missing'
    spec = importlib.util.spec_from_file_location('missing_view_report_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data), encoding='utf-8')


def synthetic_run(tmp_path):
    directory = tmp_path / 'R_ONLY_s2026_h1'
    directory.mkdir()
    logits = np.zeros((3, 12), dtype=np.float32)
    logits[np.arange(3), [0, 1, 0]] = 2
    y = np.array([0, 1, 2])
    val_logits = logits[:2]
    val_ce = float(torch.nn.functional.cross_entropy(torch.tensor(val_logits), torch.tensor(y[:2])))
    config = {'arm': 'R_ONLY', 'fold': 1, 'seed': 2026, 'shuffle': False,
              'split_seed': 2026, 'cache_sha256': 'synthetic-cache', 'code_sha256': {},
              'matrix_sha256': 'synthetic-matrix', 'sampler_seed': 7778, 'mask_seed': 31338}
    write_json(directory / 'config.json', config)
    torch.save({'config': config, 'encoder_identity': {}, 'state': {},
                'right_mu': torch.zeros(1, 8), 'right_sd': torch.ones(1, 8)}, directory / 'deployment.pt')
    np.savez(directory / 'predictions.npz', ids=np.array([4, 5, 6]), y=y, logits=logits,
             pred=logits.argmax(1), base_logits=logits, train_ids=np.array([0, 1]),
             val_ids=np.array([2, 3]), val_y=y[:2], val_logits=val_logits, val_base_logits=val_logits)
    history = [{'epoch': 0, 'val_ce': val_ce, 'val_accuracy': 1.},
               {'epoch': 1, 'val_ce': val_ce + .1, 'val_accuracy': 1.}]
    write_json(directory / 'history.json', history)
    status = {'status': 'complete', 'name': directory.name, 'arm': 'R_ONLY', 'fold': 1,
              'seed': 2026, 'shuffle': False, 'accuracy': 2/3, 'base_accuracy': 2/3,
              'macro_f1': float(f1_score(y, logits.argmax(1), average='macro')),
              'best_epoch': 0, 'best_val_ce': val_ce, 'epochs': 1, 'sampling_hashes': ['epoch-one'],
              'n_test': 3, 'parameters': 100, 'trainable_parameters': 100, 'seconds': .1,
              'predictions_sha256': sha(directory / 'predictions.npz'),
              'deployment_sha256': sha(directory / 'deployment.pt')}
    write_json(directory / 'status.json', status)
    return directory


def modify_predictions(directory, change):
    with np.load(directory / 'predictions.npz') as data:
        arrays = {key: data[key].copy() for key in data.files}
    change(arrays)
    np.savez(directory / 'predictions.npz', **arrays)
    status = json.loads((directory / 'status.json').read_text())
    status['predictions_sha256'] = sha(directory / 'predictions.npz')
    write_json(directory / 'status.json', status)


def test_audit_recomputes_valid_saved_predictions_and_selection(tmp_path):
    record = module().audit_run(synthetic_run(tmp_path), verify_project_split=False)
    assert record['accuracy'] == pytest.approx(2/3)
    assert record['best_epoch'] == 0
    assert record['val_ce'] > 0
    assert record['stage'] == 1


def test_direct_readout_epoch_zero_is_zero_head_not_frozen_anchor(tmp_path):
    directory = synthetic_run(tmp_path)
    modify_predictions(directory, lambda arrays: (arrays['logits'].fill(0),
                                                 arrays['val_logits'].fill(0), arrays['pred'].fill(0)))
    config = json.loads((directory / 'config.json').read_text())
    config['readout'] = 'direct'
    write_json(directory / 'config.json', config)
    deployment = torch.load(directory / 'deployment.pt', weights_only=False)
    deployment['config'] = config
    torch.save(deployment, directory / 'deployment.pt')
    status = json.loads((directory / 'status.json').read_text())
    status.update(accuracy=1/3, macro_f1=1/6, best_val_ce=float(np.log(12)),
                  deployment_sha256=sha(directory / 'deployment.pt'))
    write_json(directory / 'status.json', status)
    write_json(directory / 'history.json', [
        {'epoch': 0, 'val_ce': float(np.log(12)), 'val_accuracy': .5},
        {'epoch': 1, 'val_ce': float(np.log(12)) + .1, 'val_accuracy': .5}])
    result = module().audit_run(directory, verify_project_split=False)
    assert result['readout'] == 'direct' and result['base_accuracy'] == pytest.approx(2/3)


def test_summary_rejects_mixed_readout_modes():
    records = [record('R_ONLY', 1, .5), record('JOINT_ZERO', 2, .6)]
    records[0]['readout'] = 'residual'
    records[1]['readout'] = 'direct'
    with pytest.raises(ValueError):
        module().summarize_records(records)


def test_cache_selection_matches_encoder_identity_when_content_hash_is_shared(tmp_path):
    report = module()
    report.HERE = tmp_path
    for name, identity in [('cache', {'source': 'old'}), ('cache_v2', {'source': 'current'})]:
        directory = tmp_path / name
        directory.mkdir()
        write_json(directory / 'fold1.json', {'cache_sha256': 'same-bytes', 'identity': identity})
    path = report._find_cache({'fold': 1, 'cache_sha256': 'same-bytes'}, {'source': 'current'})
    assert path.parent.name == 'cache_v2'


def test_audit_rejects_tampered_saved_source_snapshot_without_checking_worktree(tmp_path):
    directory = synthetic_run(tmp_path)
    snapshot = tmp_path / 'source_code'
    snapshot.mkdir()
    source = snapshot / 'runner.py'
    source.write_text('original frozen training source', encoding='utf-8')
    config = json.loads((directory / 'config.json').read_text())
    config['code_sha256'] = {'runner.py': sha(source)}
    write_json(directory / 'config.json', config)
    deployment = torch.load(directory / 'deployment.pt', weights_only=False)
    deployment['config'] = config
    torch.save(deployment, directory / 'deployment.pt')
    status = json.loads((directory / 'status.json').read_text())
    status['deployment_sha256'] = sha(directory / 'deployment.pt')
    write_json(directory / 'status.json', status)
    module().audit_run(directory, verify_project_split=False)
    source.write_text('modified snapshot', encoding='utf-8')
    with pytest.raises(ValueError):
        module().audit_run(directory, verify_project_split=False)


@pytest.mark.parametrize('fault', ['pred', 'overlap', 'duplicate', 'label', 'nan'])
def test_audit_rejects_prediction_and_partition_corruption(tmp_path, fault):
    directory = synthetic_run(tmp_path)
    def corrupt(arrays):
        if fault == 'pred':
            arrays['pred'][0] = 2
        elif fault == 'overlap':
            arrays['train_ids'][0] = arrays['ids'][0]
        elif fault == 'duplicate':
            arrays['ids'][0] = arrays['ids'][1]
        elif fault == 'label':
            arrays['val_y'][0] = 11
        else:
            arrays['logits'][0, 0] = np.nan
    modify_predictions(directory, corrupt)
    with pytest.raises(ValueError):
        module().audit_run(directory, verify_project_split=False)


@pytest.mark.parametrize('fault', ['hash', 'selection'])
def test_audit_rejects_deployment_tampering_and_wrong_best_epoch(tmp_path, fault):
    directory = synthetic_run(tmp_path)
    if fault == 'hash':
        with (directory / 'deployment.pt').open('ab') as stream:
            stream.write(b'changed')
    else:
        status = json.loads((directory / 'status.json').read_text())
        status['best_epoch'] = 1
        write_json(directory / 'status.json', status)
    with pytest.raises(ValueError):
        module().audit_run(directory, verify_project_split=False)


def record(arm, fold, accuracy, n=10, seed=2026):
    return {'arm': arm, 'fold': fold, 'seed': seed, 'shuffle': False, 'stage': 1,
            'accuracy': accuracy, 'base_accuracy': .5, 'macro_f1': accuracy,
            'best_epoch': 0, 'n_test': n, 'cache_sha256': f'cache-{fold}',
            'sampling_hashes': ['one', 'two'], 'parameters': 100,
            'trainable_parameters': 100, 'seconds': 1.}


def test_summary_weights_subjects_equally_and_never_gates_partial_results():
    records = [record('R_ONLY', 1, .5, 1), record('R_ONLY', 2, .5, 100),
               record('JOINT_ZERO', 1, .9, 1), record('JOINT_ZERO', 2, .5, 100)]
    summary = module().summarize_records(records)
    row = next(row for row in summary['summaries'] if row['arm'] == 'JOINT_ZERO')
    assert row['accuracy'] == pytest.approx(.7)
    assert row['paired_gain_pp'] == pytest.approx(20)
    assert row['n_folds'] == 2 and row['users_improved'] == 1
    assert row['gate'] is False and row['complete'] is False
    assert len(row['paired_effects_pp']) == 2


def test_gate_requires_eight_users_mean_gain_and_anchor_improvement():
    records = [record(arm, fold, acc) for fold in range(1, 9)
               for arm, acc in [('R_ONLY', .5), ('JOINT_ZERO', .53)]]
    first = module().summarize_records(records)
    second = module().summarize_records(records)
    row = next(row for row in first['summaries'] if row['arm'] == 'JOINT_ZERO')
    assert row['gate'] and row['users_improved'] == 8
    assert row['bootstrap_ci95_pp'] == pytest.approx([3., 3.])
    assert first == second
    for item in records:
        item['base_accuracy'] = .6
    row = next(row for row in module().summarize_records(records)['summaries'] if row['arm'] == 'JOINT_ZERO')
    assert row['gate'] is False


@pytest.mark.parametrize('fault', ['cache', 'sampler', 'duplicate'])
def test_cross_run_comparison_requires_identical_cache_and_sampler_prefix(fault):
    records = [record('R_ONLY', 1, .5), record('JOINT_ZERO', 1, .6)]
    if fault == 'cache':
        records[1]['cache_sha256'] = 'different'
    elif fault == 'sampler':
        records[1]['sampling_hashes'][0] = 'different'
    else:
        records.append(records[0].copy())
    with pytest.raises(ValueError):
        module().summarize_records(records)
