"""Saved-result integrity and user-level comparisons for trainable fusion."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score


def module():
    path = Path(__file__).resolve().parents[1] / 'audit.py'
    assert path.exists(), 'The E2E audit has not been implemented'
    spec = importlib.util.spec_from_file_location('e2e_audit_contract', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(name, value):
    digest = hashlib.sha256()
    digest.update(name.encode())
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_run(tmp_path, budget=1, completed_epochs=1, minimum=12, patience=8):
    directory = tmp_path / 'E2E_R_s2026_h1'
    directory.mkdir()
    config = dict(fold=1, mode='R', seed=2026, split_seed=2026, sampler_seed=7778,
                  dropout_seed=102026, epochs=budget, batch=128, microbatch=16,
                  encoder_lr=1e-5, fuser_lr=1e-3, weight_decay=1e-4,
                  minimum_epochs=minimum, patience=patience, source_identity={}, source_code={}, cache_sha256='cache')
    write(directory / 'config.json', config)
    initial = tensor_hash('weight', torch.zeros(2))
    final = tensor_hash('weight', torch.ones(2))
    lineage = dict(initial_encoder_sha256=initial, initial_fuser_sha256='fuser', checkpoint_sha256='checkpoint',
                   checkpoint_path='synthetic.pt', source_seed=2026, source_fold=1, model_seed=2026)
    write(directory / 'initialization.json', lineage)
    state = {'encoder.weight': torch.ones(2), 'fuser.weight': torch.zeros(2),
             'mu': torch.zeros(1, 3, 1664), 'sd': torch.ones(1, 3, 1664)}
    torch.save(dict(state=state, config=config, initialization=lineage, best_epoch=1), directory / 'deployment.pt')
    logits = np.zeros((3, 12), dtype=np.float32)
    logits[np.arange(3), [0, 1, 0]] = 2
    y = np.array([0, 1, 2])
    np.savez(directory / 'predictions.npz', ids=np.array([4, 5, 6]), y=y, pred=logits.argmax(1),
             logits=logits, base_logits=logits, train_ids=np.array([0, 1]), val_ids=np.array([2, 3]),
             val_y=y[:2], val_logits=logits[:2])
    ce = float(torch.nn.functional.cross_entropy(torch.tensor(logits[:2]), torch.tensor(y[:2])))
    history = [dict(epoch=0, val_ce=float(np.log(12)), val_accuracy=.5),
               dict(epoch=1, val_ce=ce, val_accuracy=1., max_encoder_gradient_norm=.3,
                    train_loss=1., seconds=1., peak_memory_mb=10.)]
    history.extend({**history[1], 'epoch': epoch, 'val_ce': ce + .1}
                   for epoch in range(2, completed_epochs + 1))
    write(directory / 'history.json', history)
    status = dict(status='complete', fold=1, mode='R', seed=2026, accuracy=2/3, base_accuracy=2/3,
                  macro_f1=float(f1_score(y, logits.argmax(1), average='macro')), n_test=3,
                  best_epoch=1, best_val_ce=ce, epochs=completed_epochs,
                  sampling_hashes=['one'] * completed_epochs,
                  initial_encoder_sha256=initial, final_encoder_sha256=final, encoder_changed=True,
                  encoder_gradient_observed=True, trainable_parameters=4,
                  elapsed_training_seconds=1., invocation_seconds=2.,
                  artifacts={name: sha(directory / name) for name in ('deployment.pt', 'predictions.npz')})
    write(directory / 'status.json', status)
    return directory


def test_saved_run_metrics_and_updated_encoder_are_recomputed(tmp_path):
    record = module().audit_run(make_run(tmp_path), verify_project=False)
    assert record['accuracy'] == pytest.approx(2/3)
    assert record['encoder_changed'] and record['encoder_gradient_observed']
    assert record['updated_encoder_tensors'] is None
    assert record['resume_identity_verified'] is False


@pytest.mark.parametrize('completed,minimum,patience', [(1, 12, 8), (11, 12, 8), (12, 12, 12)])
def test_premature_completion_without_budget_or_early_stop_is_rejected(tmp_path, completed, minimum, patience):
    directory = make_run(tmp_path, budget=40, completed_epochs=completed,
                         minimum=minimum, patience=patience)
    with pytest.raises(ValueError, match='early.stop|budget'):
        module().audit_run(directory, verify_project=False)


@pytest.mark.parametrize('budget,completed', [(1, 1), (40, 12)])
def test_completed_budget_or_valid_early_stop_is_accepted(tmp_path, budget, completed):
    record = module().audit_run(make_run(tmp_path, budget=budget, completed_epochs=completed),
                               verify_project=False)
    assert record['epochs'] == completed


def test_normalization_recomputation_uses_training_thread_count_before_reduction():
    audit = module()
    torch.set_num_threads(2)
    features = torch.randn(11, 3, 1664, dtype=torch.float16)
    mu, sd = audit._training_stats(features)
    assert torch.get_num_threads() == 4
    torch.testing.assert_close(mu, features.float().mean(0, keepdim=True), rtol=0, atol=0)
    torch.testing.assert_close(sd, features.float().std(0, unbiased=False, keepdim=True).clamp_min(.05), rtol=0, atol=0)


def test_saved_progress_config_identity_is_checked_when_present(tmp_path):
    directory = make_run(tmp_path)
    history = json.loads((directory / 'history.json').read_text())
    deployment = torch.load(directory / 'deployment.pt', weights_only=False)
    info = dict(epoch=1, best_epoch=1, best=history[1]['val_ce'], history=history,
                sampling_hashes=['one'], best_state=deployment['state'], run_identity='wrong-config-identity')
    torch.save({'info': info}, directory / 'progress.pt')
    with pytest.raises(ValueError):
        module().audit_run(directory, verify_project=False)
    config = deployment['config']
    info['run_identity'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    torch.save({'info': info}, directory / 'progress.pt')
    assert module().audit_run(directory, verify_project=False)['resume_identity_verified'] is True
    del info['run_identity']
    torch.save({'info': info}, directory / 'progress.pt')
    assert module().audit_run(directory, verify_project=False)['resume_identity_verified'] is False


@pytest.mark.parametrize('fault', ['pred', 'nan', 'overlap', 'duplicate', 'ce', 'gradient', 'encoder_hash', 'artifact'])
def test_integrity_failures_are_rejected_even_when_file_hash_is_updated(tmp_path, fault):
    directory = make_run(tmp_path)
    status = json.loads((directory / 'status.json').read_text())
    if fault in ('pred', 'nan', 'overlap', 'duplicate', 'ce'):
        with np.load(directory / 'predictions.npz') as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if fault == 'pred':
            arrays['pred'][0] = 5
        elif fault == 'nan':
            arrays['logits'][0, 0] = np.nan
        elif fault == 'overlap':
            arrays['train_ids'][0] = arrays['ids'][0]
        elif fault == 'duplicate':
            arrays['ids'][0] = arrays['ids'][1]
        else:
            arrays['val_y'][0] = 11
        np.savez(directory / 'predictions.npz', **arrays)
        status['artifacts']['predictions.npz'] = sha(directory / 'predictions.npz')
    elif fault == 'gradient':
        status['encoder_gradient_observed'] = False
    elif fault == 'encoder_hash':
        status['final_encoder_sha256'] = 'wrong'
    else:
        with (directory / 'deployment.pt').open('ab') as stream:
            stream.write(b'changed')
    write(directory / 'status.json', status)
    with pytest.raises(ValueError):
        module().audit_run(directory, verify_project=False)


def row(mode, fold, accuracy, n=10):
    return dict(mode=mode, fold=fold, seed=2026, accuracy=accuracy, macro_f1=accuracy,
                base_accuracy=.5, n_test=n, best_epoch=3, epochs=12, maximum_epochs=40,
                encoder_changed=True, encoder_gradient_observed=True, updated_encoder_tensors=5,
                resume_identity_verified=True,
                trainable_parameters=100, elapsed_training_seconds=1.,
                initial_encoder_sha256='initial', initial_fuser_sha256='fuser',
                normalization_sha256=f'stats-{fold}', sampling_hashes=['one'],
                cache_sha256=f'cache-{fold}', source_identity={'fold': fold})


def test_equal_user_aggregation_does_not_pool_samples_or_gate_incomplete_users():
    records = [row('R', 1, .5, 1), row('R', 2, .5, 100), row('MV', 1, .9, 1), row('MV', 2, .5, 100)]
    summary = module().summarize_records(records)
    mv = next(item for item in summary['summaries'] if item['mode'] == 'MV')
    assert mv['accuracy'] == pytest.approx(.7)
    assert mv['paired_gain_pp'] == pytest.approx(20)
    assert mv['users_improved'] == 1 and mv['gate'] is False
    assert mv['complete'] is False and mv['n_folds'] == 2


def test_complete_gate_bootstrap_and_anchor_requirement():
    records = [row(mode, fold, accuracy) for fold in range(1, 9) for mode, accuracy in [('R', .5), ('MV', .53)]]
    a, b = module().summarize_records(records), module().summarize_records(records)
    assert a == b
    mv = next(item for item in a['summaries'] if item['mode'] == 'MV')
    assert mv['gate'] and mv['bootstrap_ci95_pp'] == pytest.approx([3., 3.])
    for record in records:
        record['base_accuracy'] = .6
    mv = next(item for item in module().summarize_records(records)['summaries'] if item['mode'] == 'MV')
    assert mv['gate'] is False


def test_summary_exposes_unverified_resume_identity():
    records = [row('R', 1, .5), row('R', 2, .5)]
    records[1]['resume_identity_verified'] = False
    audit = module().summarize_records(records)
    assert audit['summaries'][0]['resume_identity_verified_runs'] == 1
    assert '恢复身份已核验 1/2' in module()._markdown(audit)
    assert '缺失身份标记的旧记录不计为已核验' in module()._markdown(audit)


@pytest.mark.parametrize('field', ['initial_encoder_sha256', 'initial_fuser_sha256', 'normalization_sha256',
                                  'sampling_hashes', 'cache_sha256', 'source_identity'])
def test_paired_arms_require_identical_initialization_sources_and_sampling(field):
    records = [row('R', 1, .5), row('MV', 1, .53)]
    records[1][field] = ['changed'] if field == 'sampling_hashes' else 'changed'
    with pytest.raises(ValueError):
        module().summarize_records(records)
