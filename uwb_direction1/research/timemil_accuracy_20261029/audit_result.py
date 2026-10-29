"""Independent read-only completed TimeMIL verification; no training imports.

Metadata/freeze guard precedes all array/checkpoint decoding. Scientific files
stay frozen; this separate verifier cannot select a new model or configuration.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
ARMS = ('timemil', 'gap')

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for value in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(value)
    return h.hexdigest()

def ready(base):
    run = base/'run01'
    if not (run/'STATE.json').exists():
        raise RuntimeError('not_ready: TimeMIL has not started')
    state = read(run/'STATE.json')
    if state.get('state') != 'complete' or state.get('pid') is not None:
        raise RuntimeError('not_ready: incomplete TimeMIL run')
    if set(state.get('completed_arms', [])) != set(ARMS):
        raise RuntimeError('not_ready: missing arm')
    termination = read(base/'supervisor01/TERMINATION.json')
    if termination['exit_code'] != 0 or termination['wall_timeout']:
        raise RuntimeError('not_ready: training supervisor failed')
    assert termination['supervisor_wall_seconds'] <= 43200
    assert termination['child_cpu_seconds'] <= 43200
    freeze = read(run/'ALL_PREDICTIONS_FROZEN.json')
    assert set(freeze) == set(ARMS)
    for arm in ARMS:
        folder = run/arm
        done = read(folder/'ARM_COMPLETE.json')
        assert done['epochs'] == 64
        assert sha(folder/'predictions_unscored.npz') == freeze[arm] == done['prediction_sha256']
        assert sha(folder/'best.pt') == done['checkpoint_sha256']
    return termination

def selection(history):
    # Deliberately implements the ordering separately from the fitting driver.
    return sorted(history, key=lambda row: (-row['validation_macro'], row['validation_bce'], row['epoch']))[0]

def metrics(y, predicted):
    from sklearn.metrics import recall_score
    return float(recall_score(y, predicted, labels=np.unique(y), average='macro', zero_division=0))

def gates(scores, reference):
    return dict(gate_passed=bool(scores['timemil']-scores['gap'] >= .005 and scores['timemil'] >= reference-.005),
                gap_baseline_expansion_eligible=bool(scores['gap'] >= reference+.01))

def roles(split, ids, users):
    indices = dict(train=np.flatnonzero(np.isin(users, [3,4,5,6])),
                   validation=np.flatnonzero(users == 2), diagnostic=np.flatnonzero(users == 1))
    for role, ix in indices.items():
        np.testing.assert_array_equal(split[role], ids[ix])
    assert not (set(split['train']) & set(split['validation']) or set(split['train']) & set(split['diagnostic'])
                or set(split['validation']) & set(split['diagnostic']))
    return indices

def independent_preprocess(x):
    energy = np.mean(np.square(x), axis=(2,3), keepdims=True, dtype=np.float32)
    return (x/np.maximum(np.sqrt(energy), np.float32(1e-8))).transpose(0,2,1,3).reshape(len(x),90,567)

def check_optimizer(model, saved, steps, unused):
    import torch
    from adapter import StableLookahead
    named = list(model.named_parameters())
    active = {i for i, (name, _) in enumerate(named) if name not in unused}
    assert saved['count'] == steps and saved['alpha'] == .5 and saved['k'] == 6
    fast = saved['fast']
    assert len(fast['param_groups']) == 1
    group = fast['param_groups'][0]
    assert group['params'] == list(range(len(named)))
    assert group['lr'] == .001 and group['weight_decay'] == .0001
    assert set(fast['state']) == active
    assert {int(value['step']) for value in fast['state'].values()} == {steps}
    assert set(saved['slow']) == (active if steps >= 6 else set())
    for i in active:
        shape = named[i][1].shape
        for key in ('exp_avg', 'exp_avg_sq'):
            value = fast['state'][i][key]
            assert value.shape == shape and torch.isfinite(value).all()
        if steps >= 6:
            assert saved['slow'][i].shape == shape and torch.isfinite(saved['slow'][i]).all()
    restored = StableLookahead(model.parameters())
    restored.load_state_dict(saved)
    assert restored.count == steps
    assert all(torch.equal(value, restored.slow[key]) for key, value in saved['slow'].items())

def audit(base, output):
    assert not output.exists()
    termination = ready(base)  # Before imports that instantiate any real model.
    import torch
    from torch.nn import functional as functional
    from adapter import make_model
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    run = base/'run01'
    cfg, identity = read(base/'PROTOCOL.json'), read(run/'IDENTITY.json')
    frozen = read(base/'FROZEN_HASHES.json')
    hashes = {}
    def verify(path, expected):
        assert sha(path) == expected, str(path)
        hashes[str(path)] = expected
    for name, expected in frozen.items(): verify(base/name, expected)
    for name, expected in read(base/'AUDIT_INPUT_HASHES.json').items(): verify(base/name, expected)
    assert identity['frozen_hashes'] == frozen and identity['protocol_sha256'] == sha(base/'PROTOCOL.json')
    assert identity['torch'] == torch.__version__ and identity['numpy'] == np.__version__ and identity['python'] == sys.version
    for name, expected in read(run/'FILES.json').items(): verify(run/name, expected)
    verify(cfg['source_manifest'], cfg['source_manifest_sha256'])
    cache = Path(cfg['data'])/'cache'
    for name, expected in cfg['input_sha256'].items(): verify(cache/name, expected)
    info = read(run/'SOURCE_CACHE.json')
    verify(run/'source_maps.npy', info['sha256'])
    assert info['shape'] == [6900,90,567] and info['sample_ids_sha256'] == cfg['source_manifest_sha256']
    with np.load(cfg['source_manifest']) as data:
        ids, users = data['ids'], data['users']
    assert len(ids) == len(np.unique(ids)) == 6900 and set(users) == set(range(1,7))
    index = roles(read(run/'split.json'), ids, users)
    assert {key:len(ix) for key,ix in index.items()} == cfg['counts']
    source = np.load(run/'source_maps.npy', mmap_mode='r')
    raw = np.load(cache/'X_clutterremoved.npy', mmap_mode='r')
    # Rebuild every record's deterministic preprocessing independently in NumPy.
    # Float32 reduction order differs from torch; tolerance fixed before outcomes.
    for offset in range(0, len(ids), 64):
        x = np.array(raw[ids[offset:offset+64]], copy=True)
        expected = independent_preprocess(x)
        np.testing.assert_allclose(source[offset:offset+64], expected, rtol=2e-5, atol=2e-6)
    labels = np.load(cache/'y_clutterremoved.npy', mmap_mode='r')
    yv = np.array(labels[ids[index['validation']]], copy=True)
    yd = np.array(labels[ids[index['diagnostic']]], copy=True)
    summary = read(run/'SUMMARY.json')
    assert summary['state'] == 'complete_exploratory_source_pilot'
    synthetic = read(base/'SYNTHETIC_CHECK.json')
    scores, details = {}, {}
    for arm in ARMS:
        folder = run/arm
        checkpoints = {name:torch.load(folder/f'{name}.pt', map_location='cpu', weights_only=False)
                       for name in ('initial','first_update','best','last')}
        initial, first, best, last = [checkpoints[name] for name in ('initial','first_update','best','last')]
        assert all(value['identity'] == identity for value in checkpoints.values())
        history = [json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
        assert [row['epoch'] for row in history] == list(range(1,65))
        chosen = selection(history)
        assert initial['epoch'] == 0 and last['epoch'] == 64
        assert best['epoch'] == chosen['epoch'] == read(folder/'ARM_COMPLETE.json')['selected_epoch']
        assert best['validation_macro'] == chosen['validation_macro'] and best['validation_loss'] == chosen['validation_bce']
        assert tuple(last['best']) == (chosen['validation_macro'], -chosen['validation_bce'], -chosen['epoch'])
        model = make_model(arm, cfg['seed'])
        assert sum(p.numel() for p in model.parameters()) == synthetic['arms'][arm]['parameters']
        assert all(torch.equal(value, initial['model'][key]) for key,value in model.state_dict().items())
        unused = set(synthetic['arms'][arm]['unused_parameter_names'])
        expected_first = torch.randperm(len(index['train']), generator=torch.Generator().manual_seed(cfg['seed']+1)).numpy()[:64]
        np.testing.assert_array_equal(first['training_ids'], ids[index['train'][expected_first]])
        check_optimizer(model, first['optimizer'], 1, unused)
        steps = 64*math.ceil(len(index['train'])/64)
        check_optimizer(model, last['optimizer'], steps, unused)
        assert any(not torch.equal(value, first['model'][key]) for key,value in initial['model'].items())
        for checkpoint in checkpoints.values():
            assert all(torch.isfinite(value).all() for value in checkpoint['model'].values() if value.is_floating_point())
            assert all(torch.equal(checkpoint['model'][name], initial['model'][name]) for name in unused)
        assert all(int(value) == steps for key,value in last['model'].items() if key.endswith('num_batches_tracked'))
        torch.set_rng_state(last['torch_rng'])
        np.random.RandomState().set_state(last['numpy_rng'])
        random.Random().setstate(last['python_rng'])
        model.load_state_dict(best['model'])
        if arm == 'timemil':
            for name in ('wave1','wave2','wave3'):
                assert getattr(model,name).data_ptr() == getattr(model,name+'_').data_ptr()
                assert torch.equal(best['model'][name],best['model'][name+'_'])
        model.eval()
        def replay(ix):
            values = []
            with torch.inference_mode():
                for offset in range(0,len(ix),64):
                    x = torch.from_numpy(np.array(source[ix[offset:offset+64]], copy=True))
                    values.append(model(x).numpy())
            result = np.concatenate(values)
            assert np.isfinite(result).all()
            return result
        validation = replay(index['validation'])
        assert abs(metrics(yv,validation.argmax(1))-best['validation_macro']) < 1e-12
        bce = float(functional.binary_cross_entropy_with_logits(torch.from_numpy(validation), functional.one_hot(torch.from_numpy(yv).long(),12).float()))
        np.testing.assert_allclose(bce,best['validation_loss'],rtol=1e-4,atol=1e-5)
        diagnostic = replay(index['diagnostic'])
        with np.load(folder/'predictions_unscored.npz') as data:
            np.testing.assert_array_equal(data['ids'],ids[index['diagnostic']])
            np.testing.assert_allclose(diagnostic,data['logits'],rtol=1e-4,atol=1e-5)
            np.testing.assert_array_equal(diagnostic.argmax(1),data['logits'].argmax(1))
            scores[arm] = metrics(yd,data['logits'].argmax(1))
        assert abs(scores[arm]-summary['scores'][arm]) < 1e-12
        details[arm] = dict(selected_epoch=best['epoch'],optimizer_steps=steps,
                           all_validation_logits_replayed=len(yv),all_diagnostic_logits_replayed=len(yd),
                           optimizer_rng_bn_initial_and_unused_verified=True)
        del model,checkpoints
    cnn = read(cfg['previous_cnn_summary'])['scores']
    assert summary['cnn_reference'] == cnn
    expected_gates = gates(scores,max(cnn['full'],cnn['local']))
    for name,value in expected_gates.items(): assert summary[name] == value
    assert summary['epochs'] == 64 and summary['users7_8_labels_used'] is False
    assert summary['wall_seconds'] <= 43200 and summary['cpu_seconds'] <= 43200
    for path,expected in hashes.items(): assert sha(path) == expected, 'changed during read-only audit'
    result = dict(state='pass',scope='completed_timeMIL_read_only',scores=scores,arms=details,
                  gates=expected_gates,artifact_hashes=hashes,source_preprocessing_records_verified=6900,
                  training_termination=termination,auditor_sha256=sha(Path(__file__)),
                  audit_wall_seconds=time.monotonic()-started,users7_8_labels_used=False,
                  independent_validation=False,novel_contribution_validated=False,
                  limitation='All selected-validation/diagnostic logits and preprocessing replayed; nonselected epochs not retrained or replayed.')
    assert not output.exists()
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(state='pass',report=str(output))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    audit(BASE,args.out)
