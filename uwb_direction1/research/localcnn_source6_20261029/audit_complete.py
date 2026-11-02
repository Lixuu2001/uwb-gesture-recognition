"""Read-only completed-run verifier. Refuses all arrays until all folds freeze.

No training code imports, fitting, resampling, selection changes or partial scores.
Run through audit_launch.py so hard kills also leave a terminal receipt.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
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

def read(path):
    return json.loads(Path(path).read_text())

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def require_complete(base):
    """Metadata and byte hashes only; never opens source arrays/checkpoints."""
    run = base / 'run01'
    state = read(run / 'STATE.json')
    if state.get('state') != 'complete' or state.get('pid') is not None:
        raise RuntimeError('not_ready: training not complete')
    if not state.get('all_new_diagnostic_predictions_frozen'):
        raise RuntimeError('not_ready: aggregate prediction freeze absent')
    if set(state['completed_folds']) != set(range(1, 7)):
        raise RuntimeError('not_ready: missing fold')
    term = read(base / 'supervisor01/TERMINATION.json')
    if term['exit_code'] != 0 or term['wall_timeout']:
        raise RuntimeError('not_ready: unsuccessful training supervisor')
    assert term['supervisor_wall_seconds'] <= 43200
    assert term['child_cpu_seconds'] <= 129600
    freeze = read(run / 'ALL_NEW_PREDICTIONS_FROZEN.json')
    assert set(freeze) == set(map(str, range(2, 7)))
    for held in range(2, 7):
        folder = run / f'held{held}'
        done = read(folder / 'FOLD_COMPLETE.json')
        assert done['held'] == held and done['epochs'] == 24
        assert digest(folder / 'predictions_unscored.npz') == freeze[str(held)] == done['prediction_sha256']
        assert digest(folder / 'best.pt') == done['checkpoint_sha256']
    return term

def choose_epoch(rows):
    return sorted(rows, key=lambda row: (-row['validation_macro'], row['validation_loss'], row['epoch']))[0]

def independent_macro(y, predicted):
    classes = np.unique(y)
    assert len(y) == len(predicted) and len(classes)
    return float(sum(np.count_nonzero((y == c) & (predicted == c)) / np.count_nonzero(y == c)
                     for c in classes) / len(classes))

def gate(gains):
    return bool(np.mean(gains) >= .01 and np.count_nonzero(np.asarray(gains) > 0) >= 4
                and min(gains) >= -.02)

def verify_split(split, ids, users, held):
    val = held % 6 + 1
    index = {'train': np.where((users != held) & (users != val))[0],
             'validation': np.where(users == val)[0], 'diagnostic': np.where(users == held)[0]}
    for role, ix in index.items():
        np.testing.assert_array_equal(split[role], ids[ix])
    assert not (set(split['train']) & set(split['validation']) or
                set(split['train']) & set(split['diagnostic']) or
                set(split['validation']) & set(split['diagnostic']))
    return index

def audit(base, output):
    # This guard deliberately precedes torch import and every array load.
    term = require_complete(base)
    import torch
    import torch.nn.functional as functional
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    cfg = read(base / 'PROTOCOL.json')
    run = base / 'run01'
    identity = read(run / 'IDENTITY.json')
    frozen = read(base / 'FROZEN_HASHES.json')
    hashes = {}
    def checked_hash(path, expected):
        actual = digest(path)
        assert actual == expected, str(path)
        hashes[str(path)] = actual
    for path, expected in frozen.items():
        checked_hash(base / path, expected)
    assert identity['protocol_sha256'] == digest(base / 'PROTOCOL.json')
    assert identity['frozen_hashes'] == frozen
    assert identity['torch'] == torch.__version__ and identity['numpy'] == np.__version__
    for path, expected in read(run / 'FILES.json').items():
        checked_hash(run / path, expected)
    checked_hash(cfg['source_manifest'], cfg['source_manifest_sha256'])
    checked_hash(cfg['source_maps'], cfg['source_maps_sha256'])
    for name, expected in cfg['input_sha256'].items():
        checked_hash(Path(cfg['data']) / 'cache' / name, expected)
    reused = cfg['reused_held1']
    for key in ('prediction', 'checkpoint', 'audit'):
        checked_hash(reused[key], reused[key + '_sha256'])
    assert read(reused['audit'])['state'] == 'pass'
    with np.load(cfg['source_manifest']) as z:
        ids, users = z['ids'], z['users']
    assert len(ids) == len(np.unique(ids)) == 6900 and set(users) == set(range(1, 7))
    sys.path.insert(0, str(Path(cfg['model_code']).parent))
    from model import MVGLF
    source = np.load(cfg['source_maps'], mmap_mode='r')
    assert source.shape == (6900, 3, 64, 128)
    # The completed-run guard has now passed; index only source 1..6 labels.
    label_map = np.load(Path(cfg['data']) / 'cache/y_clutterremoved.npy', mmap_mode='r')
    summary = read(run / 'SUMMARY.json')
    assert summary['state'] == 'complete_source_confirmation'
    assert summary['users7_8_labels_used'] is False
    rows, details = [], []
    for held in range(1, 7):
        directory = Path(reused['checkpoint']).parent if held == 1 else run / f'held{held}'
        split = read(directory.parent / 'split.json') if held == 1 else read(directory / 'split.json')
        index = verify_split(split, ids, users, held)
        assert [len(index[k]) for k in ('train', 'validation', 'diagnostic')] == cfg['counts'][str(held)]
        checkpoints = {name: torch.load(directory / f'{name}.pt', map_location='cpu', weights_only=False)
                       for name in ('initial', 'first_update', 'best', 'last')}
        history = [json.loads(line) for line in (directory / 'epochs.jsonl').read_text().splitlines()]
        assert [row['epoch'] for row in history] == list(range(1, 25))
        chosen = choose_epoch(history)
        initial, first, best, last = [checkpoints[name] for name in ('initial', 'first_update', 'best', 'last')]
        assert initial['epoch'] == 0 and last['epoch'] == 24
        assert best['epoch'] == chosen['epoch']
        if held != 1:
            assert best['epoch'] == read(directory / 'FOLD_COMPLETE.json')['selected_epoch']
        for key in ('validation_macro', 'validation_loss'):
            assert best[key] == chosen[key]
        assert tuple(last['best']) == (chosen['validation_macro'], -chosen['validation_loss'], -chosen['epoch'])
        if held != 1:
            assert all(c['identity'] == identity and c['held'] == held for c in checkpoints.values())
        torch.manual_seed(cfg['seed'])
        model = MVGLF('local').to(memory_format=torch.channels_last)
        assert sum(p.numel() for p in model.parameters()) == 350220
        assert all(torch.equal(v, initial['model'][key]) for key, v in model.state_dict().items())
        assert any(not torch.equal(value, first['model'][key]) for key, value in initial['model'].items())
        expected_first = torch.randperm(len(index['train']), generator=torch.Generator().manual_seed(cfg['seed'] + 1)).numpy()[:32]
        np.testing.assert_array_equal(first['training_ids'], ids[index['train'][expected_first]])
        for checkpoint, step in ((first, 1), (last, 24 * math.ceil(len(index['train']) / 32))):
            optimizer = checkpoint['optimizer']
            assert len(optimizer['param_groups']) == 1
            group = optimizer['param_groups'][0]
            assert group['lr'] == .001 and group['weight_decay'] == .0005
            assert len(optimizer['state']) == len(list(model.parameters()))
            assert {int(v['step']) for v in optimizer['state'].values()} == {step}
            assert all(torch.isfinite(v[key]).all() for v in optimizer['state'].values() for key in ('exp_avg', 'exp_avg_sq'))
            restored = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0005)
            restored.load_state_dict(optimizer)
            for parameter in model.parameters():
                assert restored.state[parameter]['exp_avg'].shape == parameter.shape
                assert restored.state[parameter]['exp_avg_sq'].shape == parameter.shape
        for checkpoint in checkpoints.values():
            assert all(torch.isfinite(v).all() for v in checkpoint['model'].values() if v.is_floating_point())
        expected_steps = 24 * math.ceil(len(index['train']) / 32)
        assert all(int(v) == expected_steps for k, v in last['model'].items() if k.endswith('num_batches_tracked'))
        torch.set_rng_state(last['torch_rng'])
        np.random.RandomState().set_state(last['numpy_rng'])
        random.Random().setstate(last['python_rng'])
        if best['epoch'] == 24:
            assert all(torch.equal(v, last['model'][key]) for key, v in best['model'].items())
        model.load_state_dict(best['model'])
        model.eval()
        def replay(ix):
            pieces = []
            with torch.inference_mode():
                for start in range(0, len(ix), 64):
                    batch = torch.from_numpy(np.array(source[ix[start:start+64]], copy=True))
                    pieces.append(model(batch.contiguous(memory_format=torch.channels_last)).numpy())
            value = np.concatenate(pieces)
            assert np.isfinite(value).all()
            return value
        validation_logits = replay(index['validation'])
        yv = np.array(label_map[ids[index['validation']]], copy=True)
        assert abs(independent_macro(yv, validation_logits.argmax(1)) - best['validation_macro']) < 1e-12
        loss = float(functional.cross_entropy(torch.from_numpy(validation_logits), torch.from_numpy(yv).long()))
        np.testing.assert_allclose(loss, best['validation_loss'], rtol=1e-4, atol=1e-5)
        prediction_path = reused['prediction'] if held == 1 else directory / 'predictions_unscored.npz'
        diagnostic_logits = replay(index['diagnostic'])
        with np.load(prediction_path) as z:
            np.testing.assert_array_equal(z['ids'], ids[index['diagnostic']])
            np.testing.assert_allclose(diagnostic_logits, z['logits'], rtol=1e-4, atol=1e-5)
            np.testing.assert_array_equal(diagnostic_logits.argmax(1), z['logits'].argmax(1))
            y = np.array(label_map[z['ids']], copy=True)
            score = independent_macro(y, z['logits'].argmax(1))
        ref = cfg['minirocket_references'][str(held)]
        checked_hash(ref['path'], ref['sha256'])
        with np.load(ref['path']) as z:
            selected = z['users'] == held
            np.testing.assert_array_equal(z['ids'][selected], ids[index['diagnostic']])
            np.testing.assert_array_equal(z['y'][selected], y)
            reference = independent_macro(y, z['minirocket'][selected])
        row = dict(held=held, local=score, minirocket=reference, difference=score-reference)
        original = next(row for row in summary['rows'] if row['held'] == held)
        for key in ('local', 'minirocket', 'difference'):
            assert abs(row[key] - original[key]) < 1e-12
        rows.append(row)
        details.append(dict(held=held, selected_epoch=best['epoch'], optimizer_steps=expected_steps,
                            validation_logits_replayed=len(yv), diagnostic_logits_replayed=len(y),
                            initial_first_update_optimizer_rng_bn_verified=True))
        del model, checkpoints
    gains = [row['difference'] for row in rows]
    for key in ('local', 'minirocket'):
        assert abs(np.mean([row[key] for row in rows]) - summary['means'][key]) < 1e-12
    assert abs(np.mean(gains) - summary['mean_difference']) < 1e-12
    assert summary['wins'] == sum(g > 0 for g in gains)
    assert summary['descriptive_continuation_gate'] == gate(gains)
    assert summary['new_epochs'] == 120 and summary['new_trajectories'] == 5 and summary['reused_fold'] == 1
    assert summary['wall_seconds'] <= 43200 and summary['cpu_seconds'] <= 129600
    for path, expected in hashes.items():
        assert digest(path) == expected, 'Artifact changed during read-only audit: ' + path
    result = dict(state='pass', scope='completed_source_sixfold_read_only', rows=rows, folds=details,
                  descriptive_continuation_gate=gate(gains), artifact_hashes=hashes,
                  training_termination=term, auditor_sha256=digest(Path(__file__)),
                  audit_wall_seconds=time.monotonic()-started, users7_8_labels_used=False,
                  independent_validation=False, novel_contribution_validated=False,
                  limitation='Selected-validation and all diagnostic logits replayed; historical nonselected epochs not retrained or functionally replayed.')
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(state='pass', report=str(output), folds=6)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(BASE, args.out)
