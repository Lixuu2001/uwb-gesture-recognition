"""Independent CPU-only audit of the formal B result (no training/source imports).

Only this script and INDEPENDENT_AUDIT.{json,md} may be written. All experiment
inputs, checkpoints, predictions, manifests, plans, and source files are read-only.
The NumPy/SciPy replay below does not import the original analysis or training code.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
import gc
import hashlib
import itertools
import json
import math
import platform
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack
import torch
torch.set_num_threads(2)

ROOT = Path('D:/uwb_gestures/experiments/p1_20260928')
OUT = ROOT/'budget_s2026'
BASE = ROOT/'baseline_s2026'
MV = ROOT/'multiview_s2026'
PREP = ROOT.parent/'p1_20260905'/'prepared'
CODE = Path('C:/Users/19252/.codex/worktrees/uwb-user-calibration/uwb_gestures/experiments/new_routes')
NAMES = ['L', 'T', 'R', 'LT', 'LR', 'TR', 'LTR']
VIEWS = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
COST = np.array([1, 1, 1, 2, 2, 2, 3], dtype=np.int64)
MASKS = np.array([[int(v in s) for v in range(3)] for s in VIEWS], dtype=np.int8)
BUDGETS = np.arange(1., 3.01, .25)
ROLES = ['train', 'val', 'test']
REPORT = {'status': 'RUNNING', 'audit_scope': 'B budget_s2026 and complete multiview/baseline provenance; value_policy and latency excluded',
          'started_utc': datetime.now(timezone.utc).isoformat(), 'checks': Counter(), 'failures': [],
          'file_inventory': {}, 'array_inventory': {}, 'folds': [], 'numerical_max_abs_error': {},
          'method': 'Independent NumPy/SciPy calculation; CPU checkpoint loading; no original analysis/training imports, CUDA or retraining.'}
HASH_CACHE = {}


def check(condition, name, detail=None):
    REPORT['checks'][name.split('/')[0]] += 1
    if not bool(condition):
        REPORT['failures'].append({'check': name, 'detail': detail})


def close(actual, expected, name, atol=1e-10):
    a, e = np.asarray(actual), np.asarray(expected)
    ok = a.shape == e.shape and np.all(np.isfinite(a)) and np.all(np.isfinite(e))
    err = float(np.max(np.abs(a.astype(float)-e.astype(float)))) if ok and a.size else (0. if ok else None)
    key = name.split('/')[0]
    if err is not None:
        REPORT['numerical_max_abs_error'][key] = max(REPORT['numerical_max_abs_error'].get(key, 0.), err)
    check(ok and err <= atol, name, {'max_abs_error': err, 'tolerance': atol, 'actual_shape': list(a.shape), 'expected_shape': list(e.shape)})


def exact(actual, expected, name):
    if isinstance(actual, (np.ndarray, torch.Tensor)) or isinstance(expected, (np.ndarray, torch.Tensor)):
        a, b = np.asarray(actual), np.asarray(expected)
        check(a.shape == b.shape and np.array_equal(a, b), name, {'actual_shape': list(a.shape), 'expected_shape': list(b.shape)})
    else:
        check(actual == expected, name, {'actual': actual, 'expected': expected})


def compare_tree(actual, expected, name, atol=1e-10):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            check(False, name, 'Expected dictionary'); return
        exact(sorted(actual), sorted(expected), name+'/keys')
        for k in expected:
            if k in actual:
                compare_tree(actual[k], expected[k], name+'/'+k, atol)
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            check(False, name, 'Sequence length mismatch'); return
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare_tree(a, e, name+'/'+str(i), atol)
    elif isinstance(expected, float):
        close(actual, expected, name, atol)
    else:
        exact(actual, expected, name)


def sha(path):
    path = Path(path).resolve()
    key = str(path)
    before = path.stat()
    signature = (before.st_size, before.st_mtime_ns)
    if key in HASH_CACHE and HASH_CACHE[key][0] == signature:
        return HASH_CACHE[key][1]
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    after = path.stat()
    check(signature == (after.st_size, after.st_mtime_ns), 'immutability/hash_read/'+key)
    value = h.hexdigest()
    HASH_CACHE[key] = (signature, value)
    REPORT['file_inventory'][key] = {'sha256': value, 'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns}
    return value


def load_json(path):
    sha(path)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def arrhash(a):
    a = np.asarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+np.ascontiguousarray(a).tobytes()).hexdigest()


def archive(path):
    sha(path)
    with np.load(path, allow_pickle=False) as z:
        result = {k: z[k] for k in z.files}
    REPORT['array_inventory'][str(Path(path).resolve())] = {}
    for k, a in result.items():
        finite = bool(np.isfinite(a).all())
        check(finite, 'array_finiteness/'+str(path)+'/'+k)
        REPORT['array_inventory'][str(Path(path).resolve())][k] = {
            'dtype': str(a.dtype), 'shape': list(a.shape), 'elements': int(a.size),
            'array_sha256': arrhash(a), 'finite': finite}
    return result


def statehash(state):
    h = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(tensor.dtype).encode()); h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def load_checkpoint(path):
    sha(path)
    # Locally generated provenance-checked checkpoints contain RNG/NumPy state.
    return torch.load(path, map_location='cpu', weights_only=False)


def macro(prediction, y):
    return float(np.mean([np.count_nonzero(prediction[y == c] == c)/np.count_nonzero(y == c)
                          for c in np.unique(y)]))


def accuracy7(z, y):
    return np.array([macro(np.argmax(z[:, j, :], axis=1), y) for j in range(7)])


def sample_weights(y):
    _, inverse, count = np.unique(y, return_inverse=True, return_counts=True)
    return 1.0/(len(count)*count[inverse])


def cross_entropy(z, y):
    z = np.asarray(z, dtype=np.float64)
    maximum = np.max(z, axis=1)
    return float(np.mean(maximum+np.log(np.exp(z-maximum[:, None]).sum(axis=1))-z[np.arange(len(y)), y]))


def score_uncertainty(z, mode):
    z = np.asarray(z, dtype=np.float64)
    probabilities = np.exp(z-z.max(axis=1)[:, None])
    probabilities /= probabilities.sum(axis=1)[:, None]
    if mode == 'confidence':
        return 1.-np.max(probabilities, axis=1)
    return -np.sum(probabilities*np.log(np.maximum(probabilities, 1e-30)), axis=1)/math.log(z.shape[1])


def causal_replay(z, parameters):
    """Only materialize an observed row/subset on each stage, without labels."""
    f, p = parameters['first'], parameters['pair']
    n = len(z)
    route = np.empty(n, dtype=np.int64)
    pred = np.empty(n, dtype=np.int64)
    requested = np.zeros((n, 3), dtype=bool)
    all_rows = np.arange(n)
    def acquire(rows, subset):
        requested[np.ix_(rows, VIEWS[subset])] = True
        return z[rows, subset, :]
    observed1 = acquire(all_rows, f)
    route[:] = f; pred[:] = observed1.argmax(axis=1)
    next_rows = all_rows[score_uncertainty(observed1, parameters['mode']) > parameters['threshold1']]
    if len(next_rows):
        observed2 = acquire(next_rows, p)
        route[next_rows] = p; pred[next_rows] = observed2.argmax(axis=1)
        last_rows = next_rows[score_uncertainty(observed2, parameters['mode']) > parameters['threshold2']]
        if len(last_rows):
            observed3 = acquire(last_rows, 6)
            route[last_rows] = 6; pred[last_rows] = observed3.argmax(axis=1)
    return pred, route, requested


def select_validation(z, y):
    """Exhaustive independent cached grid: accuracy objective and stated ties."""
    acc = accuracy7(z, y)
    first = int(np.flatnonzero(acc[:3] >= np.max(acc[:3])-1e-12)[0])
    pair_candidates = [i for i in (3, 4, 5) if first in VIEWS[i]]
    pair = sorted(pair_candidates, key=lambda j: (-round(float(acc[j]), 12), j))[0]
    predicted = z.argmax(axis=2)
    w = sample_weights(y)
    candidates = []
    for mode in ['confidence', 'entropy']:
        u1, u2 = score_uncertainty(z[:, first], mode), score_uncertainty(z[:, pair], mode)
        grids = [np.unique(np.concatenate(([-1.], np.quantile(u, np.arange(21)/20), [2.]))) for u in (u1, u2)]
        for t1, t2 in itertools.product(*grids):
            more = u1 > t1
            last = more & (u2 > t2)
            chosen = np.where(last, 6, np.where(more, pair, first))
            correct = predicted[np.arange(len(y)), chosen] == y
            candidates.append({'params': {'first': first, 'pair': pair, 'mode': mode,
                                          'threshold1': float(t1), 'threshold2': float(t2)},
                               'val_accuracy': float(np.sum(w*correct)),
                               'val_cost': float((len(y)+np.count_nonzero(more)+np.count_nonzero(last))/len(y))})
    answer = {}
    for ceiling in BUDGETS:
        candidates_here = [c for c in candidates if c['val_cost'] <= ceiling+1e-12]
        best = max(c['val_accuracy'] for c in candidates_here)
        tied = [c for c in candidates_here if c['val_accuracy'] >= best-1e-12]
        answer[str(float(ceiling))] = sorted(tied, key=lambda c: (
            c['val_cost'], c['params']['mode'], -c['params']['threshold1'], -c['params']['threshold2']))[0]
    return answer, acc, len(candidates)


def mixture(acc, ceiling):
    """All one/two-subset vertices of the simplex intersected by cost <= B."""
    vertices = [np.eye(7)[i] for i in range(7) if COST[i] <= ceiling+1e-12]
    for a, b in itertools.combinations(range(7), 2):
        if COST[a] == COST[b]:
            continue
        fraction = (ceiling-COST[a])/(COST[b]-COST[a])
        if 0 < fraction < 1:
            v = np.zeros(7); v[a] = 1-fraction; v[b] = fraction
            vertices.append(v)
    values = [float(v@acc) for v in vertices]
    top = max(values)
    chosen = min([v for v, value in zip(vertices, values) if value >= top-1e-12],
                 key=lambda v: (float(v@COST), tuple(-v)))
    lp = linprog(-acc, A_ub=COST[None, :], b_ub=[ceiling], A_eq=np.ones((1, 7)), b_eq=[1.],
                 bounds=(0, 1), method='highs')
    check(lp.success, 'mixture_lp/success')
    if lp.success:
        close(float(chosen@acc), -float(lp.fun), 'mixture_lp/optimal_value')
    check(np.count_nonzero(chosen) <= 2 and np.min(chosen) >= 0 and chosen@COST <= ceiling+1e-12,
          'mixture_lp/valid_cost_ceiling')
    return chosen


def oracle(z, y, first, ceiling):
    """Analytic fractional program, retaining correct cheap decisions.

    Each error's only non-dominated useful option is its cheapest correct
    superset. Sort by class-balanced gain / extra physical-view cost.
    """
    correct = np.argmax(z, axis=2) == y[:, None]
    allowed = np.array([j for j in range(7) if first in VIEWS[j]], dtype=np.int64)
    initial = correct[:, first]
    w = sample_weights(y)
    add = COST[allowed]-1
    candidates = np.where(correct[:, allowed], add[None, :], 99)
    minimum = candidates.min(axis=1)
    rescue = np.flatnonzero((~initial) & (minimum < 99))
    # Lexsort explicitly reproduces descending gain/cost, descending row ID ties.
    order = rescue[np.lexsort((-rescue, -(w[rescue]/minimum[rescue])))]
    requirements = minimum[order].astype(float)
    starts = np.cumsum(requirements)-requirements
    available = len(y)*(ceiling-1.)
    allocation = np.clip((available-starts)/requirements, 0, 1)
    gain = float(np.sum(w[order]*allocation))
    result = {'macro_accuracy': float(np.sum(w[initial])+gain),
              'mean_views': float(1.+np.sum(allocation*requirements)/len(y)),
              'uses_test_labels': True, 'rescuable_rows': len(rescue)}
    return result


def oracle_lp(z, y, first, ceiling):
    """A separate full row-by-action LP, without the cheapest-rescue reduction."""
    allowed = np.array([j for j in range(7) if first in VIEWS[j]], dtype=int)
    n, k = len(y), len(allowed)
    reward = ((z[:, allowed].argmax(2) == y[:, None])*sample_weights(y)[:, None]).ravel()
    eq = coo_matrix((np.ones(n*k), (np.repeat(np.arange(n), k), np.arange(n*k))), shape=(n, n*k)).tocsr()
    c = np.tile(COST[allowed]/n, n)
    solution = linprog(-reward, A_ub=c[None, :], b_ub=[ceiling], A_eq=eq, b_eq=np.ones(n),
                       bounds=(0, 1), method='highs')
    check(solution.success, 'oracle_lp/success')
    return -float(solution.fun) if solution.success else float('nan')


def bootstrap(delta):
    rng = np.random.default_rng(20260928)
    indices = rng.integers(0, len(delta), size=(10000, len(delta)))
    means = delta[indices].mean(axis=1)
    return {'n_users': len(delta), 'mean_difference_pp': float(delta.mean()),
            'per_user_difference_pp': delta.tolist(), 'positive_users': int(np.count_nonzero(delta > 1e-10)),
            'worst_user_difference_pp': float(delta.min()), 'ci95_pp': np.quantile(means, [.025, .975]).tolist()}


def verify_history(result, parts, fold, kind):
    history = result['history']
    exact([r['epoch'] for r in history], list(range(len(history))), 'history/contiguous/'+kind+str(fold))
    selected = int(np.argmin([r['val_ce'] for r in history]))
    exact(result['best_epoch'], selected, 'history/earliest_best/'+kind+str(fold))
    close(result['best_val_ce'], history[selected]['val_ce'], 'history/best_ce/'+kind+str(fold))
    best, stale, should_stop = history[0]['val_ce'], 0, False
    shuffle = np.random.default_rng(2026+fold)
    subset_rng = np.random.default_rng(20260+fold)
    expected_steps = math.ceil(len(parts['train'])/128)
    for row in history[1:]:
        check(not should_stop, 'history/no_epoch_after_stop/'+kind+str(fold))
        if row['val_ce'] < best:
            best, stale = row['val_ce'], 0
        else:
            stale += 1
        should_stop = row['epoch'] >= 12 and stale >= 8
        exact(row['source_rows'], len(parts['train']), 'history/source_rows/'+kind+str(fold))
        exact(row['steps'], expected_steps, 'history/full_epoch_steps/'+kind+str(fold))
        check(row['seconds'] > 0 and np.isfinite(row['train_loss']), 'history/finite_training/'+kind+str(fold))
        order = shuffle.permutation(parts['train'])
        trace = hashlib.sha256(); counts = np.zeros(7, dtype=int)
        for offset in range(0, len(order), 128):
            trace.update(order[offset:offset+128].tobytes())
            if kind == 'multiview':
                s = int(subset_rng.integers(7)); trace.update(bytes(VIEWS[s])); counts[s] += 1
        exact(row['episode_hash'], trace.hexdigest(), 'history/rng_trace/'+kind+str(fold))
        if kind == 'multiview':
            exact(row['subset_batch_counts'], counts.tolist(), 'history/subset_counts/'+str(fold))
    for row in history:
        if kind == 'multiview':
            close(row['val_ce'], float(np.mean(row['val_ce_by_subset'])), 'history/mean_seven_ce/'+str(fold))
    check(should_stop or len(history)-1 == 40, 'history/valid_termination/'+kind+str(fold))
    exact(result['stop_reason'], 'patience' if should_stop else 'max_epochs', 'history/stop_reason/'+kind+str(fold))
    return {'best_ce': best, 'best_epoch': selected, 'stale': stale}, shuffle, subset_rng


def verify_progress(directory, result, parts, fold, kind, selection, shuffle, subset_rng):
    progress = load_checkpoint(directory/'progress.pt')
    info = progress['info']
    compare_tree(info['identity'], result['identity'], 'progress/identity/'+kind+str(fold))
    compare_tree(info['history'], result['history'], 'progress/history/'+kind+str(fold))
    compare_tree(info['selection'], selection, 'progress/selection/'+kind+str(fold))
    exact(info['epoch'], len(result['history'])-1, 'progress/last_epoch/'+kind+str(fold))
    exact(info['best_file_sha256'], sha(directory/'best.pt'), 'progress/best_file_sha/'+kind+str(fold))
    exact(statehash(info['best_checkpoint']['state']), result['selected_state_sha256'], 'progress/embedded_best_state/'+kind+str(fold))
    exact(info['best_checkpoint']['epoch'], result['best_epoch'], 'progress/embedded_best_epoch/'+kind+str(fold))
    close(info['best_checkpoint']['val_ce'], result['best_val_ce'], 'progress/embedded_best_ce/'+kind+str(fold))
    exact(progress['sampler']['ids'], parts['train'], 'progress/sampler_ids/'+kind+str(fold))
    compare_tree(progress['sampler']['rng'], shuffle.bit_generator.state, 'progress/shuffle_rng/'+kind+str(fold))
    if kind == 'multiview':
        compare_tree(progress['sampler']['subset_rng'], subset_rng.bit_generator.state, 'progress/subset_rng/'+str(fold))
        compare_tree(progress['sampler']['identity'], {'ids': arrhash(parts['train']), 'shuffle_seed': 2026+fold, 'subset_seed': 20260+fold}, 'progress/sampler_identity/'+str(fold))
    for group in progress['optimizer']['param_groups']:
        close(group['lr'], 1e-4, 'progress/lr/'+kind+str(fold))
        close(group['weight_decay'], 1e-4, 'progress/wd/'+kind+str(fold))
        close(group['betas'], [.9, .999], 'progress/betas/'+kind+str(fold))
        close(group['eps'], 1e-8, 'progress/eps/'+kind+str(fold))
    expected_steps = math.ceil(len(parts['train'])/128)*(len(result['history'])-1)
    exact(sorted(set(int(o['step']) for o in progress['optimizer']['state'].values())), [expected_steps], 'progress/optimizer_steps/'+kind+str(fold))
    check(all(torch.isfinite(v).all().item() for v in progress['state'].values()), 'progress/finite_weights/'+kind+str(fold))
    del progress


def verify_sources():
    b, m = load_json(BASE/'manifest.json'), load_json(MV/'manifest.json')
    for root, manifest in [(BASE, b), (MV, m)]:
        for i, (path, expected) in enumerate(manifest['source']['code'].items()):
            exact(sha(path), expected, 'source/live/'+str(path))
            exact(sha(root/'source'/f'{i:02d}_{Path(path).name}'), expected, 'source/frozen/'+str(path))
        for path, expected in manifest['source']['data'].items():
            exact(sha(path), expected, 'source/data/'+str(path))
        exact(sha(ROOT/'PREREGISTRATION.md'), manifest['source']['plan_sha256'], 'source/preregistration/'+root.name)
    exact(m['source']['baseline_manifest_sha256'], sha(BASE/'manifest.json'), 'source/baseline_manifest')
    for name, wanted in {'seed': 2026, 'fusion_seed': 2026, 'batch': 128, 'microbatch': 16, 'lr': 1e-4,
                         'weight_decay': 1e-4, 'epochs': 40, 'min_epochs': 12, 'patience': 8,
                         'smoke': False, 'smoke_steps': 0, 'eval_batch': 16, 'subsets': [list(s) for s in VIEWS],
                         'criterion': 'cross_entropy only', 'attention': 'MATH', 'view_order': ['Left', 'Top', 'Right'],
                         'selection': 'earliest minimum mean seven-subset validation CE, including epoch 0',
                         'precision': 'float32 weights; float16 inputs; bfloat16 autocast',
                         'subset_sampling': 'one uniform nonempty subset per effective batch'}.items():
        exact(m['config'].get(name), wanted, 'configuration/multiview/'+name)
    for name, wanted in {'seed': 2026, 'warmup_batch': 128, 'warmup_microbatch': 32, 'warmup_lr': 1e-4,
                         'weight_decay': 1e-4, 'warmup_epochs': 40, 'warmup_min': 12, 'warmup_patience': 8,
                         'smoke': False, 'smoke_steps': 0, 'eval_batch': 32, 'attention': 'MATH'}.items():
        exact(b['config'].get(name), wanted, 'configuration/baseline/'+name)
    for number in (1, 2, 3):
        path = ROOT/f'AMENDMENT_{number:03d}.md'
        frozen_plan = CODE.parents[1]/'docs'/'superpowers'/'plans'/f'2026-09-28-uwb-amendment-{number:03d}.md'
        sha(path)
        if frozen_plan.exists():
            exact(sha(path), sha(frozen_plan), 'source/amendment_copy/'+str(number))
    for root in [BASE, MV]:
        q = load_json(root/'queue_status.json')
        exact(q['state'], 'complete', 'completion/queue/'+root.name)
        exact(q['folds'], list(range(1, 9)), 'completion/queue_folds/'+root.name)
        exact(q['smoke'], False, 'completion/no_smoke/'+root.name)
    return b, m


def verify_split_archive(data, parts, y, subjects, context):
    ids = np.concatenate([parts[k] for k in ROLES])
    exact(data['ids'], ids, 'partition/'+context+'/ids_order')
    exact(data['y'], y[ids], 'partition/'+context+'/labels')
    exact(data['subjects'], subjects[ids], 'partition/'+context+'/subjects')
    exact(data['role'], np.concatenate([np.full(len(parts[k]), i, dtype=np.int8) for i, k in enumerate(ROLES)]), 'partition/'+context+'/role')
    for role in ROLES:
        if role+'_ids' in data:
            exact(data[role+'_ids'], parts[role], 'partition/'+context+'/'+role+'_ids')


def verify_matrix(fold, parts, y, subjects, baseline_identity):
    file = BASE/'matrices'/f'fold{fold}.npz'
    z, meta = archive(file), load_json(file.with_suffix('.json'))
    exact(sha(file), meta['matrix_sha256'], 'smoothing/file_hash/'+str(fold))
    exact(meta['matrix_sha256'], baseline_identity['matrix_sha256'], 'smoothing/model_lineage/'+str(fold))
    for role in ROLES:
        exact(z[role+'_ids'], parts[role], 'smoothing/partition/'+role+str(fold))
    exact(meta['identity']['training_subjects'], np.unique(subjects[parts['train']]).tolist(), 'smoothing/source_users/'+str(fold))
    exact(meta['identity']['partitions'], {k: arrhash(v) for k, v in parts.items()}, 'smoothing/partition_hashes/'+str(fold))
    exact(meta['identity']['fold'], fold, 'smoothing/fold/'+str(fold))
    exact(meta['identity']['validation_subject'], fold%8+1, 'smoothing/val/'+str(fold))
    check(meta['validation_used_for_fit'] is False and meta['test_used_for_fit'] is False, 'smoothing/no_target_fit_flags/'+str(fold))
    exact(meta['class_counts'], np.bincount(y[parts['train']], minlength=12).tolist(), 'smoothing/counts/'+str(fold))
    exact(meta['warnings'], 0, 'smoothing/no_warning/'+str(fold))
    exact(len(meta['oof_fits']), 6, 'smoothing/oof_six/'+str(fold))
    for row in meta['oof_fits']:
        held = row['held_subject']
        exact(row['fit_ids'], parts['train'][subjects[parts['train']] != held].tolist(), 'smoothing/fit_ids/'+str(fold)+str(held))
        exact(row['predict_ids'], parts['train'][subjects[parts['train']] == held].tolist(), 'smoothing/predict_ids/'+str(fold)+str(held))
        exact(row['warnings'], [], 'smoothing/fit_warnings/'+str(fold)+str(held))
    close(z['oof_train'].sum(1), np.ones(len(parts['train'])), 'smoothing/oof_normalized/'+str(fold))
    check(np.min(z['oof_train']) >= 0 and np.max(z['oof_train']) <= 1, 'smoothing/oof_probabilities/'+str(fold))
    diagonal = float(z['oof_train'][np.arange(len(parts['train'])), y[parts['train']]].mean())
    expected = np.full((12, 12), (1-diagonal)/11, dtype=np.float32); np.fill_diagonal(expected, diagonal)
    exact(z['matrix'], expected, 'smoothing/reconstructed_matrix/'+str(fold))
    close(meta['diagonal'], expected[0, 0], 'smoothing/diagonal/'+str(fold))
    for path, expected_hash in meta['identity']['data'].items():
        exact(sha(path), expected_hash, 'smoothing/data_hash/'+str(fold))
    for path, expected_hash in meta['identity']['codes'].items():
        exact(sha(path), expected_hash, 'smoothing/code_hash/'+str(fold))
    proxy = meta['identity']['proxy']
    exact(sha(proxy['path']), proxy['sha256'], 'smoothing/proxy_hash/'+str(fold))
    exact(proxy['fitted_parameters'], False, 'smoothing/raw_proxy/'+str(fold))


def cpu_fuser_diagnostic(features, logits, weights):
    """Every saved MV feature/logit element; cross-device BF16 is diagnostic."""
    counts = Counter(); worst = 0.; square = 0.
    W1, b1, W2, b2 = [weights[k].to(torch.bfloat16) for k in ['fuser.0.weight', 'fuser.0.bias', 'fuser.3.weight', 'fuser.3.bias']]
    with torch.no_grad():
        for start in range(0, len(features), 256):
            f = torch.from_numpy(features[start:start+256]); n = len(f)
            for j, views in enumerate(VIEWS):
                x = torch.zeros((n, 771), dtype=torch.bfloat16)
                x[:, :768].reshape(n, 3, 256)[:, views] = f[:, views].to(torch.bfloat16)
                x[:, 768+np.array(views)] = 1
                hidden = torch.nn.functional.gelu(torch.nn.functional.linear(x, W1, b1))
                predicted = torch.nn.functional.linear(hidden, W2, b2).float().numpy()
                reference = logits[start:start+n, j]
                error = np.abs(predicted-reference)
                worst = max(worst, float(error.max())); square += float(np.sum(error.astype(float)**2))
                counts['elements'] += reference.size
                counts['bitwise_equal_elements'] += int(np.count_nonzero(predicted == reference))
                counts['argmax_differences'] += int(np.count_nonzero(predicted.argmax(1) != reference.argmax(1)))
    return {**counts, 'max_abs_logit_difference': worst, 'rms_logit_difference': math.sqrt(square/counts['elements']),
            'interpretation': 'CPU BF16 fusion replay from every saved feature; numerical differences from GPU BF16 are recorded, not treated as exact original-device replay.'}


def fold_audit(fold, y, subjects, keep, base_manifest, mv_manifest):
    before_failures = len(REPORT['failures'])
    parts = {'train': np.flatnonzero(keep & ~np.isin(subjects, [fold, fold%8+1])),
             'val': np.flatnonzero(keep & (subjects == fold%8+1)), 'test': np.flatnonzero(keep & (subjects == fold))}
    ids = np.concatenate(list(parts.values()))
    exact(np.sort(ids), np.flatnonzero(keep), 'partition/union/'+str(fold))
    exact(len(np.unique(ids)), 9300, 'partition/no_overlap/'+str(fold))
    expected_split = {'ids': {k: v.tolist() for k, v in parts.items()},
                      'training_subjects': np.unique(subjects[parts['train']]).tolist(), 'val': fold%8+1, 'test': fold}
    bdir, mdir, odir = BASE/f'fold{fold}'/'warmup', MV/f'fold{fold}', OUT/f'fold{fold}'
    for path in [BASE/f'fold{fold}'/'split.json', mdir/'split.json']:
        compare_tree(load_json(path), expected_split, 'partition/split/'+str(path))
    br, mr = load_json(bdir/'result.json'), load_json(mdir/'result.json')
    for result, manifest, directory, kind in [(br, base_manifest, bdir, 'baseline'), (mr, mv_manifest, mdir, 'multiview')]:
        ident = result['identity']
        compare_tree({'source': ident['source'], 'config': ident['config']}, manifest, 'identity/manifest/'+kind+str(fold))
        exact(ident['fold'], fold, 'identity/fold/'+kind+str(fold))
        exact(ident['arm'], 'warmup' if kind == 'baseline' else kind, 'identity/arm/'+kind+str(fold))
        exact(ident['partitions'], {k: arrhash(v) for k, v in parts.items()}, 'identity/partition_hashes/'+kind+str(fold))
        exact(result['training_subjects'], expected_split['training_subjects'], 'identity/training_users/'+kind+str(fold))
        exact(result['validation_subject'], fold%8+1, 'identity/val_user/'+kind+str(fold))
        exact(result['test_subject'], fold, 'identity/test_user/'+kind+str(fold))
        for name, expected_hash in result['files'].items():
            exact(sha(directory/name), expected_hash, 'artifact/referenced_hash/'+kind+str(fold)+'/'+name)
        selection, shuffle, rng = verify_history(result, parts, fold, kind)
        verify_progress(directory, result, parts, fold, kind, selection, shuffle, rng)
        status = load_json(directory/'status.json')
        exact(status['state'], 'complete', 'completion/fold/'+kind+str(fold))
        exact(status['fold'], fold, 'completion/status_fold/'+kind+str(fold))
        exact(status['best_epoch'], result['best_epoch'], 'completion/selected/'+kind+str(fold))
        if kind == 'baseline':
            exact(status['arm'], 'warmup', 'completion/status_arm/'+str(fold))
            for key, wanted in [('max_epochs', 40), ('min_epochs', 12), ('patience', 8)]:
                exact(result[key], wanted, 'configuration/baseline_result/'+key+str(fold))
        else:
            close(status['best_val_ce'], result['best_val_ce'], 'completion/status_ce/'+str(fold))
    verify_matrix(fold, parts, y, subjects, br['identity'])
    lineage = mr['identity']['initial_checkpoint']
    expected_lineage = {'right_path': str((bdir/'right.pt').resolve()), 'right_sha256': sha(bdir/'right.pt'),
                        'result_sha256': sha(bdir/'result.json'), 'manifest_sha256': sha(BASE/'manifest.json'),
                        'model_identity': br['identity'], 'selected_epoch': br['best_epoch'],
                        'reused_training_seconds': sum(r['seconds'] for r in br['history'])}
    compare_tree(lineage, expected_lineage, 'lineage/right_manifest/'+str(fold))
    bc, right = load_checkpoint(bdir/'best.pt'), load_checkpoint(bdir/'right.pt')
    exact(statehash(bc['state']), br['selected_state_sha256'], 'checkpoint/baseline_state_hash/'+str(fold))
    exact(bc['epoch'], br['best_epoch'], 'checkpoint/baseline_epoch/'+str(fold))
    close(bc['val_ce'], br['best_val_ce'], 'checkpoint/baseline_ce/'+str(fold))
    prefix_right = {k[6:]: v for k, v in bc['state'].items() if k.startswith('right.')}
    exact(statehash(prefix_right), statehash(right), 'lineage/right_selected_weights/'+str(fold))
    del bc
    initial = load_checkpoint(mdir/'initial.pt')
    exact(statehash(initial['state']), mr['identity']['initial_state_sha256'], 'checkpoint/initial_mv_state_hash/'+str(fold))
    exact(initial['epoch'], 0, 'checkpoint/initial_epoch/'+str(fold))
    close(initial['val_ce'], mr['history'][0]['val_ce'], 'checkpoint/initial_ce/'+str(fold))
    init_ces = [cross_entropy(np.asarray(initial['val_logits'])[:, j], y[parts['val']]) for j in range(7)]
    close(init_ces, mr['history'][0]['val_ce_by_subset'], 'ce_replay/initial_seven/'+str(fold), 1e-6)
    for k, tensor in initial['state'].items():
        if k.startswith('encoder.'):
            exact(tensor, right[k], 'lineage/initial_encoder/'+str(fold)+'/'+k)
        elif k.startswith('projection.'):
            exact(tensor, right[k.replace('projection.', 'head.net.', 1)], 'lineage/initial_projection/'+str(fold)+'/'+k)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        fuser = torch.nn.Sequential(torch.nn.Linear(771, 256), torch.nn.GELU(), torch.nn.Dropout(.1), torch.nn.Linear(256, 12))
    for k, tensor in fuser.state_dict().items():
        exact(tensor, initial['state']['fuser.'+k], 'lineage/seeded_fuser/'+str(fold)+'/'+k)
    del right, initial, fuser
    bp, bf, fm = archive(bdir/'predictions.npz'), archive(bdir/'features.npz'), load_json(bdir/'features.json')
    exact(sha(bdir/'features.npz'), fm['sha256'], 'artifact/features_sha/'+str(fold))
    compare_tree(fm['identity'], {'checkpoint': sha(bdir/'right.pt'), 'extractor': sha(CODE/'train_baseline.py'), 'model_identity': br['identity']}, 'lineage/feature_identity/'+str(fold))
    exact(bf['features'].shape, (9300, 256), 'array_shapes/baseline_features/'+str(fold))
    verify_split_archive(bf, parts, y, subjects, 'baseline'+str(fold))
    for role in ROLES:
        exact(bp[role+'_ids'], parts[role], 'partition/baseline_prediction/'+role+str(fold))
    exact(bp['ids'], parts['test'], 'partition/baseline_prediction/ids'+str(fold))
    exact(bp['y'], y[parts['test']], 'partition/baseline_prediction/y'+str(fold))
    exact(bp['val_y'], y[parts['val']], 'partition/baseline_prediction/val_y'+str(fold))
    exact(bf['logits'][bf['role'] == 1], bp['val_logits'], 'array_values/baseline_val_logit_replay/'+str(fold))
    exact(bf['logits'][bf['role'] == 2], bp['logits'], 'array_values/baseline_test_logit_replay/'+str(fold))
    close(np.mean(bp['logits'].argmax(1) == bp['y']), br['accuracy'], 'metric_replay/baseline_accuracy/'+str(fold))
    close(cross_entropy(bp['val_logits'], bp['val_y']), br['best_val_ce'], 'ce_replay/baseline_selected/'+str(fold), 1e-6)
    compare_tree({k: fm[k] for k in ['feature_shape', 'logit_replay', 'roles']}, {'feature_shape': [9300, 256], 'logit_replay': 'bitwise_equal', 'roles': dict(zip(ROLES, range(3)))}, 'array_values/features_metadata/'+str(fold))
    del bp, bf
    data = archive(mdir/'predictions.npz')
    verify_split_archive(data, parts, y, subjects, 'multiview'+str(fold))
    exact(data['features'].shape, (9300, 3, 256), 'array_shapes/multiview_features/'+str(fold))
    exact(data['logits'].shape, (9300, 7, 12), 'array_shapes/multiview_logits/'+str(fold))
    exact(data['subset_masks'], MASKS, 'array_values/seven_masks/'+str(fold))
    exact(data['val_logits'], data['logits'][data['role'] == 1], 'array_values/multiview_val_duplicate/'+str(fold))
    exact(data['val_y'], y[parts['val']], 'array_values/multiview_val_labels/'+str(fold))
    best = load_checkpoint(mdir/'best.pt')
    exact(statehash(best['state']), mr['selected_state_sha256'], 'checkpoint/multiview_selected_state/'+str(fold))
    exact(best['epoch'], mr['best_epoch'], 'checkpoint/multiview_selected_epoch/'+str(fold))
    close(best['val_ce'], mr['best_val_ce'], 'checkpoint/multiview_selected_ce/'+str(fold))
    exact(np.asarray(best['val_logits']), data['val_logits'], 'array_values/multiview_checkpoint_logits/'+str(fold))
    selected_ces = [cross_entropy(data['val_logits'][:, j], data['val_y']) for j in range(7)]
    close(selected_ces, mr['val_ce_by_subset'], 'ce_replay/multiview_selected_seven/'+str(fold), 1e-6)
    close(np.mean(selected_ces), mr['best_val_ce'], 'ce_replay/multiview_selected_mean/'+str(fold), 1e-6)
    close(mr['val_ce_by_subset'], mr['history'][mr['best_epoch']]['val_ce_by_subset'], 'ce_replay/selected_history/'+str(fold))
    exact(mr['feature_shape'], [9300, 3, 256], 'array_shapes/result_features/'+str(fold))
    exact(mr['logit_shape'], [9300, 7, 12], 'array_shapes/result_logits/'+str(fold))
    exact(mr['roles'], dict(zip(ROLES, range(3))), 'array_values/result_roles/'+str(fold))
    exact(mr['validation_replay'], 'bitwise_equal', 'array_values/result_validation_replay/'+str(fold))
    check(np.isfinite(mr['offline_export_seconds']) and mr['offline_export_seconds'] > 0,
          'cost/finite_offline_export_seconds/'+str(fold))
    close(mr['new_training_seconds'], sum(r['seconds'] for r in mr['history']), 'cost/training_seconds/'+str(fold))
    close(mr['reused_right_training_seconds'], sum(r['seconds'] for r in br['history']), 'cost/reused_seconds/'+str(fold))
    exact(mr['peak_allocated_bytes'], max(r['peak_allocated_bytes'] for r in mr['history']), 'cost/recorded_peak_memory/'+str(fold))
    for row in mr['history'][1:]:
        path = mdir/'epochs'/f'{row["epoch"]:02d}.json'
        epoch_status = load_json(path)
        prefix = mr['history'][:row['epoch']+1]
        epoch_best = int(np.argmin([v['val_ce'] for v in prefix]))
        compare_tree(epoch_status, {'state': 'running', 'fold': fold, 'epoch': row['epoch'],
                                    'best_epoch': epoch_best, 'best_val_ce': prefix[epoch_best]['val_ce'], 'last': row},
                     'epoch_artifacts/'+str(fold)+'/'+str(row['epoch']))
    fuser_check = cpu_fuser_diagnostic(data['features'], data['logits'], best['state'])
    del best
    zval, yval = data['logits'][data['role'] == 1], data['y'][data['role'] == 1]
    ztest, ytest = data['logits'][data['role'] == 2], data['y'][data['role'] == 2]
    curves, valacc, candidate_count = select_validation(zval, yval)
    testacc = accuracy7(ztest, ytest)
    close(mr['test_accuracy_by_subset'], (ztest.argmax(2) == ytest[:, None]).mean(0), 'metric_replay/multiview_test_accuracy/'+str(fold))
    identity = {'predictions_sha256': sha(mdir/'predictions.npz'), 'upstream_result_sha256': sha(mdir/'result.json'),
                'manifest_sha256': sha(MV/'manifest.json'), 'smoke': False,
                'analysis_sha256': sha(CODE/'budget_analysis.py'), 'statistics_sha256': sha(CODE/'run_calibration.py')}
    saved_selection, saved_result = load_json(odir/'selection.json'), load_json(odir/'result.json')
    compare_tree(saved_selection, {'identity': identity, 'curves': curves, 'validation_fixed_accuracy': valacc.tolist(),
                                  'uses_target_labels': False}, 'selection_replay/'+str(fold))
    arrays = archive(odir/'cascade_predictions.npz')
    expected_array_keys = {'ids', 'y'} | {p+str(float(b)) for p in ['predictions_b', 'choices_b'] for b in BUDGETS}
    exact(set(arrays), expected_array_keys, 'array_values/cascade_keys/'+str(fold))
    exact(arrays['ids'], parts['test'], 'array_values/cascade_ids/'+str(fold))
    exact(arrays['y'], ytest, 'array_values/cascade_labels/'+str(fold))
    first = curves['2.0']['params']['first']
    result_curves, details = {}, {}
    for ceiling in BUDGETS:
        key = str(float(ceiling)); params = curves[key]['params']
        pred, chosen, requested = causal_replay(ztest, params)
        exact(arrays['predictions_b'+key], pred, 'array_values/cascade_predictions/'+str(fold)+'/'+key)
        exact(arrays['choices_b'+key], chosen, 'array_values/cascade_choices/'+str(fold)+'/'+key)
        exact(requested.sum(1), COST[chosen], 'causality/requested_cost/'+str(fold)+'/'+key)
        exact(requested.astype(np.int8), MASKS[chosen], 'causality/monotone_subsets/'+str(fold)+'/'+key)
        actual_cost = float(requested.sum(1).mean())
        weights = mixture(valacc, actual_cost)
        oracle_here = oracle(ztest, ytest, first, actual_cost)
        result_curves[key] = {'target_val_budget': float(ceiling), 'test_cost': actual_cost,
                             'test_accuracy': macro(pred, ytest), 'fixed_mixture_within_cost_accuracy': float(weights@testacc),
                             'fixed_mixture_within_cost_actual_views': float(weights@COST),
                             'fixed_mixture_cost_ceiling': actual_cost,
                             'fixed_mixture_within_cost_weights': weights.tolist(), 'oracle_at_actual_cost': oracle_here}
        check(oracle_here['macro_accuracy'] >= macro(pred, ytest)-1e-12, 'oracle_bounds/above_cascade/'+str(fold)+'/'+key)
        check(oracle_here['mean_views'] <= actual_cost+1e-12, 'oracle_bounds/cost_ceiling/'+str(fold)+'/'+key)
        # Poison every unavailable subset, and every unvisited stage, to certify
        # the exact saved route can be reproduced without unobserved information.
        poisoned = np.full_like(ztest, np.nan)
        poisoned[:, params['first']] = ztest[:, params['first']]
        more = COST[chosen] >= 2; last = COST[chosen] == 3
        poisoned[more, params['pair']] = ztest[more, params['pair']]
        poisoned[last, 6] = ztest[last, 6]
        pp, pc, pr = causal_replay(poisoned, params)
        exact(pp, pred, 'causality/poisoned_unobserved_predictions/'+str(fold)+'/'+key)
        exact(pc, chosen, 'causality/poisoned_unobserved_decisions/'+str(fold)+'/'+key)
        lp_acc = oracle_lp(ztest, ytest, first, actual_cost)
        close(oracle_here['macro_accuracy'], lp_acc, 'oracle_lp/actual_cost/'+str(fold)+'/'+key)
        details[key] = {'counts_one_two_three_views': np.bincount(COST[chosen], minlength=4)[1:].tolist(),
                        'validation_cost': curves[key]['val_cost'], 'test_cost': actual_cost,
                        'mixture_cost': float(weights@COST), 'mixture_strictly_below_ceiling': bool(weights@COST < actual_cost-1e-12),
                        'validation_selection_parameters': params, 'oracle_lp_accuracy': lp_acc}
    weights2 = mixture(valacc, 2.)
    oracle2 = oracle(ztest, ytest, first, 2.)
    close(oracle2['macro_accuracy'], oracle_lp(ztest, ytest, first, 2.), 'oracle_lp/budget2/'+str(fold))
    selected_fixed = {str(cost): sorted(np.flatnonzero(COST == cost), key=lambda j: (-round(float(valacc[j]), 12), j))[0].item()
                      for cost in (1, 2, 3)}
    expected_result = {'identity': identity, 'fold': fold, 'subset_names': NAMES, 'fixed_accuracy': testacc.tolist(),
                       'fixed_validation_selected': selected_fixed,
                       'fixed_mixture_at_2': {'accuracy': float(weights2@testacc), 'cost': float(weights2@COST), 'weights': weights2.tolist()},
                       'oracle_at_2': oracle2, 'cascade_curves': result_curves,
                       'files': {name: sha(odir/name) for name in ['selection.json', 'cascade_predictions.npz']}}
    compare_tree(saved_result, expected_result, 'result_replay/'+str(fold))
    correctness = ztest.argmax(2) == ytest[:, None]
    pairs = [j for j in range(3, 6) if first in VIEWS[j]]
    negative_counts = {NAMES[j]: int(np.count_nonzero(correctness[:, first] & ~correctness[:, j])) for j in pairs+[6]}
    joint_only = int(np.count_nonzero(~correctness[:, first] & ~correctness[:, pairs].any(1) & correctness[:, 6]))
    curve_upper = [oracle(ztest, ytest, first, float(b)) for b in BUDGETS]
    check(np.all(np.diff([v['macro_accuracy'] for v in curve_upper]) >= -1e-12), 'oracle_bounds/monotone_budget/'+str(fold))
    check(all(v['mean_views'] <= b+1e-12 for v, b in zip(curve_upper, BUDGETS)), 'oracle_bounds/all_budget_costs/'+str(fold))
    record = {'fold': fold, 'training_users': expected_split['training_subjects'], 'validation_user': fold%8+1, 'test_user': fold,
              'partition_rows': {k: len(v) for k, v in parts.items()}, 'partition_sha256': {k: arrhash(v) for k, v in parts.items()},
              'baseline_best_epoch': br['best_epoch'], 'multiview_best_epoch': mr['best_epoch'],
              'baseline_training_epochs': len(br['history'])-1, 'multiview_training_epochs': len(mr['history'])-1,
              'validation_grid_candidates': candidate_count, 'validation_selected_first': NAMES[first],
              'validation_selected_pair': NAMES[curves['2.0']['params']['pair']], 'fixed_macro_accuracy': testacc.tolist(),
              'primary': result_curves['2.0'], 'fixed_mixture_at_2': expected_result['fixed_mixture_at_2'], 'oracle_at_2': oracle2,
              'oracle_negative_addition_rows': negative_counts, 'oracle_joint_only_rescue_rows': joint_only,
              'oracle_budget_curve': {str(float(b)): value for b, value in zip(BUDGETS, curve_upper)},
              'budget_details': details, 'cpu_fuser_diagnostic': fuser_check,
              'new_training_seconds': mr['new_training_seconds'], 'reused_training_seconds': mr['reused_right_training_seconds'],
              'failures_in_fold': len(REPORT['failures'])-before_failures}
    REPORT['folds'].append(record)
    del data, br, mr, arrays
    gc.collect()
    print(f'FOLD {fold}/8 checked; failures={record["failures_in_fold"]}; selected epoch={record["multiview_best_epoch"]}; primary cost={record["primary"]["test_cost"]:.12f}', flush=True)
    return expected_result


def synthetic_oracle_test():
    # Row 0 would be damaged by every addition; row 1 needs one addition;
    # row 2 needs both jointly; row 3 cannot be rescued.
    correctness = np.zeros((4, 7), dtype=bool)
    correctness[0, 2] = True; correctness[1, 4] = True; correctness[2, 6] = True
    y = np.zeros(4, dtype=int)
    z = np.stack([correctness.astype(float), (~correctness).astype(float)], axis=-1)
    answer = oracle(z, y, 2, 1.375)
    close(answer['macro_accuracy'], .5625, 'oracle_synthetic/harm_and_joint_and_fraction')
    close(answer['mean_views'], 1.375, 'oracle_synthetic/fractional_cost')
    close(oracle_lp(z, y, 2, 1.375), .5625, 'oracle_synthetic/full_lp')
    # Unequal classes ensure a macro-weighted upper bound, not micro accuracy.
    y2 = np.array([0, 0, 0, 1]); pred = np.where(correctness, y2[:, None], 1-y2[:, None])
    z2 = np.eye(2)[pred]
    close(oracle(z2, y2, 2, 1.375)['macro_accuracy'], oracle_lp(z2, y2, 2, 1.375), 'oracle_synthetic/unequal_classes')
    # Explicitly distinguish <=budget from an artificial exact-cost equality.
    low_cost_wins = np.array([.9, .5, .4, .8, .7, .75, .85])
    mix = mixture(low_cost_wins, 2.5)
    exact(mix, np.eye(7)[0], 'mixture_synthetic/retain_better_cheaper_subset')
    close(mix@COST, 1., 'mixture_synthetic/cost_below_ceiling')
    tied = mixture(np.ones(7), 2.5)
    exact(tied, np.eye(7)[0], 'mixture_synthetic/lowest_cost_then_declared_order')


def summarize(rows):
    fixed = np.array([r['fixed_accuracy'] for r in rows])
    primary = [r['cascade_curves']['2.0'] for r in rows]
    cascade = np.array([r['test_accuracy'] for r in primary])
    mixture_actual = np.array([r['fixed_mixture_within_cost_accuracy'] for r in primary])
    oracle2 = np.array([r['oracle_at_2']['macro_accuracy'] for r in rows])
    mixture2 = np.array([r['fixed_mixture_at_2']['accuracy'] for r in rows])
    oracleactual = np.array([r['oracle_at_actual_cost']['macro_accuracy'] for r in primary])
    expected = {'state': 'complete', 'subset_names': NAMES, 'fixed_macro_accuracy': fixed.mean(0).tolist(),
                'fixed_per_user': fixed.tolist(), 'primary_cascade_accuracy': float(cascade.mean()),
                'primary_cascade_cost': float(np.mean([r['test_cost'] for r in primary])),
                'primary_cascade_per_user_cost': [r['test_cost'] for r in primary],
                'cascade_vs_fixed_mixture_within_cost': bootstrap((cascade-mixture_actual)*100),
                'oracle_minus_cascade_at_same_cost_pp': float(np.mean((oracleactual-cascade)*100)),
                'oracle_at_2_accuracy': float(oracle2.mean()), 'fixed_mixture_at_2_accuracy': float(mixture2.mean()),
                'oracle_minus_mixture_at_2_pp': float(np.mean(oracle2-mixture2)*100),
                'note': 'Oracle uses labels; conditional opportunity only. Validation cost limits may shift on target users.'}
    room = expected['oracle_minus_mixture_at_2_pp'] >= 2-1e-10 and expected['oracle_minus_cascade_at_same_cost_pp'] >= 2-1e-10
    expected['opportunity_decision'] = 'POLICY_PILOT_WARRANTED' if room else 'STOP_NO_ORACLE_ROOM'
    compare_tree(load_json(OUT/'summary.json'), expected, 'summary_replay')
    REPORT['recomputed_summary'] = expected
    REPORT['additional_user_bootstraps'] = {
        'oracle_minus_mixture_at_2': bootstrap((oracle2-mixture2)*100),
        'oracle_minus_cascade_at_actual_cost_ceiling': bootstrap((oracleactual-cascade)*100)}
    REPORT['deployment_gate_diagnostic_for_simple_cascade'] = {
        'cost_at_most_2': bool(expected['primary_cascade_cost'] <= 2),
        'excess_cost_above_2': expected['primary_cascade_cost']-2,
        'gain_at_least_2pp_within_same_cost_ceiling': expected['cascade_vs_fixed_mixture_within_cost']['mean_difference_pp'] >= 2-1e-10,
        'positive_at_least_6_users': expected['cascade_vs_fixed_mixture_within_cost']['positive_users'] >= 6,
        'gap_to_fixed_all_views_pp': float((fixed[:, 6]-cascade).mean()*100),
        'gap_to_fixed_all_views_at_most_1pp': bool((fixed[:, 6]-cascade).mean()*100 <= 1+1e-10),
        'sample_pooled_actual_mean_views': float(np.average([p['test_cost'] for p in primary], weights=[f['partition_rows']['test'] for f in REPORT['folds']])),
        'passed': False,
        'interpretation': 'Conditional oracle-opportunity gate passes. This simple cascade does not pass the deployment criteria.'}
    REPORT['counts'] = {'outer_users': 8, 'fold_budget_points': 72, 'test_rows_once': 9300,
                        'mv_exported_rows_across_folds': 8*9300, 'mv_exported_logits': 8*9300*7*12,
                        'cascade_prediction_elements_compared': 9300*9,
                        'cascade_choice_elements_compared': 9300*9,
                        'all_mv_feature_elements_checked_finite_and_hashed': 8*9300*3*256,
                        'cpu_fusion_argmax_comparisons': 8*9300*7,
                        'cpu_fusion_argmax_differences': sum(f['cpu_fuser_diagnostic']['argmax_differences'] for f in REPORT['folds']),
                        'array_archives_audited': len(REPORT['array_inventory']),
                        'validation_grid_candidates_total': sum(r['validation_grid_candidates'] for r in REPORT['folds']),
                        'mixture_strictly_below_cost_ceiling_points': sum(v['mixture_strictly_below_ceiling'] for r in REPORT['folds'] for v in r['budget_details'].values()),
                        'primary_test_users_cost_above_2': sum(p['test_cost'] > 2 for p in primary)}


def write_report(started):
    REPORT['status'] = 'PASS' if not REPORT['failures'] else 'FAIL'
    REPORT['finished_utc'] = datetime.now(timezone.utc).isoformat()
    REPORT['elapsed_seconds'] = time.perf_counter()-started
    REPORT['environment'] = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                             'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__,
                             'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
                             'torch_cuda_initialized': torch.cuda.is_initialized(), 'torch_num_threads': torch.get_num_threads()}
    check(not torch.cuda.is_initialized(), 'execution/no_cuda_initialized')
    REPORT['status'] = 'PASS' if not REPORT['failures'] else 'FAIL'
    REPORT['audit_script_sha256'] = sha(Path(__file__))
    REPORT['checks'] = dict(REPORT['checks'])
    REPORT['checks_total'] = sum(REPORT['checks'].values())
    REPORT['input_files_in_inventory'] = len(REPORT['file_inventory'])
    REPORT['limitations'] = [
        'Opportunity only: the oracle reads outer-test labels, uses fractional per-row selection, and is not deployable.',
        'The predeclared test-label oracle decides whether to run a conditional policy pilot. These test users are not new external confirmatory data for that later pilot.',
        'Cost means number of processed physical radar views per completed synchronized clip, including the first view. It is not measured elapsed time, acquisition power, or hardware reduction.',
        'The primary cascade has exact user-mean test cost 2.0046155303030306 (>2); its gap to fixed LTR is 1.52083333333333 pp (>1). Neither failure may be rounded away.',
        'Fixed mixture selection obeys a cost CEILING; the chosen mixture can be cheaper. The oracle is also an upper bound under the same ceiling and may spend less. Labels do not choose the fixed mixture.',
        'The audit independently replays every formal budget metric and saved route from raw exported logits and checks source/weight/partition hashes. It does not retrain or rerun the complete radar encoder.',
        'Full CPU BF16 fusion replay from saved features is an additional cross-device diagnostic; small CPU/GPU BF16 differences and any argmax differences are explicitly recorded, and original exported logits remain the audited metric input.',
        'Only one model seed and eight users. Whole-user bootstrap resamples paired user results; overlapping cross-validation training sets and class omissions limit independence and generalization.',
        'The shared subset model is one architecture, not seven separately optimized models; the fixed names are descriptive, while validation-selected fixed subsets and mixtures are the fair selection controls.',
        'Current hashes and internal lineage establish present consistency, not independent historical timestamp attestation. Intermediate epoch logits/weights are not all retained; earliest selection is verified against the complete saved history and embedded progress transaction.',
        'Source-only smoothing is verified from its saved OOF probabilities, fit/predict ID evidence, matrix reconstruction and fingerprints; the proxy logistic regressions were not refit.',
        'value_policy and real CPU/GPU latency results are outside this audit; this report makes no claim about their completion status or outcomes.'
    ]
    result = REPORT.get('recomputed_summary', {})
    lines = ['# Independent audit of B: fixed multiradar processing budget', '',
             f'**{REPORT["status"]}** — {REPORT["checks_total"]} checks; {len(REPORT["failures"])} failures; CPU only.', '',
             'Run: `D:/D_conda/envs/pytorch/python.exe D:/uwb_gestures/experiments/p1_20260928/budget_s2026/independent_audit.py`', '',
             'The standalone script imports no original analysis/training module. It reads the formal eight-fold artifacts and writes only this report and its JSON evidence companion.', '']
    if result:
        lines += ['## Independently recomputed results', '', '| Quantity | Result |', '|---|---:|']
        lines += [f'| Fixed {name} macro accuracy | {value*100:.9f}% |' for name, value in zip(NAMES, result['fixed_macro_accuracy'])]
        comp = result['cascade_vs_fixed_mixture_within_cost']
        lines += [f'| Primary cascade accuracy | {result["primary_cascade_accuracy"]*100:.9f}% |',
                  f'| Primary cascade actual mean views | **{result["primary_cascade_cost"]:.14f} (>2)** |',
                  f'| Cascade minus validation-selected mixture within its actual cost ceiling | {comp["mean_difference_pp"]:.9f} pp |',
                  f'| Paired user bootstrap 95% interval | [{comp["ci95_pp"][0]:.9f}, {comp["ci95_pp"][1]:.9f}] pp |',
                  f'| Positive users | {comp["positive_users"]}/8 |',
                  f'| Oracle at ceiling 2 minus mixture at ceiling 2 | {result["oracle_minus_mixture_at_2_pp"]:.9f} pp |',
                  f'| Oracle at cascade actual cost ceiling minus cascade | {result["oracle_minus_cascade_at_same_cost_pp"]:.9f} pp |',
                  f'| Conditional opportunity decision | **{result["opportunity_decision"]}** |', '',
                  'The cascade fails the strict <=2 average-view deployment budget and the <=1 pp gap-to-LTR condition. Passing the oracle opportunity test only authorizes a conditional policy pilot under the frozen plan; it is not a successful learned policy.', '',
                  '## Fold evidence', '',
                  '| Test user | Validation user | Train/val/test rows | Baseline/MV selected epoch | First→pair | Cascade accuracy | Actual views | Joint-only rescues |',
                  '|---:|---:|---|---|---|---:|---:|---:|']
        for f in REPORT['folds']:
            p = f['partition_rows']
            lines.append(f'| {f["fold"]} | {f["validation_user"]} | {p["train"]}/{p["val"]}/{p["test"]} | {f["baseline_best_epoch"]}/{f["multiview_best_epoch"]} | {f["validation_selected_first"]}→{f["validation_selected_pair"]} | {f["primary"]["test_accuracy"]*100:.6f}% | {f["primary"]["test_cost"]:.12f} | {f["oracle_joint_only_rescue_rows"]} |')
        lines += ['', '## Verification coverage', '',
                  '- Raw labels/subjects: 9593 records, 293 excluded by the frozen quality rule, 9300 retained; exact 6/1/1 whole-user split and all IDs/roles/labels checked.',
                  '- Live source/data hashes, frozen source snapshots, manifests, same-fold Right checkpoint lineage, actual inherited encoder/projection tensors and seeded fusion-head tensors checked.',
                  '- All listed artifacts hashed; each epoch history and epoch JSON checked; shuffle/subset RNG traces and optimizer step counts reconstructed; initial candidate, earliest mean-seven-CE selection, best state and embedded progress checkpoint checked.',
                  '- All seven masks, all nine validation threshold selections, every saved cascade prediction/choice, fixed/mixture/oracle result leaf and summary field recomputed. CPU-only full fusion-head replay additionally checks every exported MV logit against the saved features, with cross-device numerical differences retained as diagnostic evidence.',
                  '- Mixture vertices independently cross-checked by a seven-variable LP. Oracle bounds independently cross-checked by a full sample-action LP at all 72 actual cascade costs and eight primary cost-2 points. Harmful additions, joint-only rescue, fractional allocation and unequal-class weighting are covered.',
                  '- Unobserved data are replaced with NaNs in every saved cascade route replay; predictions/decisions remain identical. Unique requested views exactly reproduce reported sample costs. This is causal replay of saved logits, not a latency measurement.',
                  f'- {REPORT.get("input_files_in_inventory",0)} file fingerprints and per-array shape/dtype/content hashes are in INDEPENDENT_AUDIT.json.', '',
                  '## Scope and limitations', '']
        lines += ['- '+v for v in REPORT['limitations']]
        lines += ['', '## CPU fusion replay numerical diagnostic', '', '| Fold | Logit elements | Exact elements | Max absolute difference | Argmax differences |', '|---:|---:|---:|---:|---:|']
        for f in REPORT['folds']:
            d = f['cpu_fuser_diagnostic']
            lines.append(f'| {f["fold"]} | {d["elements"]} | {d["bitwise_equal_elements"]} | {d["max_abs_logit_difference"]:.9g} | {d["argmax_differences"]} |')
    if REPORT['failures']:
        lines += ['', '## Exact failures', '', '```json', json.dumps(REPORT['failures'], ensure_ascii=False, indent=2), '```']
    lines += ['', f'Audit script SHA-256: `{REPORT["audit_script_sha256"]}`.',
              f'Elapsed: {REPORT["elapsed_seconds"]:.3f} seconds. CUDA initialized: {REPORT["environment"]["torch_cuda_initialized"]}.', '']
    (OUT/'INDEPENDENT_AUDIT.json').write_text(json.dumps(REPORT, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (OUT/'INDEPENDENT_AUDIT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    started = time.perf_counter()
    original = {}
    try:
        # Record complete initial inventories, including otherwise unreferenced
        # root summaries, statuses and snapshots. Audit outputs are excluded.
        for root in (BASE, MV, OUT):
            for p in sorted(root.rglob('*')):
                if p.is_file() and p.name not in {'independent_audit.py', 'INDEPENDENT_AUDIT.json', 'INDEPENDENT_AUDIT.md'}:
                    sha(p); original[str(p.resolve())] = (p.stat().st_size, p.stat().st_mtime_ns)
        print(f'INPUT INVENTORY {len(original)} files hashed', flush=True)
        b, m = verify_sources()
        y, subjects = np.load(PREP/'y.npy'), np.load(PREP/'subj.npy')
        exact(y.shape, (9593,), 'dataquality/raw_count'); exact(subjects.shape, (9593,), 'dataquality/subject_count')
        keep = ~(((subjects == 2) & np.isin(y, [10, 11])) | ((subjects == 6) & (y == 3)))
        exact(int(keep.sum()), 9300, 'dataquality/kept_count'); exact(int((~keep).sum()), 293, 'dataquality/excluded_count')
        exact(np.unique(subjects).tolist(), list(range(1, 9)), 'dataquality/eight_users')
        exact(np.unique(y).tolist(), list(range(12)), 'dataquality/twelve_classes')
        REPORT['dataquality'] = {'raw_rows': len(y), 'retained_rows': int(keep.sum()), 'excluded_rows': int((~keep).sum()),
                                 'raw_class_counts_by_user': {str(h): np.bincount(y[subjects == h], minlength=12).tolist() for h in range(1, 9)},
                                 'retained_class_counts_by_user': {str(h): np.bincount(y[(subjects == h) & keep], minlength=12).tolist() for h in range(1, 9)},
                                 'retained_ids_sha256': arrhash(np.flatnonzero(keep))}
        synthetic_oracle_test()
        rows = [fold_audit(fold, y, subjects, keep, b, m) for fold in range(1, 9)]
        summarize(rows)
        for key, signature in original.items():
            stat = Path(key).stat()
            check((stat.st_size, stat.st_mtime_ns) == signature, 'immutability/end_of_run/'+key)
        check(not torch.cuda.is_initialized(), 'execution/cpu_only')
    except Exception as error:
        REPORT['failures'].append({'check': 'execution/exception', 'detail': repr(error), 'traceback': traceback.format_exc()})
    write_report(started)
    print(json.dumps({'status': REPORT['status'], 'checks': REPORT['checks_total'], 'failures': len(REPORT['failures']),
                      'reports': [str(OUT/'INDEPENDENT_AUDIT.json'), str(OUT/'INDEPENDENT_AUDIT.md')],
                      'summary': REPORT.get('recomputed_summary')}, ensure_ascii=False), flush=True)
    return 0 if REPORT['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
