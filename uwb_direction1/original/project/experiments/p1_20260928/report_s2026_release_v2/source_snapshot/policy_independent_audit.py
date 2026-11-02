"""Independent CPU replay of the frozen conditional-policy pilot.

No original new_routes module is imported. Original pickle policy containers
are mapped to inert local classes, sklearn prediction is permitted, and each
ExtraTrees model is separately refit in memory from independently rebuilt
validation-fit inputs. No experiment input, model, source or plan is modified.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
import hashlib
import itertools
import json
import math
import pickle
import platform
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits

ROOT = Path('D:/uwb_gestures/experiments/p1_20260928')
OUT = ROOT/'value_policy_s2026'
MV = ROOT/'multiview_s2026'
BUD = ROOT/'budget_s2026'
PREP = ROOT.parent/'p1_20260905'/'prepared'
CODE = Path('C:/Users/19252/.codex/worktrees/uwb-user-calibration/uwb_gestures/experiments/new_routes')
SOURCE_SHA = '0222130b12461521f5bd45fc628caec6da67d0d448796c5ff43cb83f76cc25df'
TEST_SHA = 'fe138d320bc0c9d85a046fa810d821b323f90f9b4c6f089e7a9f4934f263c8a1'
NAMES = ['L', 'T', 'R', 'LT', 'LR', 'TR', 'LTR']
SUBSETS = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
COST = np.array([1, 1, 1, 2, 2, 2, 3], dtype=np.int64)
MASK = np.array([[v in subset for v in range(3)] for subset in SUBSETS])
PRICES = [0., .005, .01, .02, .05, .1, .15, .2, .3, .4, .5, 1., 2.]
METHODS = ['class_gain', 'extra_trees']
COMPARATORS = ['frozen_budget2_cascade', 'class_gain', 'fixed_mixture_within_candidate_cost']
PARAMS = {'n_estimators': 200, 'max_depth': 4, 'min_samples_leaf': 20, 'random_state': 2026, 'n_jobs': 1}
SPEC = {
    'policy_features': '12 current-subset probabilities, normalized entropy, top-two margin',
    'fit_fraction_per_class': .6,
    'split_seed': '33000 + fold*100 + class; ascending sample IDs before permutation',
    'price_grid': PRICES, 'price_selection_budget': 2.,
    'gain_target': 'correctness after requesting a superset minus current correctness',
    'class_gain_shrinkage_examples': 20., 'extra_trees': PARAMS,
    'action_ties': 'stop, fewer additional views, smaller subset index; tolerance 1e-12',
    'price_ties': 'higher macro accuracy, lower mean views, higher price; tolerance 1e-12',
    'main_candidate': 'extra_trees', 'fixed_comparator_family': COMPARATORS,
    'strong_simple_rule': 'candidate must gain >=2pp and improve >=6/8 users against EACH comparator; no target winner selection',
    'cost_fairness_rule': 'candidate population mean views must not exceed either adaptive comparator population mean; fixed mixture uses candidate cost as ceiling',
    'aggregation': 'each user receives equal weight for primary accuracy and cost; per-user cost matching is descriptive only',
    'runtime_rule': 'no speedup claim without a separate actual on-demand total-runtime benchmark',
    'validation_reuse_limit': 'policy user had no encoder gradient updates, but was reused for encoder checkpoint selection',
}
REPORT = {'status': 'RUNNING', 'scope': 'Formal value_policy_s2026 results only; no latency audit or changes to B evidence',
          'started_utc': datetime.now(timezone.utc).isoformat(), 'checks': Counter(), 'failures': [],
          'files': {}, 'arrays': {}, 'folds': [], 'numerical_max_error': {}, 'tree_refit_records': []}
CACHE = {}


def check(condition, name, detail=None):
    REPORT['checks'][name.split('/')[0]] += 1
    if not bool(condition):
        REPORT['failures'].append({'check': name, 'detail': detail})


def exact(a, b, name):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = np.asarray(a), np.asarray(b)
        nan = np.issubdtype(x.dtype, np.floating) and np.issubdtype(y.dtype, np.floating)
        ok = x.shape == y.shape and np.array_equal(x, y, equal_nan=nan)
        check(ok, name, None if ok else {'actual_shape': list(x.shape), 'expected_shape': list(y.shape)})
    else:
        check(a == b, name, None if a == b else {'actual': a, 'expected': b})


def close(a, b, name, atol=1e-10):
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = x.shape == y.shape and np.all(np.isfinite(x)) and np.all(np.isfinite(y))
    delta = float(np.max(np.abs(x-y))) if ok and x.size else (0. if ok else None)
    group = name.split('/')[0]
    if delta is not None:
        REPORT['numerical_max_error'][group] = max(delta, REPORT['numerical_max_error'].get(group, 0.))
    check(ok and delta <= atol, name, {'max_abs_error': delta, 'atol': atol})


def tree_compare(a, b, name):
    if isinstance(b, dict):
        check(isinstance(a, dict), name+'/is_dict')
        if not isinstance(a, dict): return
        exact(sorted(a), sorted(b), name+'/keys')
        for k in b:
            if k in a: tree_compare(a[k], b[k], name+'/'+k)
    elif isinstance(b, (list, tuple)):
        check(isinstance(a, (list, tuple)) and len(a) == len(b), name+'/length')
        if isinstance(a, (list, tuple)) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)): tree_compare(x, y, name+'/'+str(i))
    elif isinstance(b, float): close(a, b, name)
    else: exact(a, b, name)


def sha(path):
    p = Path(path).resolve(); stat = p.stat(); stamp = (stat.st_size, stat.st_mtime_ns)
    if str(p) in CACHE and CACHE[str(p)][0] == stamp: return CACHE[str(p)][1]
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    stat2 = p.stat()
    check(stamp == (stat2.st_size, stat2.st_mtime_ns), 'immutability/hash_read/'+str(p))
    digest = h.hexdigest(); CACHE[str(p)] = (stamp, digest)
    REPORT['files'][str(p)] = {'sha256': digest, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    return digest


def read_json(p):
    sha(p)
    return json.loads(Path(p).read_text(encoding='utf-8'))


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def read_arrays(p, keys=None):
    sha(p)
    with np.load(p, allow_pickle=False) as archive:
        values = {k: archive[k] for k in (archive.files if keys is None else keys)}
    REPORT['arrays'][str(Path(p).resolve())] = {}
    for key, a in values.items():
        allow_nan = key.endswith('_pair_gain')
        finite = bool(np.isfinite(a).all())
        check(finite or (allow_nan and not np.isinf(a).any()), 'arrays/finite/'+str(p)+'/'+key)
        REPORT['arrays'][str(Path(p).resolve())][key] = {
            'shape': list(a.shape), 'dtype': str(a.dtype), 'elements': int(a.size),
            'array_sha256': array_sha(a), 'nan_count': int(np.isnan(a).sum()), 'nan_allowed_unvisited_pair': allow_nan}
    return values


def macro(prediction, labels):
    return float(np.mean([np.count_nonzero(prediction[labels == c] == c)/np.count_nonzero(labels == c)
                          for c in np.unique(labels)]))


def fixed_accuracy(logits, labels):
    return np.array([macro(logits[:, j].argmax(1), labels) for j in range(7)])


def weights(labels):
    _, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return 1./(len(counts)*counts[inv])


def features(current_logits):
    z = np.asarray(current_logits, dtype=np.float64)
    check(z.ndim == 2 and z.shape[1] == 12 and np.isfinite(z).all(), 'features/current_only_12_logits')
    p = np.exp(z-z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    entropy = -(p*np.log(np.maximum(p, 1e-30))).sum(1)/np.log(12)
    ordered = np.sort(p, axis=1)
    return np.column_stack([p, entropy, ordered[:, -1]-ordered[:, -2]])


def split_validation(ids, y, fold):
    fit_ids, tune_ids = [], []
    for c in sorted(np.unique(y)):
        ordered_ids = np.sort(ids[y == c], kind='stable')
        rng = np.random.default_rng(33000+fold*100+int(c))
        shuffled_ids = ordered_ids[rng.permutation(len(ordered_ids))]
        n = int(math.floor(.6*len(shuffled_ids)))
        fit_ids.extend(shuffled_ids[:n]); tune_ids.extend(shuffled_ids[n:])
    fit = np.flatnonzero(np.isin(ids, fit_ids)); tune = np.flatnonzero(np.isin(ids, tune_ids))
    return fit, tune


class SavedPolicy:
    """Inert pickle container; original policy methods are never executed."""
    pass


class SavedClassGain:
    def predict(self, x):
        return self.table_[np.argmax(x[:, :12], axis=1)]


class IndependentUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('value_policy', '__main__'):
            if name == 'GainPolicy': return SavedPolicy
            if name == 'ClassGainRegressor': return SavedClassGain
            raise ValueError('Unexpected original-module pickle class: '+module+'.'+name)
        return super().find_class(module, name)


def destinations(first, state):
    return tuple(j for j in range(7) if set(SUBSETS[state]) < set(SUBSETS[j])) if state == first else (6,)


def policy_route(z, policy, price):
    """Independent state-machine implementation, with explicit acquisitions.

    Only the logits returned by acquire are passed to the gain predictor. IDs,
    users, labels and logits for unrequested subsets do not enter decisions.
    """
    n = len(z); first = int(policy.first)
    final = np.full(n, first, dtype=np.int64)
    pred = np.empty(n, dtype=np.int64)
    path = np.full((n, 3), -1, dtype=np.int64); path[:, 0] = first
    added = np.zeros((n, 3), dtype=np.int64); added[:, 0] = 1
    acquired = np.zeros((n, 3), dtype=bool)
    pair_gain = np.full(n, np.nan); first_gain = None
    pending = [(np.arange(n), first, 0)]
    read_calls = []
    while pending:
        rows, state, stage = pending.pop(0)
        if not len(rows): continue
        before = acquired[rows].sum(1)
        acquired[np.ix_(rows, SUBSETS[state])] = True
        check(np.all(acquired[rows].sum(1)-before == added[rows, stage]), 'causality/newly_acquired_view_cost')
        current = z[rows, state]
        check(current.shape == (len(rows), 12) and np.isfinite(current).all(), 'causality/read_requested_only')
        read_calls.append({'stage': stage, 'subset': int(state), 'rows': int(len(rows))})
        final[rows] = state; pred[rows] = current.argmax(1)
        if state == 6: continue
        action_states = (state,)+destinations(first, state)
        gain = np.asarray(policy.models[state].predict(features(current)), dtype=float)
        if gain.ndim == 1: gain = gain.reshape(-1, 1)
        check(gain.shape == (len(rows), len(action_states)-1) and np.isfinite(gain).all(), 'causality/gain_shape')
        if stage == 0: first_gain = gain
        else: pair_gain[rows] = gain[:, 0]
        extra = COST[np.array(action_states)]-COST[state]
        utilities = np.concatenate((np.zeros((len(rows), 1)), gain), axis=1)-price*extra
        max_utility = utilities.max(axis=1)
        eligible = utilities >= max_utility[:, None]-1e-12
        choose = np.array(action_states)[eligible.argmax(axis=1)]
        for target in action_states[1:]:
            moving = rows[choose == target]
            if len(moving):
                path[moving, stage+1] = target
                added[moving, stage+1] = int(COST[target]-COST[state])
                pending.append((moving, target, stage+1))
    route = {'prediction': pred, 'subset_indices': final, 'costs': acquired.sum(1).astype(np.int64),
             'path': path, 'added_views': added, 'requested_mask': acquired,
             'first_gains': first_gain, 'pair_gain': pair_gain}
    exact(acquired, MASK[final], 'causality/final_mask')
    exact(route['costs'], COST[final], 'causality/final_unique_cost')
    exact(added.sum(1), route['costs'], 'causality/summed_increment_cost')
    return route, read_calls


def frozen_cascade(z, params):
    first, pair = params['first'], params['pair']; n = len(z)
    choice = np.full(n, first, dtype=np.int64)
    f = features(z[:, first])
    u = 1-f[:, :12].max(1) if params['mode'] == 'confidence' else f[:, 12]
    onward = np.flatnonzero(u > params['threshold1'])
    choice[onward] = pair
    if len(onward):
        f2 = features(z[onward, pair]); u2 = 1-f2[:, :12].max(1) if params['mode'] == 'confidence' else f2[:, 12]
        choice[onward[u2 > params['threshold2']]] = 6
    return {'prediction': z[np.arange(n), choice].argmax(1), 'subset_indices': choice, 'costs': COST[choice]}


def choose_frozen_cascade(z, y):
    acc = fixed_accuracy(z, y)
    first = int(np.flatnonzero(acc[:3] >= acc[:3].max()-1e-12)[0])
    pairs = [j for j in range(3, 6) if first in SUBSETS[j]]
    pair = sorted(pairs, key=lambda j: (-round(float(acc[j]), 12), j))[0]
    f1, f2 = features(z[:, first]), features(z[:, pair])
    candidates = []
    pred = z.argmax(2); w = weights(y)
    for mode in ('confidence', 'entropy'):
        u1, u2 = [(1-f[:, :12].max(1) if mode == 'confidence' else f[:, 12]) for f in (f1, f2)]
        t1, t2 = [np.unique(np.r_[-1., np.quantile(u, np.linspace(0, 1, 21)), 2.]) for u in (u1, u2)]
        for a, b in itertools.product(t1, t2):
            more = u1 > a; all3 = more & (u2 > b)
            choice = np.where(all3, 6, np.where(more, pair, first))
            cost = float((len(y)+np.count_nonzero(more)+np.count_nonzero(all3))/len(y))
            if cost <= 2+1e-12:
                candidates.append({'params': {'first': first, 'pair': pair, 'mode': mode,
                                             'threshold1': float(a), 'threshold2': float(b)},
                                   'val_accuracy': float(np.sum(w*(pred[np.arange(len(y)), choice] == y))), 'val_cost': cost})
    top = max(c['val_accuracy'] for c in candidates)
    tied = [c for c in candidates if c['val_accuracy'] >= top-1e-12]
    best = min(tied, key=lambda c: (c['val_cost'], c['params']['mode'], -c['params']['threshold1'], -c['params']['threshold2']))
    return best, acc


def fixed_mixture(acc, ceiling):
    vertices = [np.eye(7)[j] for j in range(7) if COST[j] <= ceiling+1e-12]
    for a, b in itertools.combinations(range(7), 2):
        if COST[a] == COST[b]: continue
        alpha = (ceiling-COST[a])/(COST[b]-COST[a])
        if 0 < alpha < 1:
            v = np.zeros(7); v[a] = 1-alpha; v[b] = alpha; vertices.append(v)
    top = max(float(v@acc) for v in vertices)
    best = min([v for v in vertices if v@acc >= top-1e-12], key=lambda v: (float(v@COST), tuple(-v)))
    lp = linprog(-acc, A_ub=COST[None, :], b_ub=[ceiling], A_eq=np.ones((1, 7)), b_eq=[1.], bounds=(0, 1), method='highs')
    check(lp.success, 'mixture/independent_lp_success')
    if lp.success: close(best@acc, -lp.fun, 'mixture/optimal_under_ceiling')
    check(best@COST <= ceiling+1e-12 and np.count_nonzero(best) <= 2, 'mixture/ceiling_and_two_components')
    return best


def oracle_space(z, y, first, ceiling):
    correctness = z.argmax(2) == y[:, None]; w = weights(y)
    allowed = np.array([j for j in range(7) if first in SUBSETS[j]])
    extra = COST[allowed]-1
    mincost = np.where(correctness[:, allowed], extra, 99).min(1)
    rows = np.flatnonzero((~correctness[:, first]) & (mincost < 99))
    rows = rows[np.lexsort((-rows, -w[rows]/mincost[rows]))]
    cost = mincost[rows].astype(float); prior = np.cumsum(cost)-cost
    fraction = np.clip((len(y)*(ceiling-1)-prior)/cost, 0, 1)
    return float(w[correctness[:, first]].sum()+(w[rows]*fraction).sum())


def choose_price(policy, z, y):
    grid = []
    for price in PRICES:
        route, _ = policy_route(z, policy, price)
        grid.append({'price': price, 'macro_accuracy': macro(route['prediction'], y), 'mean_views': float(route['costs'].mean())})
    feasible = [r for r in grid if r['mean_views'] <= 2+1e-12]
    check(bool(feasible), 'price_selection/feasible_fixed_grid')
    best_accuracy = max(r['macro_accuracy'] for r in feasible)
    tied = [r for r in feasible if r['macro_accuracy'] >= best_accuracy-1e-12]
    return min(tied, key=lambda r: (r['mean_views'], -r['price'])), grid


def tree_state_sha(state):
    h = hashlib.sha256()
    for name in state['nodes'].dtype.names:
        h.update(name.encode()); h.update(array_sha(state['nodes'][name]).encode())
    h.update(array_sha(state['values']).encode())
    h.update(str(state['max_depth']).encode()); h.update(str(state['node_count']).encode())
    return h.hexdigest()


def verify_fitted_models(policies, z, y, first, fold):
    """Exact independent recipe replay, in memory; no file/model replacement."""
    correctness = (z.argmax(2) == y[:, None]).astype(float)
    expected_log = {}; fold_negative = 0; proof = []
    states = [first]+[j for j in range(3, 6) if first in SUBSETS[j]]
    for method in METHODS:
        policy = policies[method]
        exact(policy.method, method, 'pickle/method/'+str(fold))
        exact(policy.first, first, 'pickle/first/'+str(fold))
        exact(sorted(policy.models), sorted(states), 'pickle/states/'+str(fold)+method)
        logs = {}
        for state in states:
            targets_at = destinations(first, state)
            target = correctness[:, targets_at]-correctness[:, state, None]
            x = features(z[:, state])
            exact(x.shape, (len(y), 14), 'fit_recipe/fourteen_state_features')
            check(np.all(np.isin(target, [-1., 0., 1.])), 'fit_recipe/signed_gain_range')
            negative = int(np.count_nonzero(target == -1))
            if method == 'extra_trees': fold_negative += negative
            counts = [[int(np.count_nonzero(target[:, j] == v)) for v in (-1, 0, 1)] for j in range(len(targets_at))]
            logs[NAMES[state]] = {'input_shape': [len(y), 14], 'destinations': [NAMES[j] for j in targets_at],
                                  'gain_counts_minus1_zero_plus1': counts}
            model = policy.models[state]
            evidence = {'fold': fold, 'method': method, 'state': NAMES[state], 'state_index': state,
                        'input_shape': list(x.shape), 'input_sha256': array_sha(x),
                        'signed_gain_shape': list(target.shape), 'signed_gain_sha256': array_sha(target),
                        'gain_counts_minus1_zero_plus1': counts}
            if method == 'class_gain':
                global_mean = target.mean(0); current_class = x[:, :12].argmax(1)
                table = np.array([(target[current_class == c].sum(0)+20*global_mean)/(np.count_nonzero(current_class == c)+20) for c in range(12)])
                exact(model.global_mean_, global_mean, 'class_gain_refit/global_mean/'+str(fold)+NAMES[state])
                exact(model.table_, table, 'class_gain_refit/shrunk_table/'+str(fold)+NAMES[state])
                evidence.update(global_mean_sha256=array_sha(global_mean), table_sha256=array_sha(table), shrinkage_examples=20)
            else:
                exact(type(model), ExtraTreesRegressor, 'tree_refit/estimator_type/'+str(fold))
                expected_model = ExtraTreesRegressor(**PARAMS)
                tree_compare(model.get_params(), expected_model.get_params(), 'tree_refit/all_hyperparameters/'+str(fold)+NAMES[state])
                exact(model.n_features_in_, 14, 'tree_refit/no_unobserved_feature_dimensions/'+str(fold))
                exact(model.n_outputs_, len(targets_at), 'tree_refit/action_outputs/'+str(fold))
                started = time.perf_counter()
                expected_model.fit(x, target[:, 0] if target.shape[1] == 1 else target)
                evidence['audit_refit_seconds_not_runtime_benchmark'] = time.perf_counter()-started
                exact(len(model.estimators_), 200, 'tree_refit/tree_count/'+str(fold)+NAMES[state])
                records = []
                for index, (saved_tree, refit_tree) in enumerate(zip(model.estimators_, expected_model.estimators_)):
                    a, b = saved_tree.tree_.__getstate__(), refit_tree.tree_.__getstate__()
                    exact(saved_tree.random_state, refit_tree.random_state, 'tree_refit/tree_rng/'+str(fold))
                    exact(a['nodes'], b['nodes'], 'tree_refit/all_node_fields/'+str(fold)+NAMES[state]+str(index))
                    exact(a['values'], b['values'], 'tree_refit/all_signed_target_values/'+str(fold)+NAMES[state]+str(index))
                    exact(a['max_depth'], b['max_depth'], 'tree_refit/depth/'+str(fold))
                    check(a['max_depth'] <= 4, 'tree_refit/max_depth_limit')
                    leaf = a['nodes']['left_child'] == -1
                    check(np.all(a['nodes']['n_node_samples'][leaf] >= 20), 'tree_refit/min_leaf_samples')
                    exact(int(a['nodes']['n_node_samples'][0]), len(y), 'tree_refit/all_fit_rows_at_root')
                    records.append({'tree': index, 'random_state': saved_tree.random_state,
                                    'node_count': a['node_count'], 'depth': a['max_depth'],
                                    'saved_tree_sha256': tree_state_sha(a), 'refit_tree_sha256': tree_state_sha(b),
                                    'nodes_equal': bool(np.array_equal(a['nodes'], b['nodes'])),
                                    'values_equal': bool(np.array_equal(a['values'], b['values']))})
                evidence['trees'] = records
                exact(model.predict(x), expected_model.predict(x), 'tree_refit/fit_prediction_exact/'+str(fold)+NAMES[state])
                REPORT['tree_refit_records'].append(evidence)
            proof.append({k: v for k, v in evidence.items() if k != 'trees'})
        expected_log[method] = {'states': logs, 'fit_count': len(y)}
    return expected_log, proof, fold_negative


def paired_bootstrap(a, b):
    differences = (np.asarray(a)-np.asarray(b))*100
    indices = np.random.default_rng(20260928).integers(0, 8, size=(10000, 8))
    samples = differences[indices].mean(1)
    return {'n_users': 8, 'mean_difference_pp': float(differences.mean()),
            'per_user_difference_pp': differences.tolist(), 'positive_users': int(np.count_nonzero(differences > 1e-10)),
            'worst_user_difference_pp': float(differences.min()), 'ci95_pp': np.quantile(samples, [.025, .975]).tolist()}


def audit_fold(fold, yall, subjects, keep, manifest):
    started = time.perf_counter(); old_failures = len(REPORT['failures'])
    directory = OUT/f'fold{fold}'; upstream = MV/f'fold{fold}'; bud = BUD/f'fold{fold}'
    result, selection, fit_log = [read_json(directory/name) for name in ('result.json', 'selection.json', 'fit_log.json')]
    br, bs, mr = read_json(bud/'result.json'), read_json(bud/'selection.json'), read_json(upstream/'result.json')
    tree_compare(result['protocol'], SPEC, 'protocol/fold_result/'+str(fold))
    tree_compare(selection['protocol'], SPEC, 'protocol/selection/'+str(fold))
    expected_identity = {'fold': fold, 'smoke': False, 'source': manifest['source'], 'config': SPEC,
                         'budget_result_sha256': sha(bud/'result.json'), 'budget_selection_sha256': sha(bud/'selection.json'),
                         'budget_identity': br['identity'], 'opportunity_summary_sha256': sha(BUD/'summary.json')}
    tree_compare(result['identity'], expected_identity, 'identity/result/'+str(fold))
    tree_compare(selection['identity'], expected_identity, 'identity/selection/'+str(fold))
    exact(sorted(result['files']), sorted(['selection.json', 'models.pkl', 'predictions.npz', 'fit_log.json']), 'identity/artifact_inventory/'+str(fold))
    for name, expected in result['files'].items(): exact(sha(directory/name), expected, 'identity/artifact_hash/'+str(fold)+name)
    exact(sha(upstream/'predictions.npz'), mr['files']['predictions.npz'], 'identity/mv_prediction_hash/'+str(fold))
    exact(sha(upstream/'predictions.npz'), br['identity']['predictions_sha256'], 'identity/budget_prediction_hash/'+str(fold))
    exact(sha(upstream/'result.json'), br['identity']['upstream_result_sha256'], 'identity/budget_upstream_result_hash/'+str(fold))
    exact(sha(MV/'manifest.json'), br['identity']['manifest_sha256'], 'identity/budget_upstream_manifest_hash/'+str(fold))
    exact(sha(bud/'selection.json'), br['files']['selection.json'], 'identity/frozen_selection_hash/'+str(fold))
    tree_compare(bs['identity'], br['identity'], 'identity/budget_selection/'+str(fold))
    exact(br['identity']['smoke'], False, 'identity/no_smoke/'+str(fold))
    exact(sha(CODE/'budget_analysis.py'), br['identity']['analysis_sha256'], 'identity/budget_analysis_live/'+str(fold))
    exact(sha(CODE/'run_calibration.py'), br['identity']['statistics_sha256'], 'identity/statistics_live/'+str(fold))
    data = read_arrays(upstream/'predictions.npz', ['ids', 'y', 'subjects', 'role', 'logits', 'subset_masks', 'train_ids', 'val_ids', 'test_ids'])
    parts = {'train': np.flatnonzero(keep & ~np.isin(subjects, [fold, fold%8+1])),
             'val': np.flatnonzero(keep & (subjects == fold%8+1)), 'test': np.flatnonzero(keep & (subjects == fold))}
    expected_ids = np.concatenate(list(parts.values()))
    exact(data['ids'], expected_ids, 'partitions/mv_id_order/'+str(fold))
    exact(data['y'], yall[expected_ids], 'partitions/labels/'+str(fold))
    exact(data['subjects'], subjects[expected_ids], 'partitions/subjects/'+str(fold))
    exact(data['role'], np.concatenate([np.full(len(v), i, dtype=np.int8) for i, v in enumerate(parts.values())]), 'partitions/roles/'+str(fold))
    for name, ids in parts.items(): exact(data[name+'_ids'], ids, 'partitions/'+name+str(fold))
    exact(data['subset_masks'].astype(bool), MASK, 'partitions/seven_mask_order/'+str(fold))
    exact(mr['identity']['partitions'], {name: array_sha(ids) for name, ids in parts.items()}, 'partitions/model_partition_hashes/'+str(fold))
    exact(mr['training_subjects'], sorted(set(range(1, 9))-{fold, fold%8+1}), 'partitions/encoder_six_users/'+str(fold))
    exact(mr['validation_subject'], fold%8+1, 'partitions/encoder_checkpoint_user/'+str(fold))
    val_rows, target_rows = data['role'] == 1, data['role'] == 2
    validation = {k: data[k][val_rows] for k in ('ids', 'y', 'logits')}
    target = {k: data[k][target_rows] for k in ('ids', 'y', 'logits')}
    fit, tune = split_validation(validation['ids'], validation['y'], fold)
    exact(selection['validation_fit_ids'], validation['ids'][fit].tolist(), 'split/stratified_fit_ids/'+str(fold))
    exact(selection['validation_tune_ids'], validation['ids'][tune].tolist(), 'split/stratified_price_ids/'+str(fold))
    check(not np.intersect1d(fit, tune).size and np.array_equal(np.sort(np.r_[fit, tune]), np.arange(len(val_rows.nonzero()[0]))), 'split/exhaustive_disjoint/'+str(fold))
    for selected in (fit, tune):
        ids = validation['ids'][selected]
        exact(np.unique(subjects[ids]).tolist(), [fold%8+1], 'split/whole_validation_subject/'+str(fold))
        check(not np.intersect1d(ids, parts['train']).size and not np.intersect1d(ids, parts['test']).size, 'split/no_encoder_train_or_test_overlap/'+str(fold))
    for c in np.unique(validation['y']):
        n = int(np.count_nonzero(validation['y'] == c))
        exact(int(np.count_nonzero(validation['y'][fit] == c)), int(.6*n), 'split/per_class_60pct/'+str(fold)+str(c))
        exact(int(np.count_nonzero(validation['y'][tune] == c)), n-int(.6*n), 'split/per_class_40pct/'+str(fold)+str(c))
    exact(selection['validation_array_sha256'], {k: array_sha(v) for k, v in validation.items()}, 'identity/validation_array_content/'+str(fold))
    frozen, valacc = choose_frozen_cascade(validation['logits'], validation['y'])
    tree_compare(bs['curves']['2.0'], frozen, 'cascade/independent_validation_selection/'+str(fold))
    close(bs['validation_fixed_accuracy'], valacc, 'cascade/validation_fixed_values/'+str(fold))
    exact(selection['frozen_cascade_params'], frozen['params'], 'cascade/frozen_policy_cascade/'+str(fold))
    close(selection['validation_fixed_accuracy'], valacc, 'cascade/policy_fixed_values/'+str(fold))
    first = frozen['params']['first']
    exact(selection['first_subset_index'], first, 'selection/first_index/'+str(fold))
    exact(selection['first_subset_name'], NAMES[first], 'selection/first_name/'+str(fold))
    exact(selection['uses_target_labels'], False, 'selection/no_target_labels_flag/'+str(fold))
    with (directory/'models.pkl').open('rb') as stream: policies = IndependentUnpickler(stream).load()
    exact(sorted(policies), sorted(METHODS), 'pickle/two_methods/'+str(fold))
    logs, proof, negative = verify_fitted_models(policies, validation['logits'][fit], validation['y'][fit], first, fold)
    for method in METHODS: logs[method]['tune_count'] = len(tune)
    tree_compare(fit_log, logs, 'fit_log/recomputed/'+str(fold))
    selected_prices, price_grids = {}, {}
    for method in METHODS:
        selected_prices[method], price_grids[method] = choose_price(policies[method], validation['logits'][tune], validation['y'][tune])
    tree_compare(selection['selected_prices'], selected_prices, 'price_selection/selection_file/'+str(fold))
    tree_compare(selection['validation_price_grid'], price_grids, 'price_selection/all_13_values/'+str(fold))
    tree_compare(result['selected_prices'], selected_prices, 'price_selection/result_file/'+str(fold))
    exact(selection['fold'], fold, 'identity/selection_fold/'+str(fold))
    expected_selection = {'identity': expected_identity, 'fold': fold, 'protocol': SPEC,
                          'first_subset_index': first, 'first_subset_name': NAMES[first],
                          'validation_fit_ids': validation['ids'][fit].tolist(), 'validation_tune_ids': validation['ids'][tune].tolist(),
                          'validation_array_sha256': {k: array_sha(v) for k, v in validation.items()},
                          'selected_prices': selected_prices, 'validation_price_grid': price_grids,
                          'frozen_cascade_params': frozen['params'], 'validation_fixed_accuracy': valacc.tolist(), 'uses_target_labels': False}
    tree_compare(selection, expected_selection, 'selection/full_reconstruction/'+str(fold))
    arrays = read_arrays(directory/'predictions.npz'); routes = {}; traces = {}
    z, labels = target['logits'], target['y']
    for method in METHODS:
        routes[method], traces[method] = policy_route(z, policies[method], selected_prices[method]['price'])
        route = routes[method]
        poisoned = np.full_like(z, np.nan)
        for stage in range(3):
            for subset in np.unique(route['path'][:, stage]):
                if subset < 0: continue
                visited = route['path'][:, stage] == subset
                poisoned[visited, subset] = z[visited, subset]
        replay, _ = policy_route(poisoned, policies[method], selected_prices[method]['price'])
        for key, value in route.items(): exact(replay[key], value, 'causality/poison_unvisited_exact/'+str(fold)+method+key)
    routes['frozen_budget2_cascade'] = frozen_cascade(z, frozen['params'])
    testacc = fixed_accuracy(z, labels)
    close(testacc, br['fixed_accuracy'], 'metrics/upstream_fixed_accuracy/'+str(fold))
    cascade_cost = float(routes['frozen_budget2_cascade']['costs'].mean())
    cascade_acc = macro(routes['frozen_budget2_cascade']['prediction'], labels)
    close(cascade_cost, br['cascade_curves']['2.0']['test_cost'], 'metrics/upstream_cascade_cost/'+str(fold))
    close(cascade_acc, br['cascade_curves']['2.0']['test_accuracy'], 'metrics/upstream_cascade_accuracy/'+str(fold))
    actual_cost = float(routes['extra_trees']['costs'].mean())
    mixture = fixed_mixture(valacc, actual_cost)
    expected_arrays = {'ids': target['ids'], 'y': labels, 'fixed_mixture_weights': mixture}
    metrics = {}
    for method, route in routes.items():
        expected_arrays.update({method+'_'+name: value for name, value in route.items()})
        metrics[method] = {'macro_accuracy': macro(route['prediction'], labels), 'mean_views': float(route['costs'].mean()),
                           'view_count_histogram': [int(np.count_nonzero(route['costs'] == c)) for c in (1, 2, 3)]}
    metrics['fixed_mixture_within_candidate_cost'] = {'macro_accuracy': float(mixture@testacc), 'mean_views': float(mixture@COST),
        'cost_ceiling': actual_cost, 'weights': mixture.tolist(),
        'accuracy_kind': 'exact random-mixture expectation; weights chosen with validation accuracy'}
    metrics['three_views'] = {'macro_accuracy': float(testacc[6]), 'mean_views': 3.}
    expected_arrays['three_views_prediction'] = z[:, 6].argmax(1)
    exact(sorted(arrays), sorted(expected_arrays), 'prediction_replay/array_keys/'+str(fold))
    for key in expected_arrays: exact(arrays[key], expected_arrays[key], 'prediction_replay/every_array_value/'+str(fold)+key)
    expected_result = {'identity': expected_identity, 'fold': fold, 'n_test': len(labels),
                       'selected_prices': selected_prices, 'metrics': metrics, 'fixed_subset_accuracy': testacc.tolist(),
                       'protocol': SPEC, 'can_claim_speedup': False, 'runtime_validation': 'pending actual on-demand benchmark',
                       'files': {name: sha(directory/name) for name in ['selection.json', 'models.pkl', 'predictions.npz', 'fit_log.json']}}
    tree_compare(result, expected_result, 'result_replay/every_field/'+str(fold))
    oracle2 = oracle_space(z, labels, first, 2.)
    mixture2 = fixed_mixture(valacc, 2.)
    oracle_cascade = oracle_space(z, labels, first, cascade_cost)
    REPORT['folds'].append({'fold': fold, 'source_users': mr['training_subjects'], 'validation_user': fold%8+1, 'test_user': fold,
        'validation_reused_for_encoder_checkpoint_selection': True,
        'fit_count': len(fit), 'price_validation_count': len(tune), 'test_count': len(labels),
        'fit_ids_sha256': array_sha(validation['ids'][fit]), 'price_validation_ids_sha256': array_sha(validation['ids'][tune]),
        'test_ids_sha256': array_sha(target['ids']), 'first': NAMES[first],
        'selected_prices': selected_prices, 'price_grid': price_grids, 'metrics': metrics,
        'model_fit_proof': proof, 'negative_training_target_count': negative, 'causal_reader_calls': traces,
        'opportunity_recomputed': {'oracle_at_2': oracle2, 'mixture_at_2': float(mixture2@testacc),
            'oracle_at_cascade_cost': oracle_cascade, 'cascade_accuracy': cascade_acc},
        'audit_seconds_not_latency': time.perf_counter()-started, 'new_failures': len(REPORT['failures'])-old_failures})
    print(f'POLICY AUDIT fold={fold}/8; new_failures={len(REPORT["failures"])-old_failures}; all 600 trees replayed; seconds={time.perf_counter()-started:.3f}', flush=True)
    return expected_result


def summarize(rows):
    methods = ['extra_trees', 'class_gain', 'frozen_budget2_cascade', 'fixed_mixture_within_candidate_cost', 'three_views']
    accuracy = {m: np.array([r['metrics'][m]['macro_accuracy'] for r in rows]) for m in methods}
    costs = {m: np.array([r['metrics'][m]['mean_views'] for r in rows]) for m in methods}
    n = np.array([r['n_test'] for r in rows])
    comparisons = {m: paired_bootstrap(accuracy['extra_trees'], accuracy[m]) for m in COMPARATORS}
    gap = float(np.mean(accuracy['extra_trees']-accuracy['three_views'])*100)
    gate = {'mean_actual_views_at_most_2': bool(costs['extra_trees'].mean() <= 2+1e-12),
            'within_1pp_of_three_views': gap >= -1-1e-10}
    fairness = {}
    for m in COMPARATORS:
        comp = comparisons[m]
        gate['gain_at_least_2pp_vs_'+m] = comp['mean_difference_pp'] >= 2-1e-10
        gate['positive_at_least_6users_vs_'+m] = comp['positive_users'] >= 6
        d = costs['extra_trees']-costs[m]
        fairness[m] = {'candidate_minus_comparator_mean_views': float(d.mean()), 'per_user_cost_difference': d.tolist(),
                       'per_user_cost_no_greater': (d <= 1e-12).tolist(),
                       'note': 'Per-user matching is descriptive; the adaptive-comparator gate uses equal-user population mean cost'}
        if m != 'fixed_mixture_within_candidate_cost': gate['population_cost_no_greater_than_'+m] = bool(d.mean() <= 1e-12)
    table = {m: {'macro_accuracy': float(accuracy[m].mean()), 'per_user_accuracy': accuracy[m].tolist(),
                 'mean_views': float(costs[m].mean()), 'per_user_mean_views': costs[m].tolist(),
                 'pooled_sample_mean_views_descriptive': float(np.average(costs[m], weights=n))} for m in methods}
    result = {'state': 'complete', 'users': 8, 'protocol': SPEC, 'methods': table, 'comparisons': comparisons, 'fairness': fairness, 'gate': gate,
              'candidate_minus_three_views_pp': gap,
              'decision': 'STATISTICAL_GATE_PASSED_RUNTIME_PENDING' if all(gate.values()) else 'STOP_VALUE_POLICY_MECHANISM',
              'can_claim_speedup': False, 'runtime_validation': 'separate actual on-demand benchmark required',
              'note': 'Exploratory eight-user, one-model-seed pilot. Bootstrap resamples users. No target-selected comparator winner.'}
    tree_compare(read_json(OUT/'summary.json'), result, 'summary_replay/every_field')
    REPORT['recomputed_summary'] = result
    opportunity = [f['opportunity_recomputed'] for f in REPORT['folds']]
    room1 = float(np.mean([o['oracle_at_2']-o['mixture_at_2'] for o in opportunity])*100)
    room2 = float(np.mean([o['oracle_at_cascade_cost']-o['cascade_accuracy'] for o in opportunity])*100)
    allowed = room1 >= 2-1e-10 and room2 >= 2-1e-10
    REPORT['conditional_trigger_recomputed'] = {'oracle_minus_mixture_at_2_pp': room1, 'oracle_minus_cascade_at_its_cost_ceiling_pp': room2,
        'policy_pilot_warranted': allowed, 'test_labels_used_in_predeclared_trigger': True,
        'not_external_confirmatory_data': True}
    check(allowed, 'opportunity/predeclared_conditional_trigger_met')
    saved_budget_summary = read_json(BUD/'summary.json')
    close(room1, saved_budget_summary['oracle_minus_mixture_at_2_pp'], 'opportunity/oracle_vs_mixture_replay')
    close(room2, saved_budget_summary['oracle_minus_cascade_at_same_cost_pp'], 'opportunity/oracle_vs_cascade_replay')
    exact(saved_budget_summary['opportunity_decision'], 'POLICY_PILOT_WARRANTED', 'opportunity/decision')
    REPORT['counts'] = {'outer_folds': 8, 'test_rows_once': int(n.sum()), 'methods': 2,
        'price_grid_points_replayed': 8*2*13, 'policy_test_routes_replayed': int(n.sum()*2),
        'policy_test_paths_and_added_view_elements': int(n.sum()*2*3*2),
        'pickle_models_verified': 8*2*3, 'extra_trees_models_independently_refit': len(REPORT['tree_refit_records']),
        'individual_trees_independently_refit_and_compared': sum(len(v['trees']) for v in REPORT['tree_refit_records']),
        'all_tree_nodes_equal': all(t['nodes_equal'] for v in REPORT['tree_refit_records'] for t in v['trees']),
        'all_tree_values_equal': all(t['values_equal'] for v in REPORT['tree_refit_records'] for t in v['trees']),
        'negative_fit_targets_preserved': sum(f['negative_training_target_count'] for f in REPORT['folds']),
        'formal_policy_prediction_archives_audited': 8, 'derived_result_fields_recomputed': True,
        'mixture_cost_strictly_lower_than_candidate_points': sum(r['metrics']['fixed_mixture_within_candidate_cost']['mean_views'] < r['metrics']['extra_trees']['mean_views']-1e-12 for r in rows)}


def final_report(started):
    REPORT['finished_utc'] = datetime.now(timezone.utc).isoformat()
    REPORT['audit_elapsed_seconds_not_latency_benchmark'] = time.perf_counter()-started
    REPORT['environment'] = {'python': platform.python_version(), 'python_executable': sys.executable,
        'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'cpu_only': True,
        'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'torch_imported': 'torch' in sys.modules,
        'original_value_policy_module_imported': 'value_policy' in sys.modules}
    check('torch' not in sys.modules and 'value_policy' not in sys.modules, 'execution/no_torch_or_original_policy_import')
    REPORT['script_sha256'] = sha(Path(__file__))
    REPORT['checks'] = dict(REPORT['checks']); REPORT['checks_total'] = sum(REPORT['checks'].values())
    REPORT['status'] = 'PASS' if not REPORT['failures'] else 'FAIL'
    REPORT['limitations'] = [
        'The validation user received no encoder gradient updates but was used to choose the encoder checkpoint, the frozen cascade and now the policy fit/price partition. This is internal reuse, not a fresh independent policy-development user.',
        'The predeclared oracle trigger uses outer-test labels. Conditional policy results from these same eight users are exploratory and cannot be called new external confirmatory evidence.',
        'Within run_arrays, choices are serialized before target_loader is invoked, and independently reconstructed fit/price inputs contain only validation IDs. Earlier budget/oracle analysis and policy preflight already read target arrays; this audit does NOT claim that all target I/O occurred after selection.',
        'Every saved sklearn ExtraTrees model was independently refit in memory with the frozen 14 observed-state features and signed -1/0/+1 targets; all 4800 trees are compared field by field. sklearn fitting/prediction is reused, while routing, features, costs, selection, mixtures, metrics and gates are independently implemented.',
        'Pickle files are fingerprint-checked and original policy classes are mapped to inert audit classes; original value_policy.py, checked_existing, fit_policy, route_policy and summarize are never imported or called.',
        'Hashes and saved lineage prove current internal consistency. They do not provide independent historical timestamp attestation. The exact in-memory refits support fit-data/recipe provenance, not reconstruction of historical process timing.',
        'Processing cost counts unique physical views including the initial view and any direct add-both action. This audit measures no latency, speedup, energy use, hardware saving or acquisition saving.',
        'The fixed random mixture is optimized under the candidate cost CEILING and can retain a cheaper better subset. This is not an equality-constrained cost match.',
        'The candidate must beat EACH frozen comparator by at least 2 pp with at least 6 positive users, cost no more than BOTH adaptive comparators and 2 views, and remain within 1 pp of LTR. Passing one comparison cannot repair another failed gate.',
        'The conclusion concerns one fixed ExtraTrees value predictor, one model seed and eight users under this data/model protocol. It does not show that all active sensing/value policies fail.',
        'Whole-user bootstrap is exploratory. Outer-fold training sets overlap, users are few, and quality filtering removes specified subject/class combinations.',
        'B model-training/encoder replay was audited separately. This audit checks the linked raw prediction artifacts and current identities without retraining or rerunning the neural encoder. B audit files remain unchanged.',
        'All elapsed/refit times in this evidence file are audit-work durations only, not inference-runtime measurements. Latency is explicitly outside scope.'
    ]
    lines = ['# Independent conditional-policy result audit', '',
             f'**{REPORT["status"]}**: {REPORT["checks_total"]} elementary consistency/numerical checks; {len(REPORT["failures"])} errors.', '',
             'Run: `D:/D_conda/envs/pytorch/python.exe D:/uwb_gestures/experiments/p1_20260928/value_policy_s2026/independent_audit.py`', '',
             'This is a CPU-only scientific-result audit. It does not import the original analysis module and writes only its script and the JSON/Markdown audit reports.', '']
    summary = REPORT.get('recomputed_summary')
    if summary:
        lines += ['## Recomputed outcome', '', f'**{summary["decision"]}**.', '', '| Method | User-mean macro accuracy | User-mean actual views |', '|---|---:|---:|']
        for name, row in summary['methods'].items(): lines.append(f'| {name} | {row["macro_accuracy"]*100:.9f}% | {row["mean_views"]:.12f} |')
        lines += ['', '| ExtraTrees minus comparator | Difference | Positive users | User bootstrap 95% interval |', '|---|---:|---:|---:|']
        for name, c in summary['comparisons'].items():
            lines.append(f'| {name} | {c["mean_difference_pp"]:.9f} pp | {c["positive_users"]}/8 | [{c["ci95_pp"][0]:.9f}, {c["ci95_pp"][1]:.9f}] pp |')
        lines += ['', f'ExtraTrees minus LTR: {summary["candidate_minus_three_views_pp"]:.9f} pp.', '', '## Gate reconstruction', '', '| Gate | Pass |', '|---|---|']
        lines += [f'| {name} | {value} |' for name, value in summary['gate'].items()]
        lines += ['', 'Every comparator is mandatory. Better accuracy than class-gain or the fixed mixture does not override the cascade accuracy failure, excess cost, adaptive-cost fairness failures, or the LTR gap.', '',
                  '## Evidence coverage', '',
                  '- Eight complete folds; live and frozen source/plan hashes, model pickle hashes, result identities, validation IDs, raw labels/subjects and role alignment checked.',
                  '- Per-class 60/40 split independently regenerated with seed 33000 + fold*100 + class. Source-gradient, validation and test users remain disjoint; checkpoint-selection reuse of validation is explicitly retained as a limitation.',
                  '- All class-gain tables reconstructed with 20-example shrinkage. All 24 ExtraTrees models refit in memory with 200 trees, depth 4, minimum leaf 20, seed 2026 and n_jobs=1. All 4800 tree node/value states have complete comparison records and canonical hashes in the JSON evidence.',
                  '- All signed negative, zero and positive gain targets reconstructed. All 208 price-grid points and tie/budget selection rules independently replayed using only price-validation rows.',
                  '- Loaded-pickle causal routes reproduced every test prediction, state path, added-view count, requested mask and saved gain. Poisoning all unvisited subsets with NaNs leaves outputs unchanged. Direct add-both costs two extra views; all final costs equal unique requested physical views.',
                  '- Fixed mixtures checked against an independent linear program with a cost ceiling, including a synthetic case where the optimal solution is strictly cheaper. All result fields and final summary/bootstraps/gates recomputed from raw arrays.',
                  '- The previous labeled-oracle opportunity trigger is independently recomputed and recorded as conditional/exploratory, not external confirmation.', '',
                  '## Per-fold decisions', '', '| Test user | Validation user | Fit/tune/test rows | Class-gain price | ExtraTrees price | ExtraTrees accuracy | ExtraTrees views |',
                  '|---:|---:|---|---:|---:|---:|---:|']
        for f in REPORT['folds']:
            et = f['metrics']['extra_trees']
            lines.append(f'| {f["fold"]} | {f["validation_user"]} | {f["fit_count"]}/{f["price_validation_count"]}/{f["test_count"]} | {f["selected_prices"]["class_gain"]["price"]} | {f["selected_prices"]["extra_trees"]["price"]} | {et["macro_accuracy"]*100:.6f}% | {et["mean_views"]:.12f} |')
        lines += ['', '## Limits', '']+['- '+s for s in REPORT['limitations']]
    if REPORT['failures']: lines += ['', '## Errors', '', '```json', json.dumps(REPORT['failures'], ensure_ascii=False, indent=2), '```']
    lines += ['', f'Audit script SHA-256: `{REPORT["script_sha256"]}`.',
              f'Frozen policy source SHA-256: `{SOURCE_SHA}`; test source SHA-256: `{TEST_SHA}`.',
              f'Audit duration: {REPORT["audit_elapsed_seconds_not_latency_benchmark"]:.3f} s. This is not a latency benchmark.', '']
    (OUT/'INDEPENDENT_AUDIT.json').write_text(json.dumps(REPORT, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (OUT/'INDEPENDENT_AUDIT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    started = time.perf_counter(); originals = {}
    try:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.name not in ['independent_audit.py', 'INDEPENDENT_AUDIT.json', 'INDEPENDENT_AUDIT.md']:
                sha(p); originals[str(p)] = (p.stat().st_size, p.stat().st_mtime_ns)
        # B audit outputs are read-only and explicitly checked for preservation.
        for name in ['independent_audit.py', 'INDEPENDENT_AUDIT.json', 'INDEPENDENT_AUDIT.md']:
            p = BUD/name; sha(p); originals[str(p)] = (p.stat().st_size, p.stat().st_mtime_ns)
        manifest = read_json(OUT/'manifest.json')
        tree_compare(manifest['config'], SPEC, 'protocol/frozen_manifest')
        exact(manifest['models'], str(MV.resolve()), 'identity/model_directory')
        exact(manifest['budget'], str(BUD.resolve()), 'identity/budget_directory')
        exact(manifest['smoke'], False, 'identity/formal_non_smoke')
        exact(manifest['summary_sha256'], sha(BUD/'summary.json'), 'identity/opportunity_summary')
        exact(sha(CODE/'value_policy.py'), SOURCE_SHA, 'source/explicit_policy_sha')
        exact(sha(CODE/'test_value_policy.py'), TEST_SHA, 'source/explicit_test_sha')
        for source, expected in manifest['source']['files'].items():
            exact(sha(source), expected, 'source/live/'+source)
            exact(sha(OUT/'source'/Path(source).name), expected, 'source/frozen/'+source)
        exact(platform.python_version(), manifest['source']['python'], 'runtime/python_version_for_exact_refit')
        exact(np.__version__, manifest['source']['numpy'], 'runtime/numpy_version_for_exact_refit')
        exact(sklearn.__version__, manifest['source']['sklearn'], 'runtime/sklearn_version_for_exact_refit')
        exact(sorted(p.name for p in OUT.glob('fold*') if p.is_dir()), ['fold'+str(i) for i in range(1, 9)], 'completion/exactly_eight_folds')
        y, subjects = np.load(PREP/'y.npy'), np.load(PREP/'subj.npy')
        sha(PREP/'y.npy'); sha(PREP/'subj.npy')
        keep = ~(((subjects == 2) & np.isin(y, [10, 11])) | ((subjects == 6) & (y == 3)))
        exact(len(y), 9593, 'dataquality/raw_9593'); exact(int(keep.sum()), 9300, 'dataquality/retained_9300')
        cheap = fixed_mixture(np.array([.9, .5, .4, .8, .7, .75, .85]), 2.5)
        exact(cheap, np.eye(7)[0], 'mixture/synthetic_cost_ceiling_not_equality')
        with threadpool_limits(limits=1):
            rows = [audit_fold(i, y, subjects, keep, manifest) for i in range(1, 9)]
        summarize(rows)
        for p, expected in originals.items():
            stat = Path(p).stat(); exact((stat.st_size, stat.st_mtime_ns), expected, 'immutability/end/'+p)
    except Exception as error:
        REPORT['failures'].append({'check': 'execution/exception', 'detail': repr(error), 'traceback': traceback.format_exc()})
    final_report(started)
    print(json.dumps({'status': REPORT['status'], 'checks': REPORT['checks_total'], 'errors': len(REPORT['failures']),
                      'decision': REPORT.get('recomputed_summary', {}).get('decision'),
                      'tree_refit_models': len(REPORT['tree_refit_records']),
                      'seconds_audit_only': REPORT['audit_elapsed_seconds_not_latency_benchmark']}, ensure_ascii=False), flush=True)
    return 0 if REPORT['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
