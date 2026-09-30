"""Conditional, causal extra-view value prediction with a frozen pilot protocol.

Only current-subset probabilities, entropy and margin enter the policy. The
array-backed reader is for offline evaluation; a runtime reader can encode and
cache requested views through the same callback interface. Offline timings do
not establish an end-to-end speedup.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import platform
import shutil
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor

from budget_analysis import (SUBSETS, COSTS, NAMES, analyze_fold, fixed_mixture,
                             oracle_bound, route_cascade, subset_accuracies, tune_cascades)
from run_calibration import ROOT, digest, dump, macro_accuracy, paired_summary

PRICES = (0., .005, .01, .02, .05, .1, .15, .2, .3, .4, .5, 1., 2.)
TREE_PARAMETERS = dict(n_estimators=200, max_depth=4, min_samples_leaf=20,
                       random_state=2026, n_jobs=1)
METHODS = ('class_gain', 'extra_trees')
COMPARATORS = ('frozen_budget2_cascade', 'class_gain', 'fixed_mixture_within_candidate_cost')
ARTIFACTS = ('selection.json', 'models.pkl', 'predictions.npz', 'fit_log.json')
TOLERANCE = 1e-12
CONFIG = {
    'policy_features': '12 current-subset probabilities, normalized entropy, top-two margin',
    'fit_fraction_per_class': .6,
    'split_seed': '33000 + fold*100 + class; ascending sample IDs before permutation',
    'price_grid': list(PRICES), 'price_selection_budget': 2.,
    'gain_target': 'correctness after requesting a superset minus current correctness',
    'class_gain_shrinkage_examples': 20., 'extra_trees': TREE_PARAMETERS,
    'action_ties': 'stop, fewer additional views, smaller subset index; tolerance 1e-12',
    'price_ties': 'higher macro accuracy, lower mean views, higher price; tolerance 1e-12',
    'main_candidate': 'extra_trees',
    'fixed_comparator_family': ['frozen_budget2_cascade', 'class_gain',
                               'fixed_mixture_within_candidate_cost'],
    'strong_simple_rule': 'candidate must gain >=2pp and improve >=6/8 users against EACH comparator; no target winner selection',
    'cost_fairness_rule': 'candidate population mean views must not exceed either adaptive comparator population mean; fixed mixture uses candidate cost as ceiling',
    'aggregation': 'each user receives equal weight for primary accuracy and cost; per-user cost matching is descriptive only',
    'runtime_rule': 'no speedup claim without a separate actual on-demand total-runtime benchmark',
    'validation_reuse_limit': 'policy user had no encoder gradient updates, but was reused for encoder checkpoint selection',
}


def state_features(logits):
    """Fourteen features from one already-observed subset, never another view."""
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 12 or not np.isfinite(z).all():
        raise ValueError('Current logits must be a finite [N,12] array')
    exp = np.exp(z-z.max(axis=1, keepdims=True))
    probability = exp/exp.sum(axis=1, keepdims=True)
    entropy = -(probability*np.log(np.maximum(probability, 1e-30))).sum(1)/np.log(12)
    ordered = np.sort(probability, axis=1)
    margin = ordered[:, -1]-ordered[:, -2]
    return np.column_stack((probability, entropy, margin))


def internal_split(ids, labels, fold):
    """Deterministic, class-stratified 60/40 validation-user partition."""
    ids, labels = np.asarray(ids), np.asarray(labels)
    if (ids.ndim != 1 or labels.shape != ids.shape or len(np.unique(ids)) != len(ids)
            or fold not in range(1, 9) or not np.isin(labels, np.arange(12)).all()):
        raise ValueError('Unique aligned IDs, labels0..11 and fold1..8 required')
    fit, tune = [], []
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        if len(rows) < 2:
            raise ValueError('Each validation class needs both fit and tune examples')
        rows = rows[np.argsort(ids[rows], kind='stable')]
        rng = np.random.default_rng(33000+fold*100+int(label))
        rows = rows[rng.permutation(len(rows))]
        split = int(np.floor(.6*len(rows)))
        fit.extend(rows[:split])
        tune.extend(rows[split:])
    return {'fit_indices': np.sort(np.asarray(fit, dtype=np.int64)),
            'tune_indices': np.sort(np.asarray(tune, dtype=np.int64))}


class ClassGainRegressor:
    """Predict gains by current predicted class with fixed empirical shrinkage."""
    def fit(self, features, targets):
        x, targets = np.asarray(features), np.asarray(targets, dtype=np.float64)
        if targets.ndim == 1:
            targets = targets[:, None]
        if x.shape != (len(targets), 14) or not len(x) or not np.isfinite(targets).all():
            raise ValueError('Aligned nonempty state features and finite gain targets required')
        predicted_class = x[:, :12].argmax(1)
        self.global_mean_ = targets.mean(0)
        self.table_ = np.empty((12, targets.shape[1]), dtype=np.float64)
        for cls in range(12):
            rows = predicted_class == cls
            self.table_[cls] = (targets[rows].sum(0)+20*self.global_mean_)/(int(rows.sum())+20)
        return self

    def predict(self, features):
        x = np.asarray(features)
        if x.ndim != 2 or x.shape[1] != 14 or not np.isfinite(x).all():
            raise ValueError('Finite [N,14] state features required')
        return self.table_[x[:, :12].argmax(1)]


def action_destinations(first, state):
    if first not in (0, 1, 2):
        raise ValueError('Starting subset must contain exactly one view')
    if state == first:
        return tuple(i for i in range(3, 7) if first in SUBSETS[i])
    if state in (3, 4, 5) and first in SUBSETS[state]:
        return (6,)
    raise ValueError('A gain predictor is only defined for a reachable single or pair')


@dataclass
class GainPolicy:
    method: str
    first: int
    models: dict

    def predict_gain(self, state, current_logits):
        gain = np.asarray(self.models[state].predict(state_features(current_logits)))
        if gain.ndim == 1:
            gain = gain[:, None]
        expected = (len(current_logits), len(action_destinations(self.first, state)))
        if gain.shape != expected or not np.isfinite(gain).all():
            raise ValueError('Gain regressor returned malformed or nonfinite predictions')
        return gain


def fit_policy(logits, labels, first, method):
    """Fit only from the supplied policy-fit rows; no tune or target argument."""
    z, y = np.asarray(logits), np.asarray(labels)
    if (z.shape != (len(y), 7, 12) or not len(y) or not np.isfinite(z).all()
            or not np.isin(y, np.arange(12)).all() or method not in METHODS):
        raise ValueError('Require finite fit logits, labels0..11 and a frozen method')
    states = (first,)+action_destinations(first, first)[:2]
    correct = (z.argmax(2) == y[:, None]).astype(np.float64)
    models = {}
    for state in states:
        destinations = action_destinations(first, state)
        targets = correct[:, destinations]-correct[:, state, None]
        features = state_features(z[:, state])
        model = ClassGainRegressor() if method == 'class_gain' else ExtraTreesRegressor(**TREE_PARAMETERS)
        # sklearn's one-output tree target is one-dimensional by convention.
        model.fit(features, targets[:, 0] if method == 'extra_trees' and len(destinations) == 1 else targets)
        models[state] = model
    return GainPolicy(method, first, models)


def route_policy(logits_or_reader, policy, price, n_rows=None):
    """Causal routing with a reader(rows, subset_index) acquisition boundary.

    A callback may encode/cache physical views. The router asks it only for
    requested supersets. All arrays returned here describe actual route paths.

    Saved ``models.pkl`` is a dict with ``class_gain`` and ``extra_trees``
    GainPolicy instances. After verified unpickling, obtain the price from
    ``selection['selected_prices'][method]['price']`` and call
    ``route_policy(reader, models[method], price, n_rows=1)``. The reader receives
    an integer row-index vector and a SUBSETS index, and returns [len(rows),12]
    logits. It must preprocess/encode only newly requested physical views and
    reuse its cached feature tensors. The policy holds only CPU regressors.
    """
    if not np.isfinite(price) or price < 0:
        raise ValueError('Price must be finite and nonnegative')
    action_destinations(policy.first, policy.first)
    if callable(logits_or_reader):
        reader = logits_or_reader
        if n_rows is None or int(n_rows) != n_rows or n_rows <= 0:
            raise ValueError('A callback requires a positive n_rows')
        n = int(n_rows)
    else:
        z = np.asarray(logits_or_reader)
        if z.ndim != 3 or z.shape[1:] != (7, 12) or not len(z):
            raise ValueError('Offline logits must have shape [N,7,12]')
        n = len(z)
        reader = lambda rows, subset: z[rows, subset]

    def read(rows, subset):
        current = np.asarray(reader(rows, subset))
        if current.shape != (len(rows), 12) or not np.isfinite(current).all():
            raise ValueError('Requested subset reader returned invalid logits')
        return current

    def decide(state, current):
        destinations = action_destinations(policy.first, state)
        gain = policy.predict_gain(state, current)
        candidates = (state,)+destinations
        extra = np.array([0]+[int(COSTS[d]-COSTS[state]) for d in destinations])
        score = np.column_stack((np.zeros(len(current)), gain))-price*extra
        # Candidate order already implements stop, cheaper action, lower index.
        tie = score >= score.max(1, keepdims=True)-TOLERANCE
        selected = np.asarray(candidates, dtype=np.int64)[tie.argmax(1)]
        return selected, gain

    first = policy.first
    initial_rows = np.arange(n)
    initial = read(initial_rows, first)
    choice, first_gain = decide(first, initial)
    first_choice = choice.copy()
    prediction = initial.argmax(1)
    path = np.full((n, 3), -1, dtype=np.int64)
    path[:, 0] = first
    added = np.zeros((n, 3), dtype=np.int64)
    added[:, 0] = 1
    pair_gain = np.full(n, np.nan)
    for destination in action_destinations(first, first):
        rows = np.flatnonzero(first_choice == destination)
        if not len(rows):
            continue
        current = read(rows, destination)
        prediction[rows] = current.argmax(1)
        path[rows, 1] = destination
        added[rows, 1] = COSTS[destination]-1
        if destination == 6:
            continue
        last_choice, gains = decide(destination, current)
        pair_gain[rows] = gains[:, 0]
        final_rows = rows[last_choice == 6]
        if len(final_rows):
            final = read(final_rows, 6)
            prediction[final_rows] = final.argmax(1)
            choice[final_rows] = 6
            path[final_rows, 2] = 6
            added[final_rows, 2] = 1
    requested = np.array([[view in SUBSETS[s] for view in range(3)] for s in choice])
    return {'prediction': prediction, 'subset_indices': choice, 'costs': COSTS[choice],
            'path': path, 'added_views': added, 'requested_mask': requested,
            'first_gains': first_gain, 'pair_gain': pair_gain}


def select_price(policy, logits, labels):
    """Select the fixed price using only the internal tune partition."""
    rows = []
    for price in PRICES:
        route = route_policy(logits, policy, price)
        rows.append({'price': price, 'macro_accuracy': macro_accuracy(route['prediction'], labels),
                     'mean_views': float(route['costs'].mean())})
    feasible = [row for row in rows if row['mean_views'] <= 2+TOLERANCE]
    if not feasible:
        raise ValueError('No price satisfies the frozen internal budget')
    best = max(row['macro_accuracy'] for row in feasible)
    tied = [row for row in feasible if row['macro_accuracy'] >= best-TOLERANCE]
    return min(tied, key=lambda row: (row['mean_views'], -row['price'])), rows


def array_digest(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def checked_existing(directory, identity):
    """Completed runs are immutable and bound to every input and live source."""
    directory = Path(directory)
    if not (directory/'result.json').exists():
        return None
    saved = json.loads((directory/'result.json').read_text(encoding='utf-8'))
    if saved['identity'] != identity:
        raise ValueError('Value-policy result identity changed (source, input or protocol)')
    if set(saved['files']) != set(ARTIFACTS):
        raise ValueError('Value-policy artifact inventory changed')
    for name, sha in saved['files'].items():
        if digest(directory/name) != sha:
            raise ValueError('Saved value-policy artifact changed: '+name)
    return saved


def validate_opportunity(summary, rows):
    """Recompute the two preregistered opportunity conditions, using all users."""
    if sorted(row['fold'] for row in rows) != list(range(1, 9)):
        raise ValueError('Opportunity requires all eight verified outer users')
    if any(row['identity']['smoke'] for row in rows):
        raise ValueError('Smoke evidence cannot authorize a formal opportunity pilot')
    if summary.get('state') != 'complete' or summary.get('opportunity_decision') != 'POLICY_PILOT_WARRANTED':
        raise ValueError('Formal value policy requires POLICY_PILOT_WARRANTED opportunity')
    values = {
        'oracle_minus_mixture_at_2_pp': np.mean([
            100*(row['oracle_at_2']['macro_accuracy']-row['fixed_mixture_at_2']['accuracy']) for row in rows]),
        'oracle_minus_cascade_at_same_cost_pp': np.mean([
            100*(row['cascade_curves']['2.0']['oracle_at_actual_cost']['macro_accuracy']
                 -row['cascade_curves']['2.0']['test_accuracy']) for row in rows])}
    for key, expected in values.items():
        if (not np.isfinite(expected) or expected < 2-1e-10
                or not np.isclose(summary.get(key, np.nan), expected, rtol=0, atol=1e-10)):
            raise ValueError('Opportunity summary differs from verified folds or fails gate: '+key)


def validate_budget_numbers(data, row, selection):
    """Recompute upstream gate inputs so editable JSON cannot authorize a run."""
    val, test = data['role'] == 1, data['role'] == 2
    selected, validation_accuracy = tune_cascades(data['logits'][val], data['y'][val])
    if (set(selection['curves']) != set(selected)
            or not np.allclose(selection['validation_fixed_accuracy'], validation_accuracy,
                               rtol=0, atol=TOLERANCE)):
        raise ValueError('Frozen budget selection does not reproduce validation')
    for name, expected in selected.items():
        actual = selection['curves'][name]
        if (actual['params'] != expected['params']
                or not np.allclose([actual['val_accuracy'], actual['val_cost']],
                                   [expected['val_accuracy'], expected['val_cost']], rtol=0, atol=TOLERANCE)):
            raise ValueError('Frozen budget selection does not reproduce validation')
    logits, labels = data['logits'][test], data['y'][test]
    primary = selected['2.0']['params']
    route = route_cascade(logits, primary)
    cost = float(route['costs'].mean())
    fixed = subset_accuracies(logits, labels)
    mixture = fixed_mixture(validation_accuracy, 2.)
    oracle2 = oracle_bound(logits, labels, primary['first'], 2.)
    oracle_cost = oracle_bound(logits, labels, primary['first'], cost)
    checks = [(row['fixed_accuracy'], fixed),
              (row['fixed_mixture_at_2']['weights'], mixture),
              ([row['fixed_mixture_at_2']['accuracy'], row['fixed_mixture_at_2']['cost']],
               [float(mixture @ fixed), float(mixture @ COSTS)]),
              ([row['cascade_curves']['2.0']['test_cost'], row['cascade_curves']['2.0']['test_accuracy']],
               [cost, macro_accuracy(route['prediction'], labels)])]
    for key, expected in oracle2.items():
        checks.append((row['oracle_at_2'][key], expected))
    for key, expected in oracle_cost.items():
        checks.append((row['cascade_curves']['2.0']['oracle_at_actual_cost'][key], expected))
    if any(not np.allclose(actual, expected, rtol=0, atol=TOLERANCE) for actual, expected in checks):
        raise ValueError('Frozen budget statistics do not reproduce prediction arrays')


def source_record():
    here = Path(__file__).resolve().parent
    sources = [here/name for name in ('value_policy.py', 'test_value_policy.py',
                                     'budget_analysis.py', 'run_calibration.py')]
    sources += [ROOT/'CONDITIONAL_POLICY_PLAN.md', ROOT/'AMENDMENT_003.md',
                ROOT/'POLICY_COMPARATOR_CLARIFICATION.md']
    return {'files': {str(path): digest(path) for path in sources},
            'python': platform.python_version(), 'numpy': np.__version__,
            'sklearn': sklearn.__version__}


def validate_output_paths(models, budget, out, allow_smoke):
    out = Path(out).resolve()
    for source in (Path(models).resolve(), Path(budget).resolve()):
        if out == source or source in out.parents or out in source.parents:
            raise ValueError('Policy output must be independent of both input trees')
    if allow_smoke and out == (ROOT/'value_policy_s2026').resolve():
        raise ValueError('Smoke policy requires an independent output directory')


def preflight(models, budget, folds, allow_smoke=False):
    """Verify existing budget artifacts; never create or amend budget results."""
    models, budget = Path(models), Path(budget)
    if not folds or len(set(folds)) != len(folds) or any(f not in range(1, 9) for f in folds):
        raise ValueError('Distinct folds1..8 required')
    summary = None
    if not allow_smoke:
        summary_path = budget/'summary.json'
        if not summary_path.is_file():
            raise ValueError('Formal policy requires an existing complete opportunity summary')
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        # Reject the no-go before opening any model or target prediction input.
        if summary.get('state') != 'complete' or summary.get('opportunity_decision') != 'POLICY_PILOT_WARRANTED':
            raise ValueError('Formal policy requires POLICY_PILOT_WARRANTED opportunity')
    verified, selections, result_hashes = {}, {}, {}
    for fold in (folds if allow_smoke else range(1, 9)):
        directory = budget/f'fold{fold}'
        if not (directory/'result.json').is_file() or not (directory/'selection.json').is_file():
            raise ValueError('Budget inputs must already exist before conditional policy: fold'+str(fold))
        # analyze_fold performs full upstream hashes/configuration/partition
        # validation and returns the existing result only after digest checks.
        before = digest(directory/'result.json')
        row = analyze_fold(models, budget, fold, allow_smoke=allow_smoke)
        if digest(directory/'result.json') != before:
            raise ValueError('Budget result changed during preflight')
        if bool(row['identity']['smoke']) != bool(allow_smoke):
            raise ValueError('Formal/smoke policy and upstream evidence must match')
        selection = json.loads((directory/'selection.json').read_text(encoding='utf-8'))
        if selection['identity'] != row['identity'] or selection.get('uses_target_labels') is not False:
            raise ValueError('Frozen budget selection identity or label-use declaration differs')
        with np.load(models/f'fold{fold}'/'predictions.npz', allow_pickle=False) as archive:
            validate_budget_numbers({key: archive[key] for key in ('logits', 'y', 'role')}, row, selection)
        verified[fold], selections[fold] = row, selection
        result_hashes[str(fold)] = before
    if summary is not None:
        validate_opportunity(summary, list(verified.values()))
    return {'rows': verified, 'selections': selections,
            'budget_result_sha256': result_hashes,
            'summary_sha256': digest(budget/'summary.json') if summary is not None else None,
            'models': str(models.resolve()), 'budget': str(budget.resolve()),
            'source': source_record(), 'config': CONFIG, 'smoke': bool(allow_smoke)}


def run_arrays(directory, fold, identity, validation, target_loader, frozen_selection):
    """Fit/tune from validation arrays, serialize choices, then open target data.

    Production callers obtain these arrays/identities from the verified
    preflight. The callback boundary permits direct synthetic leakage tests.
    """
    directory = Path(directory)
    existing = checked_existing(directory, identity)
    if existing is not None:
        return existing
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'selection.json').exists():
        previous = json.loads((directory/'selection.json').read_text(encoding='utf-8'))
        if previous['identity'] != identity:
            raise ValueError('Interrupted selection identity differs from this run')
    ids, y, z = (np.asarray(validation[key]) for key in ('ids', 'y', 'logits'))
    if z.shape != (len(ids), 7, 12) or y.shape != ids.shape:
        raise ValueError('Validation inputs must be aligned [N,7,12] logits and IDs/labels')
    split = internal_split(ids, y, fold)
    fit, tune = split['fit_indices'], split['tune_indices']
    cascade_params = frozen_selection['curves']['2.0']['params']
    first = cascade_params['first']
    if frozen_selection.get('uses_target_labels') is not False:
        raise ValueError('Frozen selection must declare validation-only selection')
    policies, selected, grids, logs = {}, {}, {}, {}
    for method in METHODS:
        policies[method] = fit_policy(z[fit], y[fit], first, method)
        selected[method], grids[method] = select_price(policies[method], z[tune], y[tune])
        states = {}
        correctness = (z[fit].argmax(2) == y[fit, None]).astype(np.int64)
        for state in policies[method].models:
            destinations = action_destinations(first, state)
            target = correctness[:, destinations]-correctness[:, state, None]
            states[NAMES[state]] = {
                'input_shape': [len(fit), 14], 'destinations': [NAMES[s] for s in destinations],
                'gain_counts_minus1_zero_plus1': [[int(np.sum(target[:, j] == value))
                         for value in (-1, 0, 1)] for j in range(len(destinations))]}
        logs[method] = {'states': states, 'fit_count': len(fit), 'tune_count': len(tune)}
    selection = {'identity': identity, 'fold': fold, 'protocol': CONFIG,
                 'first_subset_index': first, 'first_subset_name': NAMES[first],
                 'validation_fit_ids': ids[fit].tolist(), 'validation_tune_ids': ids[tune].tolist(),
                 'validation_array_sha256': {key: array_digest(validation[key]) for key in ('ids', 'y', 'logits')},
                 'selected_prices': selected, 'validation_price_grid': grids,
                 'frozen_cascade_params': cascade_params,
                 'validation_fixed_accuracy': frozen_selection['validation_fixed_accuracy'],
                 'uses_target_labels': False}
    # These writes occur before even invoking the target loader.
    with (directory/'models.pkl').open('wb') as stream:
        pickle.dump(policies, stream, protocol=4)
    dump(directory/'fit_log.json', logs)
    dump(directory/'selection.json', selection)

    target = target_loader()
    target_ids, target_logits = np.asarray(target['ids']), np.asarray(target['logits'])
    if (target_ids.ndim != 1 or len(np.unique(target_ids)) != len(target_ids)
            or np.intersect1d(target_ids, ids).size
            or target_logits.shape != (len(target_ids), 7, 12)):
        raise ValueError('Target IDs/logits are malformed or overlap validation')
    routes = {method: route_policy(target_logits, policies[method], selected[method]['price']) for method in METHODS}
    routes['frozen_budget2_cascade'] = route_cascade(target_logits, cascade_params)
    actual_cost = float(routes['extra_trees']['costs'].mean())
    # This comparator choice reads target cost, but never target correctness.
    mixture = fixed_mixture(np.asarray(selection['validation_fixed_accuracy']), actual_cost)
    labels = np.asarray(target['y'])
    if labels.shape != target_ids.shape or not np.isin(labels, np.arange(12)).all():
        raise ValueError('Target labels must align with IDs and lie in0..11')
    metrics, arrays = {}, {'ids': target_ids, 'y': labels, 'fixed_mixture_weights': mixture}
    for method, route in routes.items():
        metrics[method] = {'macro_accuracy': macro_accuracy(route['prediction'], labels),
                           'mean_views': float(route['costs'].mean()),
                           'view_count_histogram': [int(np.sum(route['costs'] == cost)) for cost in (1, 2, 3)]}
        arrays.update({method+'_'+name: value for name, value in route.items()})
    fixed = subset_accuracies(target_logits, labels)
    metrics['fixed_mixture_within_candidate_cost'] = {
        'macro_accuracy': float(mixture @ fixed), 'mean_views': float(mixture @ COSTS),
        'cost_ceiling': actual_cost, 'weights': mixture.tolist(),
        'accuracy_kind': 'exact random-mixture expectation; weights chosen with validation accuracy'}
    metrics['three_views'] = {'macro_accuracy': float(fixed[6]), 'mean_views': 3.}
    arrays['three_views_prediction'] = target_logits[:, 6].argmax(1)
    np.savez_compressed(directory/'predictions.npz', **arrays)
    result = {'identity': identity, 'fold': fold, 'n_test': len(labels),
              'selected_prices': selected, 'metrics': metrics,
              'fixed_subset_accuracy': fixed.tolist(), 'protocol': CONFIG,
              'can_claim_speedup': False, 'runtime_validation': 'pending actual on-demand benchmark',
              'files': {name: digest(directory/name) for name in ARTIFACTS}}
    dump(directory/'result.json', result)
    return result


def ensure_manifest(out, context):
    out = Path(out)
    manifest = {key: context[key] for key in ('models', 'budget', 'summary_sha256', 'source', 'config', 'smoke')}
    path = out/'manifest.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Value-policy manifest source/input/protocol changed')
        for source, sha in context['source']['files'].items():
            if digest(out/'source'/Path(source).name) != sha:
                raise ValueError('Value-policy source snapshot changed')
        return
    (out/'source').mkdir(parents=True, exist_ok=True)
    for source, sha in context['source']['files'].items():
        destination = out/'source'/Path(source).name
        shutil.copyfile(source, destination)
        if digest(destination) != sha:
            raise ValueError('Could not freeze source snapshot')
    dump(path, manifest)


def run_fold(models, budget, out, fold, context):
    """Run a fold only with the context returned by this run's preflight."""
    models, budget, out = Path(models), Path(budget), Path(out)
    if (str(models.resolve()) != context['models'] or str(budget.resolve()) != context['budget']
            or fold not in context['rows'] or context['source'] != source_record()):
        raise ValueError('Preflight source/input context changed')
    if not context['smoke'] and context['summary_sha256'] != digest(budget/'summary.json'):
        raise ValueError('Opportunity summary changed after preflight')
    budget_directory = budget/f'fold{fold}'
    if digest(budget_directory/'result.json') != context['budget_result_sha256'][str(fold)]:
        raise ValueError('Budget result changed after preflight')
    budget_row = context['rows'][fold]
    path = models/f'fold{fold}'/'predictions.npz'
    if digest(path) != budget_row['identity']['predictions_sha256']:
        raise ValueError('Multiview predictions changed after budget verification')
    identity = {'fold': fold, 'smoke': context['smoke'], 'source': context['source'], 'config': CONFIG,
                'budget_result_sha256': digest(budget_directory/'result.json'),
                'budget_selection_sha256': digest(budget_directory/'selection.json'),
                'budget_identity': budget_row['identity'],
                'opportunity_summary_sha256': context['summary_sha256']}
    if identity['budget_selection_sha256'] != budget_row['files']['selection.json']:
        raise ValueError('Budget selection changed after preflight')
    directory = out/f'fold{fold}'
    existing = checked_existing(directory, identity)
    if existing is not None:
        return existing
    with np.load(path, allow_pickle=False) as archive:
        role = archive['role']
        validation = {key: archive[key][role == 1] for key in ('ids', 'y', 'logits')}
        def target_loader():
            return {key: archive[key][role == 2] for key in ('ids', 'y', 'logits')}
        return run_arrays(directory, fold, identity, validation, target_loader, context['selections'][fold])


def summarize(rows, out):
    if sorted(row['fold'] for row in rows) != list(range(1, 9)):
        raise ValueError('Eight completed outer users are required for a policy decision')
    rows = sorted(rows, key=lambda row: row['fold'])
    if any(row['identity']['smoke'] for row in rows):
        raise ValueError('Smoke results cannot be summarized as scientific outcomes')
    if any(row['identity']['source'] != rows[0]['identity']['source'] for row in rows):
        raise ValueError('Policy sources differ between outer users')
    names = ('extra_trees', 'class_gain', 'frozen_budget2_cascade',
             'fixed_mixture_within_candidate_cost', 'three_views')
    accuracy = {name: np.array([row['metrics'][name]['macro_accuracy'] for row in rows]) for name in names}
    costs = {name: np.array([row['metrics'][name]['mean_views'] for row in rows]) for name in names}
    if any(not np.isfinite(accuracy[name]).all() or not np.isfinite(costs[name]).all()
           or np.any((accuracy[name] < 0) | (accuracy[name] > 1))
           or np.any((costs[name] < 1) | (costs[name] > 3)) for name in names):
        raise ValueError('Finite macro accuracies0..1 and processing costs1..3 required')
    candidate = accuracy['extra_trees']
    comparisons = {name: paired_summary(candidate[:, None], accuracy[name][:, None]) for name in COMPARATORS}
    fairness = {}
    gate = {'mean_actual_views_at_most_2': float(costs['extra_trees'].mean()) <= 2+TOLERANCE,
            'within_1pp_of_three_views': float((candidate-accuracy['three_views']).mean()*100) >= -1-1e-10}
    for name, comparison in comparisons.items():
        gate['gain_at_least_2pp_vs_'+name] = comparison['mean_difference_pp'] >= 2-1e-10
        gate['positive_at_least_6users_vs_'+name] = comparison['positive_users'] >= 6
        difference = costs['extra_trees']-costs[name]
        fairness[name] = {'candidate_minus_comparator_mean_views': float(difference.mean()),
                          'per_user_cost_difference': difference.tolist(),
                          'per_user_cost_no_greater': (difference <= TOLERANCE).tolist(),
                          'note': 'Per-user matching is descriptive; the adaptive-comparator gate uses equal-user population mean cost'}
        if name != 'fixed_mixture_within_candidate_cost':
            gate['population_cost_no_greater_than_'+name] = float(difference.mean()) <= TOLERANCE
    sample_counts = np.array([row['n_test'] for row in rows])
    table = {name: {'macro_accuracy': float(accuracy[name].mean()),
                    'per_user_accuracy': accuracy[name].tolist(),
                    'mean_views': float(costs[name].mean()), 'per_user_mean_views': costs[name].tolist(),
                    'pooled_sample_mean_views_descriptive': float(np.average(costs[name], weights=sample_counts))}
             for name in names}
    result = {'state': 'complete', 'users': 8, 'protocol': CONFIG, 'methods': table,
              'comparisons': comparisons, 'fairness': fairness, 'gate': gate,
              'candidate_minus_three_views_pp': float((candidate-accuracy['three_views']).mean()*100),
              'decision': 'STATISTICAL_GATE_PASSED_RUNTIME_PENDING' if all(gate.values()) else 'STOP_VALUE_POLICY_MECHANISM',
              'can_claim_speedup': False, 'runtime_validation': 'separate actual on-demand benchmark required',
              'note': 'Exploratory eight-user, one-model-seed pilot. Bootstrap resamples users. No target-selected comparator winner.'}
    dump(Path(out)/'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, default=ROOT/'multiview_s2026')
    parser.add_argument('--budget', type=Path, default=ROOT/'budget_s2026')
    parser.add_argument('--out', type=Path, default=ROOT/'value_policy_s2026')
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(1, 9)))
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    validate_output_paths(args.models, args.budget, args.out, args.allow_smoke)
    context = preflight(args.models, args.budget, args.folds, args.allow_smoke)
    ensure_manifest(args.out, context)
    rows = []
    for fold in args.folds:
        rows.append(run_fold(args.models, args.budget, args.out, fold, context))
        print(f'VALUE_POLICY_COMPLETE fold={fold}; results saved, no partial gate', flush=True)
    if sorted(args.folds) == list(range(1, 9)) and not args.allow_smoke:
        result = summarize(rows, args.out)
        print(json.dumps({'state': result['state'], 'decision': result['decision']}), flush=True)


if __name__ == '__main__':
    # Keep pickle class names importable by independent audit/benchmark scripts.
    from value_policy import main as canonical_main
    canonical_main()
