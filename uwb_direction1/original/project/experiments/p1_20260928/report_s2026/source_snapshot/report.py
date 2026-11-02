"""CPU-only, provenance-checked report for the frozen new-route pilot.

No model is trained, no CUDA context is created, and no input is modified.
Formal output is possible only after every mandatory eight-user result and the
prespecified on-demand latency measurement is complete. --audit-core prints an
explicitly incomplete A/B audit and never writes a research deliverable.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import platform
import shutil
import sys

import numpy as np

import budget_analysis as B
import run_calibration as A
import benchmark_latency as L
from benchmark_latency import HashInventory

HERE = Path(__file__).resolve().parent
ROOT = Path('D:/uwb_gestures/experiments/p1_20260928')
OLD = ROOT.parent/'p1_20260905'
FOLDS = tuple(range(1, 9))
A_METHODS = ('frozen', 'source_prototype', 'supported_only', 'translation',
             'ridge_affine', 'lowrank', 'validation_selected_simple', 'allclass_lowrank')
DOCS = ('REPORT_REQUIREMENTS.md', 'PREREGISTRATION.md', 'AMENDMENT_001.md',
        'AMENDMENT_002.md', 'AMENDMENT_003.md', 'CONDITIONAL_POLICY_PLAN.md',
        'POLICY_COMPARATOR_CLARIFICATION.md', 'LATENCY_PROTOCOL.md', 'LITERATURE_NOTES.md')
A_LABELS = {'frozen': '冻结 Right 分类器', 'source_prototype': '冻结源原型（诊断）',
            'supported_only': '仅支持类中心移动', 'translation': '共同平移',
            'ridge_affine': 'ridge affine', 'lowrank': 'rank2 用户因素',
            'validation_selected_simple': '验证选择的简单校准',
            'allclass_lowrank': '全类别高预算低秩参考'}


def fail(message):
    raise ValueError(message)


def read_json(path, inventory=None):
    if inventory is not None:
        inventory.verify(path)
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError('Missing or invalid JSON: '+str(path)) from error


def read_npz(path, inventory=None):
    if inventory is not None:
        inventory.verify(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def reconcile(actual, expected, context='result', atol=1e-10, exact_keys=False):
    """Compare the recomputed field tree; never coerce NaN to agreement."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or (exact_keys and set(actual) != set(expected)):
            fail(context+': field inventory differs')
        for key, value in expected.items():
            if key not in actual:
                fail(context+': missing '+key)
            reconcile(actual[key], value, context+'.'+key, atol, exact_keys)
    elif isinstance(expected, (list, tuple, np.ndarray)):
        if not isinstance(actual, (list, tuple, np.ndarray)) or len(actual) != len(expected):
            fail(context+': array size differs')
        for i, value in enumerate(expected):
            reconcile(actual[i], value, context+f'[{i}]', atol, exact_keys)
    elif isinstance(expected, (bool, str)) or expected is None:
        if actual != expected:
            fail(context+': value differs')
    elif isinstance(expected, (int, float, np.number)):
        if (isinstance(actual, bool) or not isinstance(actual, (int, float, np.number))
                or not np.isfinite(actual) or not np.isfinite(expected)
                or abs(actual-expected) > atol):
            fail(context+': numerical value differs')
    elif actual != expected:
        fail(context+': value differs')


def assert_array(actual, expected, context):
    if not np.array_equal(np.asarray(actual), np.asarray(expected)):
        fail(context+': arrays differ')


def verify_artifacts(directory, files, inventory):
    L.verify_files(directory, files, inventory)


def require_eight_folds(rows):
    if sorted(row.get('fold') for row in rows) != list(FOLDS):
        fail('Exactly eight distinct outer folds1..8 are required')
    for row in rows:
        identity = row.get('identity', {})
        if identity.get('smoke') or identity.get('config', {}).get('smoke'):
            fail('smoke results cannot support a formal report')


def validate_ids(ids, labels, raw_y, raw_subjects, fold=None):
    ids, labels = np.asarray(ids), np.asarray(labels)
    if (ids.ndim != 1 or labels.shape != ids.shape or ids.dtype.kind not in 'iu'
            or len(np.unique(ids)) != len(ids)):
        fail('IDs must be a unique integer vector aligned with labels')
    if ((ids < 0) | (ids >= len(raw_y))).any():
        fail('ID outside the source data')
    if not np.array_equal(labels, np.asarray(raw_y)[ids]):
        fail('Raw label identity differs')
    if fold is not None and not np.all(np.asarray(raw_subjects)[ids] == fold):
        fail('Raw user identity differs')


def within_budget(actual, limit=2.):
    return bool(np.isfinite(actual) and actual <= limit+1e-12)


def paired_user_effect(candidate, baseline):
    row = A.paired_summary(candidate, baseline)
    row.update(bootstrap_unit='user', bootstrap_draws=10000,
               bootstrap_seed=20260928, support_repeats_averaged_within_user=True)
    return row


def macro_f1(prediction, labels, classes):
    prediction, labels = np.asarray(prediction), np.asarray(labels)
    if prediction.shape != labels.shape or labels.ndim != 1 or not len(labels):
        fail('Aligned nonempty predictions and labels required for F1')
    scores = []
    for c in classes:
        tp = np.sum((prediction == c) & (labels == c))
        denominator = np.sum(prediction == c)+np.sum(labels == c)
        scores.append(2*tp/denominator if denominator else 0.)
    if not scores:
        fail('No classes for F1')
    return float(np.mean(scores))


def classification_metrics(prediction, labels, selected):
    prediction, labels = np.asarray(prediction), np.asarray(labels)
    present = np.unique(labels)
    excluded = present[~np.isin(present, selected)]
    mask = np.isin(labels, excluded)
    return {'all': A.macro_accuracy(prediction, labels),
            'calibrated': A.macro_accuracy(prediction, labels, selected),
            'uncalibrated': A.macro_accuracy(prediction, labels, excluded) if len(excluded) else None,
            'f1_all_present': macro_f1(prediction, labels, present),
            'f1_all_fixed12': macro_f1(prediction, labels, range(12)),
            'f1_uncalibrated': macro_f1(prediction[mask], labels[mask], excluded) if len(excluded) else None}


def align_query(data, query_ids, query_y, fold):
    ids, query_ids = np.asarray(data['ids']), np.asarray(query_ids)
    if len(np.unique(ids)) != len(ids) or len(np.unique(query_ids)) != len(query_ids):
        fail('Cross-route IDs must be unique')
    lookup = {int(value): i for i, value in enumerate(ids)}
    if any(int(value) not in lookup for value in query_ids):
        fail('Cross-route query ID missing')
    rows = np.array([lookup[int(value)] for value in query_ids])
    assert_array(data['y'][rows], query_y, 'Cross-route label')
    if not np.all(data['subjects'][rows] == fold) or not np.all(data['role'][rows] == 2):
        fail('Cross-route fold/user role differs')
    return data['logits'][rows]


def group_metrics(prediction, costs, labels):
    prediction, costs, labels = map(np.asarray, (prediction, costs, labels))
    result = {}
    for name, mask in [('gesture', labels != 11), ('idle', labels == 11)]:
        result[name] = ({'n': int(mask.sum()),
                         'macro_accuracy': A.macro_accuracy(prediction[mask], labels[mask]),
                         'mean_views': float(costs[mask].mean())} if mask.any() else None)
    return result


def rescue_decomposition(predictions, labels, first):
    correct = np.asarray(predictions) == np.asarray(labels)[:, None]
    pairs = [i for i in range(3, 6) if B.SUBSETS[first][0] in B.SUBSETS[i]]
    one, full = correct[:, first], correct[:, 6]
    return {'n': len(labels), 'first_subset': B.NAMES[first],
            'single_wrong_full_correct': int(np.sum(~one & full)),
            'single_correct_full_wrong': int(np.sum(one & ~full)),
            'single_wrong_any_pair_correct': int(np.sum(~one & correct[:, pairs].any(1))),
            'joint_only_rescue': int(np.sum(~one & ~correct[:, pairs].any(1) & full)),
            'single_and_full_wrong': int(np.sum(~one & ~full))}


def validate_latency_rows(rows, selected, methods, raw_y, raw_subjects, repeats):
    expected = {(fold, int(i), method, repeat) for fold, ids in selected.items()
                for i in ids for method in methods for repeat in range(repeats)}
    seen = set()
    for row in rows:
        key = (row['fold'], row['clip_id'], row['method'], row['repeat'])
        if key in seen:
            fail('Latency duplicate clip/method/repeat')
        seen.add(key)
        if key not in expected:
            fail('Latency row is not in the prespecified clip/method/repeat set')
        validate_ids(np.array([row['clip_id']]), [row['label']], raw_y, raw_subjects, row['fold'])
        if not row.get('replay', {}).get('bitwise_equal'):
            fail('Latency batch-one replay failed')
        subset = row['subset_index']
        if subset not in range(7):
            fail('Latency subset outside fixed seven subsets')
        views = list(B.SUBSETS[subset])
        if (sorted(row['raw_accesses']) != views or sorted(row['encoded_views']) != views
                or row['actual_views'] != len(views)):
            fail('Latency actual observations/encoding do not agree with subset cost')
        path = [s for s in row['path'] if s != -1]
        if (not path or path[-1] != subset or len(set(path)) != len(path)
                or any(not set(B.SUBSETS[a]) < set(B.SUBSETS[b]) for a, b in zip(path, path[1:]))):
            fail('Latency path is not monotone append with encoding reuse')
        if (len(row['final_logits']) != 12 or not np.isfinite(row['final_logits']).all()
                or int(np.argmax(row['final_logits'])) != row['prediction']):
            fail('Latency prediction does not match actual final logits')
    if seen != expected:
        fail('Latency clips/methods/repeats incomplete')
    return L.summarize(rows)


def require_inputs(root, formal=True):
    root = Path(root)
    files = [root/name for name in DOCS]
    files += [root/name/'summary.json' for name in ('calibration_s2026', 'budget_s2026')]
    for fold in FOLDS:
        files.extend(root/name/f'fold{fold}'/suffix for name, suffix in (
            ('baseline_s2026', 'warmup/result.json'), ('multiview_s2026', 'result.json'),
            ('calibration_s2026', 'result.json'), ('budget_s2026', 'result.json')))
    if formal:
        budget_path = root/'budget_s2026/summary.json'
        needs_policy = (not budget_path.exists() or
                        read_json(budget_path).get('opportunity_decision') == 'POLICY_PILOT_WARRANTED')
        if needs_policy:
            files += [root/'value_policy_s2026/summary.json']
            files += [root/'value_policy_s2026'/f'fold{fold}'/'result.json' for fold in FOLDS]
        files += [root/'latency_s2026/summary.json', root/'latency_s2026/preflight.json']
        files += [root/'latency_s2026'/f'fold{fold}'/'observations.json' for fold in FOLDS]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        fail('Formal report inputs incomplete; no report written:\n'+'\n'.join(missing))


def summarize_a(rows):
    require_eight_folds(rows)
    arrays = {name: np.array([[m['uncalibrated'] for m in r['methods'][name]['metrics']]
                              for r in rows]) for name in A_METHODS}
    if any(value.shape != (8, 20) for value in arrays.values()):
        fail('A requires exactly twenty support repetitions within each of eight users')
    table = {}
    for name in A_METHODS:
        per_user = arrays[name].mean(1)
        metrics = [r['methods'][name]['metrics'] for r in rows]
        mse = np.array([[m['uncalibrated_center_mse'] for m in mm] for mm in metrics])
        mse0 = np.array([[m['uncalibrated_center_mse'] for m in r['methods']['frozen']['metrics']] for r in rows])
        table[name] = dict(uncalibrated_macro_accuracy=float(per_user.mean()),
            uncalibrated_per_user=per_user.tolist(), worst_user_uncalibrated=float(per_user.min()),
            within_user_support_std=arrays[name].std(1, ddof=1).tolist(),
            all_macro_accuracy=float(np.mean([[m['all'] for m in mm] for mm in metrics])),
            calibrated_macro_accuracy=float(np.mean([[m['calibrated'] for m in mm] for mm in metrics])),
            vs_frozen=A.paired_summary(arrays[name], arrays['frozen']),
            support_budget_per_user=[r['methods'][name].get('support_budget',
                  0 if name in ('frozen', 'source_prototype') else 9) for r in rows],
            uncalibrated_center_mse=float(mse.mean()), center_mse_reduction=float((mse0-mse).mean()))
        for metric in ('f1_all_present', 'f1_all_fixed12', 'f1_uncalibrated'):
            values = np.array([[m[metric] for m in mm] for mm in metrics])
            table[name][metric] = float(values.mean())
            table[name][metric+'_per_user'] = values.mean(1).tolist()
        table[name]['all_per_user'] = [float(np.mean([m['all'] for m in mm])) for mm in metrics]
        table[name]['calibrated_per_user'] = [float(np.mean([m['calibrated'] for m in mm])) for mm in metrics]
    comparison = A.paired_summary(arrays['lowrank'], arrays['validation_selected_simple'])
    gate = A.decision_gate(comparison, table['lowrank']['vs_frozen'])
    return {'state': 'complete', 'users': 8, 'support_repeats': 20, 'methods': table,
            'lowrank_vs_simple': comparison, 'gate': gate,
            'decision': 'GO' if all(gate.values()) else 'STOP_LOW_RANK_MECHANISM'}


def validate_archive(data, raw_y, raw_subjects, fold, multiview=False):
    validate_ids(data['ids'], data['y'], raw_y, raw_subjects)
    keep = ~(((raw_subjects == 2) & np.isin(raw_y, [10, 11]))
             | ((raw_subjects == 6) & (raw_y == 3)))
    if keep.sum() != 9300:
        fail('Frozen quality protocol must retain 9300 observations')
    assert_array(np.sort(data['ids']), np.flatnonzero(keep), 'Full archive legal IDs')
    assert_array(data['subjects'], raw_subjects[data['ids']], 'Full archive subjects')
    roles = np.where(data['subjects'] == fold, 2, np.where(data['subjects'] == fold % 8+1, 1, 0))
    assert_array(data['role'], roles, 'Six/one/one whole-user roles')
    expected_shape = (9300, 7, 12) if multiview else (9300, 12)
    if data['logits'].shape != expected_shape or not np.isfinite(data['logits']).all():
        fail('Malformed or nonfinite logits')
    if multiview:
        masks = [[v in subset for v in range(3)] for subset in B.SUBSETS]
        assert_array(data['subset_masks'], masks, 'Fixed L/T/R/LT/LR/TR/LTR order')
    elif data['features'].shape != (9300, 256) or not np.isfinite(data['features']).all():
        fail('Malformed frozen 256-dimensional feature archive')


def audit_a_fold(root, fold, baseline, data, inventory):
    from calibration import fixed_partition, sample_support, correction_scores, prototype_scores
    directory = root/'calibration_s2026'/f'fold{fold}'
    row = read_json(directory/'result.json', inventory)
    require = ('selection.json', 'source_model.pkl', 'predictions.npz')
    if set(row['files']) != set(require):
        fail('A artifact inventory differs')
    verify_artifacts(directory, row['files'], inventory)
    selection = read_json(directory/'selection.json', inventory)
    identity = row['identity']
    feature_path = root/'baseline_s2026'/f'fold{fold}'/'warmup/features.npz'
    expected_identity = dict(features_sha256=inventory.verify(feature_path),
        feature_metadata_sha256=inventory.verify(feature_path.with_suffix('.json')),
        upstream_result_sha256=inventory.verify(feature_path.parent/'result.json'),
        source={name: inventory.verify(HERE/name) for name in ('run_calibration.py', 'calibration.py')},
        reg=list(A.REGS), tau=list(A.TAUS), seeds=list(A.SEEDS), fold=fold)
    reconcile(identity, expected_identity, 'A identity', exact_keys=True)
    reconcile(selection['identity'], identity, 'A selection identity', exact_keys=True)
    with (directory/'source_model.pkl').open('rb') as stream:
        model = pickle.load(stream)  # Only after its upstream-bound SHA256 passed.
    train = data['role'] == 0
    assert_array(model.source_ids, data['ids'][train], 'A source model training IDs')
    assert_array(model.source_user_ids, sorted(set(range(1, 9))-{fold, fold % 8+1}), 'A source users')
    assert_array(selection['source_users'], model.source_user_ids, 'A selection source users')
    reconcile(selection['source_model_metadata'], model.metadata, 'A source metadata')
    reconcile(selection['fit_diagnostics'], model.diagnostics, 'A fit diagnostics')
    selected = list(map(int, model.select_classes()))
    assert_array(row['support_classes'], selected, 'A support class choice')
    assert_array(selection['classes'], selected, 'A source-only D-optimal classes')
    # Grid choices are independently regenerated from the held validation user.
    validation = {key: value[data['role'] == 1] for key, value in data.items()}
    vp = fixed_partition(validation['ids'], validation['y'], validation['subjects'])
    choices, grids = {}, {}
    for method in (*A.METHODS, 'allclass_lowrank'):
        classes = sorted(np.unique(validation['y']).tolist()) if method == 'allclass_lowrank' else selected
        choices[method], grids[method] = A.tune_method(model, validation, vp, fold % 8+1,
            classes, 'lowrank' if method == 'allclass_lowrank' else method)
    reconcile(selection['validation_grid'], grids, 'A validation grid', atol=1e-12)
    reconcile(selection['hyperparameters'], choices, 'A selected hyperparameters', atol=1e-12)
    reconcile(row['hyperparameters'], choices, 'A result hyperparameters', atol=1e-12)
    best = max(choices[method]['score'] for method in A.METHODS[:-1])
    simple = next(method for method in A.METHODS[:-1] if choices[method]['score'] >= best-1e-12)
    if selection['simple_method'] != simple or row['validation_selected_simple'] != simple:
        fail('A simple method differs from its validation-only choice')
    target = {key: value[data['role'] == 2] for key, value in data.items()}
    tp = fixed_partition(target['ids'], target['y'], target['subjects'])
    arrays = read_npz(directory/'predictions.npz', inventory)
    y, query = target['y'][tp.query_indices], tp.query_indices
    for key, expected in [('query_ids', tp.query_ids), ('query_y', y),
                          ('support_pool_ids', tp.support_ids), ('selected_classes', selected)]:
        assert_array(arrays[key], expected, 'A '+key)
    if row['query_count'] != len(query) or row['support_budget'] != 9:
        fail('A fixed query or support count differs')
    uncalibrated = [c for c in np.unique(y) if c not in selected]
    z = model.transform(target['features'][query])
    centers = {c: z[y == c].mean(0) for c in uncalibrated}
    methods = {}
    for name in A_METHODS:
        predictions = arrays[name+'_predictions']
        if predictions.shape != (20, len(y)) or not np.isin(predictions, range(12)).all():
            fail('A predictions require20 repetitions and fixed query rows')
        if name == 'validation_selected_simple':
            assert_array(predictions, arrays[simple+'_predictions'], 'A simple alias')
            methods[name] = methods[simple]
            continue
        metrics = []
        classes = list(np.unique(y)) if name == 'allclass_lowrank' else selected
        for i, seed in enumerate(A.SEEDS):
            delta = np.zeros_like(model.mu)
            if name in ('frozen', 'source_prototype'):
                scores = target['logits'][query] if name == 'frozen' else prototype_scores(z, model.mu)
            else:
                support = sample_support(tp, fold, classes, seed)
                assert_array(arrays[name+'_support_ids'][i], support.ids, 'A deterministic support sample')
                if np.intersect1d(support.ids, tp.query_ids).size:
                    fail('A support/query leakage')
                config = choices[name]
                delta = model.estimate_delta(target['features'][support.indices], support.labels,
                    'lowrank' if name == 'allclass_lowrank' else name, config['reg'])
                if not np.allclose(arrays[name+'_delta'][i], delta, rtol=0, atol=1e-10):
                    fail('A saved delta differs from support-only estimate')
                scores = correction_scores(target['logits'][query], z, model.mu, delta, config['tau'])
            assert_array(predictions[i], scores.argmax(1), 'A saved prediction replay')
            # The higher-budget reference uses all-class accuracy for its supported group.
            mm = classification_metrics(predictions[i], y, classes)
            mm['uncalibrated'] = A.macro_accuracy(predictions[i], y, uncalibrated)
            unmask = np.isin(y, uncalibrated)
            mm['f1_uncalibrated'] = macro_f1(predictions[i][unmask], y[unmask], uncalibrated)
            mm['uncalibrated_center_mse'] = float(np.mean([
                np.mean((centers[c]-model.mu[c]-delta[c])**2) for c in uncalibrated]))
            reconcile(row['methods'][name]['metrics'][i],
                      {k: mm[k] for k in ('all', 'calibrated', 'uncalibrated', 'uncalibrated_center_mse')},
                      'A raw-array metrics', atol=1e-10)
            metrics.append(mm)
        budget = 0 if name in ('frozen', 'source_prototype') else len(classes)*3
        if name == 'allclass_lowrank' and row['methods'][name]['support_budget'] != budget:
            fail('A high-budget reference count differs')
        methods[name] = {'metrics': metrics, 'support_budget': budget}
        times = row['methods'][name].get('adapt_and_query_ms')
        if times is not None:
            if len(times) != 20 or not np.isfinite(times).all() or min(times) < 0:
                fail('A cached-feature support/query timings malformed')
            methods[name]['adapt_and_query_ms'] = times
    reconcile(row['methods']['validation_selected_simple']['metrics'],
              row['methods'][simple]['metrics'], 'A simple metrics alias')
    return {'fold': fold, 'identity': identity, 'support_classes': selected,
            'query_count': len(y), 'hyperparameters': choices,
            'validation_selected_simple': simple, 'fit_diagnostics': model.diagnostics,
            'methods': methods}, arrays


def compute_budget(data, fold):
    val, test = data['role'] == 1, data['role'] == 2
    selected, val_accuracy = B.tune_cascades(data['logits'][val], data['y'][val])
    selection = {'curves': selected, 'validation_fixed_accuracy': val_accuracy.tolist(),
                 'uses_target_labels': False}
    z, y = data['logits'][test], data['y'][test]
    fixed = B.subset_accuracies(z, y)
    first = selected['2.0']['params']['first']
    curves, arrays = {}, {'ids': data['ids'][test], 'y': y}
    for budget in B.BUDGETS:
        route = B.route_cascade(z, selected[str(budget)]['params'])
        cost = float(route['costs'].mean())
        mix = B.fixed_mixture(val_accuracy, cost)
        curves[str(budget)] = dict(target_val_budget=budget, test_cost=cost,
            test_accuracy=A.macro_accuracy(route['prediction'], y),
            fixed_mixture_within_cost_accuracy=float(mix @ fixed),
            fixed_mixture_within_cost_actual_views=float(mix @ B.COSTS),
            fixed_mixture_cost_ceiling=cost, fixed_mixture_within_cost_weights=mix.tolist(),
            oracle_at_actual_cost=B.oracle_bound(z, y, first, cost))
        arrays[f'predictions_b{budget}'] = route['prediction']
        arrays[f'choices_b{budget}'] = route['subset_indices']
    mixture = B.fixed_mixture(val_accuracy, 2.)
    choices = L.fixed_selections(selection)
    row = {'fold': fold, 'subset_names': list(B.NAMES), 'fixed_accuracy': fixed.tolist(),
           'fixed_validation_selected': choices,
           'fixed_mixture_at_2': {'accuracy': float(mixture @ fixed),
                                  'cost': float(mixture @ B.COSTS), 'weights': mixture.tolist()},
           'oracle_at_2': B.oracle_bound(z, y, first, 2.), 'cascade_curves': curves}
    return row, selection, arrays


def summarize_b(rows):
    require_eight_folds(rows)
    fixed = np.array([row['fixed_accuracy'] for row in rows])
    primary = [row['cascade_curves']['2.0'] for row in rows]
    cascade = np.array([r['test_accuracy'] for r in primary])
    mixture = np.array([r['fixed_mixture_within_cost_accuracy'] for r in primary])
    oracle = np.array([r['oracle_at_2']['macro_accuracy'] for r in rows])
    mixture2 = np.array([r['fixed_mixture_at_2']['accuracy'] for r in rows])
    result = dict(state='complete', subset_names=list(B.NAMES),
        fixed_macro_accuracy=fixed.mean(0).tolist(), fixed_per_user=fixed.tolist(),
        primary_cascade_accuracy=float(cascade.mean()),
        primary_cascade_cost=float(np.mean([r['test_cost'] for r in primary])),
        primary_cascade_per_user_cost=[r['test_cost'] for r in primary],
        cascade_vs_fixed_mixture_within_cost=A.paired_summary(cascade[:, None], mixture[:, None]),
        oracle_minus_cascade_at_same_cost_pp=float(np.mean([
            (r['oracle_at_actual_cost']['macro_accuracy']-r['test_accuracy'])*100 for r in primary])),
        oracle_at_2_accuracy=float(oracle.mean()), fixed_mixture_at_2_accuracy=float(mixture2.mean()),
        oracle_minus_mixture_at_2_pp=float((oracle-mixture2).mean()*100))
    room = result['oracle_minus_mixture_at_2_pp'] >= 2-1e-10 and result['oracle_minus_cascade_at_same_cost_pp'] >= 2-1e-10
    result['opportunity_decision'] = 'POLICY_PILOT_WARRANTED' if room else 'STOP_NO_ORACLE_ROOM'
    return result


def audit_b_fold(root, fold, data, inventory):
    directory = root/'budget_s2026'/f'fold{fold}'
    saved = read_json(directory/'result.json', inventory)
    verify_artifacts(directory, saved['files'], inventory)
    if set(saved['files']) != {'selection.json', 'cascade_predictions.npz'}:
        fail('B budget artifact inventory differs')
    identity = dict(predictions_sha256=inventory.verify(root/'multiview_s2026'/f'fold{fold}'/'predictions.npz'),
        upstream_result_sha256=inventory.verify(root/'multiview_s2026'/f'fold{fold}'/'result.json'),
        manifest_sha256=inventory.verify(root/'multiview_s2026/manifest.json'), smoke=False,
        analysis_sha256=inventory.verify(HERE/'budget_analysis.py'),
        statistics_sha256=inventory.verify(HERE/'run_calibration.py'))
    reconcile(saved['identity'], identity, 'B source identity', exact_keys=True)
    row, selection, arrays = compute_budget(data, fold)
    reconcile(saved, row, 'B raw-array result', atol=1e-12)
    expected_selection = dict(identity=identity, **selection)
    reconcile(read_json(directory/'selection.json', inventory), expected_selection,
              'B validation-only selection', atol=1e-12, exact_keys=True)
    saved_arrays = read_npz(directory/'cascade_predictions.npz', inventory)
    if set(saved_arrays) != set(arrays):
        fail('B cascade array inventory differs')
    for key, expected in arrays.items():
        assert_array(saved_arrays[key], expected, 'B '+key)
    row['identity'] = identity
    return row, selection


def summarize_policy(rows):
    require_eight_folds(rows)
    names = ('extra_trees', 'class_gain', 'frozen_budget2_cascade',
             'fixed_mixture_within_candidate_cost', 'three_views')
    comparators = names[1:4]
    accuracy = {name: np.array([r['metrics'][name]['macro_accuracy'] for r in rows]) for name in names}
    costs = {name: np.array([r['metrics'][name]['mean_views'] for r in rows]) for name in names}
    if any(not np.isfinite(accuracy[n]).all() or not np.isfinite(costs[n]).all()
           or np.any((accuracy[n] < 0) | (accuracy[n] > 1))
           or np.any((costs[n] < 1) | (costs[n] > 3)) for n in names):
        fail('Policy finite accuracy/cost range differs')
    candidate = accuracy['extra_trees']
    comparisons = {name: A.paired_summary(candidate[:, None], accuracy[name][:, None]) for name in comparators}
    gate = {'mean_actual_views_at_most_2': within_budget(float(costs['extra_trees'].mean())),
            'within_1pp_of_three_views': float((candidate-accuracy['three_views']).mean()*100) >= -1-1e-10}
    fairness = {}
    for name, comparison in comparisons.items():
        gate['gain_at_least_2pp_vs_'+name] = comparison['mean_difference_pp'] >= 2-1e-10
        gate['positive_at_least_6users_vs_'+name] = comparison['positive_users'] >= 6
        difference = costs['extra_trees']-costs[name]
        fairness[name] = {'candidate_minus_comparator_mean_views': float(difference.mean()),
            'per_user_cost_difference': difference.tolist(),
            'per_user_cost_no_greater': (difference <= 1e-12).tolist()}
        if name != 'fixed_mixture_within_candidate_cost':
            gate['population_cost_no_greater_than_'+name] = float(difference.mean()) <= 1e-12
    weights = [r['n_test'] for r in rows]
    table = {name: {'macro_accuracy': float(accuracy[name].mean()),
            'per_user_accuracy': accuracy[name].tolist(), 'mean_views': float(costs[name].mean()),
            'per_user_mean_views': costs[name].tolist(),
            'pooled_sample_mean_views_descriptive': float(np.average(costs[name], weights=weights))} for name in names}
    return {'state': 'complete', 'users': 8, 'methods': table, 'comparisons': comparisons,
            'fairness': fairness, 'gate': gate,
            'candidate_minus_three_views_pp': float((candidate-accuracy['three_views']).mean()*100),
            'decision': 'STATISTICAL_GATE_PASSED_RUNTIME_PENDING' if all(gate.values()) else 'STOP_VALUE_POLICY_MECHANISM',
            'can_claim_speedup': False}


def audit_policy_fold(root, fold, data, budget_row, budget_selection, inventory):
    import value_policy as P
    directory = root/'value_policy_s2026'/f'fold{fold}'
    saved = read_json(directory/'result.json', inventory)
    manifest = read_json(root/'value_policy_s2026/manifest.json', inventory)
    source = P.source_record()
    for path, sha in source['files'].items():
        inventory.verify(path, sha)
        inventory.verify(root/'value_policy_s2026/source'/Path(path).name, sha)
    identity = {'fold': fold, 'smoke': False, 'source': source, 'config': P.CONFIG,
        'budget_result_sha256': inventory.verify(root/'budget_s2026'/f'fold{fold}'/'result.json'),
        'budget_selection_sha256': inventory.verify(root/'budget_s2026'/f'fold{fold}'/'selection.json'),
        'budget_identity': budget_row['identity'],
        'opportunity_summary_sha256': inventory.verify(root/'budget_s2026/summary.json')}
    reconcile(saved['identity'], identity, 'Policy identity', exact_keys=True)
    reconcile(manifest, dict(models=str((root/'multiview_s2026').resolve()),
        budget=str((root/'budget_s2026').resolve()), summary_sha256=identity['opportunity_summary_sha256'],
        source=source, config=P.CONFIG, smoke=False), 'Policy manifest', exact_keys=True)
    if set(saved['files']) != set(P.ARTIFACTS):
        fail('Policy artifact inventory differs')
    verify_artifacts(directory, saved['files'], inventory)
    # checked_existing reconciles all saved metrics against independently verified fixed accuracies.
    P.checked_existing(directory, identity, budget_row['fixed_accuracy'])
    selection = read_json(directory/'selection.json', inventory)
    arrays = read_npz(directory/'predictions.npz', inventory)
    val, test = data['role'] == 1, data['role'] == 2
    ids, y, z = (data[key][val] for key in ('ids', 'y', 'logits'))
    split = P.internal_split(ids, y, fold)
    assert_array(selection['validation_fit_ids'], ids[split['fit_indices']], 'Policy validation fit IDs')
    assert_array(selection['validation_tune_ids'], ids[split['tune_indices']], 'Policy price IDs')
    reconcile(selection['validation_array_sha256'],
              {key: P.array_digest(data[key][val]) for key in ('ids', 'y', 'logits')}, 'Policy validation hashes')
    reconcile(selection['frozen_cascade_params'], budget_selection['curves']['2.0']['params'], 'Policy frozen cascade')
    if selection['uses_target_labels'] is not False:
        fail('Policy selection must be validation-only')
    assert_array(arrays['ids'], data['ids'][test], 'Policy exact target ID order')
    assert_array(arrays['y'], data['y'][test], 'Policy exact target labels')
    with (directory/'models.pkl').open('rb') as stream:
        policies = pickle.load(stream)
    if set(policies) != set(P.METHODS):
        fail('Both frozen policy models are required')
    routes = {}
    for name in P.METHODS:
        policy = policies[name]
        if policy.first != selection['first_subset_index'] or policy.method != name:
            fail('Serialized policy identity differs')
        selected, grid = P.select_price(policy, z[split['tune_indices']], y[split['tune_indices']])
        reconcile(selection['selected_prices'][name], selected, 'Policy validation price', atol=1e-12)
        reconcile(selection['validation_price_grid'][name], grid, 'Policy price grid', atol=1e-12)
        route = P.route_policy(data['logits'][test], policy, selected['price'])
        for key, value in route.items():
            actual = arrays[name+'_'+key]
            same = np.array_equal(actual, value, equal_nan=True) if value.dtype.kind == 'f' else np.array_equal(actual, value)
            if not same:
                fail('Policy raw-array causal replay differs: '+name+'_'+key)
        routes[name] = route
    route = B.route_cascade(data['logits'][test], selection['frozen_cascade_params'])
    for key, value in route.items():
        assert_array(arrays['frozen_budget2_cascade_'+key], value, 'Policy frozen cascade replay')
    assert_array(arrays['three_views_prediction'], data['logits'][test, 6].argmax(1), 'Policy three-view replay')
    routes['frozen_budget2_cascade'] = route
    return saved, selection, routes


def validate_latency_mode(summary, has_environment_note):
    env, config = summary.get('environment', {}), summary.get('configuration', {})
    if (summary.get('state') != 'complete' or summary.get('folds') != list(FOLDS)
            or summary.get('classification') not in ('isolated_gpu_latency', 'desktop_context_gpu_latency')
            or config.get('allow_smoke') is not False):
        fail('Formal eight-fold GPU latency required; smoke/partial cannot be reported')
    if (env.get('device') != 'cuda' or env.get('batch') != 1 or env.get('attention') != 'MATH'
            or env.get('autocast') != 'bfloat16' or env.get('tf32') is not False):
        fail('Formal latency requires the frozen CUDA batch-one BF16 MATH configuration')
    if (summary.get('batch_one_replay') != 'bitwise_equal' or config.get('warmup', 0) < 10
            or config.get('repeats', 0) < 1):
        fail('Formal latency replay/warmup/repeat protocol differs')
    if summary['classification'] == 'desktop_context_gpu_latency':
        if not has_environment_note:
            fail('Desktop GPU context requires its frozen environment note')
        if env.get('gpu_exclusive') is not False:
            fail('Desktop GPU latency cannot be marked exclusive')


def audit_latency(root, raw_y, raw_subjects, inventory, include_policy):
    directory = root/'latency_s2026'
    frozen = read_json(directory/'FROZEN_SHA256.json', inventory)
    if frozen.get('state') != 'frozen':
        fail('Latency requires an independently frozen output SHA256 inventory')
    required = {'summary.json', 'preflight.json', 'gpu_idle_checks.json'}
    required |= {f'fold{fold}/{name}.json' for fold in FOLDS for name in ('observations', 'summary')}
    if not required.issubset(frozen.get('files', {})):
        fail('Latency frozen inventory omits required result files')
    verify_artifacts(directory, frozen['files'], inventory)
    saved = read_json(directory/'summary.json', inventory)
    validate_latency_mode(saved, (root/'LATENCY_ENVIRONMENT_NOTE.md').is_file())
    for path, sha in saved['hash_inventory'].items():
        inventory.verify(path, sha)
    note = root/'LATENCY_ENVIRONMENT_NOTE.md'
    if saved['classification'] == 'desktop_context_gpu_latency' and str(note.resolve()) not in saved['hash_inventory']:
        fail('Latency environment note is not bound into measured source identity')
    preflight = read_json(directory/'preflight.json', inventory)
    reconcile(preflight['data'], saved['sample_preflight'], 'Latency preflight data')
    if preflight['state'] != 'verified' or preflight['data']['prepared_replay'] != 'bitwise_equal':
        fail('Latency raw/prepared replay was not verified')
    selected = L.select_clips(raw_y, raw_subjects, FOLDS)
    if sum(len(ids) for ids in selected.values()) != 279:
        fail('Latency must use all279 prespecified clips')
    reconcile(saved['sample_preflight']['selected_ids'],
              {str(k): v.tolist() for k, v in selected.items()}, 'Latency prespecified IDs')
    if saved['sample_preflight']['sampled_clips'] != 279 or saved['sample_preflight']['per_view_checks'] != 837:
        fail('Latency raw preprocessing count differs')
    methods = ['baselineRight', 'fixed1', 'fixed2', 'fixed3', 'frozen_budget2_cascade']
    if include_policy:
        methods += ['class_gain', 'extra_trees']
    repeats = saved['configuration']['repeats']
    rows = []
    for fold in FOLDS:
        fold_rows = read_json(directory/f'fold{fold}'/'observations.json', inventory)
        summary = validate_latency_rows(fold_rows, {fold: selected[fold]}, methods, raw_y, raw_subjects, repeats)
        reconcile(read_json(directory/f'fold{fold}'/'summary.json', inventory), summary, 'Latency fold summary')
        rows.extend(fold_rows)
    computed = validate_latency_rows(rows, selected, methods, raw_y, raw_subjects, repeats)
    for method in methods:
        matching = [r for r in rows if r['method'] == method]
        computed[method]['offline_batch_prediction_differences'] = sum(bool(r['offline_prediction_differs']) for r in matching)
        computed[method]['offline_batch_route_differences'] = sum(bool(r['offline_path_differs']) for r in matching)
    reconcile(saved['methods'], computed, 'Latency global summary', exact_keys=True)
    checks = read_json(directory/'gpu_idle_checks.json', inventory)
    reconcile(saved['gpu_idle_checks'], checks, 'Latency recorded GPU checks', exact_keys=True)
    if len(checks) != 17 or any(c.get('state') != 'verified' for c in checks):
        fail('Latency requires verified preflight and before/after checks for eight folds')
    if saved['classification'] == 'desktop_context_gpu_latency':
        identities = checks[0]['allowed_idle_gpu_identities']
        for check in checks:
            reconcile(check['allowed_idle_gpu_identities'], identities, 'Latency desktop PID identity')
            samples = check.get('gpu_utilization_samples', [])
            if len(samples) != 5 or any(v > 5 for sample in samples[-3:] for v in sample['utilization_percent']):
                fail('Latency desktop idle sampling condition failed')
    return dict(state='complete', methods=computed, classification=saved['classification'],
                environment=saved['environment'], included=saved['included'], excluded=saved['excluded'],
                command_powershell=saved['command_powershell'], sample_count=279,
                observations_per_method=279*repeats, batch_one_replay='bitwise_equal',
                gpu_exclusive=False, frozen_inventory_sha256=inventory.verify(directory/'FROZEN_SHA256.json'))


def aggregate_groups(rows):
    result = {}
    for group in ('gesture', 'idle'):
        present = [row[group] for row in rows if row[group] is not None]
        result[group] = {'n_users': len(present), 'n_samples': sum(row['n'] for row in present),
                        'macro_accuracy': float(np.mean([row['macro_accuracy'] for row in present])),
                        'mean_views': float(np.mean([row['mean_views'] for row in present])),
                        'per_user': rows}
    return result


def group_mixture(predictions, weights, labels):
    result = {}
    for name, mask in [('gesture', labels != 11), ('idle', labels == 11)]:
        result[name] = None if not mask.any() else {
            'n': int(mask.sum()), 'macro_accuracy': float(weights @ np.array([
                A.macro_accuracy(predictions[mask, i], labels[mask]) for i in range(7)])),
            'mean_views': float(weights @ B.COSTS)}
    return result


def training_cost(row, arm):
    history = row['history']
    if [r['epoch'] for r in history] != list(range(len(history))):
        fail('Training history epochs are missing or duplicated')
    seconds = np.array([r['seconds'] for r in history], dtype=float)
    if not np.isfinite(seconds).all() or np.any(seconds < 0):
        fail('Training timing history is malformed')
    best = min(history, key=lambda r: (r['val_ce'], r['epoch']))
    reconcile({'epoch': row['best_epoch'], 'val_ce': row['best_val_ce']},
              {'epoch': best['epoch'], 'val_ce': best['val_ce']}, 'Earliest validation minimum')
    total = float(seconds.sum())
    if arm == 'multiview':
        reconcile(row['new_training_seconds'], total, 'Multiview history seconds', atol=1e-6)
    return dict(epochs=len(history)-1, selected_epoch=row['best_epoch'], history_seconds=total,
                initial_validation_seconds=float(seconds[0]))


def audit(root, formal=True):
    root = Path(root).resolve()
    require_inputs(root, formal)
    inventory = HashInventory()
    for name in DOCS:
        inventory.verify(root/name)
    for name in ('LATENCY_ENVIRONMENT_NOTE.md', 'INTERPRETATION_NOTES.md'):
        if (root/name).exists():
            inventory.verify(root/name)
    for name in ('report.py', 'test_report.py'):
        inventory.verify(HERE/name)
    raw_y, raw_subjects = (np.load(OLD/'prepared'/name, allow_pickle=False) for name in ('y.npy', 'subj.npy'))
    if len(raw_y) != 9593 or raw_y.shape != raw_subjects.shape:
        fail('Frozen source corpus must have9593 aligned rows before quality filtering')
    a_rows, b_rows, policy_rows, training_rows, cross_rows = [], [], [], [], []
    selections, groups, decomposition = [], {}, []
    b_table = {}
    policy_complete = all((root/'value_policy_s2026'/f'fold{f}'/'result.json').is_file() for f in FOLDS)
    def add_b(name, accuracy, cost, group, kind='actual', subset=None):
        entry = b_table.setdefault(name, {'per_user_accuracy': [], 'per_user_mean_views': [],
                                         'kind': kind, 'selected_subsets': []})
        entry['per_user_accuracy'].append(float(accuracy))
        entry['per_user_mean_views'].append(float(cost))
        entry['selected_subsets'].append(subset)
        if group is not None:
            groups.setdefault(name, []).append(group)
    for fold in FOLDS:
        print(f'REPORT_AUDIT fold={fold}/8 source/arrays/selection', flush=True)
        baseline = L.verify_model_run(root/'baseline_s2026', fold, 'warmup', raw_y, raw_subjects, False, inventory)
        multi = L.verify_model_run(root/'multiview_s2026', fold, 'multiview', raw_y, raw_subjects, False, inventory)
        for name, record in [('baseline_s2026', baseline), ('multiview_s2026', multi)]:
            for index, (path, sha) in enumerate(record['identity']['source']['code'].items()):
                inventory.verify(root/name/'source'/f'{index:02d}_{Path(path).name}', sha)
        directory = root/'baseline_s2026'/f'fold{fold}'/'warmup'
        metadata = read_json(directory/'features.json', inventory)
        reconcile(metadata['identity'], {'checkpoint': baseline['files']['right.pt'],
            'extractor': inventory.verify(HERE/'train_baseline.py'), 'model_identity': baseline['identity']},
            'Frozen feature checkpoint/extractor identity', exact_keys=True)
        inventory.verify(directory/'features.npz', metadata['sha256'])
        features = read_npz(directory/'features.npz', inventory)
        validate_archive(features, raw_y, raw_subjects, fold)
        baseline_arrays = read_npz(directory/'predictions.npz', inventory)
        for role, id_key, logits_key, label_key in [(2, 'ids', 'logits', 'y'), (1, 'val_ids', 'val_logits', 'val_y')]:
            mask = features['role'] == role
            assert_array(baseline_arrays[id_key], features['ids'][mask], 'Baseline exported IDs')
            assert_array(baseline_arrays[logits_key], features['logits'][mask], 'Baseline bitwise logit replay')
            assert_array(baseline_arrays[label_key], features['y'][mask], 'Baseline target labels')
        reconcile(baseline['accuracy'], float(np.mean(baseline_arrays['logits'].argmax(1) == baseline_arrays['y'])),
                  'Baseline raw prediction accuracy')
        a_row, a_arrays = audit_a_fold(root, fold, baseline, features, inventory)
        a_rows.append(a_row)
        data = read_npz(root/'multiview_s2026'/f'fold{fold}'/'predictions.npz', inventory)
        validate_archive(data, raw_y, raw_subjects, fold, multiview=True)
        b_row, b_selection = audit_b_fold(root, fold, data, inventory)
        b_rows.append(b_row)
        selections.append({'fold': fold, 'calibration_classes_zero_based': a_row['support_classes'],
            'calibration_simple': a_row['validation_selected_simple'], 'calibration_hyperparameters': a_row['hyperparameters'],
            'fixed_validation_selected': {k: B.NAMES[v] for k, v in b_row['fixed_validation_selected'].items()},
            'cascade': b_selection['curves']['2.0']['params']})
        test = data['role'] == 2
        y, z = data['y'][test], data['logits'][test]
        predictions = z.argmax(2)
        first = b_selection['curves']['2.0']['params']['first']
        primary = b_row['cascade_curves']['2.0']
        for i, name in enumerate(B.NAMES):
            add_b(name, b_row['fixed_accuracy'][i], B.COSTS[i], group_metrics(predictions[:, i], np.full(len(y), B.COSTS[i]), y))
        for cost, i in b_row['fixed_validation_selected'].items():
            add_b('val_fixed'+cost, b_row['fixed_accuracy'][i], int(cost),
                  group_metrics(predictions[:, i], np.full(len(y), int(cost)), y), subset=B.NAMES[i])
        mix = np.array(b_row['fixed_mixture_at_2']['weights'])
        add_b('fixed_mixture_at_2', b_row['fixed_mixture_at_2']['accuracy'], b_row['fixed_mixture_at_2']['cost'],
              group_mixture(predictions, mix, y), kind='exact_mixture_expectation')
        route = B.route_cascade(z, b_selection['curves']['2.0']['params'])
        add_b('cascade', primary['test_accuracy'], primary['test_cost'], group_metrics(route['prediction'], route['costs'], y))
        mix = np.array(primary['fixed_mixture_within_cost_weights'])
        add_b('mixture_at_cascade_cost', primary['fixed_mixture_within_cost_accuracy'],
              primary['fixed_mixture_within_cost_actual_views'], group_mixture(predictions, mix, y), kind='exact_mixture_expectation')
        add_b('oracle_at_2', b_row['oracle_at_2']['macro_accuracy'], b_row['oracle_at_2']['mean_views'],
              None, kind='label_informed_fractional_oracle')
        decomposition.append({'fold': fold, **rescue_decomposition(predictions, y, first),
            'gesture': rescue_decomposition(predictions[y != 11], y[y != 11], first),
            'idle': rescue_decomposition(predictions[y == 11], y[y == 11], first) if np.any(y == 11) else None})
        if policy_complete:
            p_row, p_sel, routes = audit_policy_fold(root, fold, data, b_row, b_selection, inventory)
            policy_rows.append(p_row)
            selections[-1]['policy_prices'] = p_sel['selected_prices']
            for name in ('class_gain', 'extra_trees'):
                route = routes[name]
                add_b(name, p_row['metrics'][name]['macro_accuracy'], p_row['metrics'][name]['mean_views'],
                      group_metrics(route['prediction'], route['costs'], y))
            mix = np.array(p_row['metrics']['fixed_mixture_within_candidate_cost']['weights'])
            mm = p_row['metrics']['fixed_mixture_within_candidate_cost']
            add_b('mixture_at_candidate_cost', mm['macro_accuracy'], mm['mean_views'],
                  group_mixture(predictions, mix, y), kind='exact_mixture_expectation')
        baseline_cost, multi_cost = training_cost(baseline, 'warmup'), training_cost(multi, 'multiview')
        reconcile(multi['reused_right_training_seconds'], baseline_cost['history_seconds'], 'Reused Right training seconds', atol=1e-6)
        training_rows.append({'fold': fold, 'right': baseline_cost, 'multiview_extra': multi_cost})
        matched = align_query(data, a_arrays['query_ids'], a_arrays['query_y'], fold)
        cross = {'fold': fold, 'n_queries': len(a_arrays['query_ids']),
                 'query_ids_sha256': L.array_digest(a_arrays['query_ids']), 'query_y_sha256': L.array_digest(a_arrays['query_y'])}
        for method in ('frozen', 'validation_selected_simple', 'lowrank'):
            cross['A_'+method] = float(np.mean([A.macro_accuracy(p, a_arrays['query_y'])
                                               for p in a_arrays[method+'_predictions']]))
        for method, i in [('R', 2), ('TR', 5), ('LTR', 6), ('validation_selected_2', b_row['fixed_validation_selected']['2'])]:
            cross['B_'+method] = A.macro_accuracy(matched[:, i].argmax(1), a_arrays['query_y'])
        cross_rows.append(cross)
    a_summary, b_summary = summarize_a(a_rows), summarize_b(b_rows)
    old_a = read_json(root/'calibration_s2026/summary.json', inventory)
    a_core = {key: value for key, value in a_summary.items() if key != 'methods'}
    a_core['methods'] = {name: {key: value for key, value in row.items()
                        if not key.startswith('f1_') and key not in ('all_per_user', 'calibrated_per_user')}
                        for name, row in a_summary['methods'].items()}
    reconcile(old_a, a_core, 'A summary')
    reconcile(read_json(root/'budget_s2026/summary.json', inventory), b_summary, 'B summary')
    policy_summary = summarize_policy(policy_rows) if policy_complete else {'state': 'pending'}
    if policy_complete:
        reconcile(read_json(root/'value_policy_s2026/summary.json', inventory), policy_summary, 'Policy summary')
    if formal and b_summary['opportunity_decision'] == 'POLICY_PILOT_WARRANTED' and not policy_complete:
        fail('Conditional policy results incomplete; formal report blocked')
    for entry in b_table.values():
        entry['macro_accuracy'] = float(np.mean(entry['per_user_accuracy']))
        entry['mean_views'] = float(np.mean(entry['per_user_mean_views']))
        entry['within_2_views'] = within_budget(entry['mean_views'])
    if sum(row['n_queries'] for row in cross_rows) != 7440:
        fail('Cross-route comparison requires exactly7440 fixed calibration queries')
    cross_summary = {key: float(np.mean([row[key] for row in cross_rows])) for key in cross_rows[0]
                     if key.startswith(('A_', 'B_'))}
    independent = None
    audit_path = root/'budget_s2026/INDEPENDENT_AUDIT.json'
    if audit_path.exists():
        independent = read_json(audit_path, inventory)
        if independent['status'] != 'PASS' or independent['failures']:
            fail('Independent B audit did not pass')
        for path, record in independent['file_inventory'].items():
            inventory.verify(path, record['sha256'])
        independent = {'status': 'PASS', 'checks': sum(independent['checks'].values()),
                       'scope': independent['audit_scope'], 'sha256': inventory.verify(audit_path)}
    latency = audit_latency(root, raw_y, raw_subjects, inventory, policy_complete) if formal else {'state': 'not_audited_in_core_mode'}
    # Detect concurrent changes, without repeatedly hashing unchanged multi-GB files.
    for path, sha in list(inventory.files.items()):
        inventory.verify(path, sha)
    return {'state': 'complete' if formal else 'core_audit_only_not_a_final_report',
        'schema_version': 1, 'protocol': {'seed': 2026, 'users': list(FOLDS), 'quality_retained': 9300,
            'fixed_cross_route_queries': 7440, 'support_repeats': 20, 'bootstrap_draws': 10000,
            'bootstrap_seed': 20260928, 'statistical_unit': 'user; support repetitions averaged within user',
            'accuracy_unit': 'fraction0..1', 'effects_unit': 'percentage_points', 'cost_unit': 'processed_real_views'},
        'A': a_summary, 'A_folds': a_rows, 'B': b_summary, 'B_table': b_table,
        'B_groups': {name: aggregate_groups(value) for name, value in groups.items()},
        'B_folds': b_rows, 'B_rescue_decomposition': decomposition,
        'policy': policy_summary, 'policy_folds': policy_rows, 'selections': selections,
        'cross_route': {'n_queries': 7440, 'per_user': cross_rows, 'methods': cross_summary},
        'training_cost': {'per_user': training_rows,
            'right_seconds': sum(r['right']['history_seconds'] for r in training_rows),
            'multiview_extra_seconds': sum(r['multiview_extra']['history_seconds'] for r in training_rows),
            'statistical_source_fit_seconds': None, 'policy_fit_seconds': None,
            'unrecorded_note': 'Historical statistical/policy fitting durations were not recorded; audit runtime is not substituted.'},
        'latency': latency, 'independent_B_audit': independent,
        'source_root': str(root), 'report_code': str(Path(__file__).resolve()),
        'hash_inventory': inventory.files}


B_LABELS = {**{name: '固定'+name for name in B.NAMES},
    'val_fixed1': '每折验证选固定1路', 'val_fixed2': '每折验证选固定2路',
    'val_fixed3': '每折验证选固定3路', 'fixed_mixture_at_2': '固定混合（上限2）',
    'cascade': '主置信度/熵级联', 'mixture_at_cascade_cost': '固定混合（级联成本上限）',
    'oracle_at_2': 'oracle（上限2；真实标签）', 'class_gain': '类别收益表',
    'extra_trees': 'ExtraTrees 收益预测', 'mixture_at_candidate_cost': '固定混合（候选成本上限）'}


def percent(value, digits=6):
    return f'{100*value:.{digits}f}%'


def pp(value, digits=6):
    return f'{value:+.{digits}f} pp'


def markdown_table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
                     + ['| '+' | '.join(str(value).replace('|', '/') for value in row)+' |' for row in rows])


def build_report(result):
    if result['state'] != 'complete' or result['latency']['state'] != 'complete':
        fail('Only complete verified results may produce REPORT.md')
    aa, bb, policy = result['A'], result['B'], result['policy']
    at, bt = aa['methods'], result['B_table']
    effect = aa['lowrank_vs_simple']
    sections = ['# 单雷达部分动作校准与多雷达预算识别：可否决先导报告',
        '本报告由逐样本预测、来源身份、冻结选择文件和实测记录重算。所有科学结论以本轮8名研发用户、'
        'seed=2026和既定停止条件为范围；报告生成不重新训练模型，也不修改原始产物。',
        '## 结论与投入判断',
        f"A 的 rank2 用户因素结果为 **{aa['decision']}**。未校准已知类准确率为"
        f"{percent(at['lowrank']['uncalibrated_macro_accuracy'])}，验证选择的简单校准为"
        f"{percent(at['validation_selected_simple']['uncalibrated_macro_accuracy'])}；差值"
        f"{pp(effect['mean_difference_pp'])}，{effect['positive_users']}/8名用户改善，探索性用户重采样95%区间"
        f"[{effect['ci95_pp'][0]:.6f}, {effect['ci95_pp'][1]:.6f}] pp。",
        f"B 的七子集共享模型中，固定LTR为{percent(bt['LTR']['macro_accuracy'])}，固定TR为"
        f"{percent(bt['TR']['macro_accuracy'])}，逐折验证选出的固定两路为{percent(bt['val_fixed2']['macro_accuracy'])}。"
        '固定TR是看完开发结果后的组合描述；验证选两路才是独立于目标标签的预定配置规则。',
        f"主级联为{percent(bt['cascade']['macro_accuracy'])}，实际平均{bt['cascade']['mean_views']:.10f}路。"
        '该成本按未舍入数值判断；显示为约2路不能使它通过严格的2路约束。']
    if policy.get('state') == 'complete':
        candidate = policy['methods']['extra_trees']
        versus = policy['comparisons']['frozen_budget2_cascade']
        sections += [f"ExtraTrees 条件策略结果为 **{policy['decision']}**：准确率"
            f"{percent(candidate['macro_accuracy'])}，实际平均{candidate['mean_views']:.10f}路；相对主级联"
            f"{pp(versus['mean_difference_pp'])}，{versus['positive_users']}/8名用户改善。"
            f"距固定三路{pp(policy['candidate_minus_three_views_pp'])}。"]
    if aa['decision'].startswith('STOP') and policy.get('decision', '').startswith('STOP'):
        sections += ['**停止扩展本轮两个未通过的复杂机制。保留真实多雷达观测作为后续系统的候选工程基础。** '
            '本先导未找到可作为主创新的有效机制；普通固定融合的工程价值、oracle信息空间和事后选组合都不能替代新机制成立的证据。'
            '本轮不在看过目标结果后继续搜索参数以补出正结果。']
    sections += ['## 数据、实验单位与比较边界',
        '数据来自同一 UWB 配对片段库：质量过滤前9593条，保留9300条，输出词表为12个已知类别。'
        '每折留1名用户测试、循环取下一名用户验证、其余6人训练；8折测试用户依次为HV1–HV8。'
        'HV2的零基类别10和11、HV6的类别3因既定质量规则缺失，不补造。G12对应零基标签11，是空类（idle）。',
        '主准确率是先在每名用户实际存在的类别上平均各类召回率，再对8名用户等权平均的 macro accuracy。'
        '处理成本先对每名用户的样本求平均，再对8名用户等权平均。支持抽样20次先在用户内平均；'
        'bootstrap只重采样8名用户，10000次，seed=20260928。百分比写为%，准确率差写为百分点（pp）。'
        '多次支持抽样不是160名独立用户，9300个片段也不是9300个独立统计单位。',
        'A固定查询共7440条，支持候选池与这些查询完全不交；B的主结果使用全部9300条测试记录。'
        '跨路线只在A的7440条查询上另作描述，逐位核对同折ID、标签和用户角色。'
        'A的“未校准已知类”主指标不能直接与B的“全部已知类”主指标相减。',
        '## A：9条片段支持的部分动作校准',
        '输入为独立训练的Right分类器256维冻结激活与12维logits。PCA32、共享类内方差和rank2用户因素仅由6名源用户拟合；'
        '缺失用户—类别单元采用带掩码的低秩填补。源数据的D-optimal准则选3个支持类别，每类3条。'
        '校正保留原判别logits，tau=0精确回到冻结分类器；lambda和tau仅由验证用户选择。',
        markdown_table(['方法', '未校准已知类准确率', '全部查询准确率', '支持类准确率', '每用户支持预算'],
            [[A_LABELS[name], percent(row['uncalibrated_macro_accuracy']), percent(row['all_macro_accuracy']),
              percent(row['calibrated_macro_accuracy']), '/'.join(map(str, sorted(set(row['support_budget_per_user']))))]
             for name, row in at.items()]),
        '“支持类”列对9样本方法指原先选出的3类；全类别高预算参考的支持类覆盖该用户全部可用类。'
        '高预算参考需要30/33/36条标签，并在验证用户上按全部类准确率选参数，不能作为同预算胜利，也不是oracle。'
        '所有方法包括负结果和冻结源原型诊断均保留。',
        '### A 的预注册门槛',
        markdown_table(['条件', '通过'], [[key, '是' if value else '否'] for key, value in aa['gate'].items()]),
        '条件含义依次为：对验证选简单方法平均至少+2 pp、至少6/8用户改善、优于冻结分类器、'
        '相对简单方法无用户平均下降超过2 pp、相对冻结分类器无用户平均下降超过2 pp。任一失败即停止该rank2机制。',
        markdown_table(['用户', '支持类别（零基）', '验证选简单方法', '简单方法准确率', 'rank2准确率', 'rank2−简单（pp）'],
            [[f'HV{i+1}', ','.join(map(str, result['A_folds'][i]['support_classes'])),
              result['A_folds'][i]['validation_selected_simple'],
              percent(at['validation_selected_simple']['uncalibrated_per_user'][i]),
              percent(at['lowrank']['uncalibrated_per_user'][i]), f'{effect["per_user_difference_pp"][i]:+.6f}']
             for i in range(8)]),
        '![A逐用户配对效应](A_user_effects.png)',
        '图A：每点代表1名用户，已先平均20次支持抽样。右图0 pp线为无变化，+2 pp线只表示平均增益门槛的位置；'
        '通过仍需同时满足全部逐用户条件。均值和区间来自8名用户，不把重复抽样当新增独立证据。',
        '### 描述性F1、原始中心误差与支持抽样波动',
        markdown_table(['方法', '全部查询present-class宏F1', '全部查询fixed12宏F1', '未校准查询宏F1',
                        '最差用户未校准准确率', '原始中心MSE', '相对冻结中心MSE减少'],
            [[A_LABELS[name], percent(row['f1_all_present']), percent(row['f1_all_fixed12']),
              percent(row['f1_uncalibrated']), percent(row['worst_user_uncalibrated']),
              f'{row["uncalibrated_center_mse"]:.8f}', f'{row["center_mse_reduction"]:+.8f}']
             for name, row in at.items()]),
        'present-class宏F1只平均该用户查询中真实存在的类；fixed12宏F1始终平均12个输出类，'
        '真值与预测均不存在的类按0计。未校准F1先仅保留未校准类别对应的真值查询，再只平均这些未校准类别。'
        '这些F1均是补充描述，不参与原停止门槛，不用于翻转结论。',
        '中心MSE在目标查询类中心上作事后诊断，衡量tau缩放判别修正之前的原始估计中心；'
        '不参与模型或支持类选择。完整的每用户支持标准差、20次指标、超参数与拟合诊断见JSON和CSV。'
        '数值低秩填补收敛只说明数值迭代稳定，不证明用户因素机制有效。'
        '即使高预算参考改善原始中心MSE，也可能未匹配分类readout边界；这些只是待单独验证的解释，'
        '不能将失败全部归因于用户因素不存在，也不证明所有个性化都无效。',
        '## B：共享模型的真实子集、预算基线与条件策略',
        '七个子集的顺序固定为L/T/R/LT/LR/TR/LTR。共享编码器从本折Right模型初始化，训练时均匀采样非空子集；'
        '各真实视角的256维特征放入固定L/T/R槽位并附可见掩码，经融合头分类。未请求视角在执行路径中不编码。'
        '七子集来自同一个共享网络，不能解释为七个分别优化模型的性能上限。',
        markdown_table(['方法', '全部测试macro accuracy', '平均处理路数', '≤2路', '数值性质'],
            [[B_LABELS[name], percent(row['macro_accuracy']), f'{row["mean_views"]:.10f}',
              '是' if row['within_2_views'] else '否',
              {'actual': '实际输出/实际成本', 'exact_mixture_expectation': '固定随机混合的精确期望',
               'label_informed_fractional_oracle': '真实标签+分数随机化，不可部署'}[row['kind']]]
             for name, row in bt.items()]),
        '固定混合的权重按验证准确率选择，表内准确率和成本为其精确期望，不是一次随机路由抽样结果。'
        '在级联/候选实际成本上限内，混合若选择更便宜且验证更优的固定子集则保留；因此不声称成本必然严格相等。'
        '级联先观察一路，再按当前已见logits的最大概率或归一化熵决定追加；验证预算2不能保证测试实际成本≤2。',
        '![B准确率与实际处理路数](B_accuracy_cost.png)',
        '图B：准确率来自全部9300条正式测试记录，横轴是8用户等权平均的实际处理路数；随机混合和oracle用期望成本。'
        'oracle星形点使用真实标签、不能部署。279条延迟样本没有参与此准确率图或主门槛。',
        '### oracle机会与机制验证分开',
        f"oracle在成本上限2内的准确率为{percent(bb['oracle_at_2_accuracy'])}；比固定混合高"
        f"{bb['oracle_minus_mixture_at_2_pp']:.6f} pp，比主级联在该级联实际成本下高"
        f"{bb['oracle_minus_cascade_at_same_cost_pp']:.6f} pp。两项机会门槛的决定是"
        f"{bb['opportunity_decision']}，仅用于决定是否启动条件先导，不代表策略机制成立。",
        'oracle的范围仅为当前冻结分类器、验证选出的首路、向包含首路的子集单调追加观测的动作族；'
        '保留原先正确的预测，按最低新增成本挽救错误，并可用分数随机化满足平均预算。'
        '它不是所有重新训练模型、丢弃已见视角或任意感知策略的普适信息上界。',
        '### 条件价值预测器的冻结对照与停止条件',
        '策略状态是当前已观测子集的12维概率、归一化熵和top2间隔，共14维；动作允许停止、追加一路或同时追加两路。'
        '收益标签为追加后是否正确减去当前是否正确，取−1/0/+1。验证用户按类别分为60%拟合和40%选price；'
        '类别收益表与固定ExtraTrees配方使用相同内部数据。候选预先固定为ExtraTrees，不能按测试成绩另选赢家。']
    if policy.get('state') == 'complete':
        sections += [markdown_table(['ExtraTrees相对对照', '平均准确率差（pp）', '改善用户', '用户bootstrap95%区间（pp）', '平均路数差'],
            [[name, f'{value["mean_difference_pp"]:+.6f}', f'{value["positive_users"]}/8',
              f'[{value["ci95_pp"][0]:.6f}, {value["ci95_pp"][1]:.6f}]',
              f'{policy["fairness"][name]["candidate_minus_comparator_mean_views"]:+.10f}']
             for name, value in policy['comparisons'].items()]),
            markdown_table(['预定条件', '通过'], [[key, '是' if value else '否'] for key, value in policy['gate'].items()]),
            '候选必须逐一对主级联、类别收益表、候选成本上限内固定混合达到至少+2 pp和至少6/8用户改善；'
            '同时不比两个自适应对照消耗更多平均路数，候选实际平均成本≤2且距三路≤1 pp。'
            '只赢其中一个有利对照不能覆盖其他条件的失败。逐用户成本差在JSON/CSV中保留，平均约束不等于每个用户精确同成本。',
            '当前结果限制了这个14维后验状态、固定收益预测器和当前分类器组合的扩展价值；'
            '它不证明所有主动感知无效，也未验证从其他已观测信号预测新增视角价值是否可行。']
    sections += ['### 动作、空类与收益分解',
        markdown_table(['方法', '动作准确率（8用户）', '动作平均路数', '空类准确率（7用户）', '空类平均路数'],
            [[B_LABELS[name], percent(group['gesture']['macro_accuracy']), f'{group["gesture"]["mean_views"]:.8f}',
              percent(group['idle']['macro_accuracy']), f'{group["idle"]["mean_views"]:.8f}']
             for name, group in result['B_groups'].items()]),
        '动作组排除零基标签11，空类组只含标签11。HV2的空类被质量规则排除，因此空类统计只有7名用户，'
        'HV2对应单元为缺失而非0。分组成本用于检查省处理是否主要来自空类；主门槛仍用原全类别结果。',
        markdown_table(['用户/验证首路', '单路错→全三路对', '单路对→全三路错', '任一追加一路可救', '仅两路联合追加可救'],
            [[f'HV{row["fold"]}/{row["first_subset"]}', row['single_wrong_full_correct'],
              row['single_correct_full_wrong'], row['single_wrong_any_pair_correct'], row['joint_only_rescue']]
             for row in result['B_rescue_decomposition']]),
        '分解以验证选定的首路为起点。仅联合追加可救指：首路错、两个包含首路的两路子集均错，而LTR正确。'
        '这些例数描述信息互补和有害追加，不是可部署策略已能识别这些样本；动作/空类分解明细保存在JSON。',
        '## 跨路线：相同7440条查询上的全部已知类',
        markdown_table(['方法', '同查询macro accuracy'], [[name, percent(value)] for name, value in result['cross_route']['methods'].items()]),
        '上表仅为同折、同query_ids、同query_y的描述性比较，A已先平均支持重复，B取对应外层折模型。'
        '它不能替换A的未校准类别门槛或B的9300条主门槛；B的R来自缺失视角共享模型，'
        'A的Right来自独立训练体系，二者不是同一个模型。',
        '## 训练成本与片段已可用后的实测处理延迟',
        markdown_table(['用户折', 'Right训练秒', 'Right训练epoch/选中epoch', '新增多雷达训练秒', '多雷达epoch/选中epoch'],
            [[f'HV{row["fold"]}', f'{row["right"]["history_seconds"]:.3f}',
              f'{row["right"]["epochs"]}/{row["right"]["selected_epoch"]}',
              f'{row["multiview_extra"]["history_seconds"]:.3f}',
              f'{row["multiview_extra"]["epochs"]}/{row["multiview_extra"]["selected_epoch"]}']
             for row in result['training_cost']['per_user']]),
        f"八折Right基础训练合计{result['training_cost']['right_seconds']:.3f}秒；新增多雷达阶段"
        f"{result['training_cost']['multiview_extra_seconds']:.3f}秒。两者分别计费，不把复用Right重复算成新增融合成本。"
        '这些数值由history各项seconds重算，含计时段内epoch验证；Right初始epoch0记为0，多雷达初始验证时长另有记录。'
        '它们不包含源数据准备、离线导出、队列等待和额外审计，也不是整个项目用时。',
        '源统计拟合、验证网格搜索和策略最初拟合没有完整保存历史秒数，因此标为未记录。'
        'A保存的adapt_and_query_ms仅覆盖缓存特征上的支持参数估计与查询校正，不能代替模型编码或端到端耗时。'
        '报告/独立审计的重算用时不回填为原始训练成本。']
    latency = result['latency']
    env = latency['environment']
    sections += [f"延迟测量使用{env.get('gpu_name', '记录于environment的GPU')}，PyTorch {env.get('torch', '')}、"
        f"CUDA {env.get('cuda_runtime', '')}、CPU线程{env.get('threads', '')}，batch=1、BF16 autocast、MATH attention、TF32关闭。"
        f"每种方法同一组279条预选片段，实际每方法记录{latency['observations_per_method']}次。mean/median/p95单位均为毫秒。",
        markdown_table(['方法', 'mean（ms）', 'median（ms）', 'p95（ms）', '测量子集平均路数', '相对原离线batch预测/路由变更数'],
            [[name, f'{row["mean_ms"]:.6f}', f'{row["median_ms"]:.6f}', f'{row["p95_ms"]:.6f}',
              f'{row["mean_actual_views"]:.8f}',
              f'{row["offline_batch_prediction_differences"]}/{row["offline_batch_route_differences"]}']
             for name, row in latency['methods'].items()]),
        'mean/median/p95从逐条端到端记录重算，p95为NumPy线性分位数；用户等权时延均值和逐用户均值另存JSON。'
        'batch=1直接按需路径与相同batch=1缓存重放逐位一致；与正式离线batch=16/32的数值形状差异允许产生预测/路由变化，'
        '这些变化已记录，未重新调阈值或price。279条只用于延迟，不替换全部9300条性能或任何准确率门槛。',
        '计时从主机内存中的已去静态杂波90×189片段开始，包含请求视角的幅度/解析相位运动生成、逐片段标准化、'
        'float16打包、H2D、实际新增视角编码、已编码特征复用融合、输出回CPU和CPU策略决策。'
        '不包含传感器采集、原始文件读取、此前静态杂波去除、一次性模型加载、预热和审计；预处理与冻结prepared输入逐位一致。'
        '策略CPU开销在总耗时内，阶段计时存在嵌套，不能把各阶段不加区别相加或称为独立GPU kernel耗时。']
    if latency['classification'] == 'desktop_context_gpu_latency':
        sections += ['本次环境分类为desktop_context_gpu_latency：保留显式登记的空闲QQ桌面GPU上下文，'
            '每折前后检查进程身份并采样5次GPU利用率，后3次须≤5%。这是非独占本机处理时延；'
            '采样窗口空闲不能保证整个计时期间毫无桌面噪声，也不能保证专用部署设备的时延。'
            '补充环境记录在有效计时前冻结，未改变样本、策略或准确率门槛。']
    if 'extra_trees' in latency['methods']:
        times = latency['methods']
        sections += [f"本次候选相对固定3路mean耗时差为"
            f"{times['extra_trees']['mean_ms']-times['fixed3']['mean_ms']:+.6f} ms，"
            f"相对主级联为{times['extra_trees']['mean_ms']-times['frozen_budget2_cascade']['mean_ms']:+.6f} ms。"
            '负值只表示本次边界内测得更短；正值表示更慢。不能由处理路数直接推断速度，'
            '时延结果也不能把未通过的准确率/预算门槛翻为GO。']
    sections += ['90帧、20Hz约为4.5秒已采集动作；延迟从完整片段已可用开始，不证明全过程实时、因果前缀识别或自然连续流表现。'
        '固定TR工程上可只安装2部雷达，但本开发集选择仍需新用户确认；动态路由需3部雷达全部安装并同步采集，'
        '省处理不等于省硬件、采集功耗或允许动作结束后补采同一动作。',
        '## 局限、文献依据与项目衔接',
        '这里只有8位研发用户、一个模型种子，没有外部用户、跨天或自然连续流确认。各折训练用户重叠，'
        '用户bootstrap区间仅为探索性重采样描述，不构成独立确认性统计推断。所有类已在源训练词表内，'
        'A研究的是新用户的部分已知动作支持，不是从未见过的新类别学习。',
        '策略验证用户未用于编码器梯度更新，但曾用于选encoder checkpoint，之后又用于策略拟合和选价；这种内部复用限制需保留。'
        '收益模型拟合和价格选择仅使用验证用户，selection先保存再进入该策略目标评价。'
        '但是否启动条件策略的预定oracle机会门槛和独立重算确实使用目标标签；不能声称策略运行前从未读取任何目标标签，'
        '也不能把条件先导视为完全未经探索的独立确认。',
        '文献支持本轮问题与候选机制，不证明本数据集有效性：'
        '[Task-Specific Adapters（CVPR 2022）](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Cross-Domain_Few-Shot_Learning_With_Task-Specific_Adapters_CVPR_2022_paper.html)'
        '提供跨域少样本适配背景，但原任务含未见类别/域；'
        '[Acquisition Conditioned Oracle（ICML 2024）](https://proceedings.mlr.press/v235/valancius24a.html)'
        '提示需要考虑联合信息价值；'
        '[Explainability-driven Ranking（ICML 2025）](https://proceedings.mlr.press/v267/guney25a.html)'
        '说明个例特征获取与策略学习已有研究；'
        '[AFA Evaluation（JMLR 2025）](https://www.jmlr.org/papers/v26/23-1635.html)'
        '提示离线重放评价的假设边界。没有据相邻文献标题声称已完整复现或本轮首创。',
        '旧稿30的“自适应省22.7%”不能继承为本轮精度或效率结论：原24第3.2节X2b指逐样本oracle留出视角测距误差中位数'
        '由86.4 mm降至66.8 mm的相对改善，不是已训练可部署策略的分类准确率提升，也不是处理路数或时延减少22.7%。',
        '与现有三项研究内容衔接时，若今后以新的有效证据将预算识别升为内容一，原内容二应收束为未知动作拒识与少样本注册，'
        '避免重复计算选视角贡献。第一部分已看过全部12类的encoder不能直接用于第二部分“never-seen未知类”验证，'
        '必须按后者词表重新训练和校准。这里仅给结构建议，未修改原Word或旧路线文档；第三点其他数据集工作与本轮分开。',
        '## 复现与审计',
        '生成器逐折核对源代码/输入/权重/预测哈希、9300条合法ID和六/一/一角色、A确定性支持/查询分区、'
        'A全部验证网格与目标预测重放、B七子集和全部9档预算选择、策略price/逐条路由、原result/summary一致性、'
        '跨路线7440条身份和延迟279条样本/真实成本/重放。缺折、smoke、错误ID、被修改的冻结文件或不一致摘要都会阻断正式输出。',
        '本报告的计算属于归档一致性审计，不是重新训练encoder，也不能替代外部数据验证。'
        '全部小数、逐用户指标、选择参数、成本、公平性条件及输入SHA256见computed_results.json；'
        'CSV和复现命令见同目录README.md。原始模型/数据及旧文档保留。']
    if result.get('independent_B_audit'):
        audit = result['independent_B_audit']
        sections += [f"独立B审计状态{audit['status']}，记录{audit['checks']}项检查且无失败。"
            '其范围是预算分析及模型来源；融合头CPU复算中可有BF16舍入差，但不能将这种复算表述为整个encoder重训练。'
            '该审计不覆盖策略或延迟，后两者有各自的验证边界。']
    return '\n\n'.join(sections)+'\n'


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_csvs(result, out):
    a_rows, repeat_rows = [], []
    for row in result['A_folds']:
        for method, values in row['methods'].items():
            metric = values['metrics']
            a_rows.append({'fold': row['fold'], 'method': method,
                'support_budget': values['support_budget'],
                'uncalibrated_accuracy_percent': np.mean([m['uncalibrated'] for m in metric])*100,
                'all_accuracy_percent': np.mean([m['all'] for m in metric])*100,
                'supported_accuracy_percent': np.mean([m['calibrated'] for m in metric])*100,
                'uncalibrated_support_std_pp': np.std([m['uncalibrated'] for m in metric], ddof=1)*100,
                'f1_all_present_percent': np.mean([m['f1_all_present'] for m in metric])*100,
                'f1_all_fixed12_percent': np.mean([m['f1_all_fixed12'] for m in metric])*100,
                'f1_uncalibrated_percent': np.mean([m['f1_uncalibrated'] for m in metric])*100,
                'raw_center_mse': np.mean([m['uncalibrated_center_mse'] for m in metric])})
            for i, mm in enumerate(metric):
                repeat_rows.append({'fold': row['fold'], 'method': method, 'support_seed': A.SEEDS[i], **mm})
    write_csv(out/'A_per_user.csv', a_rows)
    write_csv(out/'A_support_repeats.csv', repeat_rows)
    write_csv(out/'A_paired_effects.csv', [{'fold': i+1, 'lowrank_minus_simple_pp': value}
        for i, value in enumerate(result['A']['lowrank_vs_simple']['per_user_difference_pp'])])
    write_csv(out/'B_per_user.csv', [{'fold': i+1, 'method': method, 'accuracy_percent': accuracy*100,
        'mean_processed_views': row['per_user_mean_views'][i], 'kind': row['kind'],
        'selected_subset': row['selected_subsets'][i]} for method, row in result['B_table'].items()
        for i, accuracy in enumerate(row['per_user_accuracy'])])
    write_csv(out/'B_summary.csv', [{'method': method, 'accuracy_percent': row['macro_accuracy']*100,
        'mean_processed_views': row['mean_views'], 'within_2_views': row['within_2_views'], 'kind': row['kind']}
        for method, row in result['B_table'].items()])
    group_rows = []
    for method, groups in result['B_groups'].items():
        for group, summary in groups.items():
            for i, per_user in enumerate(summary['per_user']):
                value = per_user[group]
                group_rows.append({'method': method, 'group': group, 'fold': i+1,
                    'n': None if value is None else value['n'],
                    'accuracy_percent': None if value is None else value['macro_accuracy']*100,
                    'mean_processed_views': None if value is None else value['mean_views'],
                    'missing_reason': 'HV2 label11 quality-excluded' if value is None else ''})
    write_csv(out/'B_gesture_idle.csv', group_rows)
    write_csv(out/'B_rescue_decomposition.csv', [{k: v for k, v in row.items() if k not in ('gesture', 'idle')}
                                                for row in result['B_rescue_decomposition']])
    cross = []
    for row in result['cross_route']['per_user']:
        cross.append({key+'_percent' if key.startswith(('A_', 'B_')) else key:
                      value*100 if key.startswith(('A_', 'B_')) else value for key, value in row.items()})
    write_csv(out/'cross_route_7440.csv', cross)
    selection_rows = []
    for row in result['selections']:
        for method, params in row['calibration_hyperparameters'].items():
            selection_rows.append(dict(fold=row['fold'], method=method,
                selected_support_classes_zero_based=','.join(map(str, row['calibration_classes_zero_based'])),
                validation_simple=row['calibration_simple'], reg=params['reg'], tau=params['tau'],
                validation_objective_accuracy_percent=params['score']*100))
    write_csv(out/'A_validation_selection.csv', selection_rows)
    if result['policy'].get('state') == 'complete':
        policy_rows = []
        for method, values in result['policy']['fairness'].items():
            comp = result['policy']['comparisons'][method]
            for i in range(8):
                policy_rows.append(dict(fold=i+1, comparator=method,
                    candidate_minus_comparator_pp=comp['per_user_difference_pp'][i],
                    candidate_minus_comparator_views=values['per_user_cost_difference'][i]))
        write_csv(out/'policy_paired_comparisons.csv', policy_rows)
    write_csv(out/'training_cost.csv', [{'fold': row['fold'], 'stage': stage, **values}
        for row in result['training_cost']['per_user'] for stage, values in row.items() if stage != 'fold'])
    write_csv(out/'latency.csv', [{'method': name, 'observations': row['observations'],
        'mean_ms': row['mean_ms'], 'median_ms': row['median_ms'], 'p95_ms': row['p95_ms'],
        'per_user_macro_mean_ms': row['per_user_macro_mean_ms'], 'mean_actual_views': row['mean_actual_views'],
        'offline_batch_prediction_differences': row['offline_batch_prediction_differences'],
        'offline_batch_route_differences': row['offline_batch_route_differences']}
        for name, row in result['latency']['methods'].items()])


def plot_figures(result, out):
    # Contract: descriptive paired effects and actual cost, no post-hoc gate changes.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    font = next((name for name in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC') if name in installed), None)
    if font is None:
        fail('A Chinese plotting font is required; no unreadable fallback figure will be published')
    plt.rcParams.update({'font.family': font, 'svg.fonttype': 'none', 'axes.unicode_minus': False,
                         'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False, 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.3), layout='constrained')
    y = np.arange(8)
    methods = result['A']['methods']
    low = np.array(methods['lowrank']['uncalibrated_per_user'])*100
    simple = np.array(methods['validation_selected_simple']['uncalibrated_per_user'])*100
    for i in range(8):
        axes[0].plot([simple[i], low[i]], [i, i], color='#aaaaaa', lw=1)
    axes[0].scatter(simple, y, color='#3d6587', label='验证选择简单校准', zorder=3)
    axes[0].scatter(low, y, color='#b96363', marker='s', label='rank2用户因素', zorder=3)
    axes[0].set(yticks=y, yticklabels=[f'HV{i+1}' for i in y], xlabel='未校准已知类 macro accuracy（%）', title='A1  每用户准确率')
    axes[0].invert_yaxis()
    axes[0].legend(loc='lower left', fontsize=8)
    effect = result['A']['lowrank_vs_simple']
    delta = np.array(effect['per_user_difference_pp'])
    axes[1].barh(y, delta, color=['#568879' if d >= 0 else '#b96363' for d in delta], height=.55)
    axes[1].axvline(0, color='#555555', lw=.8)
    axes[1].axvline(2, color='#888888', ls='--', lw=.8, label='+2 pp平均门槛位置')
    for i, value in enumerate(delta):
        axes[1].annotate(f'{value:+.2f}', (value, i), xytext=(4 if value >= 0 else -4, 0),
                         textcoords='offset points', va='center', ha='left' if value >= 0 else 'right', fontsize=8)
    axes[1].set(yticks=y, yticklabels=[f'HV{i+1}' for i in y], xlabel='rank2 − 验证简单校准（百分点，pp）',
                title='A2  配对效应：每用户先平均20次支持抽样', xlim=(min(-1, float(delta.min())-3), max(5, float(delta.max())+3)))
    axes[1].invert_yaxis()
    axes[1].legend(loc='lower left', fontsize=8)
    fig.suptitle(f"8名用户；均值 {effect['mean_difference_pp']:+.2f} pp；用户bootstrap95%区间 "
                 f"[{effect['ci95_pp'][0]:.2f}, {effect['ci95_pp'][1]:.2f}] pp", fontsize=10)
    for extension in ('png', 'svg'):
        fig.savefig(out/f'A_user_effects.{extension}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10.2, 6.0), layout='constrained')
    table = result['B_table']
    for name in B.NAMES:
        row = table[name]
        ax.scatter(row['mean_views'], row['macro_accuracy']*100, c='#8097ae', s=42, marker='o')
        ax.annotate(name, (row['mean_views'], row['macro_accuracy']*100), xytext=(-8 if name == 'LTR' else 7, 0),
                    textcoords='offset points', ha='right' if name == 'LTR' else 'left', va='center', fontsize=8)
    colors = {'val_fixed1': '#4f5865', 'val_fixed2': '#4f5865', 'val_fixed3': '#4f5865',
              'fixed_mixture_at_2': '#aa9d7a', 'cascade': '#b36365',
              'class_gain': '#538777', 'extra_trees': '#d19947', 'oracle_at_2': '#86609d'}
    markers = {'cascade': 'X', 'extra_trees': 's', 'oracle_at_2': '*', 'class_gain': 'v'}
    for name, color in colors.items():
        if name not in table:
            continue
        row = table[name]
        label = B_LABELS[name]
        if name == 'oracle_at_2':
            label = 'oracle：使用真实标签，不能部署'
        elif name == 'fixed_mixture_at_2':
            label += '（精确期望）'
        ax.scatter(row['mean_views'], row['macro_accuracy']*100, color=color,
                   marker=markers.get(name, 'D'), s=110 if name == 'oracle_at_2' else 56, label=label, zorder=4)
    ax.axvline(2, color='#777777', ls='--', lw=1, label='实际处理预算上限2路')
    ax.set(xlabel='平均处理的真实视角数（8用户等权；混合/oracle为期望）',
           ylabel='全部测试 macro accuracy（%）', xlim=(.9, 3.15),
           title='B  准确率—实际处理路数：9300条正式测试观测')
    ax.legend(loc='lower center', bbox_to_anchor=(.5, -.47), ncol=3, fontsize=8)
    fig.savefig(out/'B_accuracy_cost.png', dpi=300, bbox_inches='tight')
    fig.savefig(out/'B_accuracy_cost.svg', bbox_inches='tight')
    plt.close(fig)


def write_readme(result, out):
    script = str(Path(__file__).resolve())
    executable = str(Path(sys.executable).resolve())
    quote = lambda value: "'"+str(value).replace("'", "''")+"'"
    command = '& '+quote(executable)+' '+quote(script)+' --root '+quote(result['source_root'])+' --out '+quote(str(out)+'_reproduced')
    lines = ['# 复现与文件说明',
        'REPORT.md为中文独立报告；computed_results.json保留原精度（准确率为0–1，差为pp，成本为真实处理视角数）。'
        'CSV列名显式注明percent、pp、ms或seconds；A_support_repeats.csv保留0–1准确率，不能把支持重复当独立用户。',
        'A_user_effects和B_accuracy_cost各有PNG与可编辑文字SVG；图源来自computed_results.json。'
        'A图是8用户内先平均后的效果，B图用9300条主结果与实际/期望处理成本。',
        '使用原已验证的Windows工作树和输入目录运行；只需CPU进行报告重算，不加载CUDA。'
        '完整源/数据/权重路径与SHA256见JSON的hash_inventory。旧实验数据及模型需仍然存在，不能仅凭报告目录重新训练。',
        '```powershell\n'+command+'\n```',
        '快速只读A/B/策略审计（不会生成正式报告，也不会代替完整延迟验证）：',
        '```powershell\n& '+quote(executable)+' '+quote(script)+' --root '+quote(result['source_root'])+' --audit-core\n```',
        '单元测试：',
        '```powershell\n& '+quote(executable)+' -m unittest discover -s '+quote(HERE)+' -p test_report.py -v\n```',
        '原延迟命令（需要受控测量环境，不能与报告CPU审计并行计时）：',
        '```powershell\n'+result['latency']['command_powershell']+'\n```',
        '正式报告要求8折基础/校准/多雷达/预算结果完整；机会门槛要求条件策略时须有完整8折策略。'
        '还需真实CUDA测量的279预选片段、所有方法、对应FROZEN_SHA256.json。缺失、smoke、ID错位、摘要不一致或冻结哈希变化立即失败，'
        '不会发布REPORT.md。输出目录必须新建或为空；输入目录不能作为输出。',
        'source_snapshot仅归档本次报告与数值审计实现及其测试，帮助审阅；复现命令仍指向原工作树，'
        '原因是实验身份绑定了原源码绝对路径和相邻计划文件。不会把移位后的归档脚本冒充同身份重跑。',
        'ARTIFACT_SHA256.json列出本报告交付文件摘要。图像需做渲染审阅；自动生成不替代研究解释或独立实验。']
    (out/'README.md').write_text('\n\n'.join(lines)+'\n', encoding='utf-8')


def generate(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    require_inputs(root, formal=True)
    inputs = [root/name for name in ('baseline_s2026', 'calibration_s2026', 'multiview_s2026',
                                   'budget_s2026', 'value_policy_s2026', 'latency_s2026')]
    if out == root or any(out == source or source in out.parents or out in source.parents for source in inputs):
        fail('Report output must be independent of all experiment input trees')
    if out.exists() and any(out.iterdir()):
        fail('Use a new empty report output directory')
    result = audit(root, formal=True)
    # Validation completes before any deliverable path is created.
    staging = out.with_name(out.name+'.staging-'+str(os.getpid()))
    if staging.exists():
        fail('Existing staging directory; preserve it and choose a new output name')
    staging.mkdir(parents=True)
    A.dump(staging/'computed_results.json', result)
    export_csvs(result, staging)
    plot_figures(result, staging)
    (staging/'REPORT.md').write_text(build_report(result), encoding='utf-8')
    write_readme(result, staging)
    source_out = staging/'source_snapshot'
    source_out.mkdir()
    for name in ('report.py', 'test_report.py', 'calibration.py', 'run_calibration.py',
                 'budget_analysis.py', 'value_policy.py', 'benchmark_latency.py'):
        shutil.copy2(HERE/name, source_out/name)
    # Reproduction command must name the intended final directory, not staging.
    text = (staging/'README.md').read_text(encoding='utf-8').replace(str(staging)+'_reproduced', str(out)+'_reproduced')
    (staging/'README.md').write_text(text, encoding='utf-8')
    files = {str(path.relative_to(staging)).replace('\\', '/'): A.digest(path)
             for path in sorted(staging.rglob('*')) if path.is_file()}
    A.dump(staging/'ARTIFACT_SHA256.json', {'state': 'complete', 'files': files})
    if out.exists():
        out.rmdir()  # Only the previously checked empty report directory.
    staging.rename(out)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, default=ROOT/'report_s2026')
    parser.add_argument('--audit-core', action='store_true', help='CPU-only read-only audit; no formal artifacts or latency validation')
    args = parser.parse_args(argv)
    result = audit(args.root, formal=False) if args.audit_core else generate(args.root, args.out)
    brief = {'state': result['state'], 'A_decision': result['A']['decision'],
             'A_lowrank_vs_simple': result['A']['lowrank_vs_simple'],
             'B_cascade_actual_views': result['B']['primary_cascade_cost'],
             'B_opportunity_decision': result['B']['opportunity_decision'],
             'policy_decision': result['policy'].get('decision'), 'hashes_verified': len(result['hash_inventory'])}
    print(json.dumps(brief, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    main()
