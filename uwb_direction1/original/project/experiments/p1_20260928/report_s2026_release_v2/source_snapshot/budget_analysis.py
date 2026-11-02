"""Seven real-view subsets, deployable simple cascades and labeled oracle bounds.

Offline arrays contain all views for evaluation; every cascade decision reads
only the logits of the subset already requested. Oracle outputs are diagnostic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from run_calibration import ROOT, digest, dump, macro_accuracy, paired_summary

SUBSETS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
NAMES = ('L', 'T', 'R', 'LT', 'LR', 'TR', 'LTR')
COSTS = np.array([len(s) for s in SUBSETS])
BUDGETS = (1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3.)


def class_weights(labels):
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    return np.array([1/(len(classes)*counts[np.where(classes == c)[0][0]]) for c in labels])


def uncertainty(logits, mode):
    values = np.asarray(logits, dtype=float)
    ex = np.exp(values-values.max(1, keepdims=True))
    p = ex/ex.sum(1, keepdims=True)
    if mode == 'confidence':
        return 1-p.max(1)
    if mode == 'entropy':
        return -(p*np.log(np.maximum(p, 1e-30))).sum(1)/np.log(p.shape[1])
    raise ValueError('Unknown uncertainty mode')


def fixed_mixture(validation_accuracy, budget):
    """Best validation-chosen mixture within a cost ceiling; exact expectation."""
    acc = np.asarray(validation_accuracy, dtype=float)
    if acc.shape != (7,) or not np.isfinite(acc).all() or not 1 <= budget <= 3:
        raise ValueError('Seven finite validation accuracies and budget1..3 required')
    candidates = []
    for i in range(7):
        if COSTS[i] <= budget+1e-12:
            w = np.eye(7)[i]
            candidates.append(w)
        for j in range(i+1, 7):
            if COSTS[i] < budget < COSTS[j]:
                w = np.zeros(7)
                w[j] = (budget-COSTS[i])/(COSTS[j]-COSTS[i])
                w[i] = 1-w[j]
                candidates.append(w)
    best = max(float(w @ acc) for w in candidates)
    tied = [w for w in candidates if float(w @ acc) >= best-1e-12]
    return min(tied, key=lambda w: (float(w @ COSTS), tuple(-w)))


def route_cascade(logits, params):
    z = np.asarray(logits)
    first, pair = params['first'], params['pair']
    if z.ndim != 3 or z.shape[1] != 7 or first not in (0, 1, 2):
        raise ValueError('Require [N,7,classes] and one-view starting subset')
    if len(SUBSETS[pair]) != 2 or SUBSETS[first][0] not in SUBSETS[pair]:
        raise ValueError('Pair must contain the first observed view')
    choice = np.full(len(z), first, dtype=int)
    second_rows = np.flatnonzero(uncertainty(z[:, first], params['mode']) > params['threshold1'])
    choice[second_rows] = pair
    if len(second_rows):
        third = uncertainty(z[second_rows, pair], params['mode']) > params['threshold2']
        choice[second_rows[third]] = 6
    return {'prediction': z[np.arange(len(z)), choice].argmax(1),
            'subset_indices': choice, 'costs': COSTS[choice]}


def oracle_bound(logits, labels, first, budget):
    """Fractional-knapsack upper bound using test labels; never deployable.

    Keep already-correct one-view predictions. Rescue errors with the cheapest
    correct superset, only if budget permits. Accounts for harmful additions and
    cases where both additional views are necessary. Supports unequal classes.
    """
    z, y = np.asarray(logits), np.asarray(labels)
    if z.shape[:2] != (len(y), 7) or first not in (0, 1, 2) or not 1 <= budget <= 3:
        raise ValueError('Invalid oracle arrays, initial view or cost budget')
    correct = z.argmax(2) == y[:, None]
    weights = class_weights(y)
    initial = correct[:, first]
    allowed = [i for i, subset in enumerate(SUBSETS) if SUBSETS[first][0] in subset]
    rescues = []
    for row in np.flatnonzero(~initial):
        extra = [COSTS[i]-1 for i in allowed if correct[row, i]]
        if extra:
            cost = min(extra)
            rescues.append((weights[row]/cost, row, cost))
    remaining = len(y)*(budget-1)
    accuracy = float(weights @ initial)
    spent = 0.
    for _, row, cost in sorted(rescues, reverse=True):
        fraction = min(1., remaining/cost)
        accuracy += weights[row]*fraction
        remaining -= cost*fraction
        spent += cost*fraction
        if remaining <= 1e-12:
            break
    return {'macro_accuracy': accuracy, 'mean_views': 1+spent/len(y),
            'uses_test_labels': True, 'rescuable_rows': len(rescues)}


def subset_accuracies(logits, labels):
    return np.array([macro_accuracy(logits[:, i].argmax(1), labels) for i in range(7)])


def tune_cascades(val_logits, val_y):
    acc = subset_accuracies(val_logits, val_y)
    first = next(i for i in range(3) if acc[i] >= max(acc[:3])-1e-12)
    pairs = [i for i in range(3, 6) if first in SUBSETS[i]]
    pair = min(pairs, key=lambda i: (-round(float(acc[i]), 12), i))
    weights = class_weights(val_y)
    candidates = []
    for mode in ('confidence', 'entropy'):
        a = uncertainty(val_logits[:, first], mode)
        b = uncertainty(val_logits[:, pair], mode)
        thresholds1 = np.unique(np.r_[-1., np.quantile(a, np.linspace(0, 1, 21)), 2.])
        thresholds2 = np.unique(np.r_[-1., np.quantile(b, np.linspace(0, 1, 21)), 2.])
        for t1 in thresholds1:
            for t2 in thresholds2:
                params = {'first': first, 'pair': pair, 'mode': mode,
                          'threshold1': float(t1), 'threshold2': float(t2)}
                route = route_cascade(val_logits, params)
                candidates.append({'params': params, 'val_accuracy': float(weights @ (route['prediction'] == val_y)),
                                   'val_cost': float(route['costs'].mean())})
    selected = {}
    for budget in BUDGETS:
        feasible = [r for r in candidates if r['val_cost'] <= budget+1e-12]
        best = max(r['val_accuracy'] for r in feasible)
        tied = [r for r in feasible if r['val_accuracy'] >= best-1e-12]
        selected[str(budget)] = min(tied, key=lambda r: (r['val_cost'],
                                  r['params']['mode'], -r['params']['threshold1'], -r['params']['threshold2']))
    return selected, acc


def analyze_fold(root, out, fold, allow_smoke=False):
    path = root/f'fold{fold}'/'predictions.npz'
    upstream_path = path.parent/'result.json'
    upstream = json.loads(upstream_path.read_text(encoding='utf-8'))
    uid = upstream['identity']
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if (uid['arm'] != 'multiview' or uid['fold'] != fold
            or manifest != {'source': uid['source'], 'config': uid['config']}
            or uid['config']['subsets'] != [list(v) for v in SUBSETS]
            or (uid['config']['smoke'] and not allow_smoke)):
        raise ValueError('Input is not the requested verified multiview protocol')
    smoke = uid['config']['smoke']
    required = {'seed': 2026, 'fusion_seed': 2026, 'batch': 128, 'microbatch': 16,
                'lr': 1e-4, 'weight_decay': 1e-4, 'eval_batch': 16, 'patience': 8,
                'epochs': 1 if smoke else 40, 'min_epochs': 1 if smoke else 12,
                'smoke_steps': 3 if smoke else 0, 'criterion': 'cross_entropy only', 'attention': 'MATH'}
    if any(uid['config'].get(k) != v for k, v in required.items()):
        raise ValueError('Upstream optimizer/selection configuration differs from frozen protocol')
    initial = uid['initial_checkpoint']
    if (initial['model_identity']['fold'] != fold or initial['model_identity']['arm'] != 'warmup'
            or digest(initial['right_path']) != initial['right_sha256']):
        raise ValueError('Initial classifier checkpoint lineage mismatch')
    for name, sha in upstream['files'].items():
        if digest(path.parent/name) != sha:
            raise ValueError('Upstream multiview artifact changed: '+name)
    if upstream['files']['predictions.npz'] != digest(path):
        raise ValueError('Upstream subset prediction digest mismatch')
    archive = np.load(path)
    data = {k: archive[k] for k in archive.files}
    if (tuple(data['logits'].shape) != (9300, 7, 12) or len(data['ids']) != 9300
            or len(np.unique(data['ids'])) != 9300 or not np.isfinite(data['logits']).all()):
        raise ValueError('Unexpected full-view archive')
    prepared = ROOT.parent/'p1_20260905'/'prepared'
    raw_y, raw_subjects = np.load(prepared/'y.npy'), np.load(prepared/'subj.npy')
    keep = ~(((raw_subjects == 2) & np.isin(raw_y, [10, 11])) | ((raw_subjects == 6) & (raw_y == 3)))
    np.testing.assert_array_equal(np.sort(data['ids']), np.where(keep)[0])
    np.testing.assert_array_equal(data['y'], raw_y[data['ids']])
    np.testing.assert_array_equal(data['subjects'], raw_subjects[data['ids']])
    expected_parts = {'train': np.where(keep & ~np.isin(raw_subjects, [fold, fold % 8 + 1]))[0],
                      'val': np.where(keep & (raw_subjects == fold % 8 + 1))[0],
                      'test': np.where(keep & (raw_subjects == fold))[0]}
    hashes = {k: hashlib.sha256(str(v.dtype).encode()+str(v.shape).encode()+v.tobytes()).hexdigest()
              for k, v in expected_parts.items()}
    if uid['partitions'] != hashes or initial['model_identity']['partitions'] != hashes:
        raise ValueError('Initial and multiview checkpoints do not share the exact frozen partitions')
    expected = np.array([[v in s for v in range(3)] for s in SUBSETS])
    np.testing.assert_array_equal(data['subset_masks'].astype(bool), expected)
    val = data['role'] == 1
    test = data['role'] == 2
    if (set(np.unique(data['role'])) != {0, 1, 2}
            or set(data['subjects'][val]) != {fold % 8 + 1} or set(data['subjects'][test]) != {fold}
            or set(data['subjects'][data['role'] == 0]) != set(range(1, 9))-{fold, fold % 8 + 1}):
        raise ValueError('Subset archive role/subject protocol mismatch')
    selected, val_accuracy = tune_cascades(data['logits'][val], data['y'][val])
    directory = out/f'fold{fold}'
    identity = {'predictions_sha256': digest(path), 'upstream_result_sha256': digest(upstream_path),
                'manifest_sha256': digest(root/'manifest.json'), 'smoke': uid['config']['smoke'],
                'analysis_sha256': digest(Path(__file__)),
                'statistics_sha256': digest(Path(__file__).with_name('run_calibration.py'))}
    if (directory/'result.json').exists():
        saved = json.loads((directory/'result.json').read_text(encoding='utf-8'))
        if saved['identity'] != identity:
            raise ValueError('Budget analysis source/input identity changed')
        for name, sha in saved['files'].items():
            if digest(directory/name) != sha:
                raise ValueError('Budget analysis artifact changed')
        return saved
    dump(directory/'selection.json', {'identity': identity, 'curves': selected,
          'validation_fixed_accuracy': val_accuracy.tolist(), 'uses_target_labels': False})
    logits, labels = data['logits'][test], data['y'][test]
    test_accuracy = subset_accuracies(logits, labels)
    first = selected['2.0']['params']['first']
    curves, arrays = {}, {'ids': data['ids'][test], 'y': labels}
    for budget in BUDGETS:
        route = route_cascade(logits, selected[str(budget)]['params'])
        actual_cost = float(route['costs'].mean())
        # Cost matching observes only the number of processed views, not labels.
        mixture = fixed_mixture(val_accuracy, actual_cost)
        curves[str(budget)] = {'target_val_budget': budget, 'test_cost': actual_cost,
            'test_accuracy': macro_accuracy(route['prediction'], labels),
            'fixed_mixture_within_cost_accuracy': float(mixture @ test_accuracy),
            'fixed_mixture_within_cost_actual_views': float(mixture @ COSTS),
            'fixed_mixture_cost_ceiling': actual_cost,
            'fixed_mixture_within_cost_weights': mixture.tolist(),
            'oracle_at_actual_cost': oracle_bound(logits, labels, first, actual_cost)}
        arrays[f'predictions_b{budget}'] = route['prediction']
        arrays[f'choices_b{budget}'] = route['subset_indices']
    primary_mixture = fixed_mixture(val_accuracy, 2.)
    fixed_selected = {str(cost): int(max(np.where(COSTS == cost)[0], key=lambda i: (round(float(val_accuracy[i]), 12), -i)))
                      for cost in (1, 2, 3)}
    result = {'identity': identity, 'fold': fold, 'subset_names': NAMES,
              'fixed_accuracy': test_accuracy.tolist(), 'fixed_validation_selected': fixed_selected,
              'fixed_mixture_at_2': {'accuracy': float(primary_mixture @ test_accuracy),
                                    'cost': float(primary_mixture @ COSTS), 'weights': primary_mixture.tolist()},
              'oracle_at_2': oracle_bound(logits, labels, first, 2.), 'cascade_curves': curves}
    np.savez_compressed(directory/'cascade_predictions.npz', **arrays)
    result['files'] = {n: digest(directory/n) for n in ('selection.json', 'cascade_predictions.npz')}
    dump(directory/'result.json', result)
    return result


def summarize(rows, out):
    if sorted(r['fold'] for r in rows) != list(range(1, 9)):
        raise ValueError('Eight completed outer users required')
    if any(r['identity']['smoke'] for r in rows):
        raise ValueError('Smoke results cannot be summarized as scientific outcomes')
    fixed = np.array([r['fixed_accuracy'] for r in rows])
    primary = [r['cascade_curves']['2.0'] for r in rows]
    cascade = np.array([r['test_accuracy'] for r in primary])
    mixture = np.array([r['fixed_mixture_within_cost_accuracy'] for r in primary])
    oracle = np.array([r['oracle_at_2']['macro_accuracy'] for r in rows])
    mixture2 = np.array([r['fixed_mixture_at_2']['accuracy'] for r in rows])
    result = {'state': 'complete', 'subset_names': NAMES, 'fixed_macro_accuracy': fixed.mean(0).tolist(),
              'fixed_per_user': fixed.tolist(),
              'primary_cascade_accuracy': float(cascade.mean()),
              'primary_cascade_cost': float(np.mean([r['test_cost'] for r in primary])),
              'primary_cascade_per_user_cost': [r['test_cost'] for r in primary],
              'cascade_vs_fixed_mixture_within_cost': paired_summary(cascade[:, None], mixture[:, None]),
              'oracle_minus_cascade_at_same_cost_pp': float(np.mean([
                  (r['oracle_at_actual_cost']['macro_accuracy']-r['test_accuracy'])*100 for r in primary])),
              'oracle_at_2_accuracy': float(oracle.mean()),
              'fixed_mixture_at_2_accuracy': float(mixture2.mean()),
              'oracle_minus_mixture_at_2_pp': float((oracle-mixture2).mean()*100),
              'note': 'Oracle uses labels; conditional opportunity only. Validation cost limits may shift on target users.'}
    room = (result['oracle_minus_mixture_at_2_pp'] >= 2-1e-10
            and result['oracle_minus_cascade_at_same_cost_pp'] >= 2-1e-10)
    result['opportunity_decision'] = 'POLICY_PILOT_WARRANTED' if room else 'STOP_NO_ORACLE_ROOM'
    dump(out/'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, default=ROOT/'multiview_s2026')
    parser.add_argument('--out', type=Path, default=ROOT/'budget_s2026')
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(1, 9)))
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    if args.allow_smoke and args.out.resolve() == (ROOT/'budget_s2026').resolve():
        parser.error('Smoke analysis requires an independent output directory')
    rows = [analyze_fold(args.models, args.out, fold, args.allow_smoke) for fold in args.folds]
    if args.folds == list(range(1, 9)):
        print(json.dumps(summarize(rows, args.out), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
