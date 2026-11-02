"""Nested validation, fixed support budgets and user-level paired inference."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('D:/uwb_gestures/experiments/p1_20260928')
METHODS = ('supported_only', 'translation', 'ridge_affine', 'lowrank')
REGS = (0.1, 1., 10., 100.)
TAUS = (0., .125, .25, .5, 1., 2.)
SEEDS = tuple(range(92000, 92020))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def macro_accuracy(prediction, labels, classes=None):
    pred, y = np.asarray(prediction), np.asarray(labels)
    if pred.shape != y.shape or y.ndim != 1:
        raise ValueError('Aligned one-dimensional predictions and labels required')
    classes = np.unique(y) if classes is None else np.asarray(classes)
    observed = [int(c) for c in classes if np.any(y == c)]
    if not observed:
        raise ValueError('No evaluated classes')
    return float(np.mean([np.mean(pred[y == c] == c) for c in observed]))


def select_candidate(rows):
    best = max(row['score'] for row in rows)
    tied = [row for row in rows if row['score'] >= best-1e-12]
    return min(tied, key=lambda x: (x['tau'], -x['reg']))


def paired_summary(candidate, baseline):
    a, b = np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a-b).all():
        raise ValueError('Require aligned finite [users, support repeats] arrays')
    differences = (a-b).mean(axis=1)*100
    rng = np.random.default_rng(20260928)
    draws = differences[rng.integers(0, len(differences), size=(10000, len(differences)))].mean(1)
    return {'n_users': len(differences), 'mean_difference_pp': float(differences.mean()),
            'per_user_difference_pp': differences.tolist(),
            'positive_users': int(np.sum(differences > 1e-10)),
            'worst_user_difference_pp': float(differences.min()),
            'ci95_pp': np.quantile(draws, [.025, .975]).tolist()}


def decision_gate(comparison, vs_frozen):
    tolerance_pp = 1e-10
    return {'gain_at_least_2pp': comparison['mean_difference_pp'] >= 2-tolerance_pp,
            'positive_at_least_6users': comparison['positive_users'] >= 6,
            'beats_frozen': vs_frozen['mean_difference_pp'] > tolerance_pp,
            'no_user_worse_than_minus2pp_vs_simple': comparison['worst_user_difference_pp'] >= -2-tolerance_pp,
            'no_user_worse_than_minus2pp_vs_frozen': vs_frozen['worst_user_difference_pp'] >= -2-tolerance_pp}


def tune_method(model, data, partition, user, selected, method):
    """Only one validation user's support and query arrays enter this function."""
    from calibration import correction_scores, sample_support
    query = partition.query_indices
    z = model.transform(data['features'][query])
    logits, labels = data['logits'][query], data['y'][query]
    eval_classes = [c for c in np.unique(labels) if c not in selected]
    if not eval_classes:  # Higher-budget all-class reference uses overall accuracy.
        eval_classes = np.unique(labels).tolist()
    rows = []
    for reg in REGS:
        values = {tau: [] for tau in TAUS}
        for seed in SEEDS:
            support = sample_support(partition, user, selected, seed)
            delta = model.estimate_delta(data['features'][support.indices], support.labels, method, reg)
            for tau in TAUS:
                pred = correction_scores(logits, z, model.mu, delta, tau).argmax(1)
                values[tau].append(macro_accuracy(pred, labels, eval_classes))
        for tau in TAUS:
            rows.append({'reg': reg, 'tau': tau, 'score': float(np.mean(values[tau]))})
    return select_candidate(rows), rows


def evaluate_method(model, data, partition, user, selected, method, config):
    from calibration import correction_scores, sample_support
    query = partition.query_indices
    features, labels, logits = data['features'], data['y'][query], data['logits'][query]
    z = model.transform(features[query])
    uncalibrated = [c for c in np.unique(labels) if c not in selected]
    predictions, support_ids, shifts, metrics, latencies = [], [], [], [], []
    for seed in SEEDS:
        support = sample_support(partition, user, selected, seed)
        if np.intersect1d(support.ids, partition.query_ids).size:
            raise ValueError('Support/query leakage')
        started = time.perf_counter()
        delta = model.estimate_delta(features[support.indices], support.labels, method, config['reg'])
        scores = correction_scores(logits, z, model.mu, delta, config['tau'])
        latencies.append((time.perf_counter()-started)*1000)
        pred = scores.argmax(1)
        predictions.append(pred)
        support_ids.append(support.ids)
        shifts.append(delta)
        centers_mse = [float(np.mean((z[labels == c].mean(0)-model.mu[c]-delta[c])**2))
                       for c in uncalibrated]
        metrics.append({'all': macro_accuracy(pred, labels),
                        'calibrated': macro_accuracy(pred, labels, selected),
                        'uncalibrated': macro_accuracy(pred, labels, uncalibrated) if uncalibrated else None,
                        'uncalibrated_center_mse': float(np.mean(centers_mse)) if centers_mse else None})
    return {'metrics': metrics, 'adapt_and_query_ms': latencies}, {
        'predictions': np.asarray(predictions), 'support_ids': np.asarray(support_ids),
        'delta': np.asarray(shifts)}


def run_fold(baseline, out, fold):
    from calibration import fit_source, fixed_partition, prototype_scores
    path = baseline/f'fold{fold}'/'warmup'/'features.npz'
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if digest(path) != meta['sha256']:
        raise ValueError('Input feature digest mismatch')
    upstream_path = path.parent/'result.json'
    upstream = json.loads(upstream_path.read_text(encoding='utf-8'))
    if (meta['identity']['model_identity'] != upstream['identity']
            or upstream['identity']['fold'] != fold or upstream['identity']['arm'] != 'warmup'
            or meta['identity']['checkpoint'] != upstream['files']['right.pt']
            or digest(path.parent/'right.pt') != upstream['files']['right.pt']):
        raise ValueError('Input features are not bound to this fold and classifier checkpoint')
    identity = {'features_sha256': meta['sha256'],
                'feature_metadata_sha256': digest(path.with_suffix('.json')),
                'upstream_result_sha256': digest(upstream_path),
                'source': {p.name: digest(p) for p in (Path(__file__), HERE/'calibration.py')},
                'reg': REGS, 'tau': TAUS, 'seeds': SEEDS, 'fold': fold}
    # JSON-normalize tuples before comparing a resumed completed artifact.
    identity = json.loads(json.dumps(identity))
    directory = out/f'fold{fold}'
    if (directory/'result.json').exists():
        saved = json.loads((directory/'result.json').read_text(encoding='utf-8'))
        if saved['identity'] != identity:
            raise ValueError('Calibration result identity mismatch')
        for name, sha in saved['files'].items():
            if digest(directory/name) != sha:
                raise ValueError('Saved calibration artifact changed')
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    archive = np.load(path)
    data = {k: archive[k] for k in archive.files}
    expected_users = set(range(1, 9))-{fold, fold % 8 + 1}
    if (len(data['ids']) != 9300 or len(np.unique(data['ids'])) != 9300
            or set(np.unique(data['role'])) != {0, 1, 2}
            or set(data['subjects'][data['role'] == 0]) != expected_users
            or set(data['subjects'][data['role'] == 1]) != {fold % 8 + 1}
            or set(data['subjects'][data['role'] == 2]) != {fold}
            or np.any((data['subjects'] == 2) & np.isin(data['y'], [10, 11]))
            or np.any((data['subjects'] == 6) & (data['y'] == 3))):
        raise ValueError('Feature archive violates exact six/one/one subject or quality protocol')
    tr = data['role'] == 0
    model = fit_source(data['features'][tr], data['y'][tr], data['subjects'][tr],
                       n_components=32, rank=2, source_ids=data['ids'][tr])
    selected = list(map(int, model.select_classes()))
    users = {'val': fold % 8 + 1, 'test': fold}
    val = {k: v[data['role'] == 1] for k, v in data.items()}
    if set(val['subjects']) != {users['val']}:
        raise ValueError('Validation role/whole-user mismatch')
    vp = fixed_partition(val['ids'], val['y'], val['subjects'])
    chosen, grids = {}, {}
    for method in METHODS:
        chosen[method], grids[method] = tune_method(model, val, vp, users['val'], selected, method)
    best_simple_score = max(chosen[method]['score'] for method in METHODS[:-1])
    simple = next(method for method in METHODS[:-1] if chosen[method]['score'] >= best_simple_score-1e-12)
    # Higher budget: only classes actually present for that user.
    chosen['allclass_lowrank'], grids['allclass_lowrank'] = tune_method(
        model, val, vp, users['val'], sorted(np.unique(val['y']).tolist()), 'lowrank')
    # Save model/validation choices before target support or queries are used.
    dump(directory/'selection.json', {'identity': identity, 'classes': selected,
         'hyperparameters': chosen, 'validation_grid': grids, 'simple_method': simple,
         'source_users': np.unique(data['subjects'][tr]).tolist(),
         'source_model_metadata': model.metadata, 'fit_diagnostics': model.diagnostics,
         'validation_objective': 'macro accuracy on classes excluded from the nine-label calibration',
         'allclass_reference_objective': 'all available classes; higher budget, excluded from GO comparator'})
    with (directory/'source_model.pkl').open('wb') as stream:
        pickle.dump(model, stream, protocol=4)
    test = {k: v[data['role'] == 2] for k, v in data.items()}
    if set(test['subjects']) != {users['test']}:
        raise ValueError('Test role/whole-user mismatch')
    tp = fixed_partition(test['ids'], test['y'], test['subjects'])
    query = tp.query_indices
    query_labels = test['y'][query]
    unseen_classes = [c for c in np.unique(query_labels) if c not in selected]
    saved_arrays = {'query_ids': tp.query_ids, 'query_y': query_labels,
                    'support_pool_ids': tp.support_ids, 'selected_classes': selected}
    result_methods = {}
    query_z = model.transform(test['features'][query])
    unadapted_center_mse = float(np.mean([np.mean((query_z[query_labels == c].mean(0)-model.mu[c])**2)
                                         for c in unseen_classes]))
    for name in ('frozen', 'source_prototype'):
        scores = test['logits'][query] if name == 'frozen' else prototype_scores(model.transform(test['features'][query]), model.mu)
        pred = scores.argmax(1)
        row = {'all': macro_accuracy(pred, query_labels),
               'calibrated': macro_accuracy(pred, query_labels, selected),
               'uncalibrated': macro_accuracy(pred, query_labels, unseen_classes),
               'uncalibrated_center_mse': unadapted_center_mse}
        result_methods[name] = {'metrics': [row.copy() for _ in SEEDS]}
        saved_arrays[name+'_predictions'] = np.repeat(pred[None], len(SEEDS), axis=0)
    for method in METHODS:
        summary, arrays = evaluate_method(model, test, tp, fold, selected, method, chosen[method])
        result_methods[method] = summary
        saved_arrays.update({method+'_'+k: v for k, v in arrays.items()})
    result_methods['validation_selected_simple'] = result_methods[simple]
    saved_arrays['validation_selected_simple_predictions'] = saved_arrays[simple+'_predictions']
    all_classes = sorted(np.unique(test['y']).tolist())
    summary, arrays = evaluate_method(model, test, tp, fold, all_classes, 'lowrank', chosen['allclass_lowrank'])
    # Score the same noncalibrated class subset as the nine-label methods for comparison,
    # but explicitly retain the greater all-class support cost.
    for i, pred in enumerate(arrays['predictions']):
        summary['metrics'][i]['uncalibrated'] = macro_accuracy(pred, query_labels, unseen_classes)
        summary['metrics'][i]['uncalibrated_center_mse'] = float(np.mean([
            np.mean((query_z[query_labels == c].mean(0)-model.mu[c]-arrays['delta'][i, c])**2)
            for c in unseen_classes]))
    summary['support_budget'] = len(all_classes)*3
    result_methods['allclass_lowrank'] = summary
    saved_arrays.update({'allclass_lowrank_'+k: v for k, v in arrays.items()})
    np.savez_compressed(directory/'predictions.npz', **saved_arrays)
    result = {'identity': identity, 'fold': fold, 'support_classes': selected,
              'support_budget': 9, 'query_count': len(query), 'hyperparameters': chosen,
              'fit_diagnostics': model.diagnostics,
              'validation_selected_simple': simple, 'methods': result_methods,
              'files': {n: digest(directory/n) for n in ('selection.json', 'source_model.pkl', 'predictions.npz')}}
    dump(directory/'result.json', result)
    return result


def summarize(results, out):
    if sorted(r['fold'] for r in results) != list(range(1, 9)):
        raise ValueError('Summary requires all eight users; partial results cannot decide the gate')
    names = tuple(results[0]['methods'])
    arrays = {name: np.array([[m['uncalibrated'] for m in r['methods'][name]['metrics']] for r in results]) for name in names}
    table = {}
    for name in names:
        table[name] = {'uncalibrated_macro_accuracy': float(arrays[name].mean()),
                      'uncalibrated_per_user': arrays[name].mean(1).tolist(),
                      'worst_user_uncalibrated': float(arrays[name].mean(1).min()),
                      'within_user_support_std': arrays[name].std(1, ddof=1).tolist(),
                      'all_macro_accuracy': float(np.mean([m['all'] for r in results for m in r['methods'][name]['metrics']])),
                      'calibrated_macro_accuracy': float(np.mean([m['calibrated'] for r in results for m in r['methods'][name]['metrics']])),
                      'vs_frozen': paired_summary(arrays[name], arrays['frozen'])}
        budgets = [r['methods'][name].get('support_budget', 0 if name in ('frozen', 'source_prototype') else 9) for r in results]
        mse = np.array([[m['uncalibrated_center_mse'] for m in r['methods'][name]['metrics']] for r in results])
        mse0 = np.array([[m['uncalibrated_center_mse'] for m in r['methods']['frozen']['metrics']] for r in results])
        table[name].update(support_budget_per_user=budgets,
                           uncalibrated_center_mse=float(mse.mean()),
                           center_mse_reduction=float((mse0-mse).mean()),
                           center_mse_note='Raw estimated centers, before tau scales the discriminative logit correction')
    comparison = paired_summary(arrays['lowrank'], arrays['validation_selected_simple'])
    vs_frozen = table['lowrank']['vs_frozen']
    gate = decision_gate(comparison, vs_frozen)
    summary = {'state': 'complete', 'users': 8, 'support_repeats': len(SEEDS),
               'methods': table, 'lowrank_vs_simple': comparison, 'gate': gate,
               'decision': 'GO' if all(gate.values()) else 'STOP_LOW_RANK_MECHANISM',
               'note': 'Exploratory pilot; eight users, one model seed. Bootstrap resamples users only.'}
    dump(out/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, default=ROOT/'baseline_s2026')
    parser.add_argument('--out', type=Path, default=ROOT/'calibration_s2026')
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(1, 9)))
    args = parser.parse_args()
    results = []
    for fold in args.folds:
        results.append(run_fold(args.baseline, args.out, fold))
        print(f'CALIBRATION_COMPLETE fold={fold}; numerical results saved, no partial gate', flush=True)
    if args.folds == list(range(1, 9)):
        summary = summarize(results, args.out)
        print(json.dumps({'state': summary['state'], 'decision': summary['decision']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
