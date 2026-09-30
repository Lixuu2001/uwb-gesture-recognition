"""Small source-only linear fusion screen, governed by FROZEN_PROBE_PROTOCOL.json."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from audit_restored import digest, dump, metrics, validate_cache


def fit_statistics(features, labels, subjects):
    """Accept only source rows. Missing user/class cells are never zero filled."""
    x = np.asarray(features, dtype=np.float64).reshape(len(features), -1)
    mean, scale = x.mean(0), x.std(0)
    scale = np.maximum(scale, 1e-6)
    x = (x - mean) / scale
    weights = np.zeros(len(x))
    drifts = []
    for c in range(12):
        users = np.unique(subjects[labels == c])
        assert len(users) > 0
        means = []
        for u in users:
            cell = (labels == c) & (subjects == u)
            weights[cell] = 1.0 / (12 * len(users) * cell.sum())
            means.append(x[cell].mean(0))
        means = np.stack(means)
        drifts.extend(means - means.mean(0))
    assert np.isclose(weights.sum(), 1.0)
    targets = np.eye(12)[labels]
    xmean, ymean = weights @ x, weights @ targets
    xc, yc = x - xmean, targets - ymean
    covariance = xc.T @ (xc * weights[:, None])
    rhs = xc.T @ (yc * weights[:, None])
    d = np.stack(drifts)
    drift_covariance = d.T @ d / len(d)
    return dict(mean=mean, scale=scale, xmean=xmean, ymean=ymean,
                covariance=covariance, rhs=rhs, drift_covariance=drift_covariance)


def fit_readout(stats, alpha, eta):
    a = stats['covariance'] + eta * stats['drift_covariance']
    a = a + alpha * np.eye(len(a))
    w = solve(a, stats['rhs'], assume_a='pos')
    b = stats['ymean'] - stats['xmean'] @ w
    return w, b


def predict(features, stats, w, b):
    x = features.reshape(len(features), -1).astype(np.float64)
    return ((x - stats['mean']) / stats['scale']) @ w + b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, default=Path(__file__).resolve().parents[1]/'research/FROZEN_PROBE_PROTOCOL.json')
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Use a new empty output directory')
    audit = json.loads(args.audit.read_text())
    assert audit['state'] == 'verified' and len(audit['folds']) == 8
    protocol = json.loads(args.protocol.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    dump(args.out/'protocol.json', protocol)
    y = np.load(args.data/'cache/y_clutterremoved.npy')
    users = np.load(args.data/'cache/subj_clutterremoved.npy')
    report = {'state': 'running', 'protocol_sha256': digest(args.protocol),
              'script_sha256': digest(__file__), 'audit_sha256': digest(args.audit),
              'novelty_claim': False, 'encoder_retrained': False, 'folds': []}
    for held in range(1, 9):
        start = time.perf_counter()
        foldpath = args.data/f'experiments/p1_20260928/multiview_s2026/fold{held}'
        artifact = foldpath/'predictions.npz'
        assert digest(artifact) == audit['artifact_hashes'][str(artifact.relative_to(args.data))]
        with np.load(artifact, allow_pickle=False) as cache:
            validate_cache(cache, foldpath/'split.json', y, users, held)
            f, labels, subjects, role = (cache[k] for k in ('features', 'y', 'subjects', 'role'))
            tr, va, te = (role == i for i in range(3))
            stats = fit_statistics(f[tr], labels[tr], subjects[tr])
            # Complete all validation choices before using this fold's test labels.
            selected, grid_results = {}, []
            for method, etas in [('balanced_ridge_concat', [0.0]),
                                 ('balanced_ridge_class_conditional_user_penalty', protocol['penalty_eta_grid'])]:
                best = None
                for eta in etas:
                    for alpha in protocol['ridge_alpha_grid']:
                        w, b = fit_readout(stats, alpha, eta)
                        score = metrics(labels[va], predict(f[va], stats, w, b).argmax(-1))['macro_accuracy']
                        grid_results.append({'method': method, 'alpha': alpha, 'eta': eta,
                                             'validation_macro_accuracy': score})
                        if best is None or score > best['validation_macro_accuracy']:
                            best = dict(alpha=alpha, eta=eta, validation_macro_accuracy=score, w=w, b=b)
                selected[method] = best
            foldout = args.out/f'fold{held}'
            foldout.mkdir()
            dump(foldout/'validation_selection.json', grid_results)
            predictions = {'original_LTR': cache['logits'][te, 6].argmax(-1),
                           'uniform_seven_subset_probability': softmax(cache['logits'][te], axis=-1).mean(1).argmax(-1)}
            choices = {}
            for method, best in selected.items():
                predictions[method] = predict(f[te], stats, best['w'], best['b']).argmax(-1)
                np.savez_compressed(foldout/(method+'.npz'), mean=stats['mean'], scale=stats['scale'],
                                    w=best['w'], b=best['b'], source_ids=cache['ids'][tr])
                choices[method] = {k: best[k] for k in ('alpha', 'eta', 'validation_macro_accuracy')}
            np.savez_compressed(foldout/'test_predictions.npz', ids=cache['ids'][te], y=labels[te], **predictions)
            row = {'held_user': held, 'validation_user': held % 8 + 1, 'choices': choices,
                   'metrics': {name: metrics(labels[te], p) for name, p in predictions.items()},
                   'seconds': time.perf_counter() - start}
            report['folds'].append(row)
            dump(foldout/'result.json', row)
            dump(args.out/'RESULTS.json', report)
            print(json.dumps({'fold': held, 'macro_accuracy': {k: v['macro_accuracy'] for k, v in row['metrics'].items()},
                              'seconds': row['seconds']}), flush=True)
    names = list(report['folds'][0]['metrics'])
    scores = {name: np.array([r['metrics'][name]['macro_accuracy'] for r in report['folds']]) for name in names}
    report['summary'] = {name: {'mean_user_macro_accuracy': float(v.mean()),
                               'worst_user_macro_accuracy': float(v.min())} for name, v in scores.items()}
    candidate = scores['balanced_ridge_class_conditional_user_penalty']
    report['paired'] = {name: {'mean_difference_pp': float(100 * (candidate-v).mean()),
                              'improved_users': int(np.sum(candidate > v)),
                              'per_user_difference_pp': (100 * (candidate-v)).tolist()}
                        for name, v in scores.items() if name != 'balanced_ridge_class_conditional_user_penalty'}
    report['continue_criterion_met'] = bool(
        (candidate-scores['original_LTR']).mean() >= .01 and
        (candidate-scores['balanced_ridge_concat']).mean() >= .01 and
        np.sum(candidate > scores['balanced_ridge_concat']) >= 6)
    report['state'] = 'complete'
    dump(args.out/'RESULTS.json', report)
    print(json.dumps({'state': 'complete', 'summary': report['summary'],
                      'continue_criterion_met': report['continue_criterion_met']}), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=3):
        main()
