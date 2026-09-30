"""Validation-gated additive versus joint three-radar kernel mechanism screen."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh, solve
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from audit_restored import digest, dump, metrics, validate_cache


def weights_for(y, users):
    weights = np.zeros(len(y))
    classes = np.unique(y)
    for c in classes:
        domains = np.unique(users[y == c])
        for u in domains:
            mask = (y == c) & (users == u)
            weights[mask] = 1 / (len(classes) * len(domains) * mask.sum())
    return weights


def source_map(x, y, users, seed, count):
    rng = np.random.default_rng(seed)
    mean = x.mean(0)
    scale = np.maximum(x.std(0), 1e-6)
    z = (x - mean) / scale
    # Round-robin cells prevent dominant user/class cells monopolizing landmarks.
    pools = [list(rng.permutation(np.flatnonzero((y == c) & (users == u))))
             for c in np.unique(y) for u in np.unique(users[y == c])]
    chosen = []
    while len(chosen) < min(count, len(x)):
        for p in pools:
            if p and len(chosen) < min(count, len(x)):
                chosen.append(p.pop())
    pairs = rng.integers(0, len(x), size=(2048, 2))
    distances = ((z[pairs[:, 0]] - z[pairs[:, 1]]) ** 2).sum(-1)
    median = np.maximum(np.median(distances, axis=0), 1e-6)
    return dict(mean=mean, scale=scale, landmarks=z[chosen], median=median,
                landmark_indices=np.asarray(chosen))


def kernel(z, landmarks, median, kind):
    d = np.stack([cdist(z[:, v], landmarks[:, v], 'sqeuclidean') / median[v]
                  for v in range(3)])
    if kind == 'additive':
        return np.exp(-d).mean(0)
    if kind == 'joint':
        return np.exp(-d.mean(0))
    raise ValueError(kind)


def features(x, mapping, kind, whitening):
    z = (x - mapping['mean']) / mapping['scale']
    return kernel(z, mapping['landmarks'], mapping['median'], kind) @ whitening


def fit(phi, y, weights, alpha):
    target = np.eye(12)[y]
    xm, ym = weights @ phi, weights @ target
    centered = phi - xm
    w = solve(centered.T @ (centered * weights[:, None]) + alpha * np.eye(phi.shape[1]),
              centered.T @ ((target - ym) * weights[:, None]), assume_a='pos')
    return w, ym - xm @ w


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, default=Path(__file__).resolve().parents[1]/'research/KERNEL_SCREEN_PROTOCOL.json')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    protocol = json.loads(args.protocol.read_text())
    audit = json.loads(args.audit.read_text())
    assert audit['state'] == 'verified'
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise ValueError('Use new output directory or explicit --resume')
    args.out.mkdir(parents=True, exist_ok=True)
    hashes = dict(protocol_sha256=digest(args.protocol), script_sha256=digest(__file__), audit_sha256=digest(args.audit))
    if (args.out/'RESULTS.json').exists():
        previous = json.loads((args.out/'RESULTS.json').read_text())
        assert all(previous[k] == v for k,v in hashes.items()), 'Resume inputs changed'
    dump(args.out/'protocol.json', protocol)
    y = np.load(args.data/'cache/y_clutterremoved.npy')
    users = np.load(args.data/'cache/subj_clutterremoved.npy')
    report = dict(state='running_validation', pid=os.getpid(), **hashes, folds=[], test_evaluated=False)
    for held in protocol['folds']:
        out = args.out/f'fold{held}'
        out.mkdir(exist_ok=True)
        if args.resume and (out/'validation.json').exists():
            report['folds'].append(json.loads((out/'validation.json').read_text()))
            continue
        start = time.perf_counter()
        base = args.data/f'experiments/p1_20260928/multiview_s2026/fold{held}'
        cachepath = base/'predictions.npz'
        assert digest(cachepath) == audit['artifact_hashes'][str(cachepath.relative_to(args.data))]
        with np.load(cachepath, allow_pickle=False) as cache:
            validate_cache(cache, base/'split.json', y, users, held)
            tr, va = cache['role'] == 0, cache['role'] == 1
            # Never construct new predictions on test rows in the selection stage.
            x, labels, subjects = cache['features'][tr].astype(float), cache['y'][tr], cache['subjects'][tr]
            xv, yv = cache['features'][va].astype(float), cache['y'][va]
            mapping = source_map(x, labels, subjects, protocol['seed'] + held, protocol['landmarks'])
            np.savez_compressed(out/'mapping.npz', **mapping, source_ids=cache['ids'][tr],
                                landmark_ids=cache['ids'][tr][mapping['landmark_indices']])
            weights = weights_for(labels, subjects)
            row = dict(held_user=held, validation_user=held%8+1,
                       original_LTR=metrics(yv, cache['logits'][va, 6].argmax(-1)), methods={})
            for kind in protocol['kernels']:
                kmm = kernel(mapping['landmarks'], mapping['landmarks'], mapping['median'], kind)
                eigenvalues, vectors = eigh(kmm)
                whitening = vectors / np.sqrt(np.maximum(eigenvalues, 1e-10))
                phi, phiv = features(x, mapping, kind, whitening), features(xv, mapping, kind, whitening)
                grid, best = [], None
                for alpha in protocol['alpha_grid']:
                    w,b = fit(phi, labels, weights, alpha)
                    score = metrics(yv, (phiv@w+b).argmax(-1))
                    grid.append(dict(alpha=alpha, macro_accuracy=score['macro_accuracy']))
                    if best is None or score['macro_accuracy'] > best['metrics']['macro_accuracy']:
                        best = dict(alpha=alpha, metrics=score)
                        np.savez_compressed(out/f'{kind}.npz', whitening=whitening, w=w,b=b)
                        np.savez_compressed(out/f'{kind}_validation.npz', ids=cache['ids'][va], y=yv, scores=phiv@w+b)
                row['methods'][kind] = dict(**best, grid=grid)
            row['seconds'] = time.perf_counter()-start
            dump(out/'validation.json',row)
            report['folds'].append(row)
            dump(args.out/'RESULTS.json', report)
            print(json.dumps(dict(fold=held, seconds=row['seconds'], scores={k:v['metrics']['macro_accuracy'] for k,v in row['methods'].items()})),flush=True)
    values = {k: np.array([r['methods'][k]['metrics']['macro_accuracy'] for r in report['folds']]) for k in protocol['kernels']}
    original = np.array([r['original_LTR']['macro_accuracy'] for r in report['folds']])
    diff = values['joint']-values['additive']
    gate = protocol['gate']
    report['validation_summary'] = {k:float(v.mean()) for k,v in {**values,'original_LTR':original}.items()}
    report['paired_validation'] = dict(joint_minus_additive_pp=(100*diff).tolist(),
                                      joint_minus_original_pp=(100*(values['joint']-original)).tolist())
    report['gate_passed'] = bool(100*diff.mean() >= gate['min_mean_gain_vs_additive_pp'] and
        100*(values['joint']-original).mean() >= gate['min_mean_gain_vs_old_LTR_pp'] and
        (diff>0).sum() >= gate['min_positive_folds_vs_additive'])
    report['state'] = 'validation_passed_test_pending' if report['gate_passed'] else 'rejected_on_validation'
    report['total_fit_seconds'] = sum(r['seconds'] for r in report['folds'])
    dump(args.out/'RESULTS.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='folds'}),flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=3):
        main()
