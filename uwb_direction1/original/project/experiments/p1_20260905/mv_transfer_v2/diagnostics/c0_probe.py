"""C0: the source-matched, redundancy-controlled, selection-clean redo of C.

The first C diagnostic compared a top-32 set that turned out to be entirely
Right-temporal against random draws from all 672 coordinates, so its gap mixed
predictability with branch membership. Everything here draws its controls from
the same candidate pool, repeats the random draw, keeps the probe user out of
the selection, and reports the hidden-view branches separately.

Frozen criteria: configs/c0_frozen.md. This is exploration after seeing the P1
outer results, not a pre-registered confirmation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge

import protocol as P
import predictability as C
from protocol import DIAG, dump

RIDGE_ALPHA = C.RIDGE_ALPHA
RANDOM_DRAWS = 10
DIMENSIONS = (16, 32, 64)
EPS = 1e-12

# 672 = 3 views x (192 temporal + 32 range)
BRANCHES = {}
for _v, _name in enumerate(P.VIEW_NAMES):
    BRANCHES[f'{_name}_temporal'] = np.arange(_v * 224, _v * 224 + 192)
    BRANCHES[f'{_name}_range'] = np.arange(_v * 224 + 192, (_v + 1) * 224)
CANDIDATE_POOL = BRANCHES['Right_temporal']          # what the old run actually selected


def ridge_skill(features, target, fit_users, y, subj, ids):
    """Leave-one-user-out skill inside `fit_users` only. Returns skill[user, coord]."""
    rows, users = [], []
    for user in fit_users:
        inner_fit = ids[(subj[ids] != user)]
        probe = ids[(subj[ids] == user)]
        if not len(probe):
            continue
        x_mu, x_sd = features[inner_fit].mean(0), features[inner_fit].std(0) + 1e-6
        t_mu, t_sd = target[inner_fit].mean(0), target[inner_fit].std(0) + 1e-6
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit((features[inner_fit] - x_mu) / x_sd, (target[inner_fit] - t_mu) / t_sd)
        predicted = model.predict((features[probe] - x_mu) / x_sd)
        truth = (target[probe] - t_mu) / t_sd
        centres, _ = C.class_centres((target[inner_fit] - t_mu) / t_sd, y[inner_fit])
        sse_predict = ((truth - predicted) ** 2).sum(0)
        sse_centre = ((truth - centres[y[probe]]) ** 2).sum(0)
        rows.append(1 - sse_predict / (sse_centre + EPS))
        users.append(user)
    return np.stack(rows), users


def qr_diverse(features, pool, eligible, k):
    """Rank-revealing QR pivoting: the least redundant k of the eligible columns.

    Selection sees only the fitting rows handed in, so it stays inside the same
    permission boundary as the skill ranking it is compared against.
    """
    columns = pool[eligible] if eligible.any() else pool
    block = features[:, columns]
    block = (block - block.mean(0)) / (block.std(0) + 1e-9)
    from scipy.linalg import qr
    _, _, pivots = qr(block, mode='economic', pivoting=True)
    return columns[pivots[:k]]


def probe_accuracy(target, columns, fit_ids, probe_ids, y, transform=None):
    a = target[fit_ids][:, columns] if transform is None else transform(target[fit_ids])
    b = target[probe_ids][:, columns] if transform is None else transform(target[probe_ids])
    mu, sd = a.mean(0), a.std(0) + 1e-6
    model = LogisticRegression(max_iter=400).fit((a - mu) / sd, y[fit_ids])
    return float(model.score((b - mu) / sd, y[probe_ids]))


def predicted_coordinate_probe(features, target, columns, fit_ids, probe_ids, y):
    """How much class information survives the actual prediction path?

    The classifier is trained on true teacher coordinates of the fitting users
    and tested on coordinates the ridge predicted for the held-out user from its
    radar input alone. This is what a student reading predicted targets would
    have to work with.
    """
    x_mu, x_sd = features[fit_ids].mean(0), features[fit_ids].std(0) + 1e-6
    t_mu, t_sd = target[fit_ids].mean(0), target[fit_ids].std(0) + 1e-6
    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit((features[fit_ids] - x_mu) / x_sd, (target[fit_ids] - t_mu) / t_sd)
    predicted = ridge.predict((features[probe_ids] - x_mu) / x_sd)[:, columns]
    a = ((target[fit_ids] - t_mu) / t_sd)[:, columns]
    mu, sd = a.mean(0), a.std(0) + 1e-6
    model = LogisticRegression(max_iter=400).fit((a - mu) / sd, y[fit_ids])
    return float(model.score((predicted - mu) / sd, y[probe_ids]))


def run(folds, dimensions=DIMENSIONS):
    y, subj = P.labels()
    features = C.build_probe_features()
    records = []
    branch_skill = {name: [] for name in BRANCHES}
    for fold in folds:
        target, _ = C.teacher_coordinates(fold)
        tr, va, te = P.fold_split(fold, y, subj)
        source_users = sorted(set(subj[tr].tolist()))
        for probe_user in source_users:
            # Everything below sees only the other source users until the final score.
            fit_users = [u for u in source_users if u != probe_user]
            fit_ids = tr[subj[tr] != probe_user]
            probe_ids = tr[subj[tr] == probe_user]
            skills, _ = ridge_skill(features, target, fit_users, y, subj, fit_ids)
            mean_skill = skills.mean(0)
            for name, columns in BRANCHES.items():
                branch_skill[name].append(float(mean_skill[columns].mean()))
            eligible = ((skills[:, CANDIDATE_POOL] > 0).sum(0) >= len(fit_users) - 2) \
                & (mean_skill[CANDIDATE_POOL] > 0)
            score = mean_skill - 0.5 * skills.std(0)
            rng = np.random.default_rng(1000 * fold + probe_user)
            for k in dimensions:
                order = np.argsort(-np.where(eligible, score[CANDIDATE_POOL], -np.inf))
                top = CANDIDATE_POOL[order[:min(k, max(int(eligible.sum()), 1))]]
                variants = {
                    'top_skill_same_source': top,
                    'qr_diverse_same_source': qr_diverse(target[fit_ids], CANDIDATE_POOL,
                                                         eligible, k),
                    'pca_same_source': None,                    # handled separately
                }
                row = {'fold': fold, 'probe_user': probe_user, 'k': int(k),
                       'eligible_in_pool': int(eligible.sum()), 'pool_size': int(len(CANDIDATE_POOL)),
                       'top_skill_mean': float(mean_skill[top].mean()),
                       'selected_from_hidden_views': int(sum(
                           1 for c in top if c < 448)),
                       'probe': {}}
                for name, columns in variants.items():
                    if columns is None:
                        continue
                    row['probe'][name] = probe_accuracy(target, columns, fit_ids, probe_ids, y)
                pca = PCA(n_components=k, random_state=0).fit(target[fit_ids][:, CANDIDATE_POOL])
                row['probe']['pca_same_source'] = probe_accuracy(
                    target, None, fit_ids, probe_ids, y,
                    transform=lambda a: pca.transform(a[:, CANDIDATE_POOL]))
                draws_same, draws_all = [], []
                for _ in range(RANDOM_DRAWS):
                    draws_same.append(probe_accuracy(
                        target, rng.choice(CANDIDATE_POOL, size=k, replace=False),
                        fit_ids, probe_ids, y))
                    draws_all.append(probe_accuracy(
                        target, rng.choice(target.shape[1], size=k, replace=False),
                        fit_ids, probe_ids, y))
                row['probe']['random_same_source_mean'] = float(np.mean(draws_same))
                row['probe']['random_same_source_std'] = float(np.std(draws_same))
                row['probe']['random_all_672_mean'] = float(np.mean(draws_all))
                row['probe']['random_all_672_std'] = float(np.std(draws_all))
                if k == 32:
                    row['probe']['top_skill_predicted_path'] = predicted_coordinate_probe(
                        features, target, top, fit_ids, probe_ids, y)
                records.append(row)
            print('fold', fold, 'probe user', probe_user,
                  json.dumps(records[-1]['probe']), flush=True)

    summary = {'branch_mean_skill': {k: float(np.mean(v)) for k, v in branch_skill.items()},
               'by_dimension': {}}
    for k in dimensions:
        rows = [r for r in records if r['k'] == k]
        keys = sorted({key for r in rows for key in r['probe']})
        summary['by_dimension'][int(k)] = {
            'n_probe_evaluations': len(rows),
            'eligible_in_pool_mean': float(np.mean([r['eligible_in_pool'] for r in rows])),
            'top_skill_mean': float(np.mean([r['top_skill_mean'] for r in rows])),
            'selected_from_hidden_views_total': int(sum(r['selected_from_hidden_views'] for r in rows)),
            'probe': {key: float(np.mean([r['probe'][key] for r in rows if key in r['probe']]))
                      for key in keys}}
    dump(DIAG / 'c0_probe.json', {'records': records, 'summary': summary,
                                  'candidate_pool': 'Right_temporal_192',
                                  'random_draws': RANDOM_DRAWS,
                                  'frozen_criteria': 'configs/c0_frozen.md'})
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, nargs='+', default=list(P.FOLDS))
    ap.add_argument('--dimensions', type=int, nargs='+', default=list(DIMENSIONS))
    args = ap.parse_args()
    run(args.folds, args.dimensions)
    print('C0_PROBE_OK')
