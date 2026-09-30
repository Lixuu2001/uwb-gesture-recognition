"""P0 diagnostic for method C: which teacher coordinates are predictable at all?

Fixed pooled coordinates on both sides, so the same coordinate index means the
same thing in every inner fold. A per-fold PCA would make the average of
component k meaningless, which is why the pooling is frozen instead.

The outer teacher has already seen every source user, so a positive score here
is a component-selection signal, not a statement that the teacher generalises.
"""
import argparse
import gc
import json

import numpy as np
import torch
from sklearn.linear_model import Ridge

import protocol as P
from protocol import BASE, CACHE, DIAG, RESULTS_CLEAN, digest, dump

PROBE_CACHE = CACHE / 'probe_features_right.npy'
RIDGE_ALPHA = 10.0
SELECTED = 32
MIN_ELIGIBLE = 8
MIN_POSITIVE_USERS = 5
EPS = 1e-12


def build_probe_features():
    """420 fixed radar coordinates: 10x21 pooled amplitude and phase of the Right view.

    Deliberately not the encoder output of a three-view trained model, which
    would already carry multi-view training in its parameters.
    """
    if PROBE_CACHE.exists():
        return np.load(PROBE_CACHE)
    amplitude = np.load(BASE / 'prepared/amp_stats.npy', mmap_mode='r')[:, P.DEPLOY_VIEW]
    phase = np.load(BASE / 'prepared/only.npy', mmap_mode='r')
    out = np.zeros((len(amplitude), 420), 'float32')
    for start in range(0, len(amplitude), 256):
        block = np.asarray(phase[start:start + 256, P.DEPLOY_VIEW], dtype=np.float32)
        pooled = block.reshape(len(block), 10, 9, 21, 9).mean((2, 4)).reshape(len(block), 210)
        out[start:start + len(block)] = np.concatenate(
            [np.asarray(amplitude[start:start + len(block)]), pooled], -1)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(PROBE_CACHE, out)
    return out


def teacher_coordinates(fold):
    """672 fixed teacher coordinates: six 15-frame temporal means plus a global
    range mean, per view, from the frozen three-view teacher of this fold."""
    dest = CACHE / f'teacher_coordinates_h{fold}.npz'
    checkpoint = RESULTS_CLEAN / f'teacher_concat_s2026_h{fold}.pt'
    sha = digest(checkpoint)
    if dest.exists():
        with np.load(dest) as a:
            assert str(a['teacher_sha']) == sha
            return a['features'].astype('float32'), sha
    from train import Classifier, setup
    y, subj = P.labels()
    setup(P.SPLIT_SEED)
    model = Classifier([0, 1, 2], 'off', 'concat').cuda().eval()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['fold'] == fold and state['args']['seed'] == P.SPLIT_SEED
    model.load_state_dict(state['state']); del state
    model.requires_grad_(False)
    x = np.load(BASE / 'prepared/off.npy', mmap_mode='r')
    tr, va, te = P.fold_split(fold, y, subj)
    ids = np.sort(np.concatenate([tr, va, te]))
    out = np.zeros((len(y), 672), 'float32')
    with torch.no_grad():
        for start in range(0, len(ids), 32):
            j = ids[start:start + 32]
            xx = torch.tensor(np.array(x[j], copy=True), device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, temporal, spatial = model.encoder(xx)
            b, v, t, d = temporal.shape
            coarse = temporal.reshape(b, v, 6, 15, d).mean(3).flatten(2)      # (b,v,192)
            ranges = spatial.mean(2)                                          # (b,v,32)
            out[j] = torch.cat([coarse, ranges], -1).float().flatten(1).cpu().numpy()
    np.savez_compressed(dest, features=out, teacher_sha=sha, ids=ids)
    del model; gc.collect(); torch.cuda.empty_cache()
    return out, sha


def class_centres(values, labels):
    """values: rows already restricted to the fitting set, labels aligned to them.

    The output keeps the global 0..11 row order, so a fold that is missing a
    class leaves a zero row instead of shifting every later class up one index.
    """
    assert len(values) == len(labels)
    centres = np.zeros((P.N_CLASSES, values.shape[1]), 'float64')
    present = np.zeros(P.N_CLASSES, bool)
    for cls in np.unique(labels):
        centres[cls] = values[labels == cls].mean(0)
        present[cls] = True
    return centres, present


def inner_ridge_skill(probe_features, target, train_ids, y, subj, residual=True, reserved=()):
    """Leave-one-source-user-out ridge over fixed coordinates.

    Returns skill[user, coordinate] = 1 - SSE(prediction) / SSE(class centre),
    where the class centre comes from the fit users, never from the probe user.
    """
    skills, users = [], []
    reserved = set(np.concatenate([np.asarray(r) for r in reserved]).tolist()) if reserved else set()
    for user, fit, probe in P.inner_splits(train_ids, subj):
        assert not (set(probe.tolist()) & reserved)
        assert user not in set(subj[fit].tolist())
        x_mu, x_sd = probe_features[fit].mean(0), probe_features[fit].std(0) + 1e-6
        t_mu, t_sd = target[fit].mean(0), target[fit].std(0) + 1e-6
        x_fit = (probe_features[fit] - x_mu) / x_sd
        x_probe = (probe_features[probe] - x_mu) / x_sd
        t_fit = (target[fit] - t_mu) / t_sd
        if residual:
            # Labels are a training-time diagnostic only; they never reach a
            # deployed input. Removing the class centre asks whether anything
            # beyond "which gesture is this" is predictable.
            centres, present = class_centres(t_fit, y[fit])
            x_centres, _ = class_centres(x_fit, y[fit])
            x_fit = x_fit - x_centres[y[fit]]
            x_probe = x_probe - x_centres[y[probe]]
            t_fit = t_fit - centres[y[fit]]
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(x_fit, t_fit)
        predicted = model.predict(x_probe)
        truth = (target[probe] - t_mu) / t_sd
        if residual:
            truth = truth - centres[y[probe]]
            reference = np.zeros_like(truth)          # the class centre itself
        else:
            centres, _ = class_centres((target[fit] - t_mu) / t_sd, y[fit])
            reference = centres[y[probe]]
        sse_predict = ((truth - predicted) ** 2).sum(0)
        sse_centre = ((truth - reference) ** 2).sum(0)
        skills.append(1 - sse_predict / (sse_centre + EPS))
        users.append({'user': user, 'n_fit': int(len(fit)), 'n_probe': int(len(probe)),
                      'degenerate_denominator': int((sse_centre < 1e-8).sum())})
    return np.stack(skills), users


def fold_skill(fold, residual=True):
    """The same computation wired to this fold's frozen teacher and radar probe."""
    y, subj = P.labels()
    tr, va, te = P.fold_split(fold, y, subj)
    probe_features = build_probe_features()
    target, teacher_sha = teacher_coordinates(fold)
    skills, users = inner_ridge_skill(probe_features, target, tr, y, subj, residual,
                                      reserved=(va, te))
    return skills, users, teacher_sha


def select(skills):
    mean, std = skills.mean(0), skills.std(0)
    positive = (skills > 0).sum(0)
    eligible = (positive >= MIN_POSITIVE_USERS) & (mean > 0)
    score = mean - 0.5 * std
    order = np.argsort(-np.where(eligible, score, -np.inf))
    chosen = order[:min(SELECTED, int(eligible.sum()))]
    return chosen, {'eligible': int(eligible.sum()), 'selected': int(len(chosen)),
                    'mean_skill_selected': float(mean[chosen].mean()) if len(chosen) else None,
                    'mean_skill_all': float(mean.mean()),
                    'best_skill': float(mean.max()), 'worst_skill': float(mean.min()),
                    'stop': bool(eligible.sum() < MIN_ELIGIBLE)}


def class_probe(target, chosen, fold, seed=0):
    """Do the selected coordinates carry gesture information, or user habit only?"""
    from sklearn.linear_model import LogisticRegression
    y, subj = P.labels()
    tr, va, te = P.fold_split(fold, y, subj)
    rng = np.random.default_rng(seed)
    out = {}
    variants = {'selected': chosen,
                'random': rng.choice(target.shape[1], size=len(chosen), replace=False)}
    for name, columns in variants.items():
        scores = []
        for user, fit, probe in P.inner_splits(tr, subj):
            mu, sd = target[fit][:, columns].mean(0), target[fit][:, columns].std(0) + 1e-6
            model = LogisticRegression(max_iter=300, multi_class='multinomial')
            model.fit((target[fit][:, columns] - mu) / sd, y[fit])
            scores.append(float(model.score((target[probe][:, columns] - mu) / sd, y[probe])))
        out[name] = {'mean_inner_accuracy': float(np.mean(scores)), 'per_user': scores}
    return out


def run(folds, residual=True, with_class_probe=True):
    report = {}
    for fold in folds:
        skills, users, teacher_sha = fold_skill(fold, residual)
        chosen, summary = select(skills)
        entry = {'inner_users': users, 'selection': summary,
                 'teacher_checkpoint_sha256': teacher_sha,
                 'selected_coordinates': chosen.tolist(),
                 'skill_percentiles': np.percentile(skills.mean(0), [5, 25, 50, 75, 95]).round(4).tolist(),
                 'fraction_coordinates_positive_mean_skill': float((skills.mean(0) > 0).mean())}
        if with_class_probe and len(chosen):
            target, _ = teacher_coordinates(fold)
            entry['class_probe'] = class_probe(target, chosen, fold)
        report[fold] = entry
        print('fold', fold, json.dumps(summary), flush=True)
    out = {'per_fold': report, 'ridge_alpha': RIDGE_ALPHA, 'residual_against_class_centre': residual,
           'teacher_coordinates': 672, 'radar_probe_coordinates': 420,
           'caveat': 'the outer teacher trained on every source user, so these are '
                     'component-selection scores, not unseen-user generalisation'}
    dump(DIAG / f'predictability{"_residual" if residual else ""}.json', out)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, nargs='+', default=list(P.FOLDS))
    ap.add_argument('--raw', action='store_true', help='score against the class centre baseline '
                                                       'without removing it from the inputs first')
    ap.add_argument('--no-class-probe', action='store_true')
    args = ap.parse_args()
    result = run(args.folds, residual=not args.raw, with_class_probe=not args.no_class_probe)
    print('PREDICTABILITY_OK')
