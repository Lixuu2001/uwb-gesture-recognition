"""Training-user OOF proxy confusion projection; no validation-label fitting."""
import argparse
import gc
import json
import time
import warnings
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from pipeline import BASE, dump, digest, split
import train

OUT = BASE / 'confusion_kd_results'


def inner_splits(train_ids, subjects):
    for subject in np.unique(subjects[train_ids]):
        held = train_ids[subjects[train_ids] == subject]
        rest = train_ids[subjects[train_ids] != subject]
        yield rest, held


def estimate_matrix(probabilities, y, ids):
    nclass = probabilities.shape[1]
    if not np.isfinite(probabilities[ids]).all(): raise ValueError('Missing OOF training predictions')
    matrix = np.stack([probabilities[ids[y[ids] == c]].mean(0) for c in range(nclass)])
    if not np.isfinite(matrix).all(): raise ValueError('Unsupported class in training matrix')
    np.testing.assert_allclose(matrix.sum(1), 1, atol=1e-6)
    return matrix


def uniform_control(matrix):
    d = np.diag(matrix)
    u = np.repeat(((1 - d) / (len(d) - 1))[:, None], len(d), axis=1)
    np.fill_diagonal(u, d)
    return u


def projected_kd(student, teacher, matrix, temp=2):
    p = F.softmax(teacher.detach().float() / temp, dim=-1)
    q = .5 * (p + p @ matrix.detach().float())
    return F.kl_div(F.log_softmax(student.float() / temp, dim=-1), q, reduction='batchmean') * temp**2


def features():
    path = OUT / 'proxy_R.npy'; meta = OUT / 'proxy_R.json'
    if path.exists():
        m = json.loads(meta.read_text())
        assert digest(path) == m['sha']
        return np.load(path), m
    parts = []
    for mode in ('off', 'only'):
        a = np.load(BASE / f'prepared/{mode}.npy', mmap_mode='r')
        z = np.empty((len(a), 210), dtype='float32')
        for i in range(0, len(a), 128):
            b = np.asarray(a[i:i+128, 2], dtype='float32')
            z[i:i+len(b)] = b.reshape(-1, 10, 9, 21, 9).mean((2, 4)).reshape(-1, 210)
        parts.append(z)
    out = np.concatenate(parts, axis=1)
    np.save(path, out)
    meta_obj = {'sha': digest(path), 'shape': list(out.shape), 'fitted_parameters': False,
                'sources': {mode: digest(BASE / f'prepared/{mode}.npy') for mode in ('off', 'only')},
                'definition': '10 slow-time blocks x 21 range blocks, amplitude and signed phase, Right only'}
    dump(meta, meta_obj)
    return out, meta_obj


def prepare_fold(fold, x, feature_meta):
    y = np.load(BASE / 'prepared/y.npy'); subjects = np.load(BASE / 'prepared/subj.npy')
    tr, va, te = split(y, subjects, fold, 2026)
    path = OUT / f'matrix_h{fold}.npz'; meta_path = path.with_suffix('.json')
    if path.exists():
        meta = json.loads(meta_path.read_text())
        assert meta['feature_sha'] == feature_meta['sha'] and meta['code_sha'] == digest(__file__)
        assert meta['matrix_sha'] == digest(path)
        with np.load(path) as a:
            for key, ids in [('train_ids', tr), ('val_ids', va), ('test_ids', te)]: np.testing.assert_array_equal(a[key], ids)
            return a['structure'].copy(), a['uniform'].copy(), meta
    start = time.time(); oof = np.full((len(y), 12), np.nan); history = []
    for rest, held in inner_splits(tr, subjects):
        assert not set(subjects[rest]) & set(subjects[held])
        assert not np.intersect1d(np.r_[rest, held], np.r_[va, te]).size
        scaler = StandardScaler().fit(x[rest])
        clf = LogisticRegression(C=1, solver='lbfgs', max_iter=1000, tol=1e-4, random_state=2026)
        with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=4):
            warnings.simplefilter('always')
            clf.fit(scaler.transform(x[rest]), y[rest])
            np.testing.assert_array_equal(clf.classes_, np.arange(12))
            oof[held] = clf.predict_proba(scaler.transform(x[held]))
        history.append({'held_subject': int(subjects[held[0]]), 'n_fit': len(rest), 'n_predict': len(held),
                        'iterations': clf.n_iter_.tolist(), 'warnings': [str(w.message) for w in captured],
                        'fit_ids': rest.tolist(), 'predict_ids': held.tolist()})
        print('PROXY_OOF', fold, int(subjects[held[0]]), 'iterations', clf.n_iter_.tolist(), flush=True)
    assert np.isnan(oof[np.r_[va, te]]).all()
    matrix = estimate_matrix(oof, y, tr); uniform = uniform_control(matrix)
    np.savez_compressed(path, structure=matrix, uniform=uniform, train_ids=tr, val_ids=va, test_ids=te,
                        oof_train_probabilities=oof[tr])
    meta = {'fold': fold, 'feature_sha': feature_meta['sha'], 'code_sha': digest(__file__),
            'matrix_sha': digest(path), 'n_oof': len(tr), 'oof_accuracy': float((oof[tr].argmax(1) == y[tr]).mean()),
            'diagonal': matrix.diagonal().tolist(), 'warnings': sum(len(r['warnings']) for r in history),
            'class_counts': np.bincount(y[tr], minlength=12).tolist(), 'inner_folds': history,
            'seconds': time.time() - start, 'validation_used_for_projection': False,
            'scope': 'Low-dimensional OOF proxy confusion, not physical observability ground truth'}
    dump(meta_path, meta)
    print('MATRIX_READY', fold, 'OOF accuracy', meta['oof_accuracy'], 'seconds', meta['seconds'], flush=True)
    return matrix, uniform, meta


def run(folds, mode):
    OUT.mkdir(exist_ok=True); x, xm = features()
    for fold in folds:
        c, u, meta = prepare_fold(fold, x, xm)
        if meta['warnings']: raise RuntimeError('Proxy fit warned; inspect before running KD')
        matrix = torch.tensor(c if mode == 'structure' else u, dtype=torch.float32, device='cuda')
        args = argparse.Namespace(tag=f'sv_R_project_{mode}', mode='concat', views=[2], readout='mean',
                  folds=[fold], seed=2026, epochs=40, batch=128, microbatch=32,
                  teacher='teacher_concat', save_teacher=False, kd_weight=.5, aux='none',
                  projection=mode, projection_mix=.5, projection_matrix_sha=meta['matrix_sha'],
                  proxy_feature_sha=xm['sha'], wrapper_sha=digest(__file__))
        ordinary_kd = train.kd_loss
        train.kd_loss = lambda student, teacher, temp=2: projected_kd(student, teacher, matrix, temp)
        data = train.load_gpu('concat', [2])
        try:
            train.run(args, data, fold)
        finally:
            train.kd_loss = ordinary_kd
            del data, matrix; gc.collect(); torch.cuda.empty_cache()


def smoke():
    train.setup(2026)
    model = train.Classifier([2], 'concat').cuda()
    x = train.load_gpu('concat', [2])[:4].clone()
    t = torch.randn(4, 12, device='cuda', requires_grad=True)
    c = torch.eye(12, device='cuda') * .7 + torch.ones(12, 12, device='cuda') * .025
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z, _ = model(x); loss = projected_kd(z, t, c)
    loss.backward()
    assert t.grad is None and torch.isfinite(loss)
    assert model.encoder.temporal.proj.weight.grad.abs().sum() > 0
    print('PROJECTED_KD_SMOKE_OK', float(loss.detach()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--prepare', action='store_true'); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--folds', type=int, nargs='+', default=list(range(1, 9)))
    ap.add_argument('--mode', choices=['structure', 'uniform'], default='structure')
    args = ap.parse_args(); OUT.mkdir(exist_ok=True); train.setup(2026)
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    if args.smoke: smoke()
    elif args.prepare:
        x, meta = features()
        for fold in args.folds: prepare_fold(fold, x, meta)
    else: run(args.folds, args.mode)
