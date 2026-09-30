"""Refit Right-only OOF label smoothing on the six parameter-training users."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_meta_smoothing_protocol', HERE/'protocol.py')
_protocol = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_protocol)
BASE, digest, dump = _protocol.BASE, _protocol.digest, _protocol.dump


def classification_loss(logits, labels, matrix):
    logits = logits.float(); matrix = matrix.detach().float()
    onehot = F.one_hot(labels, num_classes=matrix.shape[0]).float()
    target = .5*(onehot + onehot@matrix)
    return F.cross_entropy(logits, labels)+.5*F.kl_div(F.log_softmax(logits/2,-1),target,reduction='batchmean')*4


def fit_matrix(x, labels, subjects, train_ids, n_classes=12):
    ids = np.asarray(train_ids)
    if set(labels[ids]) != set(range(n_classes)) or len(np.unique(subjects[ids])) != 6:
        raise ValueError('Require six training users and the complete class space')
    oof = np.full((len(labels), n_classes), np.nan)
    history = []
    for user in np.unique(subjects[ids]):
        fit = ids[subjects[ids] != user]; held = ids[subjects[ids] == user]
        scaler = StandardScaler().fit(x[fit])
        classifier = LogisticRegression(C=1, solver='lbfgs', max_iter=1000, tol=1e-4, random_state=2026)
        with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=4):
            warnings.simplefilter('always')
            classifier.fit(scaler.transform(x[fit]), labels[fit])
            if not np.array_equal(classifier.classes_, np.arange(n_classes)):
                raise ValueError('Proxy training fold lacks required classes')
            oof[held] = classifier.predict_proba(scaler.transform(x[held]))
        history.append({'held_subject': int(user), 'fit_ids': fit.tolist(), 'predict_ids': held.tolist(),
                        'iterations': classifier.n_iter_.tolist(), 'warnings': [str(w.message) for w in captured]})
    if not np.isfinite(oof[ids]).all():
        raise ValueError('Missing OOF predictions')
    diagonal = float(np.mean(oof[ids, labels[ids]]))
    matrix = np.full((n_classes, n_classes), (1-diagonal)/(n_classes-1), dtype=np.float32)
    np.fill_diagonal(matrix, diagonal)
    return matrix, oof, history


def raw_proxy(source_identity):
    """Only reuse a fixed per-sample representation, never fitted proxy weights."""
    path = BASE/'confusion_kd_results/proxy_R.npy'
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if meta.get('fitted_parameters') is not False or digest(path) != meta['sha']:
        raise ValueError('Raw Right proxy provenance mismatch')
    for mode in ('off', 'only'):
        if meta['sources'][mode] != source_identity[str(BASE/f'prepared/{mode}.npy')]:
            raise ValueError('Raw proxy source changed')
    return np.load(path), {'path': str(path), 'sha256': meta['sha'], 'fitted_parameters': False}


def data_identity():
    paths = [BASE/f'prepared/{name}.npy' for name in ('off','only','y','subj')]
    return {str(p): digest(p) for p in paths}


def prepare_matrix(held, output, source_identity):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    path = output/f'fold{held}.npz'; meta_path = path.with_suffix('.json')
    _, labels, subjects, parts = _protocol.open_fold(held)
    x, proxy = raw_proxy(source_identity)
    identity = {'fold': held, 'validation_subject': held%8+1,
                'training_subjects': np.unique(subjects[parts['train']]).tolist(),
                'data': source_identity, 'proxy': proxy,
                'codes': {str(p): digest(p) for p in (Path(__file__), HERE/'protocol.py')},
                'partitions': {k: _protocol.array_hash(v) for k,v in parts.items()}}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['identity'] != identity or digest(path) != meta['matrix_sha256']:
            raise ValueError('Six-user smoothing identity changed')
        return np.load(path)['matrix'].copy(), meta
    matrix, oof, history = fit_matrix(x, labels, subjects, parts['train'])
    warning_count = sum(len(row['warnings']) for row in history)
    if warning_count:
        dump(output/f'fold{held}_proxy_failure.json', {'history': history})
        raise RuntimeError('OOF proxy produced warnings; inspect fit before training')
    np.savez_compressed(path, matrix=matrix, oof_train=oof[parts['train']],
                        train_ids=parts['train'], val_ids=parts['val'], test_ids=parts['test'])
    meta = {'identity': identity, 'matrix_sha256': digest(path), 'oof_fits': history,
            'warnings': warning_count, 'validation_used_for_fit': False, 'test_used_for_fit': False,
            'diagonal': float(matrix[0,0]), 'class_counts': np.bincount(labels[parts['train']],minlength=12).tolist()}
    dump(meta_path, meta)
    print(f'SIX_USER_MATRIX_READY fold={held} diagonal={matrix[0,0]:.6f}', flush=True)
    return matrix, meta


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--folds',nargs='+',type=int,default=list(range(1,9)))
    parser.add_argument('--out',type=Path,default=HERE/'matrices')
    args=parser.parse_args(); source=data_identity()
    for held in args.folds: prepare_matrix(held,args.out,source)


if __name__=='__main__': main()
