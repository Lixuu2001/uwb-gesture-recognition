"""Frozen protocol access for multi-view supervised, single-radar deployed study.

Everything that decides which sample a run may touch lives here. Training
scripts import this module instead of re-deriving splits, so an arm cannot
silently train on a different population than its control.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

V2 = Path(__file__).resolve().parent
BASE = V2.parent
ROOT = BASE.parents[1]
RUNS = V2 / 'runs'
CACHE = V2 / 'cache'
DIAG = V2 / 'diagnostics'
RESULTS_CLEAN = BASE / 'results_clean'
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'vendor'))
# This package shadows same-named modules in the parent study (summarize, audit),
# so its own directory must win the import race.
sys.path.insert(0, str(V2))

SPLIT_SEED = 2026            # split is frozen; model_seed varies independently
DEPLOY_VIEW = 2              # Right radar is the deployed sensor
VIEW_NAMES = ('Left', 'Top', 'Right')
N_CLASSES = 12
FOLDS = tuple(range(1, 9))
SEGMENTS = 15
FRAMES_PER_SEGMENT = 6
FRAMES = SEGMENTS * FRAMES_PER_SEGMENT   # 90 slow-time frames
RANGE_BINS = 189
SAMPLER_SEED = 7777          # relation batch order, saved with every run
AUXILIARY_SEED = 31337       # aux head init / masks; never touches the backbone


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_plain), encoding='utf-8')
    tmp.replace(path)


def _plain(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def hash_ids(ids):
    """Stable identity of a sample set; sorted so order never fakes a mismatch."""
    a = np.ascontiguousarray(np.sort(np.asarray(ids, dtype=np.int64)))
    return hashlib.sha256(a.tobytes()).hexdigest()[:32]


def labels():
    y = np.load(BASE / 'prepared/y.npy')
    subj = np.load(BASE / 'prepared/subj.npy')
    return y, subj


def fold_split(fold, y=None, subj=None):
    """train / val / test absolute cache row ids for one held-out subject."""
    from pipeline import split
    if y is None:
        y, subj = labels()
    return split(y, subj, fold, SPLIT_SEED)


def inner_splits(train_ids, subj):
    """Leave-one-source-user-out inside the outer training set (C and tuning).

    Yields (fit_ids, probe_ids); outer val/test never enter either side.
    """
    train_ids = np.asarray(train_ids)
    for user in sorted(set(subj[train_ids].tolist())):
        probe = train_ids[subj[train_ids] == user]
        fit = train_ids[subj[train_ids] != user]
        yield int(user), np.sort(fit), np.sort(probe)


def global_matrix(fold):
    """The Right-only inner-OOF derived global smoothing matrix of the base loss.

    Reused verbatim from the label_global control so the new base arm is the
    same 74.77% objective, not a guessed epsilon.
    """
    path = BASE / f'label_smoothing_results/global_matrix_h{fold}.npz'
    with np.load(path) as archive:
        matrix = archive['global_matrix'].astype('float32')
    diagonal = np.diag(matrix)
    assert np.allclose(diagonal, diagonal[0]), 'global control must be class-flat'
    off = (1 - diagonal[0]) / (N_CLASSES - 1)
    assert np.allclose(matrix[~np.eye(N_CLASSES, dtype=bool)], off)
    np.testing.assert_allclose(matrix.sum(1), 1, atol=1e-6)
    return matrix, float((1 + diagonal[0]) / 2), str(path)


def verify_manifest(write=True):
    """Every quality-retained row is tested exactly once across the eight folds."""
    from pipeline import quality_mask
    y, subj = labels()
    good = np.where(quality_mask(y, subj))[0]
    seen, folds = [], {}
    for fold in FOLDS:
        tr, va, te = fold_split(fold, y, subj)
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
        assert set(subj[te]) == {fold} and fold not in set(subj[tr]) and fold not in set(subj[va])
        assert set(tr) | set(va) | set(te) <= set(good.tolist())
        seen.extend(te.tolist())
        folds[fold] = {'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
                       'train_hash': hash_ids(tr), 'val_hash': hash_ids(va), 'test_hash': hash_ids(te),
                       'test_classes': sorted(set(y[te].tolist())),
                       'train_users': sorted(set(subj[tr].tolist()))}
    np.testing.assert_array_equal(np.sort(seen), good)
    manifest = {'cache_rows': int(len(y)), 'quality_rows': int(len(good)),
                'split_seed': SPLIT_SEED, 'deploy_view': VIEW_NAMES[DEPLOY_VIEW],
                'class_outputs': N_CLASSES, 'folds': folds,
                'quality_hash': hash_ids(good),
                'pipeline_sha256': digest(BASE / 'pipeline.py')}
    if write:
        dump(V2 / 'configs/manifest.json', manifest)
    return manifest


def provenance(**fields):
    """Information-source ledger (implementation plan section 4). Missing keys are
    an error, not a default: an unrecorded source is exactly the failure mode
    this ledger exists to catch."""
    required = ('student_input_views', 'teacher_training_views', 'teacher_target_views',
                'teacher_training_ids_hash', 'student_train_ids_hash', 'validation_ids_hash',
                'test_ids_hash', 'teacher_checkpoint_sha256', 'target_transform_fit_ids_hash',
                'label_statistics_source', 'split_seed', 'model_seed', 'sampler_seed',
                'auxiliary_seed', 'initial_classifier_hash', 'sampler_manifest_hash',
                'code_snapshot_sha256', 'aux_settings')
    missing = [k for k in required if k not in fields]
    if missing:
        raise KeyError(f'provenance missing {missing}')
    return {k: fields[k] for k in required} | {k: v for k, v in fields.items() if k not in required}


def code_snapshot():
    files = sorted(p for p in V2.glob('*.py'))
    files += [BASE / 'pipeline.py', BASE / 'train.py',
              BASE / 'vendor/uwb_osr/model.py', BASE / 'vendor/config.py']
    return {str(p.relative_to(BASE)): digest(p) for p in files}


if __name__ == '__main__':
    print(json.dumps(verify_manifest(), indent=2)[:1200])
    print('MANIFEST_OK')
