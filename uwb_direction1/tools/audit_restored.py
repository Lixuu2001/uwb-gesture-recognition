"""Audit restored original arrays and replay frozen predictions without training.

No target labels are used for fitting or selecting a method. Historical accuracy
replay is not a new training reproduction. Output goes to a separate directory.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SUBSETS = ('L', 'T', 'R', 'LT', 'LR', 'TR', 'LTR')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def quality_mask(y, subjects):
    return ~(((subjects == 2) & np.isin(y, [10, 11])) |
             ((subjects == 6) & (y == 3)))


def metrics(y, pred):
    classes = np.unique(y)
    per_class = {str(int(c)): float(np.mean(pred[y == c] == c)) for c in classes}
    cm = np.zeros((12, 12), dtype=int)
    np.add.at(cm, (y, pred), 1)
    return {'samples': len(y), 'accuracy': float(np.mean(y == pred)),
            'macro_accuracy': float(np.mean(list(per_class.values()))),
            'per_class_accuracy': per_class, 'confusion': cm.tolist()}


def expected_parts(y, subjects, held):
    good = quality_mask(y, subjects)
    return {'train': np.flatnonzero(good & ~np.isin(subjects, [held, held % 8 + 1])),
            'val': np.flatnonzero(good & (subjects == held % 8 + 1)),
            'test': np.flatnonzero(good & (subjects == held))}


def validate_cache(cache, split_path, y, subjects, held):
    saved = json.loads(Path(split_path).read_text(encoding='utf-8-sig'))
    parts = expected_parts(y, subjects, held)
    ids = cache['ids']
    assert ids.dtype.kind in 'iu' and len(np.unique(ids)) == 9300
    np.testing.assert_array_equal(cache['y'], y[ids])
    np.testing.assert_array_equal(cache['subjects'], subjects[ids])
    for code, role in enumerate(('train', 'val', 'test')):
        np.testing.assert_array_equal(saved['ids'][role], parts[role])
        np.testing.assert_array_equal(ids[cache['role'] == code], parts[role])
    assert np.isin(cache['role'], [0, 1, 2]).all()
    assert np.isfinite(cache['logits']).all() and np.isfinite(cache['features']).all()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--package', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Use a new empty audit output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    restored = json.loads((args.data/'RESTORE_VERIFIED.json').read_text())
    manifest_path = args.package/'TRANSFER_MANIFEST.json'
    assert restored['state'] == 'verified' and restored['files'] == 86
    assert restored['transfer_manifest_sha256'] == digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest['files']:
        path = args.data/entry['path']
        assert path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'], entry['path']
    print('ORIGINAL_HASHES_VERIFIED files=86', flush=True)
    y = np.load(args.data/'cache/y_clutterremoved.npy', allow_pickle=False)
    subjects = np.load(args.data/'cache/subj_clutterremoved.npy', allow_pickle=False)
    x = np.load(args.data/'cache/X_clutterremoved.npy', mmap_mode='r', allow_pickle=False)
    prepared = args.data/'experiments/p1_20260905/prepared'
    np.testing.assert_array_equal(y, np.load(prepared/'y.npy'))
    np.testing.assert_array_equal(subjects, np.load(prepared/'subj.npy'))
    assert x.shape == (9593, 3, 90, 189) and len(y) == len(subjects) == len(x)
    assert set(np.unique(y)) == set(range(12)) and set(np.unique(subjects)) == set(range(1, 9))
    good = quality_mask(y, subjects)
    assert good.sum() == 9300 and (~good).sum() == 293
    seen, duplicates = {}, []
    for i in range(len(x)):
        row = np.asarray(x[i])
        assert np.isfinite(row).all(), i
        h = hashlib.sha256(row.tobytes()).hexdigest()
        if h in seen:
            j = seen[h]
            duplicates.append({'ids': [j, i], 'labels': [int(y[j]), int(y[i])],
                               'subjects': [int(subjects[j]), int(subjects[i])],
                               'both_retained': bool(good[j] and good[i])})
        else:
            seen[h] = i
    counts = {str(s): {str(c): int(np.sum(good & (subjects == s) & (y == c)))
                       for c in range(12)} for s in range(1, 9)}
    report = {'state': 'verified', 'scope': 'original hash audit and cached prediction replay, no retraining',
              'transfer_manifest_sha256': digest(manifest_path), 'script_sha256': digest(__file__),
              'shape': list(x.shape), 'dtype': str(x.dtype), 'raw_records': len(x),
              'retained_records': int(good.sum()), 'excluded_records': int((~good).sum()),
              'finite': True, 'retained_class_counts': counts, 'exact_duplicates': duplicates,
              'evaluation': 'mean of eight user macro accuracies over present classes; exploratory users',
              'folds': [], 'artifact_hashes': {}}
    for held in range(1, 9):
        base = args.data/f'experiments/p1_20260928/baseline_s2026/fold{held}'
        multi = args.data/f'experiments/p1_20260928/multiview_s2026/fold{held}'
        feat_path, pred_path = base/'warmup/features.npz', multi/'predictions.npz'
        feat_meta = json.loads((base/'warmup/features.json').read_text())
        result = json.loads((multi/'result.json').read_text())
        assert digest(feat_path) == feat_meta['sha256']
        assert digest(pred_path) == result['files']['predictions.npz']
        with np.load(feat_path, allow_pickle=False) as b, np.load(pred_path, allow_pickle=False) as m:
            validate_cache(b, base/'split.json', y, subjects, held)
            validate_cache(m, multi/'split.json', y, subjects, held)
            assert b['features'].shape == (9300, 256) and b['logits'].shape == (9300, 12)
            assert m['features'].shape == (9300, 3, 256) and m['logits'].shape == (9300, 7, 12)
            np.testing.assert_array_equal(b['ids'], m['ids'])
            np.testing.assert_array_equal(m['subset_masks'], [[1,0,0],[0,1,0],[0,0,1],
                                                           [1,1,0],[1,0,1],[0,1,1],[1,1,1]])
            test = m['role'] == 2
            val = m['role'] == 1
            np.testing.assert_array_equal(m['val_logits'], m['logits'][val])
            np.testing.assert_array_equal(m['val_y'], m['y'][val])
            fold = {'held_user': held, 'val_user': held % 8 + 1,
                    'right_original': metrics(b['y'][test], b['logits'][test].argmax(-1)),
                    'subsets': {name: metrics(m['y'][test], m['logits'][test, j].argmax(-1))
                                for j, name in enumerate(SUBSETS)}}
            np.testing.assert_allclose([fold['subsets'][name]['accuracy'] for name in SUBSETS],
                                       result['test_accuracy_by_subset'], rtol=0, atol=1e-12)
            report['folds'].append(fold)
        report['artifact_hashes'][str(feat_path.relative_to(args.data))] = digest(feat_path)
        report['artifact_hashes'][str(pred_path.relative_to(args.data))] = digest(pred_path)
        print(f'FOLD_REPLAY_VERIFIED held={held} LTR={fold["subsets"]["LTR"]["macro_accuracy"]:.6f}', flush=True)
    report['summary'] = {name: {'mean_user_macro_accuracy': float(np.mean(values)),
                              'worst_user_macro_accuracy': float(np.min(values))}
                         for name in ('right_original', *SUBSETS)
                         for values in [[f['right_original']['macro_accuracy'] if name == 'right_original'
                                         else f['subsets'][name]['macro_accuracy'] for f in report['folds']]]}
    dump(args.out/'AUDIT.json', report)
    print(json.dumps({'state': 'audit_complete', 'summary': report['summary']}), flush=True)


if __name__ == '__main__':
    main()
