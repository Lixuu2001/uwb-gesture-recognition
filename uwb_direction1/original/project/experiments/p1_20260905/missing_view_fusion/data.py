"""Fold-local frozen features. Hidden radars are accessed only for fit rows."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
from pipeline import digest, dump, quality_mask, split
from train import Classifier, setup

CACHE = HERE / 'cache_v2'


def pool_tokens(temporal, spatial):
    b, v, t, d = temporal.shape
    if (t, d, spatial.shape[2]) != (90, 32, 378):
        raise ValueError('expected amplitude+phase temporal90/spatial378/dim32')
    return torch.cat([temporal.reshape(b, v, 10, 9, d).mean(3).flatten(2),
                      spatial.reshape(b, v, 42, 9, d).mean(3).flatten(2)], -1).float()


def fit_scale(training_features):
    x = training_features.float()
    return x.mean(0, keepdim=True), x.std(0, unbiased=False, keepdim=True).clamp_min(.05)


def prototype_banks(training_features, labels, classes=12):
    if any(not (labels == k).any() for k in range(classes)):
        raise ValueError('each prototype class requires training observations')
    return torch.stack([training_features[labels == k].mean(0) for k in range(classes)], 1)


def validate_split(tr, va, te, y, subjects, fold):
    sets = list(map(lambda a: set(a.tolist()), (tr, va, te)))
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set(subjects[te]) == {fold}
    assert fold not in subjects[tr] and fold not in subjects[va]
    assert all(quality_mask(y[a], subjects[a]).all() for a in (tr, va, te))


def source_paths(fold):
    stem = BASE / f'results_clean/sv_R_label_global_s2026_h{fold}'
    return stem.with_suffix('.pt'), Path(str(stem) + '_predictions.npz')


def raw_identity(paths):
    return {str(p): digest(p) for p in paths}


def extraction_identity(fold):
    checkpoint, predictions = source_paths(fold)
    files = [Path(__file__), checkpoint, predictions, BASE/'pipeline.py', BASE/'train.py',
             BASE/'vendor/config.py', BASE/'vendor/uwb_osr/model.py',
             BASE/'prepared/y.npy', BASE/'prepared/subj.npy', BASE/'label_global.py',
             BASE/f'label_smoothing_results/global_matrix_h{fold}.npz']
    return {str(p): digest(p) for p in files} | raw_identity(
        [BASE/'prepared/off.npy', BASE/'prepared/only.npy'])


def extract(fold, device='cuda'):
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f'fold{fold}.npz'
    meta_path = dest.with_suffix('.json')
    identity = extraction_identity(fold)
    y = np.load(BASE/'prepared/y.npy')
    subjects = np.load(BASE/'prepared/subj.npy')
    tr, va, te = split(y, subjects, fold, 2026)
    validate_split(tr, va, te, y, subjects, fold)
    checkpoint, predictions = source_paths(fold)
    with np.load(predictions) as ref:
        for key, ids in [('train_ids', tr), ('val_ids', va), ('ids', te)]:
            np.testing.assert_array_equal(ref[key], ids)
        reference_logits = ref['logits'].copy()
    if dest.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['identity'] != identity or meta['cache_sha256'] != digest(dest):
            raise ValueError(f'cache provenance changed: {dest}; use a new cache version')
        with np.load(dest) as a:
            result = {k: a[k].copy() for k in a.files}
        for key, ids in [('train_ids', tr), ('val_ids', va), ('test_ids', te)]:
            np.testing.assert_array_equal(result[key], ids)
        assert result['val_x'].ndim == result['test_x'].ndim == 2
        return result, meta
    start = time.time()
    setup(2026)
    model = Classifier([2], 'concat', 'mean').to(device).eval().requires_grad_(False)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert ck['fold'] == fold
    for key, expected in {'views': [2], 'seed': 2026, 'mode': 'concat', 'readout': 'mean',
                          'tag': 'sv_R_label_global', 'teacher': '', 'aux': 'none',
                          'soft_target': 'training_onehot_global_projection'}.items():
        assert ck['args'][key] == expected, (key, ck['args'][key])
    assert ck['args']['global_projection_matrix_sha'] == identity[
        str(BASE/f'label_smoothing_results/global_matrix_h{fold}.npz')]
    model.load_state_dict(ck['state'])
    del ck
    assert model.encoder.temporal.xmv == model.encoder.spatial.xmv == 0
    amplitude = np.load(BASE/'prepared/off.npy', mmap_mode='r')
    phase = np.load(BASE/'prepared/only.npy', mmap_mode='r')
    result = {'train_ids': tr, 'val_ids': va, 'test_ids': te,
              'train_y': y[tr], 'val_y': y[va], 'test_y': y[te], 'train_subjects': subjects[tr]}
    with torch.inference_mode():
        for role, ids, views in [('train', tr, (0, 1, 2)), ('val', va, (2,)), ('test', te, (2,))]:
            features = np.empty((len(ids), len(views), 1664), dtype=np.float16)
            logits = np.empty((len(ids), 12), dtype=np.float32)
            for v_index, view in enumerate(views):
                for begin in range(0, len(ids), 32):
                    j = ids[begin:begin+32]
                    xx = np.concatenate([np.asarray(amplitude[j, view:view+1]),
                                         np.asarray(phase[j, view:view+1])], -1)
                    xx = torch.tensor(xx, device=device)
                    with torch.autocast(device_type=torch.device(device).type,
                                        dtype=torch.bfloat16, enabled=device != 'cpu'):
                        _, t, s = model.encoder(xx)
                        features[begin:begin+len(j), v_index] = pool_tokens(t, s)[:, 0].cpu().numpy()
                        if view == 2:
                            z = model.head(torch.cat([t.flatten(2), s.flatten(2)], -1)[:, 0])
                            logits[begin:begin+len(j)] = z.float().cpu().numpy()
                print(f'EXTRACT fold={fold} role={role} view={view} rows={len(ids)}', flush=True)
            result[role+'_x'] = features if role == 'train' else features[:, 0]
            result[role+'_base'] = logits
    # Frozen reference used the same 32-row bfloat16 inference path.
    np.testing.assert_allclose(result['test_base'], reference_logits, rtol=.015, atol=.04)
    np.testing.assert_array_equal(result['test_base'].argmax(1), reference_logits.argmax(1))
    temp = dest.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **result)
    temp.replace(dest)
    meta = {'fold': fold, 'identity': identity, 'cache_sha256': digest(dest),
            'source_input': 'Right amplitude+phase', 'encoder_training_views': ['Right'],
            'hidden_view_access': 'outer train rows only', 'dimension': 1664,
            'base_accuracy': float((result['test_base'].argmax(1) == y[te]).mean()),
            'max_reference_logit_error': float(np.max(np.abs(result['test_base'] - reference_logits))),
            'seconds': time.time()-start,
            'raw_arrays': {str(p): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                           for p in [BASE/'prepared/off.npy', BASE/'prepared/only.npy']}}
    dump(meta_path, meta)
    del model
    gc.collect()
    if device != 'cpu':
        torch.cuda.empty_cache()
    print(f'CACHE_COMPLETE fold={fold} base={meta["base_accuracy"]:.6f} seconds={meta["seconds"]:.1f}', flush=True)
    return result, meta


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, nargs='+', default=list(range(1, 9)))
    args = ap.parse_args()
    for fold in args.folds:
        extract(fold)
