"""Frozen teacher feature targets for method A, with their source ledger.

Three sources are kept apart on purpose:
  multi          three-view trained teacher, all three branches read
  target_single  the same three-view trained teacher, Right branch only
  teacher_single a Right-only trained teacher, Right branch only
Only `teacher_single` is free of extra training views; `target_single`
isolates what the extra *target* views add on top of shared training.
"""
import argparse
import gc

import numpy as np
import torch

import protocol as P
from protocol import BASE, CACHE, RESULTS_CLEAN, digest, hash_ids
from model_adapter import teacher_compact

SOURCES = {
    'multi': {'checkpoint': 'teacher_concat_s2026_h{fold}.pt', 'views': [0, 1, 2],
              'target_views': [0, 1, 2], 'mode': 'off', 'readout': 'concat'},
    'target_single': {'checkpoint': 'teacher_concat_s2026_h{fold}.pt', 'views': [0, 1, 2],
                      'target_views': [2], 'mode': 'off', 'readout': 'concat'},
    'teacher_single': {'checkpoint': 'sv_R_off_s2026_h{fold}.pt', 'views': [2],
                       'target_views': [2], 'mode': 'off', 'readout': 'mean'},
}


def _teacher_train_ids(stem):
    for suffix in ('_all_logits.npz', '_predictions.npz'):
        path = RESULTS_CLEAN / (stem + suffix)
        if path.exists():
            with np.load(path) as a:
                return a['train_ids']
    raise FileNotFoundError(f'no id record for {stem}')


def _extract(fold, source):
    """Pooled 992-dim code per view from a frozen checkpoint; cached on disk."""
    spec = SOURCES[source]
    stem = spec['checkpoint'].format(fold=fold)[:-3]
    checkpoint = RESULTS_CLEAN / (stem + '.pt')
    sha = digest(checkpoint)
    if source in ('multi', 'target_single'):
        # Already computed by the earlier reconstruction study; verified, not re-run.
        cached = BASE / f'reconstruction_results/features_h{fold}.npz'
        with np.load(cached) as a:
            assert str(a['teacher_sha']) == sha, 'cached teacher features are from another checkpoint'
            features = a['features'].astype('float32')
        return features, sha, stem
    dest = CACHE / f'teacher_single_h{fold}.npz'
    if dest.exists():
        with np.load(dest) as a:
            assert str(a['teacher_sha']) == sha
            return a['features'].astype('float32'), sha, stem
    from train import Classifier, setup
    y, subj = P.labels()
    setup(P.SPLIT_SEED)
    model = Classifier([P.DEPLOY_VIEW], spec['mode'], spec['readout']).cuda().eval()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['fold'] == fold and state['args']['seed'] == P.SPLIT_SEED
    model.load_state_dict(state['state']); del state
    model.requires_grad_(False)
    assert model.encoder.temporal.xmv == 0 and model.encoder.spatial.xmv == 0
    x = np.load(BASE / 'prepared/off.npy', mmap_mode='r')
    tr, va, te = P.fold_split(fold, y, subj)
    ids = np.sort(np.concatenate([tr, va, te]))
    out = np.zeros((len(y), 1, 992), 'float16')
    with torch.no_grad():
        for start in range(0, len(ids), 64):
            j = ids[start:start + 64]
            xx = torch.tensor(np.array(x[j, P.DEPLOY_VIEW:P.DEPLOY_VIEW + 1], copy=True), device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, t, s = model.encoder(xx)
                out[j] = teacher_compact(t, s).float().cpu().numpy()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, features=out, teacher_sha=sha, ids=ids)
    del model; gc.collect(); torch.cuda.empty_cache()
    return out.astype('float32'), sha, stem


def relation_targets(fold, source, train_ids, device='cuda'):
    """Normalised relation code plus the provenance of where it came from.

    Per-view channel standardisation is fitted on the gradient-training rows
    only, then each view is L2 normalised, views are concatenated in fixed
    slot order and the concatenation is L2 normalised again.
    """
    spec = SOURCES[source]
    features, sha, stem = _extract(fold, source)
    features = torch.tensor(features, dtype=torch.float32)
    features = features[:, [spec['views'].index(v) for v in spec['target_views']]]
    fit = torch.as_tensor(np.asarray(train_ids), dtype=torch.long)
    mu = features[fit].mean(0, keepdim=True)
    sd = features[fit].std(0, unbiased=False, keepdim=True).clamp_min(0.05)
    z = torch.nn.functional.normalize((features - mu) / sd, dim=-1)
    z = torch.nn.functional.normalize(z.flatten(1), dim=-1)
    ledger = {'teacher_training_views': [P.VIEW_NAMES[v] for v in spec['views']],
              'teacher_target_views': [P.VIEW_NAMES[v] for v in spec['target_views']],
              'teacher_checkpoint_sha256': sha, 'teacher_checkpoint': stem,
              'teacher_training_ids_hash': hash_ids(_teacher_train_ids(stem)),
              'target_transform_fit_ids_hash': hash_ids(train_ids),
              'relation_source': source, 'relation_dim': int(z.shape[1])}
    return z.to(device), ledger


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, nargs='+', default=list(P.FOLDS))
    ap.add_argument('--sources', nargs='+', default=list(SOURCES))
    args = ap.parse_args()
    y, subj = P.labels()
    for fold in args.folds:
        tr, va, te = P.fold_split(fold, y, subj)
        for source in args.sources:
            z, ledger = relation_targets(fold, source, tr, device='cpu')
            norms = z[tr].norm(dim=-1)
            print(fold, source, tuple(z.shape), 'unit_norm',
                  float(norms.min()), float(norms.max()), ledger['teacher_checkpoint_sha256'][:12], flush=True)
    print('TEACHER_TARGETS_OK')
