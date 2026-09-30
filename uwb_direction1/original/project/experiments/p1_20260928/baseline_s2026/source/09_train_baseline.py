"""Retrain the existing six-source-user control and export frozen features.

The legacy trainer is imported unchanged; no historical result is overwritten.
Source fingerprints and all model states are retained in the new run directory.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

OLD = Path('D:/uwb_gestures/experiments/p1_20260905')
OUTPUT = Path('D:/uwb_gestures/experiments/p1_20260928')
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_new_routes_original_trainer', OLD/'cross_user_meta/trainer.py')
T = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = T
spec.loader.exec_module(T)


def source_parts(held):
    return T.protocol.open_fold(held)


def extract_batch(model, x):
    """Return the exact 256-dimensional penultimate activation and logits."""
    values = []
    hook = model.head.net[4].register_forward_hook(lambda _, _input, output: values.append(output))
    try:
        logits = model(x)[0]
    finally:
        hook.remove()
    if len(values) != 1 or values[0].shape != (len(x), 256):
        raise ValueError('Unexpected classifier feature topology')
    return values[0], logits


def atomic_npz(path, **values):
    path = Path(path)
    with path.with_suffix(path.suffix+'.tmp').open('wb') as f:
        np.savez_compressed(f, **values)
    path.with_suffix(path.suffix+'.tmp').replace(path)


@torch.no_grad()
def export_features(out, held, reader, labels, subjects, parts, result):
    directory = out/f'fold{held}'/'warmup'
    feature_path = directory/'features.npz'
    meta_path = directory/'features.json'
    identity = {'checkpoint': result['files']['right.pt'],
                'extractor': T.protocol.digest(Path(__file__)),
                'model_identity': result['identity']}
    if meta_path.exists():
        saved = json.loads(meta_path.read_text(encoding='utf-8'))
        if saved['identity'] != identity or saved['sha256'] != T.protocol.digest(feature_path):
            raise ValueError('Frozen feature identity/hash differs')
        print(f'FEATURES_VERIFIED fold={held}', flush=True)
        return
    model = T.models.Classifier([2], 'concat', 'mean').cuda().eval()
    model.load_state_dict(torch.load(directory/'right.pt', map_location='cpu', weights_only=True), strict=True)
    ids_list, features_list, logits_list, roles_list = [], [], [], []
    for role_index, role in enumerate(('train', 'val', 'test')):
        ids = parts[role]
        for start in range(0, len(ids), 32):
            rows = ids[start:start+32]
            x = reader.load_ids(rows, role, 'R')
            with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16):
                features, logits = extract_batch(model, x)
            ids_list.append(rows)
            features_list.append(features.float().cpu().numpy())
            logits_list.append(logits.float().cpu().numpy())
            roles_list.append(np.full(len(rows), role_index, dtype=np.int8))
    ids = np.concatenate(ids_list)
    features = np.concatenate(features_list)
    logits = np.concatenate(logits_list)
    roles = np.concatenate(roles_list)
    if not (np.isfinite(features).all() and np.isfinite(logits).all()):
        raise ValueError('Non-finite exported features or logits')
    reference = np.load(directory/'predictions.npz')
    np.testing.assert_array_equal(logits[roles == 2], reference['logits'])
    np.testing.assert_array_equal(logits[roles == 1], reference['val_logits'])
    atomic_npz(feature_path, ids=ids, features=features, logits=logits,
               y=labels[ids], subjects=subjects[ids], role=roles)
    T.protocol.dump(meta_path, {'identity': identity, 'sha256': T.protocol.digest(feature_path),
                              'feature_shape': list(features.shape), 'logit_replay': 'bitwise_equal',
                              'roles': {'train': 0, 'val': 1, 'test': 2}})
    print(f'FEATURES_COMPLETE fold={held} shape={features.shape}', flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUTPUT/'baseline_s2026')
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(1, 9)))
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original = json.loads((OLD/'cross_user_meta/runs_s2026/manifest.json').read_text(encoding='utf-8'))
    source = T.source_identity()
    if source['code'] != original['source']['code'] or source['data'] != original['source']['data']:
        raise ValueError('Historical source/data fingerprint differs; inspect before training')
    source['code'][str(Path(__file__).resolve())] = T.protocol.digest(Path(__file__))
    source['plan_sha256'] = T.protocol.digest(HERE.parents[1]/'docs/superpowers/plans/2026-09-28-uwb-new-routes.md')
    config = dict(original['config'])
    config['seed'] = args.seed
    if args.smoke:
        config.update(warmup_epochs=1, warmup_min=1, smoke_steps=3, smoke=True)
    T.freeze_manifest(args.out, source, config)
    held = None
    try:
        for held in args.folds:
            T.protocol.dump(args.out/'queue_status.json', {'state': 'running', 'fold': held,
                                                        'folds': args.folds, 'smoke': args.smoke})
            reader, labels, subjects, parts = source_parts(held)
            # Verify and copy immutable source-only matrices into the new run.
            for suffix in ('.npz', '.json'):
                src = OLD/'cross_user_meta/runs_s2026/matrices'/f'fold{held}{suffix}'
                dst = args.out/'matrices'/src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)
            matrix, matrix_meta = T.smoothing.prepare_matrix(held, args.out/'matrices', source['data'])
            T.protocol.dump(args.out/f'fold{held}'/'split.json', {
                'ids': {k: v.tolist() for k, v in parts.items()},
                'training_subjects': np.unique(subjects[parts['train']]).tolist(),
                'val': held % 8 + 1, 'test': held})
            result = T.run_arm(args.out, held, 'warmup', None, reader, labels, subjects,
                               parts, matrix, matrix_meta, source, config)
            export_features(args.out, held, reader, labels, subjects, parts, result)
    except BaseException as error:
        T.protocol.dump(args.out/'queue_status.json', {'state': 'failed', 'fold': held,
                        'error_type': type(error).__name__, 'error': str(error)})
        raise
    T.protocol.dump(args.out/'queue_status.json', {'state': 'complete', 'folds': args.folds, 'smoke': args.smoke})
    print('BASELINE_QUEUE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
