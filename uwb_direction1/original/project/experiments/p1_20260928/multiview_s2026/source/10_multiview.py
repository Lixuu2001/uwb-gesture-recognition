"""Ordinary missing-view baseline; seven real subsets in fixed L/T/R order.

Direct inference accepts only observed radar inputs. Full three-view caching is
reserved for offline evaluation/export and is not a deployment latency estimate.
"""
import argparse
import copy
from contextlib import nullcontext
import gc
import hashlib
import json
from numbers import Integral
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from train_baseline import HERE, OUTPUT, T, atomic_npz, source_parts

SUBSETS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
SUBSET_MASKS = np.array([[int(v in subset) for v in range(3)] for subset in SUBSETS], dtype=np.int8)
ROLES = ('train', 'val', 'test')


def physical_indices(indices):
    try:
        values = tuple(indices)
    except TypeError as error:
        raise ValueError('Physical indices must be an ordered nonempty subset of L0/T1/R2') from error
    if any(not isinstance(v, Integral) or isinstance(v, (bool, np.bool_)) for v in values) or values not in SUBSETS:
        raise ValueError('Physical indices must be an ordered nonempty subset of L0/T1/R2')
    return values


class SubsetModel(nn.Module):
    def __init__(self, right=None):
        super().__init__()
        right = T.models.Classifier([2], 'concat', 'mean') if right is None else right
        self.encoder, self.projection = right.encoder, right.head.net[:-1]
        if self.encoder.temporal.xmv or self.encoder.spatial.xmv:
            raise ValueError('The shared encoder must have no cross-view mixing')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026)
            self.fuser = nn.Sequential(nn.Linear(771, 256), nn.GELU(), nn.Dropout(.1), nn.Linear(256, 12))

    def encode(self, observed):
        if observed.ndim != 4 or observed.shape[0] < 1 or observed.shape[1] not in (1, 2, 3) or tuple(observed.shape[2:]) != (90, 378):
            raise ValueError('Expected observed raw inputs [B,V,90,378], B>0 and 1<=V<=3')
        if not observed.is_floating_point() or not torch.isfinite(observed).all():
            raise ValueError('Observed radar input must be finite floating point')
        # Keep the same kernel batch shape for a view alone or inside a cache.
        # The original Right classifier uses tokens, not its unused view embedding.
        features = []
        for view in range(observed.shape[1]):
            _, temporal, spatial = self.encoder(observed[:, view:view+1].contiguous())
            tokens = torch.cat((temporal.flatten(2), spatial.flatten(2)), -1)
            features.append(self.projection(tokens))
        return torch.cat(features, 1)

    def fuse(self, observed_features, indices):
        indices = physical_indices(indices)
        if observed_features.ndim != 3 or observed_features.shape[0] < 1 or tuple(observed_features.shape[1:]) != (len(indices), 256):
            raise ValueError('Expected only observed features [B,len(indices),256]')
        if not torch.isfinite(observed_features).all():
            raise ValueError('Nonfinite observed features')
        slots = observed_features.new_zeros((len(observed_features), 3, 256))
        slots[:, indices] = observed_features
        flags = observed_features.new_zeros((len(observed_features), 3))
        flags[:, indices] = 1
        return self.fuser(torch.cat((slots.flatten(1), flags), -1))

    def forward(self, observed, indices):
        indices = physical_indices(indices)
        if observed.ndim != 4 or observed.shape[1] != len(indices):
            raise ValueError('Raw inputs must contain exactly the requested physical views')
        return self.fuse(self.encode(observed), indices)


class SubsetData(T.protocol.SubjectData):
    """Guard IDs/roles before indexing; never materialize unrequested views."""
    def __init__(self, amplitude, phase, partitions):
        super().__init__(amplitude, phase, partitions)
        for ids in partitions.values():
            ids = np.asarray(ids)
            if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or not len(ids) or len(np.unique(ids)) != len(ids) or (ids < 0).any() or (ids >= len(amplitude)).any():
                raise ValueError('Partitions require unique nonempty in-range integer IDs')

    def load_ids(self, ids, role, indices, device='cuda'):
        ids, indices = np.asarray(ids), physical_indices(indices)
        if role not in self.partitions or ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or not len(ids):
            raise ValueError('Require a known role and nonempty integer IDs')
        if not np.isin(ids, self.partitions[role]).all():
            raise ValueError('Requested IDs leave the allowed whole-user partition')
        selection = (ids[:, None], np.asarray(indices)[None, :])
        observed = np.concatenate((self.amplitude[selection], self.phase[selection]), -1)
        if not np.isfinite(observed).all():
            raise ValueError('Nonfinite observed raw radar input')
        dtype = torch.float16 if torch.device(device).type == 'cuda' else torch.float32
        return torch.from_numpy(observed).to(device=device, dtype=dtype)


def validate_parts(labels, subjects, parts, held):
    expected = T.protocol.subject_split(labels, subjects, held)
    if set(parts) != set(ROLES) or any(not np.array_equal(parts[k], expected[k]) for k in ROLES):
        raise ValueError('Fold differs from the six-source/whole-validation/whole-test protocol')
    if sum(map(len, parts.values())) != 9300:
        raise ValueError('Expected exactly 9300 quality-masked paired rows')


class SubsetSampler(T.state.ShuffleSampler):
    def __init__(self, ids, held):
        super().__init__(ids, 2026 + held)
        self.subset_rng = np.random.default_rng(20260 + held)
        self.identity = {'ids': T.protocol.array_hash(self.ids), 'shuffle_seed': 2026 + held, 'subset_seed': 20260 + held}

    def next_subset(self):
        return SUBSETS[int(self.subset_rng.integers(len(SUBSETS)))]

    def state_dict(self):
        return copy.deepcopy({**super().state_dict(), 'subset_rng': self.subset_rng.bit_generator.state, 'identity': self.identity})

    def load_state_dict(self, saved):
        if saved['identity'] != self.identity:
            raise ValueError('Sampler partition or random seeds changed')
        super().load_state_dict(saved)
        self.subset_rng.bit_generator.state = copy.deepcopy(saved['subset_rng'])


def precision(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


def train_epoch(model, optimizer, sampler, reader, labels, config):
    model.train()
    device = next(model.parameters()).device
    ids, trace, total, rows_seen, steps = sampler.next_epoch(), hashlib.sha256(), 0., 0, 0
    counts = np.zeros(7, dtype=int)
    for start in range(0, len(ids), config['batch']):
        if config['smoke_steps'] and steps >= config['smoke_steps']:
            break
        rows, subset = ids[start:start+config['batch']], sampler.next_subset()
        trace.update(rows.tobytes()); trace.update(bytes(subset))
        counts[SUBSETS.index(subset)] += 1
        optimizer.zero_grad(set_to_none=True)
        value = 0.
        for offset in range(0, len(rows), config['microbatch']):
            micro = rows[offset:offset+config['microbatch']]
            raw = reader.load_ids(micro, 'train', subset, device=device)
            y = torch.as_tensor(labels[micro], device=device).long()
            with sdpa_kernel(SDPBackend.MATH), precision(device):
                loss = F.cross_entropy(model(raw, subset), y) * (len(micro) / len(rows))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite subset training loss')
            loss.backward(); value += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
        optimizer.step()
        total += value * len(rows); rows_seen += len(rows); steps += 1
    return {'train_loss': total / rows_seen, 'source_rows': rows_seen, 'steps': steps,
            'episode_hash': trace.hexdigest(), 'subset_batch_counts': counts.tolist()}


def save_epoch(directory, model, optimizer, sampler, identity, selection, history, epoch):
    T.save_epoch(directory/'progress.pt', directory, model, optimizer, sampler, identity, selection, history, epoch)


def restore_progress(directory, model, optimizer, sampler, identity):
    info = T.state.load_progress(directory/'progress.pt', model, optimizer, sampler, identity)
    best_path = directory/'best.pt'
    if not best_path.exists() or T.protocol.digest(best_path) != info['best_file_sha256']:
        T.state.atomic_save(best_path, info['best_checkpoint'])
    return T.state.Selection.from_state(info['selection']), info['history'], info['epoch']


@torch.no_grad()
def evaluate(model, reader, labels, ids, role, batch=16):
    """Offline full-view cache; each prediction uses only its actual subset."""
    model.eval()
    device = next(model.parameters()).device
    feature_batches, logit_batches = [], []
    for start in range(0, len(ids), batch):
        raw = reader.load_ids(ids[start:start+batch], role, (0, 1, 2), device=device)
        with sdpa_kernel(SDPBackend.MATH), precision(device):
            features = model.encode(raw)
            logits = torch.stack([model.fuse(features[:, subset], subset) for subset in SUBSETS], 1)
        feature_batches.append(features.float().cpu().numpy())
        logit_batches.append(logits.float().cpu().numpy())
    features, logits = np.concatenate(feature_batches), np.concatenate(logit_batches)
    if not np.isfinite(features).all() or not np.isfinite(logits).all():
        raise ValueError('Nonfinite offline features or predictions')
    y = torch.from_numpy(labels[ids]).long()
    ces = np.array([float(F.cross_entropy(torch.from_numpy(logits[:, i]), y)) for i in range(7)])
    return features, logits, ces


def export_predictions(directory, model, reader, labels, subjects, parts, best, batch=16):
    features, logits, roles = [], [], []
    for code, role in enumerate(ROLES):
        f, z, ces = evaluate(model, reader, labels, parts[role], role, batch)
        if role == 'val':
            if not np.array_equal(z, np.asarray(best['val_logits'])) or abs(float(ces.mean()) - best['val_ce']) > 1e-7:
                raise ValueError('Selected validation logits or mean CE replay differs')
            val_logits, val_ces = z, ces
        features.append(f); logits.append(z); roles.append(np.full(len(z), code, dtype=np.int8))
    ids = np.concatenate([parts[k] for k in ROLES])
    atomic_npz(directory/'predictions.npz', ids=ids, y=labels[ids], subjects=subjects[ids],
               role=np.concatenate(roles), features=np.concatenate(features), logits=np.concatenate(logits),
               subset_masks=SUBSET_MASKS, val_logits=val_logits, val_y=labels[parts['val']],
               **{role+'_ids': parts[role] for role in ROLES})
    return {'val_ce_by_subset': val_ces.tolist(), 'validation_replay': 'bitwise_equal',
            'test_accuracy_by_subset': (logits[2].argmax(2) == labels[parts['test'], None]).mean(0).tolist()}


def verify_initial(baseline, held, parts, source):
    """Accept only the completed new six-user Right baseline from this fold."""
    directory = baseline/f'fold{held}'/'warmup'
    result = json.loads((directory/'result.json').read_text(encoding='utf-8'))
    manifest = json.loads((baseline/'manifest.json').read_text(encoding='utf-8'))
    identity = result['identity']
    required = {'seed': 2026, 'smoke': False, 'warmup_epochs': 40, 'warmup_min': 12,
                'warmup_patience': 8, 'warmup_batch': 128, 'warmup_microbatch': 32,
                'warmup_lr': 1e-4, 'weight_decay': 1e-4}
    if identity['fold'] != held or identity['arm'] != 'warmup' or identity['partitions'] != {k: T.protocol.array_hash(v) for k, v in parts.items()}:
        raise ValueError('Initial Right checkpoint belongs to another fold or protocol')
    if any(identity['config'].get(k) != v for k, v in required.items()) or manifest != {'source': identity['source'], 'config': identity['config']}:
        raise ValueError('Initial baseline is incomplete, a smoke run, or has different configuration')
    if str(HERE/'train_baseline.py') not in identity['source']['code'] or any(identity['source'].get(k) != source.get(k) for k in ('code', 'data', 'plan_sha256')):
        raise ValueError('Initial checkpoint does not match the new baseline source/data')
    if not {'right.pt', 'best.pt', 'progress.pt', 'predictions.npz'}.issubset(result['files']):
        raise ValueError('Initial baseline has no complete artifact inventory')
    for name, digest in result['files'].items():
        if T.protocol.digest(directory/name) != digest:
            raise ValueError('Initial baseline artifact hash differs: '+name)
    return {'right_path': str((directory/'right.pt').resolve()), 'right_sha256': result['files']['right.pt'],
            'result_sha256': T.protocol.digest(directory/'result.json'), 'manifest_sha256': T.protocol.digest(baseline/'manifest.json'),
            'model_identity': identity, 'selected_epoch': result['best_epoch'],
            'reused_training_seconds': sum(row['seconds'] for row in result['history'])}


def configuration(smoke=False):
    return {'seed': 2026, 'fusion_seed': 2026, 'shuffle_seed': '2026+fold', 'subset_seed': '20260+fold',
            'batch': 128, 'microbatch': 16, 'lr': 1e-4, 'weight_decay': 1e-4, 'eval_batch': 16,
            'epochs': 1 if smoke else 40, 'min_epochs': 1 if smoke else 12, 'patience': 8,
            'smoke': smoke, 'smoke_steps': 3 if smoke else 0, 'criterion': 'cross_entropy only',
            'selection': 'earliest minimum mean seven-subset validation CE, including epoch 0',
            'precision': 'float32 weights; float16 inputs; bfloat16 autocast', 'attention': 'MATH',
            'subsets': [list(v) for v in SUBSETS], 'view_order': ['Left', 'Top', 'Right'],
            'subset_sampling': 'one uniform nonempty subset per effective batch',
            'export': 'offline all-view cache; not a deployment latency measurement'}


def validate_output_paths(out, baseline, smoke):
    out, baseline, production = out.resolve(), baseline.resolve(), (OUTPUT/'multiview_s2026').resolve()
    if out == baseline or baseline in out.parents or out in baseline.parents:
        raise ValueError('Multiview output must be separate from the baseline directory')
    if smoke and (out == production or production in out.parents):
        raise ValueError('--smoke requires an independent --out outside multiview_s2026')


def freeze_sources(out, source, config):
    T.freeze_manifest(out, source, config)
    for index, (path, digest) in enumerate(source['code'].items()):
        if T.protocol.digest(out/'source'/f'{index:02d}_{Path(path).name}') != digest:
            raise ValueError('Frozen source snapshot changed: '+path)


def write_epoch_status(directory, held, history):
    """Recover audit files from the authoritative progress history."""
    selection = T.state.Selection(history[0]['val_ce'])
    for row in history[1:]:
        selection.observe(row['epoch'], row['val_ce'])
        status = {'state': 'running', 'fold': held, 'epoch': row['epoch'], 'best_epoch': selection.best_epoch,
                  'best_val_ce': selection.best_ce, 'last': row}
        path = directory/'epochs'/f'{row["epoch"]:02d}.json'
        if not path.exists():
            T.protocol.dump(path, status)
    if len(history) > 1:
        T.protocol.dump(directory/'status.json', status)


def fit_model(directory, model, reader, labels, parts, held, identity, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    sampler = SubsetSampler(parts['train'], held)
    gpu = next(model.parameters()).device.type == 'cuda'

    def candidate(epoch, logits, ce):
        return {'state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'epoch': epoch, 'val_ce': ce, 'val_logits': torch.from_numpy(logits)}

    if (directory/'progress.pt').exists():
        selection, history, epoch = restore_progress(directory, model, optimizer, sampler, identity)
        write_epoch_status(directory, held, history)
        print(f'RESUME fold={held} epoch={epoch}', flush=True)
    else:
        T.seed_all(config['seed'])
        started = time.perf_counter()
        _, logits, ces = evaluate(model, reader, labels, parts['val'], 'val', config['eval_batch'])
        selection, epoch = T.state.Selection(float(ces.mean())), 0
        history = [{'epoch': 0, 'val_ce': selection.best_ce, 'val_ce_by_subset': ces.tolist(),
                    'train_loss': None, 'steps': 0, 'seconds': time.perf_counter()-started, 'peak_allocated_bytes': 0}]
        initial = candidate(0, logits, selection.best_ce)
        T.state.atomic_save(directory/'initial.pt', initial)
        T.state.atomic_save(directory/'best.pt', initial)
        save_epoch(directory, model, optimizer, sampler, identity, selection, history, 0)
    while epoch < config['epochs'] and not selection.should_stop(epoch, config['min_epochs'], config['patience']):
        started = time.perf_counter()
        if gpu:
            torch.cuda.reset_peak_memory_stats()
        epoch += 1
        row = train_epoch(model, optimizer, sampler, reader, labels, config)
        _, logits, ces = evaluate(model, reader, labels, parts['val'], 'val', config['eval_batch'])
        ce = float(ces.mean())
        row.update(epoch=epoch, val_ce=ce, val_ce_by_subset=ces.tolist(), seconds=time.perf_counter()-started,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated() if gpu else 0)
        history.append(row)
        if selection.observe(epoch, ce):
            T.state.atomic_save(directory/'best.pt', candidate(epoch, logits, ce))
        save_epoch(directory, model, optimizer, sampler, identity, selection, history, epoch)
        write_epoch_status(directory, held, history)
        print(f'EPOCH fold={held} epoch={epoch} mean_val_ce={ce:.6f} best={selection.best_epoch} seconds={row["seconds"]:.1f}', flush=True)
    reason = 'patience' if selection.should_stop(epoch, config['min_epochs'], config['patience']) else 'max_epochs'
    return selection, history, reason


def run_fold(out, baseline, held, source, baseline_source, config):
    original_reader, labels, subjects, parts = source_parts(held)
    validate_parts(labels, subjects, parts, held)
    reader = SubsetData(original_reader.amplitude, original_reader.phase, parts)
    initial = verify_initial(baseline, held, parts, baseline_source)
    directory = out/f'fold{held}'
    directory.mkdir(parents=True, exist_ok=True)
    T.seed_all(config['seed'])
    right = T.models.Classifier([2], 'concat', 'mean')
    right.load_state_dict(torch.load(initial['right_path'], map_location='cpu', weights_only=True), strict=True)
    model = SubsetModel(right)
    del right
    identity = {'config': config, 'source': source, 'fold': held, 'arm': 'multiview', 'initial_checkpoint': initial,
                'partitions': {k: T.protocol.array_hash(v) for k, v in parts.items()},
                'initial_state_sha256': T.state.state_hash(model.state_dict())}
    completed = T.verified_completed(directory, identity)
    if completed is not None:
        return completed
    T.protocol.dump(directory/'split.json', {'ids': {k: v.tolist() for k, v in parts.items()},
                    'training_subjects': np.unique(subjects[parts['train']]).tolist(), 'val': held % 8 + 1, 'test': held})
    model = model.cuda()
    selection, history, reason = fit_model(directory, model, reader, labels, parts, held, identity, config)
    best = torch.load(directory/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(best['state'], strict=True)
    started = time.perf_counter()
    scores = export_predictions(directory, model, reader, labels, subjects, parts, best, config['eval_batch'])
    files = ['initial.pt', 'best.pt', 'progress.pt', 'predictions.npz', 'split.json']
    files += [str(p.relative_to(directory)).replace('\\', '/') for p in sorted((directory/'epochs').glob('*.json'))]
    result = {'identity': identity, 'best_epoch': selection.best_epoch, 'best_val_ce': selection.best_ce,
              'selected_state_sha256': T.state.state_hash(best['state']), 'history': history, 'stop_reason': reason,
              'training_subjects': np.unique(subjects[parts['train']]).tolist(), 'validation_subject': held % 8 + 1, 'test_subject': held,
              'new_training_seconds': sum(row['seconds'] for row in history), 'offline_export_seconds': time.perf_counter()-started,
              'reused_right_training_seconds': initial['reused_training_seconds'],
              'peak_allocated_bytes': max(row['peak_allocated_bytes'] for row in history),
              'feature_shape': [9300, 3, 256], 'logit_shape': [9300, 7, 12], 'roles': dict(zip(ROLES, range(3))),
              'interpretation': 'Ordinary subset-training baseline; route decisions require all eight folds',
              **scores, 'files': {name: T.protocol.digest(directory/name) for name in files}}
    T.protocol.dump(directory/'result.json', result)
    T.protocol.dump(directory/'status.json', {'state': 'complete', 'fold': held, 'best_epoch': selection.best_epoch,
                                             'best_val_ce': selection.best_ce})
    print(f'COMPLETE fold={held} selected_epoch={selection.best_epoch}', flush=True)
    del best, model
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT/'multiview_s2026')
    parser.add_argument('--baseline', type=Path, default=OUTPUT/'baseline_s2026')
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(1, 9)))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    try:
        validate_output_paths(args.out, args.baseline, args.smoke)
        if len(set(args.folds)) != len(args.folds) or any(h not in range(1, 9) for h in args.folds):
            raise ValueError('--folds must contain unique users from 1..8')
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plans = HERE.parents[1]/'docs/superpowers/plans'
    source = T.source_identity()
    source['code'][str(HERE/'train_baseline.py')] = T.protocol.digest(HERE/'train_baseline.py')
    source['plan_sha256'] = T.protocol.digest(plans/'2026-09-28-uwb-new-routes.md')
    baseline_source = copy.deepcopy(source)
    for path in (Path(__file__).resolve(), HERE/'test_multiview.py', plans/'2026-09-28-uwb-new-routes.md', plans/'2026-09-28-uwb-amendment-001.md'):
        source['code'][str(path)] = T.protocol.digest(path)
    source['baseline_manifest_sha256'] = T.protocol.digest(args.baseline/'manifest.json')
    source['baseline_directory'] = str(args.baseline.resolve())
    config, held = configuration(args.smoke), None
    freeze_sources(args.out, source, config)
    try:
        for held in args.folds:
            T.protocol.dump(args.out/'queue_status.json', {'state': 'running', 'fold': held, 'folds': args.folds, 'smoke': args.smoke})
            run_fold(args.out, args.baseline, held, source, baseline_source, config)
    except BaseException as error:
        T.protocol.dump(args.out/'queue_status.json', {'state': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        'fold': held, 'folds': args.folds, 'error_type': type(error).__name__, 'error': str(error)})
        raise
    T.protocol.dump(args.out/'queue_status.json', {'state': 'complete', 'folds': args.folds, 'smoke': args.smoke,
                    'all_eight_folds_in_this_invocation': set(args.folds) == set(range(1, 9))})
    print('MULTIVIEW_QUEUE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
