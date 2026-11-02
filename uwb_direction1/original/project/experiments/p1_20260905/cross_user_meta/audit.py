"""Independent saved-artifact audit; imports no trainer metric/selection code.

Default execution is CPU/read-only apart from the audit report. --replay explicitly
loads only the standalone Right Classifier and uses CUDA BF16/MATH, batch 32.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ARMS = ('A0', 'A1', 'A2', 'A3')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def partitions(labels, subjects, held):
    require(labels.ndim == 1 and labels.shape == subjects.shape, 'labels/subjects shape mismatch')
    require(set(np.unique(subjects)) == set(range(1, 9)), 'expected all eight user identities')
    require(held in range(1, 9), 'invalid held user')
    valid = ~(((subjects == 2) & np.isin(labels, [10, 11])) | ((subjects == 6) & (labels == 3)))
    val = held % 8 + 1
    return {'train': np.flatnonzero(valid & ~np.isin(subjects, [held, val])),
            'val': np.flatnonzero(valid & (subjects == val)),
            'test': np.flatnonzero(valid & (subjects == held))}


def metrics(logits, labels):
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    require(x.ndim == 2 and y.shape == (len(x),) and len(x) > 0, 'invalid logits/labels shape')
    require(np.issubdtype(y.dtype, np.integer) and np.all((y >= 0) & (y < x.shape[1])), 'invalid class labels')
    require(np.isfinite(x).all(), 'nonfinite saved logits')
    pred = x.argmax(1)
    confusion = np.zeros((x.shape[1], x.shape[1]), dtype=np.int64)
    np.add.at(confusion, (y, pred), 1)
    denominator = confusion.sum(0)+confusion.sum(1)
    f1 = np.divide(2*np.diag(confusion), denominator, out=np.zeros(x.shape[1]), where=denominator != 0)
    centered = x-x.max(1, keepdims=True)
    ce = float(np.mean(np.log(np.exp(centered).sum(1))-centered[np.arange(len(y)), y]))
    return {'accuracy': float(np.mean(pred == y)), 'macro_f1': float(f1[np.unique(y)].mean()),
            'macro_f1_fixed_classes': float(f1.mean()), 'ce': ce, 'samples': len(y),
            'observed_classes': np.unique(y).tolist(), 'per_class_f1': f1.tolist(),
            'confusion': confusion.tolist()}


def audit_history(result):
    rows = result['history']
    require(bool(rows) and [r['epoch'] for r in rows] == list(range(len(rows))), 'history epochs must be consecutive from zero')
    values = [r['val_ce'] for r in rows]
    require(all(math.isfinite(x) for x in values), 'nonfinite validation CE')
    best = int(np.argmin(values))
    require(result['best_epoch'] == best, 'selected epoch is not first minimum validation CE')
    require(abs(result['best_val_ce']-values[best]) <= 1e-7, 'best validation CE differs from history')
    maximum, minimum, patience = (result[k] for k in ['max_epochs', 'min_epochs', 'patience'])
    require(maximum >= 1 and minimum >= 1 and patience >= 1, 'invalid stopping configuration')
    require(len(rows)-1 <= maximum, 'history exceeds max_epochs')
    stale = 0; current_best = values[0]; first_stop = None
    for epoch, value in enumerate(values[1:], 1):
        if value < current_best:
            current_best = value; stale = 0
        else:
            stale += 1
        require(rows[epoch]['steps'] > 0 and math.isfinite(rows[epoch]['train_loss']), 'invalid training row')
        require(rows[epoch]['seconds'] >= 0 and rows[epoch]['episode_hash'], 'invalid epoch trace/time')
        if first_stop is None and epoch >= minimum and stale >= patience:
            first_stop = epoch
    last = len(rows)-1
    require(first_stop is None or first_stop == last, 'patience required stopping at an earlier epoch')
    if result['stop_reason'] in ('patience', 'early_stop'):
        require(first_stop == last, 'premature patience stop')
    else:
        require(result['stop_reason'] == 'max_epochs' and last == maximum, 'invalid max_epochs stop')
    return {'last_epoch': last, 'best_epoch': best, 'stale': stale}


def audit_episode_prefix(results):
    arms = [a for a in ARMS if a in results]
    for i, left in enumerate(arms):
        for right in arms[i+1:]:
            for a, b in zip(results[left]['history'], results[right]['history']):
                require((a['epoch'], a['episode_hash'], a['steps']) ==
                        (b['epoch'], b['episode_hash'], b['steps']),
                        f'episode prefix differs: {left}/{right} epoch {a["epoch"]}')


def audit_sampling(result, labels, subjects, parts, saved_sampler):
    """Regenerate source-user sample IDs independently from declared seed/rules."""
    identity = result['identity']; config = identity['config']; held = identity['fold']
    train = parts['train']; warmup = identity['arm'] == 'warmup'
    if warmup:
        rng = np.random.default_rng(config['seed']+held)
        require(np.array_equal(saved_sampler['ids'], train), 'warmup sampler IDs mismatch')
    else:
        rng = np.random.default_rng(config['episode_seed']+held)
        users = np.unique(subjects[train]); per_user = config['per_user']; step = 0
        groups = {int(u): {int(c): train[(subjects[train] == u) & (labels[train] == c)]
                           for c in np.unique(labels[train[subjects[train] == u]])} for u in users}

        def sample(user):
            classes = np.array(sorted(groups[int(user)])); selected = []
            while len(selected) < per_user:
                selected.extend(rng.permutation(classes).tolist())
            return np.array([rng.choice(groups[int(user)][c]) for c in selected[:per_user]], dtype=np.int64)

        expected_identity = {'ids': array_hash(train), 'labels': array_hash(labels[train]),
                             'subjects': array_hash(subjects[train]), 'per_user': per_user}
        require(saved_sampler['identity'] == expected_identity, 'source-user sampler identity mismatch')
    for row in result['history'][1:]:
        digest = hashlib.sha256()
        if warmup:
            ids = rng.permutation(train)
            expected_steps = math.ceil(len(train)/config['warmup_batch'])
            if config['smoke_steps']: expected_steps = min(config['smoke_steps'], expected_steps)
            for start in range(0, expected_steps*config['warmup_batch'], config['warmup_batch']):
                digest.update(ids[start:start+config['warmup_batch']].tobytes())
        else:
            expected_steps = config['smoke_steps'] or math.ceil(len(train)/(6*per_user))
            counts = {}
            for _ in range(expected_steps):
                query = int(users[step % len(users)])
                support_ids = np.concatenate([sample(u) for u in users if u != query])
                query_ids = sample(query)
                digest.update(support_ids.tobytes()); digest.update(query_ids.tobytes())
                counts[str(query)] = counts.get(str(query), 0)+1; step += 1
            if 'query_subject_counts' in row:
                require(row['query_subject_counts'] == counts, 'query subject counts differ from rotation')
        require(row['steps'] == expected_steps and row['episode_hash'] == digest.hexdigest(),
                f'sampling trace differs from declared source-user protocol at epoch {row["epoch"]}')
    if not warmup:
        require(saved_sampler['step'] == step, 'sampler step differs from replay')
    require(saved_sampler['rng'] == rng.bit_generator.state, 'sampler RNG differs from independent replay')


class HashVerifier:
    def __init__(self):
        self.cache = {}

    def verify(self, path, expected):
        path = Path(path)
        stat = path.stat(); key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if key not in self.cache:
            self.cache[key] = file_hash(path)
        require(self.cache[key] == expected, f'file hash mismatch: {path}')


def audit_matrix(root, held, identity, labels, subjects, parts, hashes):
    path = Path(root)/'matrices'/f'fold{held}.npz'
    meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    hashes.verify(path, identity['matrix_sha256'])
    require(meta['matrix_sha256'] == identity['matrix_sha256'], 'matrix metadata hash mismatch')
    src = meta['identity']
    expected_users = sorted(np.unique(subjects[parts['train']]).tolist())
    require(src['fold'] == held and src['validation_subject'] == held % 8+1 and
            src['training_subjects'] == expected_users, 'matrix user partition mismatch')
    require(src['data'] == identity['source']['data'] and src['partitions'] == identity['partitions'], 'matrix source/partition mismatch')
    require(meta['warnings'] == 0 and meta['validation_used_for_fit'] is False and meta['test_used_for_fit'] is False, 'matrix fitting warnings or held-user access')
    for code, digest in src.get('codes', {}).items():
        hashes.verify(code, digest)
    if 'proxy' in src:
        require(src['proxy']['fitted_parameters'] is False, 'proxy contains fitted parameters')
        hashes.verify(src['proxy']['path'], src['proxy']['sha256'])
    require(sorted(row['held_subject'] for row in meta['oof_fits']) == expected_users, 'OOF does not cover exactly six users')
    for row in meta['oof_fits']:
        user = row['held_subject']
        for key, expected in [('fit_ids', parts['train'][subjects[parts['train']] != user]),
                              ('predict_ids', parts['train'][subjects[parts['train']] == user])]:
            require(np.array_equal(np.asarray(row[key]), expected), f'OOF {key} leaves six-user partition')
        require(not row['warnings'], 'OOF fit warnings')
    with np.load(path, allow_pickle=False) as archive:
        for role, ids in parts.items():
            require(np.array_equal(archive[role+'_ids'], ids), f'matrix {role} IDs differ')
        matrix = archive['matrix']; oof = archive['oof_train']
        require(matrix.shape == (12, 12) and oof.shape == (len(parts['train']), 12), 'invalid matrix/OOF shape')
        require(np.isfinite(oof).all() and np.all(oof >= 0) and np.allclose(oof.sum(1), 1), 'invalid OOF probabilities')
        diagonal = oof[np.arange(len(oof)), labels[parts['train']]].mean()
        expected = np.full((12, 12), (1-diagonal)/11)
        np.fill_diagonal(expected, diagonal)
        require(np.allclose(matrix, expected, atol=1e-7, rtol=1e-6), 'matrix not reconstructed from training OOF probabilities')


def _load_module(name):
    spec = importlib.util.spec_from_file_location('_cross_meta_audit_'+name, HERE/(name+'.py'))
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def audit_last_updates(last, initial, arm):
    """Compare final trained tensors to selected warmup, even if best_epoch=0."""
    require(set(last) == set(initial), 'last parameter state topology differs from warmup')
    updates = {}
    for group, prefix in [('encoder', 'right.encoder.'), ('head', 'right.head.'), ('aux', 'fuser.')]:
        keys = [key for key in initial if key.startswith(prefix)]
        require(bool(keys), f'last update audit missing {group} tensors')
        changed = 0
        for key in keys:
            require(last[key].shape == initial[key].shape and last[key].dtype == initial[key].dtype,
                    f'last tensor shape/dtype differs: {key}')
            require(bool(torch.isfinite(last[key]).all()), f'nonfinite last tensor: {key}')
            changed += int(not torch.equal(last[key], initial[key]))
        updates[group] = {'total': len(keys), 'changed': changed}
    require(updates['encoder']['changed'] > 0 and updates['head']['changed'] > 0,
            'last encoder/head did not update from selected warmup')
    require((updates['aux']['changed'] == 0) if arm in ('A0', 'A1') else (updates['aux']['changed'] > 0),
            f'last auxiliary update pattern differs from arm {arm}')
    # The unused encoder.view_emb is included in the total but need not change.
    return updates


def audit_run(root, held, arm, hashes=None, random_hashes=None):
    hashes = hashes or HashVerifier()
    root = Path(root); directory = root/f'fold{held}'/arm
    result = json.loads((directory/'result.json').read_text(encoding='utf-8'))
    identity = result['identity']
    require(identity['fold'] == held and identity['arm'] == arm, 'result identity fold/arm mismatch')
    config_keys = ('warmup_epochs', 'warmup_min', 'warmup_patience') if arm == 'warmup' else ('epochs', 'min_epochs', 'patience')
    for result_key, config_key in zip(('max_epochs', 'min_epochs', 'patience'), config_keys):
        require(config_key in identity['config'] and result[result_key] == identity['config'][config_key],
                f'stopping configuration differs from frozen identity: {result_key}/{config_key}')
    source = identity['source']
    for group in ('code', 'data'):
        require(bool(source[group]), 'empty source identity')
        for path, digest in source[group].items():
            hashes.verify(path, digest)
    manifest = root/'manifest.json'
    if manifest.exists():
        require(json.loads(manifest.read_text(encoding='utf-8')) == {'source': source, 'config': identity['config']}, 'manifest differs from result')
    paths = {Path(p).name: p for p in source['data']}
    labels = np.load(paths['y.npy']); subjects = np.load(paths['subj.npy'])
    parts = partitions(labels, subjects, held)
    require(len(np.unique(subjects[parts['train']])) == 6 and set(labels[parts['train']]) == set(range(12)), 'training partition invalid')
    require(identity['partitions'] == {k: array_hash(v) for k, v in parts.items()}, 'partition hashes differ')
    files = result['files']
    require({'best.pt','right.pt','predictions.npz','progress.pt'} <= set(files), 'required result artifacts missing')
    for filename, digest in files.items():
        require(Path(filename).name == filename, 'artifact filename leaves run directory')
        hashes.verify(directory/filename, digest)
    history = audit_history(result)
    audit_matrix(root, held, identity, labels, subjects, parts, hashes)
    with np.load(directory/'predictions.npz', allow_pickle=False) as archive:
        for role, ids in parts.items():
            require(np.array_equal(archive[role+'_ids'], ids), f'prediction {role} IDs differ')
        require(np.array_equal(archive['ids'], parts['test']), 'prediction test IDs/order differ')
        require(np.array_equal(archive['y'], labels[parts['test']]) and np.array_equal(archive['val_y'], labels[parts['val']]), 'saved labels differ from prepared source')
        require(archive['logits'].shape == (len(parts['test']), 12) and archive['val_logits'].shape == (len(parts['val']), 12), 'prediction class dimension mismatch')
        test_metrics = metrics(archive['logits'], archive['y'])
        val_metrics = metrics(archive['val_logits'], archive['val_y'])
    require(abs(val_metrics['ce']-result['best_val_ce']) <= 1e-6, 'saved validation logits CE differs from selection')
    if 'accuracy' in result:
        require(abs(test_metrics['accuracy']-result['accuracy']) < 1e-12, 'reported accuracy differs from saved predictions')
    best = torch.load(directory/'best.pt', map_location='cpu', weights_only=False)
    require(best['epoch'] == result['best_epoch'] and abs(best['val_ce']-result['best_val_ce']) <= 1e-7, 'best checkpoint selection metadata mismatch')
    selected_hash = state_hash(best['state'])
    require(all(torch.isfinite(x).all() for x in best['state'].values()), 'nonfinite selected parameters')
    if 'selected_state_sha256' in result:
        require(result['selected_state_sha256'] == selected_hash, 'selected state hash mismatch')
    right = torch.load(directory/'right.pt', map_location='cpu', weights_only=False)
    extracted = {k[len('right.'):]: v for k, v in best['state'].items() if k.startswith('right.')}
    require(set(right) == set(extracted) and all(torch.equal(right[k], v) for k, v in extracted.items()), 'standalone Right state differs from selected full state')
    del right
    progress = torch.load(directory/'progress.pt', map_location='cpu', weights_only=False)
    require({'state','optimizer','sampler','info','python_rng','numpy_rng','torch_rng','cuda_rng'} <= set(progress), 'progress recovery state missing')
    info = progress['info']
    require(info['identity'] == identity and info['history'] == result['history'] and info['epoch'] == history['last_epoch'], 'progress final epoch/history/identity mismatch')
    require(info['selection'] == {'best_ce': result['best_val_ce'], 'best_epoch': result['best_epoch'], 'stale': history['stale']}, 'progress selection differs from independent history replay')
    if 'seed' in identity['config']:
        audit_sampling(result, labels, subjects, parts, progress['sampler'])
    if 'best_checkpoint' in info:
        require(state_hash(info['best_checkpoint']['state']) == selected_hash and info['best_checkpoint']['epoch'] == result['best_epoch'], 'embedded recovery best differs')
        require(info['best_file_sha256'] == files['best.pt'], 'recovery best file hash differs')
    last_updates = None
    if arm != 'warmup':
        require(progress['sampler']['step'] == sum(row['steps'] for row in result['history']), 'episode sampler step differs from history')
        warmup_path = root/f'fold{held}'/'warmup'/'result.json'
        warmup = json.loads(warmup_path.read_text(encoding='utf-8'))
        hashes.verify(warmup_path.parent/'best.pt', warmup['files']['best.pt'])
        warmup_best = torch.load(warmup_path.parent/'best.pt', map_location='cpu', weights_only=False)
        require(identity['initial_state_sha256'] == state_hash(warmup_best['state']), 'continuation initialization differs from selected warmup')
        require(identity['config'] == warmup['identity']['config'] and source == warmup['identity']['source'], 'continuation/warmup source or config mismatch')
        require(abs(result['history'][0]['val_ce']-warmup['best_val_ce']) <= 1e-7, 'continuation initial validation differs from warmup')
        last_updates = audit_last_updates(progress['state'], warmup_best['state'], arm)
    elif 'seed' in identity['config']:
        cache = random_hashes if random_hashes is not None else {}
        seed = identity['config']['seed']
        if seed not in cache:
            fresh = _load_module('model').build_model(seed, 'cpu')
            cache[seed] = state_hash(fresh.state_dict()); del fresh
        require(identity['initial_state_sha256'] == cache[seed], 'warmup initialization is not the seeded untrained model')
    return {'fold': held, 'arm': arm, 'status': 'verified', 'metrics': test_metrics,
            'validation': val_metrics, 'selected_state_sha256': selected_hash,
            'training_seconds': float(sum(row['seconds'] for row in result['history'])),
            'last_parameter_updates': last_updates, 'last_update_verified': last_updates is not None,
            'best_epoch': result['best_epoch'], 'last_epoch': history['last_epoch'], 'result': result}


def summarize_scores(scores, repetitions=10000, seed=2026):
    out = {'complete_runs': sum(len(scores.get(a, {})) for a in ARMS), 'means': {}, 'comparisons': {}, 'gate': None}
    for arm in ARMS:
        rows = list(scores.get(arm, {}).values())
        if rows:
            out['means'][arm] = {key: float(np.mean([r[key] for r in rows])) for key in ('accuracy','macro_f1')}
    if any(set(scores.get(a, {})) != {str(h) for h in range(1, 9)} for a in ARMS):
        out['gate_reason'] = 'Requires all 32 audited outer-user runs'
        return out
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, 8, size=(repetitions, 8))
    for comparison in ('A3-A1','A3-A0','A3-A2','interaction'):
        values = {}
        for metric in ('accuracy', 'macro_f1'):
            vectors = {a: np.array([scores[a][str(h)][metric] for h in range(1,9)]) for a in ARMS}
            if comparison == 'interaction':
                differences = vectors['A3']-vectors['A1']-vectors['A2']+vectors['A0']
            else:
                left, right = comparison.split('-'); differences = vectors[left]-vectors[right]
            bootstrap = differences[resamples].mean(1)*100
            values[metric] = {'mean_pp': float(differences.mean()*100), 'per_user_pp': (differences*100).tolist(),
                              'improved_users': int((differences > 0).sum()),
                              'bootstrap_95_percentile_pp': np.quantile(bootstrap, [.025,.975]).tolist()}
        out['comparisons'][comparison] = values
    primary = out['comparisons']['A3-A1']['accuracy']
    out['gate'] = {'passed': bool(primary['mean_pp'] >= 2 and primary['improved_users'] >= 6 and
                                 out['comparisons']['A3-A0']['accuracy']['mean_pp'] > 0 and
                                 out['comparisons']['A3-A2']['accuracy']['mean_pp'] > 0),
                   'primary_metric': 'user_equal_weight_accuracy', 'required_primary_pp': 2,
                   'required_improved_users': 6, 'additional': 'A3 mean accuracy strictly above A0 and A2'}
    out['bootstrap'] = {'unit': 'outer user', 'users': 8, 'repetitions': repetitions, 'seed': seed,
                        'method': 'paired percentile; descriptive uncertainty for fixed eight-user protocol'}
    return out


def replay_right(root, held, arm, atol=1e-5, rtol=1e-5):
    """Deploy original Classifier only; accept only Right data from protocol."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    directory = Path(root)/f'fold{held}'/arm
    model = _load_module('model').Classifier([2], 'concat', 'mean')
    model.load_state_dict(torch.load(directory/'right.pt', map_location='cpu', weights_only=False), strict=True)
    model = model.cuda().eval()
    protocol = _load_module('protocol')
    reader, _, _, parts = protocol.open_fold(held)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    output = {}
    with np.load(directory/'predictions.npz') as saved, torch.no_grad():
        for role, key in [('val', 'val_logits'), ('test', 'logits')]:
            predictions = []
            for start in range(0, len(parts[role]), 32):
                x = reader.load_ids(parts[role][start:start+32], role, 'R')
                require(tuple(x.shape[1:]) == (1,90,378), 'replay must receive only Right input')
                with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)[0]
                predictions.append(logits.float().cpu().numpy())
            actual = np.concatenate(predictions); expected = saved[key]
            equal_argmax = bool(np.array_equal(actual.argmax(1), expected.argmax(1)))
            maxabs = float(np.max(np.abs(actual-expected)))
            output[role] = {'max_abs_logit_error': maxabs, 'argmax_equal': equal_argmax,
                            'allclose': bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
                            'samples': len(actual)}
    del model; torch.cuda.empty_cache()
    output.update(batch=32, atol=atol, rtol=rtol,
                  passed=all(x['argmax_equal'] and x['allclose'] for x in output.values()))
    return output


def audit_all(root, folds=range(1,9), replay=False):
    root = Path(root); hashes = HashVerifier(); random_hashes = {}
    reports = []; errors = []; missing = []; scores = {a: {} for a in ARMS}; smoke = False; config = None
    for held in folds:
        results = {}
        for arm in ('warmup',)+ARMS:
            if not (root/f'fold{held}'/arm/'result.json').exists():
                missing.append(f'fold{held}/{arm}'); continue
            try:
                row = audit_run(root, held, arm, hashes, random_hashes)
                results[arm] = row.pop('result')
                if config is None: config = results[arm]['identity']['config']
                require(config == results[arm]['identity']['config'], 'run configuration differs across audit set')
                smoke |= bool(results[arm]['identity']['config'].get('smoke', False))
                if replay and arm in ARMS:
                    row['replay'] = replay_right(root, held, arm)
                    require(row['replay']['passed'], f'Right-only replay failed: {row["replay"]}')
                reports.append(row)
                if arm in ARMS: scores[arm][str(held)] = row['metrics']
            except (ValueError, KeyError, OSError, RuntimeError) as error:
                errors.append({'run': f'fold{held}/{arm}', 'error': str(error)})
        try:
            audit_episode_prefix(results)
            initial_hashes = {results[a]['identity']['initial_state_sha256'] for a in ARMS if a in results}
            require(len(initial_hashes) <= 1, 'four-arm initialization mismatch')
        except ValueError as error:
            errors.append({'run': f'fold{held}', 'error': str(error)})
    summary = summarize_scores(scores)
    if errors or missing or smoke:
        summary['gate'] = None
        summary['gate_reason'] = 'Audit errors, incomplete runs, or smoke configuration prevent gate evaluation'
    return {'status': 'failed' if errors else 'incomplete' if missing else 'verified',
            'errors': errors, 'missing': missing, 'runs': reports, 'summary': summary,
            'config': config,
            'replay_requested': replay, 'source_hashes_verified': len(hashes.cache)}


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=HERE/'runs_s2026')
    parser.add_argument('--folds', nargs='+', type=int, default=list(range(1,9)))
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = audit_all(args.out, args.folds, args.replay)
    destination = args.report or args.out/('AUDIT_REPLAY.json' if args.replay else 'AUDIT.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'errors': report['errors'], 'missing': report['missing'],
                      'summary': report['summary'], 'report': str(destination)}, ensure_ascii=False))
    if report['errors']: raise SystemExit(1)


if __name__ == '__main__':
    main()
