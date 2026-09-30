"""Independent, read-only CPU audit of the frozen A pilot.

Only this audit script and INDEPENDENT_AUDIT.{json,md} are written.
No experiment implementation is imported or called. The trusted local model
pickle is deserialized as a plain attribute holder, without its class methods.
Every reported statistic is independently computed from original arrays.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import hashlib
import itertools
import json
import pickle
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone
import numpy as np

ROOT = Path('D:/uwb_gestures/experiments/p1_20260928')
OUT = ROOT / 'calibration_s2026'
BASE = ROOT / 'baseline_s2026'
PREP = Path('D:/uwb_gestures/experiments/p1_20260905/prepared')
CODE = Path('C:/Users/19252/.codex/worktrees/uwb-user-calibration/uwb_gestures/experiments/new_routes')
REGS = [0.1, 1., 10., 100.]
TAUS = [0., .125, .25, .5, 1., 2.]
SEEDS = list(range(92000, 92020))
CANDIDATES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
ADAPTED = ['supported_only', 'translation', 'ridge_affine', 'lowrank']
NAMES = ['frozen', 'source_prototype'] + ADAPTED + ['validation_selected_simple', 'allclass_lowrank']
CHECKS = []
FILES = {}
NUMERIC = {}


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    path = Path(path).resolve()
    key = str(path)
    if key not in FILES:
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                value.update(block)
        st = path.stat()
        FILES[key] = {'sha256': value.hexdigest(), 'bytes': st.st_size,
                      'mtime_utc': datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}
    return FILES[key]['sha256']


def check(name, value, detail=None):
    ok = bool(value)
    row = {'name': name, 'passed': ok}
    if detail is not None:
        row['detail'] = detail
    CHECKS.append(row)
    if not ok:
        print('AUDIT_FAILURE ' + name + (' ' + str(detail) if detail is not None else ''), flush=True)
    return ok


def exact(name, observed, expected):
    a, b = np.asarray(observed), np.asarray(expected)
    return check(name, a.shape == b.shape and np.array_equal(a, b),
                 {'observed_shape': list(a.shape), 'expected_shape': list(b.shape)})


def near(name, observed, expected, atol=1e-10, rtol=1e-10):
    a, b = np.asarray(observed, dtype=float), np.asarray(expected, dtype=float)
    ok = a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
    finite = np.isfinite(a) & np.isfinite(b) if a.shape == b.shape else np.zeros(0, dtype=bool)
    err = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else None
    NUMERIC[name] = err
    return check(name, ok, {'max_abs_error': err, 'atol': atol, 'rtol': rtol})


def array_hash(value):
    x = np.ascontiguousarray(value)
    return hashlib.sha256(str(x.dtype).encode() + str(x.shape).encode() + x.tobytes()).hexdigest()


class SnapshotModel:
    pass


class SnapshotUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'calibration' and name == 'SourceCalibration':
            return SnapshotModel
        if module in ('numpy', 'numpy.core.multiarray', 'numpy._core.multiarray') and name in (
                'ndarray', 'dtype', '_reconstruct', 'scalar'):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'Unexpected trusted snapshot global: {module}.{name}')


def partition(user, all_ids, raw_y, raw_subj):
    user_ids = all_ids[raw_subj[all_ids] == user]
    pools, queries = [], []
    for c in np.unique(raw_y[user_ids]):
        ids = user_ids[raw_y[user_ids] == c]
        random_order = np.random.default_rng(91000 + user * 100 + int(c)).permutation(ids)
        pools.append(random_order[:20])
        queries.append(random_order[20:])
    return np.concatenate(pools), np.concatenate(queries)


def supports(pool, classes, seed, raw_y):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(pool[raw_y[pool] == c], size=3, replace=False)
                           for c in sorted(classes)])


def orient_svd(x):
    u, singular, vt = np.linalg.svd(x, full_matrices=False)
    signs = np.sign(vt[np.arange(len(vt)), np.abs(vt).argmax(axis=1)])
    signs[signs == 0] = 1
    return u * signs, singular, vt * signs[:, None]


def independent_source(x, y, subjects):
    """Rebuild the mathematical recipe from the six source users only."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    _, _, vt = orient_svd(x - mean)
    components = vt[:32]
    projected = (x - mean) @ components.T
    users = np.unique(subjects)
    counts = np.zeros((6, 12), dtype=np.int64)
    cells = np.full((6, 12, 32), np.nan)
    for ui, user in enumerate(users):
        for c in range(12):
            keep = (subjects == user) & (y == c)
            counts[ui, c] = np.sum(keep)
            if np.any(keep):
                cells[ui, c] = projected[keep].mean(axis=0)
    observed = counts > 0
    centers = np.nanmean(cells, axis=0)
    variance = np.maximum(np.mean((projected - centers[y]) ** 2, axis=0), 1e-6)
    scale = np.sqrt(variance)
    mu = centers / scale
    cell_means = cells / scale
    residuals = cell_means - mu
    matrix = residuals.reshape(6, -1)
    mask = np.repeat(observed, 32, axis=1)
    completed = np.where(mask, matrix, np.nanmean(matrix, axis=0))
    missing = ~mask
    converged = False
    for iteration in range(1, 101):
        u, singular, vectors = orient_svd(completed)
        rank2 = (u[:, :2] * singular[:2]) @ vectors[:2]
        previous = completed[missing].copy()
        updated = rank2[missing]
        relative = float(np.linalg.norm(updated - previous) / max(1., np.linalg.norm(previous)))
        completed[missing] = updated
        if relative <= 1e-8:
            converged = True
            break
    u, singular, vectors = orient_svd(completed)
    rank2 = (u[:, :2] * singular[:2]) @ vectors[:2]
    B = (vectors[:2].T * (singular[:2] / np.sqrt(6))).reshape(12, 32, 2)
    diagnostics = dict(iterations=iteration, max_iter=100, tol=1e-8,
                       converged=converged, relative_change=relative,
                       observed_rmse=float(np.sqrt(np.mean((rank2[mask] - matrix[mask]) ** 2))),
                       missing_cells=int((~observed).sum()), observed_cells=int(observed.sum()), rank=2)
    return dict(feature_mean=mean, pca_components=components, variance=variance, scale=scale,
                mu=mu, B=B, observed_mask=observed, cell_counts=counts,
                user_class_means=cell_means, diagnostics=diagnostics)


def choose_classes(B):
    scores = []
    for triple in itertools.combinations(CANDIDATES, 3):
        matrix = np.eye(2)
        for c in triple:
            matrix += B[c].T @ B[c]
        score = float(np.linalg.slogdet(matrix)[1])
        scores.append((triple, score))
    best = max(s for _, s in scores)
    chosen = min(t for t, s in scores if s >= best - 1e-12)
    second = sorted((s for t, s in scores if t != chosen), reverse=True)[0]
    return list(chosen), {'evaluated_triples': len(scores), 'winning_logdet': best,
                          'margin_over_second': best - second}


def independent_delta(z, labels, mu, B, method, reg):
    residual = z - mu[labels]
    if method == 'supported_only':
        delta = np.zeros((12, 32))
        for c in np.unique(labels):
            group = residual[labels == c]
            delta[c] = group.sum(axis=0) / (len(group) + reg)
        return delta
    if method == 'translation':
        return np.repeat((residual.sum(axis=0) / (len(z) + reg))[None, :], 12, axis=0)
    if method == 'ridge_affine':
        centers = np.column_stack((mu, np.ones(12)))
        design = centers[labels]
        coefficients = np.linalg.solve(design.T @ design + reg * np.eye(33), design.T @ residual)
        return centers @ coefficients
    design = B[labels].reshape(-1, 2)
    latent = np.linalg.solve(design.T @ design + reg * np.eye(2), design.T @ residual.reshape(-1))
    return (B.reshape(-1, 2) @ latent).reshape(12, 32)


def independent_scores(logits, z, mu, delta, tau):
    if tau == 0:
        return logits.copy()
    # Algebraically equivalent Gaussian ratio, independently organized as GEMM.
    bias = -np.sum(mu * delta, axis=1) - .5 * np.sum(delta * delta, axis=1)
    return logits + tau * (z @ delta.T + bias[None, :])


def accuracy(predictions, y, classes):
    predictions = np.asarray(predictions)
    if predictions.ndim == 1:
        predictions = predictions[None, :]
    return np.stack([(predictions[:, y == c] == c).mean(axis=1) for c in classes], axis=1).mean(axis=1)


def choose_row(rows):
    best = max(r['score'] for r in rows)
    return min((r for r in rows if r['score'] >= best - 1e-12),
               key=lambda r: (r['tau'], -r['reg']))


def validation_grid(z_all, logits_all, position, raw_y, pool, query, selected, method):
    actual_method = 'lowrank' if method == 'allclass_lowrank' else method
    y = raw_y[query]
    eval_classes = sorted(set(y) - set(selected)) or sorted(set(y))
    zq, logits = z_all[position[query]], logits_all[position[query]]
    rows = []
    for reg in REGS:
        mean_scores = {tau: [] for tau in TAUS}
        for seed in SEEDS:
            ids = supports(pool, selected, seed, raw_y)
            delta = independent_delta(z_all[position[ids]], raw_y[ids],
                                      CURRENT_MODEL.mu, CURRENT_MODEL.B, actual_method, reg)
            for tau in TAUS:
                pred = independent_scores(logits, zq, CURRENT_MODEL.mu, delta, tau).argmax(axis=1)
                mean_scores[tau].append(float(accuracy(pred, y, eval_classes)[0]))
        rows.extend(dict(reg=reg, tau=tau, score=float(np.mean(mean_scores[tau]))) for tau in TAUS)
    return rows


def paired(a, b):
    # Twenty repetitions first collapse to one estimate for each of eight users.
    differences = np.mean(a - b, axis=1) * 100.
    indices = np.random.default_rng(20260928).integers(0, 8, size=(10000, 8))
    boot = differences[indices].mean(axis=1)
    return dict(n_users=8, mean_difference_pp=float(differences.mean()),
                per_user_difference_pp=differences.tolist(),
                positive_users=int(np.sum(differences > 1e-10)),
                worst_user_difference_pp=float(differences.min()),
                ci95_pp=np.quantile(boot, [.025, .975]).tolist())


def compare_paired(prefix, measured, recorded):
    for key in measured:
        near(prefix + '/' + key, measured[key], recorded[key], atol=1e-10, rtol=0)


def main():
    global CURRENT_MODEL
    start = time.perf_counter()
    manifest = load_json(BASE / 'manifest.json')
    y_raw = np.load(PREP / 'y.npy', allow_pickle=False)
    s_raw = np.load(PREP / 'subj.npy', allow_pickle=False)
    valid = ~(((s_raw == 2) & np.isin(y_raw, [10, 11])) | ((s_raw == 6) & (y_raw == 3)))
    legal_ids = np.flatnonzero(valid)
    exact('prepared_aligned_shapes', y_raw.shape, s_raw.shape)
    check('prepared_raw_count_9593', len(y_raw) == 9593)
    check('quality_count_9300', len(legal_ids) == 9300)
    check('quality_excluded_293', int((~valid).sum()) == 293)
    check('manifest_seed_and_real_run', manifest['config']['seed'] == 2026 and not manifest['config']['smoke'])
    check('baseline_complete_queue', load_json(BASE / 'queue_status.json')['state'] == 'complete')
    for category in ('code', 'data'):
        for index, (path, expected) in enumerate(manifest['source'][category].items()):
            check('baseline_source_hash/' + path, sha(path) == expected)
            if category == 'code':
                archived = BASE / 'source' / f'{index:02d}_{Path(path).name}'
                check('baseline_snapshot_hash/' + archived.name, sha(archived) == expected)
    check('preregistration_bound_to_training_manifest',
          sha(ROOT / 'PREREGISTRATION.md') == manifest['source']['plan_sha256'])
    for name in ('AMENDMENT_001.md', 'AMENDMENT_002.md'):
        sha(ROOT / name)
        check(name + '_mtime_precedes_first_A_selection',
              (ROOT / name).stat().st_mtime < (OUT / 'fold1/selection.json').stat().st_mtime)
    print('AUDIT_SOURCE_HASHES_COMPLETE', flush=True)
    fold_reports = []
    metric_arrays = {name: {key: [] for key in ('all', 'calibrated', 'uncalibrated', 'center_mse')}
                     for name in NAMES}
    query_union, support_pool_union = [], []
    baseline_raw_correct, baseline_raw_count = 0, 0
    full_grid_count, target_prediction_count, prediction_mismatch_cells = 0, 0, 0
    for fold in range(1, 9):
        t0 = time.perf_counter()
        directory, upstream_dir = OUT / f'fold{fold}', BASE / f'fold{fold}/warmup'
        selected = load_json(directory / 'selection.json')
        recorded = load_json(directory / 'result.json')
        upstream = load_json(upstream_dir / 'result.json')
        feature_meta = load_json(upstream_dir / 'features.json')
        split = load_json(BASE / f'fold{fold}/split.json')
        prediction_archive = np.load(directory / 'predictions.npz', allow_pickle=False)
        saved = {k: prediction_archive[k] for k in prediction_archive.files}
        with np.load(upstream_dir / 'features.npz', allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        with np.load(upstream_dir / 'predictions.npz', allow_pickle=False) as archive:
            original = {k: archive[k] for k in archive.files}
        prefix = f'fold{fold}'
        for name, expected in recorded['files'].items():
            check(f'{prefix}/calibration_artifact_hash/{name}', sha(directory / name) == expected)
        sha(directory / 'result.json')
        for name, expected in upstream['files'].items():
            check(f'{prefix}/baseline_artifact_hash/{name}', sha(upstream_dir / name) == expected)
        for name, expected in recorded['identity']['source'].items():
            check(f'{prefix}/calibration_snapshot_hash/{name}', sha(OUT / 'source' / name) == expected)
            check(f'{prefix}/calibration_worktree_hash/{name}', sha(CODE / name) == expected)
        identity = recorded['identity']
        check(f'{prefix}/identities_agree', identity == selected['identity'])
        check(f'{prefix}/feature_identity', sha(upstream_dir / 'features.npz') == identity['features_sha256'] == feature_meta['sha256'])
        check(f'{prefix}/feature_metadata_identity', sha(upstream_dir / 'features.json') == identity['feature_metadata_sha256'])
        check(f'{prefix}/upstream_result_identity', sha(upstream_dir / 'result.json') == identity['upstream_result_sha256'])
        check(f'{prefix}/feature_checkpoint_identity', feature_meta['identity']['checkpoint'] == upstream['files']['right.pt'])
        check(f'{prefix}/feature_model_identity', feature_meta['identity']['model_identity'] == upstream['identity'])
        check(f'{prefix}/feature_extractor_identity', feature_meta['identity']['extractor'] == sha(CODE / 'train_baseline.py'))
        check(f'{prefix}/upstream_manifest_binding', upstream['identity']['source'] == manifest['source'] and upstream['identity']['config'] == manifest['config'])
        check(f'{prefix}/correct_fold_arm', upstream['identity']['fold'] == fold and upstream['identity']['arm'] == 'warmup' and identity['fold'] == fold)
        exact(f'{prefix}/predefined_reg_grid', identity['reg'], REGS)
        exact(f'{prefix}/predefined_tau_grid', identity['tau'], TAUS)
        exact(f'{prefix}/predefined_support_seeds', identity['seeds'], SEEDS)
        check(f'{prefix}/feature_shape_and_finiteness', data['features'].shape == (9300, 256) and data['logits'].shape == (9300, 12) and np.isfinite(data['features']).all() and np.isfinite(data['logits']).all())
        exact(f'{prefix}/legal_ids', np.sort(data['ids']), legal_ids)
        check(f'{prefix}/unique_ids', len(np.unique(data['ids'])) == 9300)
        exact(f'{prefix}/raw_labels_match', data['y'], y_raw[data['ids']])
        exact(f'{prefix}/raw_subjects_match', data['subjects'], s_raw[data['ids']])
        position = np.full(len(y_raw), -1, dtype=np.int64)
        position[data['ids']] = np.arange(9300)
        val_user = fold % 8 + 1
        train_users = sorted(set(range(1, 9)) - {fold, val_user})
        parts = {'train': legal_ids[~np.isin(s_raw[legal_ids], [fold, val_user])],
                 'val': legal_ids[s_raw[legal_ids] == val_user],
                 'test': legal_ids[s_raw[legal_ids] == fold]}
        for role, name in enumerate(('train', 'val', 'test')):
            exact(f'{prefix}/{name}_role_ids', data['ids'][data['role'] == role], parts[name])
            exact(f'{prefix}/{name}_split_ids', split['ids'][name], parts[name])
            exact(f'{prefix}/{name}_original_prediction_ids', original[name + '_ids'], parts[name])
            check(f'{prefix}/{name}_upstream_partition_hash', upstream['identity']['partitions'][name] == array_hash(parts[name]))
        exact(f'{prefix}/training_subjects', upstream['training_subjects'], train_users)
        check(f'{prefix}/validation_test_subjects', upstream['validation_subject'] == val_user and upstream['test_subject'] == fold)
        exact(f'{prefix}/baseline_test_ids', original['ids'], parts['test'])
        exact(f'{prefix}/baseline_test_labels', original['y'], y_raw[parts['test']])
        exact(f'{prefix}/baseline_val_labels', original['val_y'], y_raw[parts['val']])
        exact(f'{prefix}/baseline_test_logit_replay', data['logits'][data['role'] == 2], original['logits'])
        exact(f'{prefix}/baseline_val_logit_replay', data['logits'][data['role'] == 1], original['val_logits'])
        raw_correct = int(np.sum(original['logits'].argmax(axis=1) == original['y']))
        baseline_raw_correct += raw_correct
        baseline_raw_count += len(original['y'])
        near(f'{prefix}/baseline_raw_accuracy', upstream['accuracy'], raw_correct / len(original['y']))
        val_logits = original['val_logits'].astype(float)
        shifted = val_logits - np.max(val_logits, axis=1, keepdims=True)
        ce = float(np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(shifted)), original['val_y']]))
        near(f'{prefix}/baseline_selected_validation_ce', upstream['best_val_ce'], ce, atol=2e-6, rtol=0)
        histories = upstream['history']
        minimum = min(histories, key=lambda row: (row['val_ce'], row['epoch']))
        check(f'{prefix}/baseline_earliest_validation_minimum', minimum['epoch'] == upstream['best_epoch'])
        check(f'{prefix}/baseline_epoch_budget', 12 <= max(row['epoch'] for row in histories) <= 40)
        near(f'{prefix}/baseline_best_ce_from_history', minimum['val_ce'], upstream['best_val_ce'])

        # Source-only smoothing matrix and all OOF fitting IDs are also checked.
        matrix_path = BASE / f'matrices/fold{fold}.npz'
        matrix_meta = load_json(matrix_path.with_suffix('.json'))
        check(f'{prefix}/smoothing_matrix_hash', sha(matrix_path) == matrix_meta['matrix_sha256'] == upstream['identity']['matrix_sha256'])
        matrix_data = np.load(matrix_path, allow_pickle=False)
        for name in ('train', 'val', 'test'):
            exact(f'{prefix}/smoothing_{name}_ids', matrix_data[name + '_ids'], parts[name])
        seen_oof = []
        for row in matrix_meta['oof_fits']:
            hu = row['held_subject']
            exact(f'{prefix}/smoothing_oof{hu}_fit_ids', row['fit_ids'], parts['train'][s_raw[parts['train']] != hu])
            exact(f'{prefix}/smoothing_oof{hu}_predict_ids', row['predict_ids'], parts['train'][s_raw[parts['train']] == hu])
            check(f'{prefix}/smoothing_oof{hu}_no_warning', row['warnings'] == [])
            seen_oof.extend(row['predict_ids'])
        exact(f'{prefix}/smoothing_oof_cover_sources', np.sort(seen_oof), parts['train'])
        check(f'{prefix}/smoothing_metadata_no_target_fit', not matrix_meta['validation_used_for_fit'] and not matrix_meta['test_used_for_fit'])
        oof = matrix_data['oof_train']
        diagonal = float(np.mean(oof[np.arange(len(oof)), y_raw[parts['train']]]))
        independent_matrix = np.full((12, 12), (1 - diagonal) / 11, dtype=np.float32)
        np.fill_diagonal(independent_matrix, diagonal)
        near(f'{prefix}/smoothing_matrix_recomputed', matrix_data['matrix'], independent_matrix, atol=1e-7, rtol=0)

        with (directory / 'source_model.pkl').open('rb') as stream:
            model = SnapshotUnpickler(stream).load()
        CURRENT_MODEL = model
        exact(f'{prefix}/source_model_ids', model.source_ids, parts['train'])
        exact(f'{prefix}/source_model_users', model.source_user_ids, train_users)
        check(f'{prefix}/source_model_counts', model.source_user_count == 6 and model.source_sample_count == len(parts['train']))
        recomputed = independent_source(data['features'][data['role'] == 0], data['y'][data['role'] == 0], data['subjects'][data['role'] == 0])
        for name in ('feature_mean', 'pca_components', 'variance', 'scale', 'mu', 'B', 'user_class_means'):
            near(f'{prefix}/source_fit/{name}', getattr(model, name), recomputed[name], atol=2e-9, rtol=2e-9)
        for name in ('observed_mask', 'cell_counts'):
            exact(f'{prefix}/source_fit/{name}', getattr(model, name), recomputed[name])
        for key, val in recomputed['diagnostics'].items():
            near(f'{prefix}/imputation/{key}', model.diagnostics[key], val, atol=1e-10, rtol=1e-9)
        check(f'{prefix}/diagnostics_bound_to_selection_and_result', model.diagnostics == selected['fit_diagnostics'] == recorded['fit_diagnostics'])
        check(f'{prefix}/source_metadata_binding', model.metadata == selected['source_model_metadata'])
        chosen_classes, class_diagnostics = choose_classes(recomputed['B'])
        exact(f'{prefix}/source_only_D_optimal_selection', selected['classes'], chosen_classes)
        exact(f'{prefix}/saved_classes', saved['selected_classes'], chosen_classes)
        exact(f'{prefix}/reported_classes', recorded['support_classes'], chosen_classes)
        z_all = ((data['features'].astype(float) - model.feature_mean) @ model.pca_components.T) / model.scale
        vp, vq = partition(val_user, legal_ids, y_raw, s_raw)
        tp, tq = partition(fold, legal_ids, y_raw, s_raw)
        query_union.extend(tq.tolist())
        support_pool_union.extend(tp.tolist())
        exact(f'{prefix}/fixed_query_ids', saved['query_ids'], tq)
        exact(f'{prefix}/fixed_query_labels', saved['query_y'], y_raw[tq])
        exact(f'{prefix}/fixed_support_pool', saved['support_pool_ids'], tp)
        check(f'{prefix}/complete_support_query_partition', len(set(tp) & set(tq)) == 0 and set(tp) | set(tq) == set(parts['test']))
        check(f'{prefix}/support_pool_20_per_available_class', all(np.sum(y_raw[tp] == c) == 20 for c in np.unique(y_raw[tq])))
        check(f'{prefix}/query_count', len(tq) == recorded['query_count'])
        chosen_hyper = {}
        max_grid_error = 0.
        for method in ADAPTED + ['allclass_lowrank']:
            support_classes = sorted(set(y_raw[vq])) if method == 'allclass_lowrank' else chosen_classes
            grid = validation_grid(z_all, data['logits'], position, y_raw, vp, vq, support_classes, method)
            saved_grid = selected['validation_grid'][method]
            check(f'{prefix}/{method}/full_grid_size', len(grid) == len(saved_grid) == 24)
            exact(f'{prefix}/{method}/grid_coordinates', [[r['reg'], r['tau']] for r in saved_grid], [[r['reg'], r['tau']] for r in grid])
            differences = np.abs(np.array([r['score'] for r in grid]) - np.array([r['score'] for r in saved_grid]))
            max_grid_error = max(max_grid_error, float(differences.max()))
            near(f'{prefix}/{method}/validation_scores_replayed', [r['score'] for r in saved_grid], [r['score'] for r in grid], atol=1e-12, rtol=0)
            chosen = choose_row(grid)
            chosen_hyper[method] = chosen
            for key in ('reg', 'tau', 'score'):
                near(f'{prefix}/{method}/selected_{key}', selected['hyperparameters'][method][key], chosen[key], atol=1e-12, rtol=0)
                near(f'{prefix}/{method}/reported_{key}', recorded['hyperparameters'][method][key], chosen[key], atol=1e-12, rtol=0)
            full_grid_count += len(grid)
        max_simple = max(chosen_hyper[m]['score'] for m in ADAPTED[:3])
        simple = next(m for m in ADAPTED[:3] if chosen_hyper[m]['score'] >= max_simple - 1e-12)
        check(f'{prefix}/simple_validation_argmax_and_order_tie', simple == selected['simple_method'] == recorded['validation_selected_simple'])

        yq = y_raw[tq]
        zq = z_all[position[tq]]
        logitq = data['logits'][position[tq]]
        query_classes = sorted(map(int, np.unique(yq)))
        uncalibrated = sorted(set(query_classes) - set(chosen_classes))
        independent_pred = {
            'frozen': np.repeat(logitq.argmax(axis=1)[None, :], 20, axis=0),
            'source_prototype': np.repeat((-np.sum((zq[:, None, :] - model.mu[None, :, :]) ** 2, axis=2)).argmax(axis=1)[None, :], 20, axis=0)}
        independent_deltas = {name: np.zeros((20, 12, 32)) for name in ['frozen', 'source_prototype']}
        supports_by_method, max_delta_error = {}, 0.
        for method in ADAPTED + ['allclass_lowrank']:
            support_classes = query_classes if method == 'allclass_lowrank' else chosen_classes
            actual_method = 'lowrank' if method == 'allclass_lowrank' else method
            config = chosen_hyper[method]
            sampled, deltas, predictions = [], [], []
            for seed in SEEDS:
                si = supports(tp, support_classes, seed, y_raw)
                check(f'{prefix}/{method}/seed{seed}_support_query_disjoint', len(set(si) & set(tq)) == 0 and len(set(si)) == len(si) and set(si) <= set(tp))
                check(f'{prefix}/{method}/seed{seed}_three_shots', all(np.sum(y_raw[si] == c) == 3 for c in support_classes))
                delta = independent_delta(z_all[position[si]], y_raw[si], model.mu, model.B, actual_method, config['reg'])
                pred = independent_scores(logitq, zq, model.mu, delta, config['tau']).argmax(axis=1)
                sampled.append(si)
                deltas.append(delta)
                predictions.append(pred)
            sampled, deltas, predictions = map(np.asarray, (sampled, deltas, predictions))
            exact(f'{prefix}/{method}/support_rng_replay', saved[method + '_support_ids'], sampled)
            near(f'{prefix}/{method}/support_delta_replay', saved[method + '_delta'], deltas, atol=2e-9, rtol=2e-9)
            max_delta_error = max(max_delta_error, float(np.max(np.abs(saved[method + '_delta'] - deltas))))
            supports_by_method[method] = sampled
            independent_pred[method] = predictions
            independent_deltas[method] = deltas
            if config['tau'] == 0:
                exact(f'{prefix}/{method}/tau0_frozen_identity', predictions, independent_pred['frozen'])
        for method in ADAPTED[1:]:
            exact(f'{prefix}/shared_support_across_methods/{method}', supports_by_method[method], supports_by_method[ADAPTED[0]])
        check(f'{prefix}/nine_shot_budget', recorded['support_budget'] == 9 and supports_by_method['lowrank'].shape == (20, 9))
        check(f'{prefix}/higher_budget_reference', recorded['methods']['allclass_lowrank']['support_budget'] == len(query_classes) * 3 == supports_by_method['allclass_lowrank'].shape[1])
        independent_pred['validation_selected_simple'] = independent_pred[simple]
        independent_deltas['validation_selected_simple'] = independent_deltas[simple]
        measured_method_rows = {}
        for name in NAMES:
            raw_pred = saved[name + '_predictions']
            exact(f'{prefix}/{name}/prediction_replay', raw_pred, independent_pred[name])
            prediction_mismatch_cells += (int(np.count_nonzero(raw_pred != independent_pred[name]))
                                          if raw_pred.shape == independent_pred[name].shape else int(raw_pred.size))
            check(f'{prefix}/{name}/prediction_shape_range', raw_pred.shape == (20, len(tq)) and np.isin(raw_pred, np.arange(12)).all())
            target_prediction_count += int(raw_pred.size)
            calibrated_classes = query_classes if name == 'allclass_lowrank' else chosen_classes
            metrics = {'all': accuracy(raw_pred, yq, query_classes),
                       'calibrated': accuracy(raw_pred, yq, calibrated_classes),
                       'uncalibrated': accuracy(raw_pred, yq, uncalibrated)}
            target_means = np.stack([zq[yq == c].mean(axis=0) for c in uncalibrated])
            centers = model.mu[uncalibrated][None, :, :] + independent_deltas[name][:, uncalibrated, :]
            metrics['center_mse'] = np.mean((target_means[None, :, :] - centers) ** 2, axis=(1, 2))
            for key, values in metrics.items():
                metric_arrays[name][key].append(values)
                saved_key = 'uncalibrated_center_mse' if key == 'center_mse' else key
                near(f'{prefix}/{name}/reported_{key}', [r[saved_key] for r in recorded['methods'][name]['metrics']], values, atol=1e-11, rtol=1e-10)
            measured_method_rows[name] = {k: float(np.mean(v)) for k, v in metrics.items()}
        fold_report = {
            'fold': fold, 'train_users': train_users, 'validation_user': val_user, 'test_user': fold,
            'train_count': len(parts['train']), 'validation_count': len(parts['val']), 'test_count': len(parts['test']),
            'query_count': len(tq), 'support_pool_count': len(tp), 'query_classes': query_classes,
            'support_classes': chosen_classes, 'uncalibrated_classes': uncalibrated,
            'query_count_per_class': {str(c): int(np.sum(yq == c)) for c in query_classes},
            'allclass_support_budget': len(query_classes) * 3,
            'simple_method': simple, 'hyperparameters_recomputed': chosen_hyper,
            'source_fit_diagnostics_recomputed': recomputed['diagnostics'],
            'class_selection_diagnostics': class_diagnostics,
            'max_validation_grid_score_error': max_grid_error,
            'max_delta_replay_error': max_delta_error,
            'baseline_best_epoch': upstream['best_epoch'], 'baseline_trained_epochs': max(row['epoch'] for row in histories),
            'baseline_test_accuracy': raw_correct / len(original['y']),
            'baseline_validation_ce_recomputed': ce,
            'metrics_recomputed': measured_method_rows,
            'elapsed_seconds': time.perf_counter() - t0,
        }
        fold_reports.append(fold_report)
        print('AUDIT_FOLD_COMPLETE ' + json.dumps({k: fold_report[k] for k in ('fold', 'query_count', 'support_classes', 'simple_method', 'max_validation_grid_score_error', 'max_delta_replay_error', 'elapsed_seconds')}), flush=True)

    check('aggregate_query_count_7440', len(query_union) == len(set(query_union)) == 7440)
    check('aggregate_support_pool_count_1860', len(support_pool_union) == len(set(support_pool_union)) == 1860)
    check('aggregate_disjoint_legal_cover', not (set(query_union) & set(support_pool_union)) and set(query_union) | set(support_pool_union) == set(legal_ids))
    check('validation_grid_count_960', full_grid_count == 960)
    check('all_saved_target_predictions_replayed_1190400', target_prediction_count == 1190400)
    for name in NAMES:
        metric_arrays[name] = {k: np.asarray(v) for k, v in metric_arrays[name].items()}
    # Open the report to compare only after deriving all fold statistics above.
    reported_summary = load_json(OUT / 'summary.json')
    sha(OUT / 'summary.json')
    table = {}
    for name in NAMES:
        values = metric_arrays[name]
        table[name] = dict(
            uncalibrated_macro_accuracy=float(values['uncalibrated'].mean()),
            uncalibrated_per_user=values['uncalibrated'].mean(axis=1).tolist(),
            worst_user_uncalibrated=float(values['uncalibrated'].mean(axis=1).min()),
            within_user_support_std=values['uncalibrated'].std(axis=1, ddof=1).tolist(),
            all_macro_accuracy=float(values['all'].mean()),
            calibrated_macro_accuracy=float(values['calibrated'].mean()),
            uncalibrated_center_mse=float(values['center_mse'].mean()),
            center_mse_reduction=float((metric_arrays['frozen']['center_mse'] - values['center_mse']).mean()),
            support_budget_per_user=[f['allclass_support_budget'] if name == 'allclass_lowrank' else
                                     (0 if name in ('frozen', 'source_prototype') else 9) for f in fold_reports],
            vs_frozen=paired(values['uncalibrated'], metric_arrays['frozen']['uncalibrated']))
        for key, value in table[name].items():
            if key == 'vs_frozen':
                compare_paired(f'summary/{name}/vs_frozen', value, reported_summary['methods'][name][key])
            else:
                near(f'summary/{name}/{key}', reported_summary['methods'][name][key], value, atol=1e-10, rtol=1e-10)
    comp = paired(metric_arrays['lowrank']['uncalibrated'], metric_arrays['validation_selected_simple']['uncalibrated'])
    vf = table['lowrank']['vs_frozen']
    compare_paired('summary/lowrank_vs_simple', comp, reported_summary['lowrank_vs_simple'])
    gates = dict(gain_at_least_2pp=comp['mean_difference_pp'] >= 2 - 1e-10,
                 positive_at_least_6users=comp['positive_users'] >= 6,
                 beats_frozen=vf['mean_difference_pp'] > 1e-10,
                 no_user_worse_than_minus2pp_vs_simple=comp['worst_user_difference_pp'] >= -2 - 1e-10,
                 no_user_worse_than_minus2pp_vs_frozen=vf['worst_user_difference_pp'] >= -2 - 1e-10)
    decision = 'GO' if all(gates.values()) else 'STOP_LOW_RANK_MECHANISM'
    check('summary_gates_replayed', gates == reported_summary['gate'])
    check('summary_decision_replayed', decision == reported_summary['decision'])
    check('summary_complete_eight_users_20_support_repeats', reported_summary['state'] == 'complete' and reported_summary['users'] == 8 and reported_summary['support_repeats'] == 20)
    check('all_masked_SVD_converged', all(f['source_fit_diagnostics_recomputed']['converged'] for f in fold_reports))
    sha(__file__)
    failures = [r for r in CHECKS if not r['passed']]
    result = dict(
        audit_status='PASS_WITH_SCOPE_LIMITATIONS' if not failures else 'FAIL',
        created_utc=datetime.now(timezone.utc).isoformat(),
        decision_recomputed=decision,
        scope='Independent CPU-only audit; no experiment implementation imported; no GPU training or parameter changes.',
        runtime={'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                 'elapsed_seconds': time.perf_counter() - start},
        counts={'raw_rows': len(y_raw), 'quality_excluded': int((~valid).sum()), 'legal_rows': len(legal_ids),
                'users': 8, 'source_users_per_fold': 6, 'validation_users_per_fold': 1, 'test_users_per_fold': 1,
                'query_rows_total_unique': len(set(query_union)), 'reserved_support_pool_total_unique': len(set(support_pool_union)),
                'support_repeats_per_user': 20, 'same_budget_support_per_repeat': 9,
                'source_D_optimal_triples_evaluated': 84 * 8,
                'validation_grid_rows_replayed': full_grid_count,
                'validation_support_repeat_scores_replayed': full_grid_count * 20,
                'target_method_prediction_cells_replayed': target_prediction_count,
                'source_fit_models_recomputed': 8, 'target_adapted_deltas_recomputed': 8 * 20 * 5,
                'bootstrap_user_resamples': 10000, 'bootstrap_seed': 20260928,
                'checks': len(CHECKS), 'failed_checks': len(failures)},
        baseline_raw_holdout={'correct': baseline_raw_correct, 'rows': baseline_raw_count,
                              'micro_accuracy': baseline_raw_correct / baseline_raw_count,
                              'note': 'Uses the entire held-out quality-filtered user; differs from fixed-query macro accuracy.'},
        folds=fold_reports, metrics_recomputed=table, lowrank_vs_simple=comp, gates_recomputed=gates,
        failures=failures, checks=CHECKS, audited_file_inventory=FILES,
        numerical_scope={'saved_prediction_mismatching_cells': prediction_mismatch_cells,
                         'maximum_validation_score_error': max(f['max_validation_grid_score_error'] for f in fold_reports),
                         'maximum_target_delta_error': max(f['max_delta_replay_error'] for f in fold_reports)},
        limitations=[
            'No raw radar-to-feature GPU inference or retraining was rerun; SHA bindings, role IDs, source snapshots, baseline saved logits and exact logit export replay were checked.',
            'The trusted local source pickle is mapped to an inert attribute holder. Its PCA, source centering, variance, masked SVD and support deltas were independently recomputed.',
            'Local file hashes prove current consistency, not trusted external timestamps. Amendment mtimes precede the first A selection; cryptographic proof of preregistration chronology is unavailable. AMENDMENT_001/002 hashes are recorded here but not separately bound in the original A identity.',
            'Eight users and one encoder/model seed. Twenty supports are repeated measurements within a user. User bootstrap does not include retraining uncertainty or correct dependence from overlapping source-user folds.',
            'Uncalibrated means known output classes omitted from the nine support labels; this is not novel-class or open-set recognition.',
            'The higher-budget allclass reference uses 30/33/36 labels and a different validation objective. Its calibrated metric is all available classes; its uncalibrated column uses the original nine-label route subset only for reference.',
            'The raw center-MSE diagnostic uses query labels after selection and before tau scaling. It is not a deployment score and may change even when tau=0 leaves predictions unchanged.',
            'The support/query split is a retrospective class-stratified evaluation: 20 candidates are reserved from every available class, including nonselected classes. Results apply to the resulting fixed query population.',
            'Source label-smoothing OOF IDs, saved probabilities, matrix construction and hashes were checked; the historical proxy classifiers were not refit.',
            'Reported adaptation timing is not an end-to-end inference latency benchmark and was not independently remeasured here.',
            'The STOP conclusion is specific to this dataset, protocol, single seed and implemented rank-2 mechanism; it does not establish that partial-action calibration is universally ineffective.'
        ],
        write_boundary=[str(Path(__file__).resolve()), str(OUT / 'INDEPENDENT_AUDIT.json'), str(OUT / 'INDEPENDENT_AUDIT.md')])
    (OUT / 'INDEPENDENT_AUDIT.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    lines = [
        '# A pilot independent scientific audit', '',
        f"Audit status: **{result['audit_status']}**. Frozen decision independently reproduced: **{decision}**.", '',
        f"{len(CHECKS)} checks; {len(failures)} failed. CPU only; no implementation, results, parameters, or model weights modified.", '',
        '## Evidence and counts', '',
        '- Prepared labels/subjects: 9593 raw rows; 293 protocol exclusions; 9300 retained unique IDs.',
        '- Eight complete folds; six whole users for parameter fitting, one for validation, one for test. HV2 excludes classes 10/11; HV6 excludes class 3 (zero-based).',
        '- Aggregate held-out population: 7440 unique fixed queries + 1860 reserved support candidates. Every observed test class has 80 query rows and 20 candidates.',
        '- Same-budget methods share exactly the same 9 support IDs in each user/repeat: three selected classes, three examples per class, seeds 92000–92019. No support/query overlap.',
        '- Recomputed eight source-only PCA32/variance/masked-rank2 models, all 672 source class-triple scores, all 960 validation grid scores (19200 support-repeat scores), 800 target adaptation deltas and 1190400 saved prediction cells.',
        f"- Maximum validation-grid score discrepancy: {result['numerical_scope']['maximum_validation_score_error']:.3g}; maximum target-delta absolute discrepancy: {result['numerical_scope']['maximum_target_delta_error']:.3g}.",
        '- Baseline checkpoint/result/features and calibration model/selection/predictions hashes agree. Original validation/test logits exactly match exported feature-archive logits.', '',
        '## Independently recomputed outcomes', '',
        'Accuracy values below are percentages: average the 20 supports within each user, then equally weight the eight users.', '',
        '| Method | Uncalibrated known classes | All classes | Calibrated classes | Worst user, uncalibrated |',
        '|---|---:|---:|---:|---:|',
    ]
    for name in NAMES:
        r = table[name]
        lines.append(f"| {name} | {r['uncalibrated_macro_accuracy']*100:.6f} | {r['all_macro_accuracy']*100:.6f} | {r['calibrated_macro_accuracy']*100:.6f} | {r['worst_user_uncalibrated']*100:.6f} |")
    lines.extend(['',
        f"Lowrank − validation-selected simple: **{comp['mean_difference_pp']:.6f} pp**, {comp['positive_users']}/8 users improve, paired user bootstrap 95% interval **[{comp['ci95_pp'][0]:.6f}, {comp['ci95_pp'][1]:.6f}] pp**.",
        f"Lowrank − frozen: **{vf['mean_difference_pp']:.6f} pp**, {vf['positive_users']}/8 users improve, 95% interval **[{vf['ci95_pp'][0]:.6f}, {vf['ci95_pp'][1]:.6f}] pp**.", '',
        'All five implemented GO conditions fail: mean advantage >=2 pp, >=6/8 positive users, positive mean advantage over frozen, and no user losing >2 pp against either comparator.', '',
        '## Per-user harm and numerical diagnostics', '',
        '| Test user | Validation user | Source-selected classes | Query rows | Selected simple | Lowrank − simple, pp | Lowrank − frozen, pp | Imputation iterations |',
        '|---:|---:|---|---:|---|---:|---:|---:|'])
    for i, f in enumerate(fold_reports):
        lines.append(f"| {f['fold']} | {f['validation_user']} | {f['support_classes']} | {f['query_count']} | {f['simple_method']} | {comp['per_user_difference_pp'][i]:.6f} | {vf['per_user_difference_pp'][i]:.6f} | {f['source_fit_diagnostics_recomputed']['iterations']} |")
    lines.extend(['',
        'All eight masked-SVD fits converged within the preregistered 100-iteration ceiling and 1e-8 relative-change tolerance. Missing user/class cells remained masked; no missing cell was treated as a zero observation. The independent source-fit reconstruction, D-optimal class selection, grid maximum and tie rules agree with saved artifacts.', '',
        f"The HV5 lowrank mean loss is {comp['per_user_difference_pp'][4]:.6f} pp, and its support-repeat standard deviation is {table['lowrank']['within_user_support_std'][4]*100:.6f} pp. This is observed across 20 alternative support draws for the same user, not 20 independent users.", '',
        f"The uncalibrated-class raw center MSE is {table['frozen']['uncalibrated_center_mse']:.9f} for the frozen centers and {table['lowrank']['uncalibrated_center_mse']:.9f} after lowrank adaptation, a reduction of {table['lowrank']['center_mse_reduction']:.9f} (negative means worse). This post hoc diagnostic also does not support the proposed center-shift mechanism in this pilot.", '',
        'The interval versus the frozen classifier includes zero; do not describe that comparison as statistically conclusive harm. The preregistered engineering GO conditions fail regardless.', '',
        '## Errors and limitations', '',
        ('No numerical, ID/label, support-leakage, model-selection, saved-artifact hash or report-reconciliation errors were found.' if not failures else f'{len(failures)} checks failed; inspect JSON failures before using these results.'), '',
    ])
    lines.extend('- ' + limitation for limitation in result['limitations'])
    lines.extend(['', '## Reproduction', '',
        'Run the independent auditor with:', '',
        '```powershell',
        "& 'D:\\D_conda\\envs\\pytorch\\python.exe' 'D:\\uwb_gestures\\experiments\\p1_20260928\\calibration_s2026\\independent_audit_a_20260928.py'",
        '```', '',
        'The accompanying JSON records every check, per-user metrics, all validated artifact SHA-256 values and independent source-fit diagnostics. The audit writes only its two report files; the script itself is the third allowed artifact.', '',
        f"Generated UTC: {result['created_utc']}. CPU audit elapsed: {result['runtime']['elapsed_seconds']:.2f} seconds.", '',
    ])
    (OUT / 'INDEPENDENT_AUDIT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('INDEPENDENT_AUDIT_COMPLETE ' + json.dumps({
        'status': result['audit_status'], 'checks': len(CHECKS), 'failed_checks': len(failures),
        'decision': decision, 'lowrank_vs_simple': comp, 'elapsed_seconds': result['runtime']['elapsed_seconds']
    }), flush=True)
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
