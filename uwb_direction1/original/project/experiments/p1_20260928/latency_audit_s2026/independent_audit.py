"""CPU-only, read-only audit of the frozen formal latency observations.

Run with D:/D_conda/envs/pytorch/python.exe.  This script writes only to its
own latency_audit_s2026 directory and never invokes CUDA or timing measurement.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np


ROOT = Path('D:/uwb_gestures/experiments/p1_20260928')
SOURCE = Path('C:/Users/19252/.codex/worktrees/uwb-user-calibration/uwb_gestures/experiments/new_routes')
FROZEN = ROOT / 'latency_s2026'
OUT = ROOT / 'latency_audit_s2026'
SUBSETS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
NAMES = ('L', 'T', 'R', 'LT', 'LR', 'TR', 'LTR')
METHODS = ('baselineRight', 'fixed1', 'fixed2', 'fixed3',
           'frozen_budget2_cascade', 'class_gain', 'extra_trees')
STAGE_KEYS = ('preprocess_pack_ms', 'reader_total_ms', 'input_transfer_host_ms',
              'encode_host_ms', 'fuse_host_ms', 'output_cpu_ms', 'routing_and_harness_ms')
issues = []
counts = Counter()


def check(condition, tag, detail=''):
    counts[tag] += 1
    if not condition:
        issues.append({'check': tag, 'detail': str(detail)[:400]})


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def near(a, b, *, atol=1e-10):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=atol)


def compare_metrics(rows, saved, where):
    t = np.array([r['elapsed_ms'] for r in rows], dtype=np.float64)
    v = np.array([r['actual_views'] for r in rows], dtype=np.float64)
    users = sorted({r['fold'] for r in rows})
    per_user = {str(u): float(np.mean([r['elapsed_ms'] for r in rows if r['fold'] == u])) for u in users}
    expected = {'unit': 'ms', 'observations': len(rows), 'mean_ms': float(t.mean()),
                'median_ms': float(np.median(t)), 'p95_ms': float(np.quantile(t, .95, method='linear')),
                'quantile_method': 'numpy linear', 'per_user_mean_ms': per_user,
                'per_user_macro_mean_ms': float(np.mean(list(per_user.values()))),
                'mean_actual_views': float(v.mean()),
                'view_count_histogram': {str(i): int((v == i).sum()) for i in (1, 2, 3)}}
    for key, value in expected.items():
        actual = saved.get(key)
        if isinstance(value, dict):
            check(set(actual or ()) == set(value), 'summary_fields', f'{where}.{key} keys')
            for child, number in value.items():
                given = (actual or {}).get(child)
                check(near(given, number) if isinstance(number, float) and given is not None else given == number,
                      'summary_numbers', f'{where}.{key}.{child}: {given} vs {number}')
        else:
            check(near(actual, value) if isinstance(value, float) and actual is not None else actual == value,
                  'summary_numbers', f'{where}.{key}: {actual} vs {value}')
    return expected


def preprocess_functions():
    namespace = {'np': np}
    specs = ((ROOT.parent / 'p1_20260905/vendor/uwb_osr/data.py', ('_analytic', 'doppler_map')),
             (ROOT.parent / 'p1_20260905/pipeline.py', ('norm', 'transform')))
    for path, names in specs:
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        check(len(nodes) == len(names), 'preprocess_source', str(path))
        module = ast.Module(body=nodes, type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace['transform']


def quality_mask(y, subject):
    return ~(((subject == 2) & np.isin(y, (10, 11))) | ((subject == 6) & (y == 3)))


def selected_ids(y, subject):
    good = quality_mask(y, subject)
    result = {}
    for fold in range(1, 9):
        selected = []
        for cls in range(12):
            candidates = np.flatnonzero(good & (subject == fold) & (y == cls))
            if not len(candidates):
                check((fold == 2 and cls in (10, 11)) or (fold == 6 and cls == 3),
                      'missing_class_rule', f'fold {fold} class {cls}')
                continue
            check(len(candidates) >= 3, 'sample_candidates', f'fold {fold} class {cls}')
            selected.extend(np.random.default_rng(44000 + fold * 100 + cls).choice(candidates, 3, replace=False))
        result[fold] = np.asarray(selected, dtype=np.int64)
    return result


def uncertainty(z, mode):
    z = np.asarray(z, dtype=np.float64)
    p = np.exp(z - z.max())
    p /= p.sum()
    if mode == 'confidence':
        return 1.0 - p.max()
    if mode == 'entropy':
        return float(-(p * np.log(np.maximum(p, 1e-30))).sum() / np.log(len(p)))
    raise ValueError(mode)


def state_features(z):
    z = np.asarray(z, dtype=np.float64)
    p = np.exp(z - z.max())
    p /= p.sum()
    ent = -(p * np.log(np.maximum(p, 1e-30))).sum() / np.log(12)
    ordered = np.sort(p)
    return np.r_[p, ent, ordered[-1] - ordered[-2]][None]


class AuditGainPolicy:
    pass


class AuditClassGain:
    pass


class PolicyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'value_policy' and name == 'GainPolicy':
            return AuditGainPolicy
        if module == 'value_policy' and name == 'ClassGainRegressor':
            return AuditClassGain
        return super().find_class(module, name)


def predict_gain(policy, state, z):
    model = policy.models[state]
    features = state_features(z)
    if policy.method == 'class_gain':
        gain = np.asarray(model.table_[features[:, :12].argmax(1)], dtype=np.float64)
    else:
        gain = np.asarray(model.predict(features), dtype=np.float64)
    return gain.reshape(-1)


def destinations(first, state):
    if state == first:
        return [i for i in range(3, 7) if first in SUBSETS[i]]
    return [6]


def offline_route(method, logits, right, selection, policy_selection, policies):
    if method == 'baselineRight':
        return 2, [2, -1, -1], int(np.argmax(right))
    fixed = {}
    accuracy = selection['validation_fixed_accuracy']
    for cost in (1, 2, 3):
        indices = [i for i, subset in enumerate(SUBSETS) if len(subset) == cost]
        fixed[f'fixed{cost}'] = max(indices, key=lambda i: (round(float(accuracy[i]), 12), -i))
    if method in fixed:
        chosen = fixed[method]
        return chosen, [chosen, -1, -1], int(np.argmax(logits[chosen]))
    if method == 'frozen_budget2_cascade':
        params = selection['curves']['2.0']['params']
        first, pair = params['first'], params['pair']
        path = [first, -1, -1]
        chosen = first
        if uncertainty(logits[first], params['mode']) > params['threshold1']:
            path[1] = pair
            chosen = pair
            if uncertainty(logits[pair], params['mode']) > params['threshold2']:
                path[2] = 6
                chosen = 6
        return chosen, path, int(np.argmax(logits[chosen]))
    policy = policies[method]
    price = policy_selection['selected_prices'][method]['price']
    first = policy.first
    path = [first, -1, -1]
    current = first
    for pos in (1, 2):
        possible = destinations(first, current)
        gain = predict_gain(policy, current, logits[current])
        check(len(gain) == len(possible), 'policy_gain_shape', f'{method} {current}')
        scores = np.r_[0.0, gain] - price * np.array([0] + [len(SUBSETS[d]) - len(SUBSETS[current]) for d in possible])
        candidate = [current] + possible
        selected = candidate[int(np.flatnonzero(scores >= scores.max() - 1e-12)[0])]
        if selected == current:
            break
        path[pos] = selected
        current = selected
        if current == 6:
            break
    return current, path, int(np.argmax(logits[current]))


def main():
    frozen = read(FROZEN / 'FROZEN_SHA256.json')
    report = read(FROZEN / 'summary.json')
    preflight = read(FROZEN / 'preflight.json')
    idle = read(FROZEN / 'gpu_idle_checks.json')
    check(frozen.get('state') == 'frozen', 'frozen_state')
    expected_files = {str(path.relative_to(FROZEN)).replace('\\', '/') for path in FROZEN.rglob('*.json')
                      if path.name != 'FROZEN_SHA256.json'}
    check(set(frozen['files']) == expected_files, 'frozen_inventory', f'{len(frozen["files"])} vs {len(expected_files)}')
    for rel, sha in frozen['files'].items():
        check(digest(FROZEN / rel) == sha, 'frozen_sha256', rel)

    # The complete saved upstream inventory is rehashed; this establishes the
    # current bytes, not historical provenance or a second execution.
    upstream_bytes = 0
    upstream_by_extension = Counter()
    for name, sha in report['hash_inventory'].items():
        path = Path(name)
        check(path.is_file(), 'upstream_exists', name)
        if path.is_file():
            upstream_bytes += path.stat().st_size
            upstream_by_extension[path.suffix.lower()] += 1
            check(digest(path) == sha, 'upstream_sha256', name)
    check(report['hash_inventory'].get(str(SOURCE / 'benchmark_latency.py')) == digest(SOURCE / 'benchmark_latency.py'),
          'benchmark_code_identity')
    for name in ('LATENCY_PROTOCOL.md', 'LATENCY_ENVIRONMENT_NOTE.md'):
        check(report['hash_inventory'].get(str(ROOT / name)) == digest(ROOT / name), 'protocol_hash', name)

    config = report['configuration']
    check(report['state'] == 'complete' and report['folds'] == list(range(1, 9)), 'formal_run_state')
    check(config['device'] == 'cuda' and config['warmup'] >= 10 and config['repeats'] == 1
          and config['order_seed'] == 45000 and config['policy'] == 'required'
          and config['allow_smoke'] is False, 'run_configuration')
    check(report['classification'] == 'desktop_context_gpu_latency' and
          report['environment']['gpu_exclusive'] is False, 'desktop_classification')
    check(report['environment']['batch'] == 1 and report['environment']['autocast'] == 'bfloat16'
          and report['environment']['attention'] == 'MATH'
          and report['environment']['tf32'] is False, 'execution_environment')
    check(report['environment']['timer'] == 'perf_counter_ns; explicit device synchronize before/after',
          'timer_metadata')

    y = np.load(ROOT.parent / 'p1_20260905/prepared/y.npy', allow_pickle=False)
    subject = np.load(ROOT.parent / 'p1_20260905/prepared/subj.npy', allow_pickle=False)
    good = quality_mask(y, subject)
    selected = selected_ids(y, subject)
    concatenated = np.concatenate([selected[i] for i in range(1, 9)])
    check(y.shape == subject.shape == (9593,) and
          np.isin(y, np.arange(12)).all() and
          np.isin(subject, np.arange(1, 9)).all(), 'raw_label_subject_domain')
    check(len(y) == 9593 and int(good.sum()) == 9300 and int((~good).sum()) == 293,
          'quality_population')
    check(len(concatenated) == 279 and len(set(concatenated.tolist())) == 279, 'selected_population')
    check(hashlib.sha256(concatenated.tobytes()).hexdigest() == preflight['data']['selected_ids_sha256'],
          'selected_sha256')
    for fold, ids in selected.items():
        check(ids.tolist() == preflight['data']['selected_ids'][str(fold)], 'sample_rng_ids', fold)
        check(np.all(subject[ids] == fold) and np.all(good[ids]), 'sample_y_subject_mask', fold)
    check(preflight['data']['sampled_clips'] == 279 and preflight['data']['per_view_checks'] == 837
          and preflight['data']['prepared_replay'] == 'bitwise_equal', 'recorded_preprocess_preflight')

    # Reexecute the frozen original preprocessing definitions, without running
    # the benchmark module, for all 837 sampled physical views on CPU.
    transform = preprocess_functions()
    raw = np.load(ROOT.parent.parent / 'cache/X_clutterremoved.npy', mmap_mode='r', allow_pickle=False)
    off = np.load(ROOT.parent / 'p1_20260905/prepared/off.npy', mmap_mode='r', allow_pickle=False)
    only = np.load(ROOT.parent / 'p1_20260905/prepared/only.npy', mmap_mode='r', allow_pickle=False)
    check(list(raw.shape) == [9593, 3, 90, 189] and off.shape == raw.shape and only.shape == raw.shape,
          'raw_prepared_shapes')
    for clip in concatenated:
        for view in range(3):
            packed = transform(np.asarray(raw[clip, view])[None, None], 'concat').astype(np.float16)
            check(np.array_equal(packed[0, 0, :, :189], off[clip, view]) and
                  np.array_equal(packed[0, 0, :, 189:], only[clip, view]),
                  'preprocess_bitwise', f'{clip}/{view}')
    del raw, off, only

    expected_phases = [('preflight', None)]
    for fold in range(1, 9):
        expected_phases.extend([('before_fold', fold), ('after_fold', fold)])
    check(len(idle) == 17 and [(x['phase'], x['fold']) for x in idle] == expected_phases,
          'environment_check_sequence')
    canonical_identity = idle[0]['allowed_idle_gpu_identities']
    check(len(canonical_identity) == 1 and canonical_identity[0]['pid'] == 52016
          and canonical_identity[0]['name'] == 'QQ.exe', 'allowed_desktop_identity')
    for pos, stage in enumerate(idle):
        check(stage['state'] == 'verified' and stage['device'] == 'cuda' and
              stage['training_finished_assertion'] and not stage['training_conflicts'] and
              stage['known_training_pids'] == [], 'environment_stage_state', pos)
        check(stage['allowed_idle_gpu_identities'] == canonical_identity and
              stage['allowed_idle_gpu_pids'] == [52016] and stage['gpu_exclusive'] is False,
              'environment_stable_identity', pos)
        snapshots = stage['gpu_compute_snapshots']
        check(len(snapshots) == 2 and all(set(s['processes']) == {'52016'} and
              s['processes']['52016'] == canonical_identity[0]['executable'] for s in snapshots),
              'environment_no_new_pid_at_snapshots', pos)
        samples = stage['gpu_utilization_samples']
        check(len(samples) == 5 and [s['index'] for s in samples] == list(range(5)) and
              all(s['returncode'] == 0 and s['utilization_percent'] and
                  all(np.isfinite(s['utilization_percent'])) for s in samples),
              'environment_five_samples', pos)
        check(all(all(value <= 5 for value in s['utilization_percent']) for s in samples[-3:]),
              'environment_last_three_idle', pos)
    check(report['gpu_idle_checks'] == idle and preflight['gpu_idle'] == idle[0],
          'environment_reports_match')

    all_rows = []
    per_fold = {}
    offline_counts = defaultdict(lambda: {'prediction': 0, 'path': 0})
    subset_counts = Counter()
    for fold in range(1, 9):
        rows = read(FROZEN / f'fold{fold}/observations.json')
        saved_fold = read(FROZEN / f'fold{fold}/summary.json')
        ids = selected[fold]
        budget_dir = ROOT / f'budget_s2026/fold{fold}'
        policy_dir = ROOT / f'value_policy_s2026/fold{fold}'
        model_dir = ROOT / f'multiview_s2026/fold{fold}'
        right_dir = ROOT / f'baseline_s2026/fold{fold}/warmup'
        selection = read(budget_dir / 'selection.json')
        budget_result = read(budget_dir / 'result.json')
        policy_selection = read(policy_dir / 'selection.json')
        policy_result = read(policy_dir / 'result.json')
        model_result = read(model_dir / 'result.json')
        right_result = read(right_dir / 'result.json')
        check(selection['identity'] == budget_result['identity'] and
              selection['identity']['predictions_sha256'] == digest(model_dir / 'predictions.npz') and
              selection['identity']['upstream_result_sha256'] == digest(model_dir / 'result.json') and
              selection['identity']['manifest_sha256'] == digest(ROOT / 'multiview_s2026/manifest.json'),
              'budget_lineage', fold)
        check(policy_selection['identity'] == policy_result['identity'] and
              policy_selection['identity']['budget_identity'] == selection['identity'] and
              policy_selection['identity']['budget_result_sha256'] == digest(budget_dir / 'result.json') and
              policy_selection['identity']['budget_selection_sha256'] == digest(budget_dir / 'selection.json') and
              selection['uses_target_labels'] is False and policy_selection['uses_target_labels'] is False,
              'policy_lineage', fold)
        check(model_result['identity']['fold'] == right_result['identity']['fold'] == fold and
              model_result['identity']['arm'] == 'multiview' and
              right_result['identity']['arm'] == 'warmup' and
              model_result['identity']['partitions'] == right_result['identity']['partitions'] and
              model_result['identity']['initial_checkpoint']['right_sha256'] ==
              digest(right_dir / 'right.pt'), 'model_lineage', fold)
        check(budget_result['fixed_validation_selected'] ==
              {str(cost): max((i for i, subset in enumerate(SUBSETS) if len(subset) == cost),
                              key=lambda i: (round(float(selection['validation_fixed_accuracy'][i]), 12), -i))
               for cost in (1, 2, 3)}, 'validation_fixed_selection', fold)
        check(policy_selection['frozen_cascade_params'] == selection['curves']['2.0']['params'],
              'frozen_cascade_params', fold)
        with (policy_dir / 'models.pkl').open('rb') as stream:
            policies = PolicyUnpickler(stream).load()
        check(set(policies) == {'class_gain', 'extra_trees'} and
              all(policies[m].method == m and
                  policies[m].first == selection['curves']['2.0']['params']['first']
                  for m in ('class_gain', 'extra_trees')), 'loaded_policy_identity', fold)
        with np.load(model_dir / 'predictions.npz', allow_pickle=False) as z:
            mv = {k: z[k] for k in ('ids', 'y', 'subjects', 'role', 'logits', 'subset_masks')}
        with np.load(right_dir / 'predictions.npz', allow_pickle=False) as z:
            br = {k: z[k] for k in ('ids', 'y', 'logits')}
        check(mv['logits'].shape == (9300, 7, 12) and
              np.array_equal(np.sort(mv['ids']), np.flatnonzero(good)) and
              np.array_equal(mv['y'], y[mv['ids']]) and
              np.array_equal(mv['subjects'], subject[mv['ids']]) and
              np.array_equal(mv['subset_masks'].astype(bool),
                             np.array([[view in subset for view in range(3)] for subset in SUBSETS])) and
              np.all(mv['role'][np.isin(mv['ids'], ids)] == 2), 'multiview_archive_alignment', fold)
        check(np.array_equal(br['y'], y[br['ids']]) and len(br['ids']) == int((good & (subject == fold)).sum()),
              'baseline_archive_alignment', fold)
        mv_pos = {int(clip): pos for pos, clip in enumerate(mv['ids'])}
        br_pos = {int(clip): pos for pos, clip in enumerate(br['ids'])}
        check(all(int(clip) in mv_pos and int(clip) in br_pos for clip in ids), 'offline_selected_coverage', fold)
        check(len(rows) == len(ids) * len(METHODS), 'fold_row_count', fold)
        check(set(saved_fold) == set(METHODS), 'fold_summary_methods', fold)
        warmup = report['warmup'][str(fold)]
        check(warmup['iterations_per_method'] >= 10 and warmup['iterations_per_forced_path'] >= 10
              and set(warmup['methods']) == set(METHODS) and len(warmup['forced_shared_paths']) == 9,
              'warmup_metadata', fold)
        rng = np.random.default_rng(45000 + fold)
        row_pos = 0
        for clip in ids:
            order = rng.permutation(warmup['methods']).tolist()
            for position, method in enumerate(order):
                row = rows[row_pos]
                row_pos += 1
                location = f'{fold}/{int(clip)}/{method}'
                check(row['fold'] == fold and row['clip_id'] == int(clip) and
                      row['label'] == int(y[clip]) and row['method'] == method and
                      row['repeat'] == 0 and row['order_position'] == position and
                      row['method_order'] == order, 'sample_method_order', location)
                index = row['subset_index']
                path = [int(s) for s in row['path'] if s != -1]
                access = row['raw_accesses']
                requested = row['requested_subsets']
                check(0 <= index <= 6 and row['subset_name'] == NAMES[index] and
                      path and path[-1] == index and len(path) == len(requested) and
                      path == requested, 'route_path', location)
                check(all(set(SUBSETS[a]).issubset(SUBSETS[b]) for a, b in zip(path, path[1:])),
                      'route_monotone', location)
                expected_views = list(SUBSETS[index])
                check(len(access) == len(expected_views) and len(set(access)) == len(access) and
                      set(access) == set(expected_views) and access == row['encoded_views'] and
                      row['actual_views'] == len(expected_views), 'unique_physical_views', location)
                check(row['replay']['bitwise_equal'] is True and
                      row['replay']['actual_views'] == row['actual_views'] and
                      row['replay']['checked_subset_indices'] == sorted(set(requested)) and
                      set(map(int, row['visited_logits_sha256'])) == set(requested),
                      'recorded_bitwise_replay', location)
                check(len(row['final_logits']) == 12 and np.isfinite(row['final_logits']).all() and
                      row['prediction'] == int(np.argmax(row['final_logits'])),
                      'final_logits_prediction', location)
                check(np.isfinite(row['elapsed_ms']) and row['elapsed_ms'] >= 0 and
                      set(row['stages']) == set(STAGE_KEYS) and
                      all(np.isfinite(v) and v >= 0 for v in row['stages'].values()),
                      'valid_host_timings', location)
                check(near(row['elapsed_ms'], row['stages']['reader_total_ms'] +
                           row['stages']['routing_and_harness_ms'], atol=1e-4),
                      'elapsed_stage_residual', location)
                expected_index, expected_path, expected_prediction = offline_route(
                    method, mv['logits'][mv_pos[int(clip)]], br['logits'][br_pos[int(clip)]],
                    selection, policy_selection, policies)
                check(row['offline_subset_index'] == expected_index and
                      row['offline_path'] == expected_path and
                      row['offline_prediction'] == expected_prediction,
                      'offline_route_replay', location)
                pred_diff = int(row['prediction'] != expected_prediction)
                path_diff = int(row['path'] != expected_path)
                check(row['offline_prediction_differs'] == bool(pred_diff) and
                      row['offline_path_differs'] == bool(path_diff),
                      'offline_difference_flags', location)
                offline_counts[method]['prediction'] += pred_diff
                offline_counts[method]['path'] += path_diff
                subset_counts[(method, index)] += 1
        for method in METHODS:
            matching = [row for row in rows if row['method'] == method]
            check(len(matching) == len(ids), 'fold_method_coverage', f'{fold}/{method}')
            compare_metrics(matching, saved_fold[method], f'fold{fold}/{method}')
        per_fold[str(fold)] = {'clips': len(ids), 'rows': len(rows)}
        all_rows.extend(rows)

    recomputed = {}
    check(set(report['methods']) == set(METHODS), 'overall_summary_methods')
    for method in METHODS:
        matching = [row for row in all_rows if row['method'] == method]
        recomputed[method] = compare_metrics(matching, report['methods'][method], method)
        for field, key in (('offline_batch_prediction_differences', 'prediction'),
                           ('offline_batch_route_differences', 'path')):
            check(report['methods'][method][field] == offline_counts[method][key],
                  'overall_offline_difference_counts', f'{method}/{field}')
    check(len(all_rows) == 1953 and len({r['clip_id'] for r in all_rows}) == 279 and
          set(r['method'] for r in all_rows) == set(METHODS), 'overall_observation_coverage')
    check(offline_counts['baselineRight']['prediction'] == 1 and
          sum(v['prediction'] for k, v in offline_counts.items() if k != 'baselineRight') == 0 and
          sum(v['path'] for v in offline_counts.values()) == 0, 'offline_difference_expected_pattern')
    check(frozen['verification']['rows'] == len(all_rows) and
          frozen['verification']['clips'] == len(concatenated) and
          frozen['verification']['environment_checks'] == len(idle), 'frozen_verification_counts')

    stage_means = {}
    for method in METHODS:
        matching = [r for r in all_rows if r['method'] == method]
        stage_means[method] = {key: float(np.mean([r['stages'][key] for r in matching])) for key in STAGE_KEYS}
    output = {'status': 'PASS' if not issues else 'FAIL', 'issues': issues,
              'checks': dict(counts), 'total_checks': sum(counts.values()),
              'evidence': {'frozen_files_sha256': len(frozen['files']),
                           'upstream_inventory_sha256': len(report['hash_inventory']),
                           'upstream_inventory_bytes': upstream_bytes,
                           'upstream_inventory_extensions': dict(upstream_by_extension),
                           'clips': len(concatenated), 'observations': len(all_rows),
                           'methods': len(METHODS), 'folds': len(selected),
                           'preprocess_view_bitwise_checks': counts['preprocess_bitwise'],
                           'environment_checks': len(idle), 'folds_detail': per_fold,
                           'offline_differences': dict(offline_counts),
                           'subset_counts': {f'{m}/{NAMES[i]}': count for (m, i), count in subset_counts.items()},
                           'metrics': recomputed, 'stage_mean_ms': stage_means},
              'limitations': [
                  'The audit replays frozen records and CPU preprocessing; it is not a second GPU hardware measurement.',
                  'Reexecuted preprocessing uses the hashed original definitions. The recorded direct-versus-batch-one replay flag and intermediate logit hashes cannot be independently reconstructed without running GPU inference again.',
                  'Current SHA-256 agreement establishes byte identity with saved inventories, not independent historical timestamp attestation.',
                  'The allowed QQ desktop GPU context means the device was not exclusive. Five samples before/after each fold cannot guarantee zero interference during every timed interval.',
                  'The saved warmup counts and hashed benchmark loop were checked, but no independent trace of individual warmup calls exists in the archive.',
                  'Host wall-clock stage fields overlap and are not additive CUDA kernel durations; only synchronized elapsed_ms is the total latency.',
                  '279 representative clips measure latency only. Accuracy and stop gates come from the formal 9300-clip, eight-user evaluation.',
                  'Already clutter-removed 90-frame clips were host-resident. Sensor acquisition, raw-file I/O, prior clutter removal, and one-time model/policy loading are excluded. At 20 Hz, 90 frames represent about 4.5 seconds of prior acquisition.'
              ]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'INDEPENDENT_AUDIT.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = [
        '# Independent latency audit', '',
        f"**{output['status']}** — {output['total_checks']} checks; {len(issues)} issues. CPU-only; no GPU timing rerun.", '',
        f"Frozen files: {len(frozen['files'])}/19 SHA-256 matches; upstream inventory: {len(report['hash_inventory'])} byte hashes; "
        f"{len(concatenated)} clips, {len(all_rows)} method observations, {len(idle)} GPU-environment records.", '',
        '| Method | Mean ms | Median ms | p95 ms | Mean actual views | Offline prediction/path differences |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for method in METHODS:
        m = recomputed[method]
        d = offline_counts[method]
        lines.append(f"| {method} | {m['mean_ms']:.6f} | {m['median_ms']:.4f} | {m['p95_ms']:.5f} | "
                     f"{m['mean_actual_views']:.6f} | {d['prediction']}/{d['path']} |")
    lines.extend(['', 'Every per-fold and pooled mean, median, NumPy-linear p95, per-user mean, user macro mean, '
                  'view histogram, and pooled physical-view cost was recomputed from the observations. '
                  'All sampled IDs were regenerated from labels/subjects and the frozen RNG; all 837 original '
                  'preprocessing outputs were rerun on CPU and compared bitwise with prepared data. '
                  'All row path/access/count/logit assertions and the frozen offline prediction/path fields were checked.', '',
                  'The saved source and run metadata specify at least 10 warmups per method and forced path on each fold. '
                  'The archive does not include a separate trace of each warmup call.', '',
                  'The recorded 17 environment checks consist of preflight and before/after checks for every fold. '
                  'Each has two process snapshots, five utilization samples and last-three readings <=5%; '
                  'the same QQ.exe PID and process identity remain throughout the records. This was a nonexclusive '
                  'desktop-context measurement. Sampling cannot establish continuous absence of interference.', '',
                  'The synchronized elapsed time includes preprocessing and float16 packing for requested views, '
                  'CPU-to-GPU transfer, incremental view encoding, fusion/output transfer, and CPU policy decisions. '
                  'Host wall-clock stage measurements overlap, so their individual means should not be summed. '
                  f"For ExtraTrees, recorded routing/harness mean is {stage_means['extra_trees']['routing_and_harness_ms']:.3f} ms "
                  f"versus {stage_means['fixed3']['routing_and_harness_ms']:.3f} ms for fixed3; "
                  'this is descriptive and does not isolate CPU decision time.', '',
                  'The frozen formal 9300-clip policy result remains STOP_VALUE_POLICY_MECHANISM. '
                  'The ExtraTrees route processed fewer views on this 279-clip timing set but had a larger '
                  'measured mean than fixed3. The simple cascade is a measured local tradeoff; its formal '
                  'eight-user mean cost is 2.004615530303 (>2), and its accuracy is more than 1 percentage point '
                  'below fixed three-view. Fixed2 is the validation-selected two-view combination, not an '
                  'after-the-fact choice from these latency rows.', '',
                  'A 90-frame, 20 Hz clip already represents about 4.5 seconds of acquisition before this '
                  'measurement begins. The result is processing latency after a host-resident, clutter-removed '
                  'clip is available; it excludes sensing, file I/O, earlier clutter removal and model loading. '
                  'It does not establish full interaction latency, causal-prefix recognition, power savings '
                  'or a second independent hardware measurement.', '',
                  '## Limits of independent verification', '',
                  '- Upstream and frozen hashes match the saved inventories today; this is internal byte consistency, not an external timestamp witness.',
                  '- Direct-versus-batch-one replay flags were structurally audited and tied to saved visited-logit hashes. '
                  'The GPU logits were not regenerated.',
                  '- Original frozen preprocessing definitions were reexecuted and compared with prepared data. '
                  'This checks reproducibility of the frozen transform, not a separately authored transform.',
                  ])
    if issues:
        lines.extend(['', '## Issues', ''] + [f"- {x['check']}: {x['detail']}" for x in issues[:30]])
    (OUT / 'INDEPENDENT_AUDIT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': output['status'], 'checks': output['total_checks'],
                      'issues': len(issues), 'frozen_files': len(frozen['files']),
                      'upstream_files': len(report['hash_inventory']), 'rows': len(all_rows)}, ensure_ascii=False))
    return 0 if not issues else 1


if __name__ == '__main__':
    sys.exit(main())
