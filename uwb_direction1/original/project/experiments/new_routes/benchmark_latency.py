"""Actual batch-one, on-demand UWB inference latency with auditable boundaries.

The measured input is one host-resident, already clutter-removed 90x189 clip
per requested radar. Acquisition, disk I/O, clutter removal and model loading
are excluded. CPU/smoke results validate execution only, never deployment speed.
CUDA execution requires --training-finished and refuses detected competition.
An explicit idle desktop PID exception is sampled and is never called exclusive.
"""
import argparse
import ast
from contextlib import nullcontext
import gc
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = Path('D:/uwb_gestures/experiments/p1_20260905')
OUTPUT = Path('D:/uwb_gestures/experiments/p1_20260928')
SUBSETS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
NAMES = ('L', 'T', 'R', 'LT', 'LR', 'TR', 'LTR')
EXCLUDED = ['sensor acquisition (90 frames at 20 Hz is about 4.5 seconds)',
            'disk I/O and raw-file parsing', 'prior static-clutter removal',
            'one-time model/policy loading', 'preflight, warmup and replay verification']
INCLUDED = ['requested-view amplitude/analytic-phase-motion preprocessing',
            'per-clip normalization and float16 packing', 'CPU-to-device transfer',
            'newly requested view encoding, cached-feature fusion and CPU output',
            'CPU probabilities and causal routing decisions']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


class HashInventory:
    """Hash shared multi-GB inputs once; recheck when their stat changes."""
    def __init__(self):
        self.files, self.stats = {}, {}

    def verify(self, path, expected=None):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError('Missing required input: '+str(path))
        stat = path.stat()
        stamp, key = (stat.st_size, stat.st_mtime_ns), str(path)
        if self.stats.get(key) != stamp:
            self.files[key] = digest(path)
            self.stats[key] = stamp
        actual = self.files[key]
        if expected is not None and actual != expected:
            raise ValueError('Input hash differs: '+str(path))
        return actual


def verify_files(directory, files, inventory):
    directory = Path(directory).resolve()
    if not isinstance(files, dict) or not files:
        raise ValueError('Missing artifact inventory')
    for name, expected in files.items():
        path = (directory/name).resolve()
        if path == directory or directory not in path.parents:
            raise ValueError('Artifact inventory escapes its directory')
        inventory.verify(path, expected)


def require_mode(config, allow_smoke):
    if config.get('smoke') is not bool(allow_smoke):
        raise ValueError('Formal/smoke artifact mix is forbidden')


def validate_run(allow_smoke, folds, out, sources, device, warmup):
    folds = list(folds)
    if not folds or len(set(folds)) != len(folds) or any(f not in range(1, 9) for f in folds):
        raise ValueError('Distinct folds from 1..8 required')
    if not allow_smoke and sorted(folds) != list(range(1, 9)):
        raise ValueError('Formal latency requires all eight folds')
    out = Path(out).resolve()
    if allow_smoke and (out == (OUTPUT/'latency_s2026').resolve()
                        or (OUTPUT/'latency_s2026').resolve() in out.parents):
        raise ValueError('Smoke requires an independent --out outside latency_s2026')
    for source in sources:
        source = Path(source).resolve()
        if out == source or source in out.parents or out in source.parents:
            raise ValueError('Latency output must be independent of all artifact inputs')
    if device == 'cpu' and not allow_smoke:
        raise ValueError('CPU execution is a smoke validation, not the frozen CUDA benchmark')
    if warmup < 10:
        raise ValueError('At least 10 warmups per path are required')


def process_inventory():
    if os.name == 'nt':
        command = ('[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); '
                   "$ErrorActionPreference = 'Stop'; "
                   'Get-CimInstance Win32_Process | Select-Object ProcessId,Name,ExecutablePath,CreationDate,CommandLine '
                   '| ConvertTo-Json -Compress')
        result = subprocess.run(['powershell', '-NoProfile', '-Command', command],
                                capture_output=True, text=True, encoding='utf-8', errors='strict',
                                check=True, timeout=30)
        rows = json.loads(result.stdout)
        rows = rows if isinstance(rows, list) else [rows]
        if not rows or any(not isinstance(r, dict) or 'ProcessId' not in r for r in rows):
            raise ValueError('Process inventory is empty or malformed')
        return [{'pid': int(r['ProcessId']), 'name': r.get('Name') or '',
                 'executable': r.get('ExecutablePath') or '', 'creation_time': r.get('CreationDate') or '',
                 'cmdline': r.get('CommandLine') or ''} for r in rows]
    import psutil
    rows = [{'pid': p.info['pid'], 'name': p.info['name'] or '', 'executable': p.info['exe'] or '',
             'creation_time': p.info['create_time'], 'cmdline': ' '.join(p.info['cmdline'] or [])}
            for p in psutil.process_iter(['pid', 'name', 'exe', 'create_time', 'cmdline'])]
    if not rows:
        raise ValueError('Process inventory is empty')
    return rows


class GpuCheckError(ValueError):
    """Keep partial sampling evidence when a fail-closed environment check stops."""
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def check_gpu_idle(device, training_finished, training_pids, processes=None,
                   allow_idle_gpu_pids=(), allowed_identities=None):
    if device != 'cuda':
        if allow_idle_gpu_pids:
            raise ValueError('--allow-idle-gpu-pid applies only to CUDA')
        return {'device': device, 'training_finished_assertion': False, 'gpu_checks': 'not applicable'}
    allowed = list(allow_idle_gpu_pids)
    audit = {'device': device, 'state': 'checking', 'training_finished_assertion': bool(training_finished),
             'known_training_pids': list(training_pids), 'training_conflicts': [],
             'allowed_idle_gpu_pids': allowed, 'allowed_idle_gpu_identities': [],
             'gpu_compute_snapshots': [], 'gpu_utilization_samples': [], 'gpu_exclusive': False,
             'limitations': 'Process snapshots and idle sampling do not guarantee an exclusive GPU or '
                 'absence of desktop interference during timing.'}
    def reject(message):
        audit.update(state='rejected', reason=message)
        raise GpuCheckError(message, audit)
    if not training_finished:
        reject('CUDA requires explicit --training-finished after all GPU training stops')
    if os.getpid() in allowed:
        reject('The benchmark own PID cannot be an idle desktop exception')
    if (len(set(allowed)) != len(allowed) or
            any(not isinstance(pid, Integral) or isinstance(pid, bool) or pid <= 0 for pid in allowed)):
        reject('Idle desktop exceptions require distinct positive PIDs')
    markers = ('multiview.py', 'train_baseline.py', 'cross_user_meta/trainer.py',
               'cross_user_meta\\trainer.py')
    def query(command):
        return subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='strict', timeout=20)
    def snapshot(expected):
        try:
            rows = process_inventory() if processes is None else processes
            conflicts = [r['pid'] for r in rows if r['pid'] != os.getpid() and
                         (r['pid'] in training_pids or ('python' in r['cmdline'].lower()
                          and any(m in r['cmdline'].lower() for m in markers)))]
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
            reject('Cannot read process inventory: '+str(error))
        audit['training_conflicts'] = conflicts
        if conflicts:
            reject('Active training process conflict: '+str(conflicts))
        by_pid = {r['pid']: r for r in rows}
        if len(by_pid) != len(rows):
            reject('Process inventory has duplicate PID identities')
        identities = []
        for pid in allowed:
            row = by_pid.get(pid)
            if row is None:
                reject('Allowed desktop process identity disappeared: '+str(pid))
            text = ' '.join(str(row.get(k) or '') for k in ('name', 'executable', 'cmdline')).lower()
            if ('python' in text or any(m in text for m in markers)
                    or any(m in text for m in ('torchrun', 'jupyter', 'tensorflow', 'pytorch'))):
                reject('Python or identifiable training process cannot be an idle desktop exception: '+str(pid))
            identity = {k: row.get(k) for k in ('pid', 'name', 'executable', 'creation_time', 'cmdline')}
            if (any(not identity[k] for k in ('name', 'executable', 'creation_time'))
                    or not os.path.isabs(identity['executable'])
                    or os.path.basename(identity['executable']).casefold() != identity['name'].casefold()):
                reject('Incomplete desktop process identity: '+str(pid))
            identities.append(identity)
        if expected is not None and identities != expected:
            reject('Allowed desktop process identity changed')
        try:
            result = query(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'])
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            reject('Cannot read GPU compute process inventory: '+str(error))
        if result.returncode:
            reject('Cannot read GPU compute process inventory: '+result.stderr.strip())
        compute = {}
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(',', 1)]
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1]:
                reject('Malformed GPU compute process inventory: '+line)
            if int(parts[0]) != os.getpid():
                compute[int(parts[0])] = parts[1]
        audit['gpu_compute_snapshots'].append({'observed_at_unix': time.time(),
                                              'processes': {str(pid): name for pid, name in compute.items()}})
        unexpected = {pid: name for pid, name in compute.items() if pid not in allowed}
        if unexpected:
            reject('Other GPU compute processes are active: '+str(unexpected))
        for row in identities:
            pid = row['pid']
            if expected is None and pid not in compute:
                reject('Allowed desktop PID has no current GPU image path to verify: '+str(pid))
            if pid in compute and os.path.normcase(os.path.normpath(compute[pid])) != os.path.normcase(os.path.normpath(row['executable'])):
                reject('Allowed desktop GPU image path differs from its process identity: '+str(pid))
        audit['other_gpu_compute_processes'] = {str(pid): name for pid, name in compute.items()}
        return identities
    identities = snapshot(allowed_identities)
    audit['allowed_idle_gpu_identities'] = identities
    if allowed:
        audit['idle_sampling_rule'] = {'samples': 5, 'interval_seconds': 1., 'last_samples_required': 3,
                                       'max_utilization_percent': 5., 'scope': 'all GPUs returned by nvidia-smi'}
        # Delay before each observation. The first two values may retain this
        # benchmark's just-finished activity; all final three must be <= 5%.
        for index in range(5):
            time.sleep(1.)
            sample = {'index': index, 'observed_at_unix': time.time()}
            audit['gpu_utilization_samples'].append(sample)
            try:
                result = query(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
                sample.update(raw=result.stdout.strip(), stderr=result.stderr.strip(), returncode=result.returncode)
                if result.returncode:
                    reject('Cannot sample GPU utilization: '+result.stderr.strip())
                values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
                if not values or any(not np.isfinite(value) or value < 0 or value > 100 for value in values):
                    reject('GPU utilization sample is unavailable or invalid')
                sample['utilization_percent'] = values
            except GpuCheckError:
                raise
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                sample['error'] = str(error)
                reject('Cannot sample GPU utilization: '+str(error))
        samples = audit['gpu_utilization_samples']
        if (len({len(s['utilization_percent']) for s in samples}) != 1 or
                any(value > 5. for s in samples[-3:] for value in s['utilization_percent'])):
            reject('GPU utilization was not idle in all final three samples')
        snapshot(identities)
    audit['state'] = 'verified'
    return audit


def select_clips(labels, subjects, folds):
    labels, subjects, folds = np.asarray(labels), np.asarray(subjects), list(folds)
    if (labels.ndim != 1 or labels.shape != subjects.shape or not len(labels)
            or not np.isin(labels, np.arange(12)).all()
            or not np.isin(subjects, np.arange(1, 9)).all()
            or not folds or len(set(folds)) != len(folds) or any(f not in range(1, 9) for f in folds)):
        raise ValueError('Aligned labels0..11/subjects1..8 and distinct folds1..8 required')
    good = ~(((subjects == 2) & np.isin(labels, [10, 11])) | ((subjects == 6) & (labels == 3)))
    selected = {}
    for fold in folds:
        ids = []
        for label in range(12):
            candidates = np.flatnonzero(good & (subjects == fold) & (labels == label))
            if not len(candidates):
                if (fold == 2 and label in (10, 11)) or (fold == 6 and label == 3):
                    continue
                raise ValueError('Unexpected missing held-user class')
            if len(candidates) < 3:
                raise ValueError('At least three clips per available held-user class required')
            ids.extend(np.random.default_rng(44000+fold*100+label).choice(candidates, 3, replace=False))
        selected[fold] = np.asarray(ids, dtype=np.int64)
    return selected


def subset_index(value):
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value not in range(7):
        raise ValueError('Subset index must be an integer0..6')
    return int(value)


def batch_one(rows):
    rows = np.asarray(rows)
    if rows.shape != (1,) or not np.issubdtype(rows.dtype, np.integer) or rows[0] != 0:
        raise ValueError('Runtime reader supports exactly batch-one row [0]')


class IncrementalReader:
    """Only acquisition boundary used by timed shared-model routing.

    The read_view callback touches one host view. Each unique requested view is
    preprocessed/encoded once; features occupy sorted physical slots at fusion.
    """
    def __init__(self, read_view, prepare_view, backend):
        self.read_view, self.prepare_view, self.backend = read_view, prepare_view, backend
        self.features, self.logits = {}, {}
        self.raw_accesses, self.encoded_views, self.requested_subsets = [], [], []
        self.stages = {'preprocess_pack_ms': 0., 'reader_total_ms': 0.}

    def __call__(self, rows, subset):
        batch_one(rows)
        subset = subset_index(subset)
        indices = SUBSETS[subset]
        if not set(self.features).issubset(indices):
            raise ValueError('A causal request must be a superset of already observed views')
        started = time.perf_counter_ns()
        self.requested_subsets.append(subset)
        if subset not in self.logits:
            for view in indices:
                if view not in self.features:
                    begin = time.perf_counter_ns()
                    raw = self.read_view(view)
                    self.raw_accesses.append(view)
                    packed = self.prepare_view(raw)
                    self.stages['preprocess_pack_ms'] += (time.perf_counter_ns()-begin)/1e6
                    self.features[view] = self.backend.encode(packed)
                    self.encoded_views.append(view)
            logits = np.asarray(self.backend.fuse([self.features[v] for v in indices], indices))
            if logits.shape != (1, 12) or not np.isfinite(logits).all():
                raise ValueError('Runtime backend must return finite [1,12] CPU logits')
            self.logits[subset] = logits
        self.stages['reader_total_ms'] += (time.perf_counter_ns()-started)/1e6
        return self.logits[subset]


class ArrayReader:
    """Separate batch-one offline replay; never used in measured paths."""
    def __init__(self, logits):
        self.logits = np.asarray(logits)
        if self.logits.shape != (1, 7, 12) or not np.isfinite(self.logits).all():
            raise ValueError('Replay requires finite [1,7,12] logits')

    def __call__(self, rows, subset):
        batch_one(rows)
        return self.logits[:, subset_index(subset)]


def single_route(reader, subset):
    subset = subset_index(subset)
    logits = reader(np.array([0], dtype=np.int64), subset)
    return {'prediction': logits.argmax(1), 'subset_indices': np.array([subset]),
            'costs': np.array([len(SUBSETS[subset])]), 'path': np.array([[subset, -1, -1]])}


def causal_cascade(reader, params):
    from budget_analysis import uncertainty
    first, pair = subset_index(params['first']), subset_index(params['pair'])
    if first > 2 or pair not in range(3, 6) or not set(SUBSETS[first]).issubset(SUBSETS[pair]):
        raise ValueError('Cascade needs a singleton and its containing pair')
    if any(not np.isfinite(params[k]) for k in ('threshold1', 'threshold2')):
        raise ValueError('Finite frozen cascade thresholds required')
    current, path = first, [first, -1, -1]
    logits = reader(np.array([0], dtype=np.int64), first)
    if uncertainty(logits, params['mode'])[0] > params['threshold1']:
        current, path[1] = pair, pair
        logits = reader(np.array([0], dtype=np.int64), pair)
        if uncertainty(logits, params['mode'])[0] > params['threshold2']:
            current, path[2] = 6, 6
            logits = reader(np.array([0], dtype=np.int64), 6)
    return {'prediction': logits.argmax(1), 'subset_indices': np.array([current]),
            'costs': np.array([len(SUBSETS[current])]), 'path': np.array([path])}


def all_subset_cache(reader):
    if reader.features or reader.raw_accesses:
        raise ValueError('Offline replay must start with a separate empty feature cache')
    reader(np.array([0], dtype=np.int64), 6)
    logits = [reader.backend.fuse([reader.features[v] for v in subset], subset) for subset in SUBSETS]
    return np.stack(logits, axis=1)


def verify_replay(reader, direct, cache, replay):
    for subset, logits in reader.logits.items():
        if not np.array_equal(logits, cache[:, subset]):
            raise ValueError('Direct/replay batch-one logits differ for '+NAMES[subset])
    for key in ('prediction', 'subset_indices', 'costs', 'path'):
        if not np.array_equal(direct[key], replay[key]):
            raise ValueError('Direct/replay action path or prediction differs: '+key)
    count = int(direct['costs'][0])
    expected = set(SUBSETS[int(direct['subset_indices'][0])])
    if (set(reader.raw_accesses) != expected or set(reader.encoded_views) != expected
            or len(reader.raw_accesses) != count or len(reader.encoded_views) != count):
        raise ValueError('Actual raw access/encoding count differs from route cost')
    return {'bitwise_equal': True, 'checked_subset_indices': sorted(reader.logits),
            'actual_views': count}


def summarize(rows):
    if not rows:
        raise ValueError('Cannot summarize empty latency observations')
    result = {}
    for method in sorted(set(row['method'] for row in rows)):
        selected = [row for row in rows if row['method'] == method]
        times = np.array([row['elapsed_ms'] for row in selected], dtype=float)
        costs = np.array([row['actual_views'] for row in selected], dtype=float)
        if not np.isfinite(times).all() or (times < 0).any() or not np.isin(costs, [1, 2, 3]).all():
            raise ValueError('Finite nonnegative milliseconds and actual view counts1..3 required')
        users = sorted(set(row['fold'] for row in selected))
        per_user = {str(user): float(np.mean([r['elapsed_ms'] for r in selected if r['fold'] == user])) for user in users}
        result[method] = {'unit': 'ms', 'observations': len(selected), 'mean_ms': float(times.mean()),
                          'median_ms': float(np.median(times)), 'p95_ms': float(np.quantile(times, .95)),
                          'quantile_method': 'numpy linear', 'per_user_mean_ms': per_user,
                          'per_user_macro_mean_ms': float(np.mean(list(per_user.values()))),
                          'mean_actual_views': float(costs.mean()),
                          'view_count_histogram': {str(c): int(np.sum(costs == c)) for c in (1, 2, 3)}}
    return result


def extract_functions(path, names, namespace):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names) or len(nodes) != len(names):
        raise ValueError('Original preprocessing definition missing or duplicated: '+str(path))
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)


class Preprocessor:
    """Execute only the frozen original definitions, not pipeline module code."""
    def __init__(self, pipeline=OLD/'pipeline.py', vendor=OLD/'vendor/uwb_osr/data.py'):
        self.namespace = {'np': np}
        extract_functions(vendor, ['_analytic', 'doppler_map'], self.namespace)
        extract_functions(pipeline, ['norm', 'transform'], self.namespace)

    def __call__(self, raw):
        raw = np.asarray(raw)
        if raw.shape != (90, 189) or not np.isfinite(raw).all():
            raise ValueError('One finite already clutter-removed [90,189] view required')
        return self.namespace['transform'](raw[None, None], 'concat').astype(np.float16)


def partition_hashes(labels, subjects, fold):
    good = ~(((subjects == 2) & np.isin(labels, [10, 11])) | ((subjects == 6) & (labels == 3)))
    parts = {'train': np.flatnonzero(good & ~np.isin(subjects, [fold, fold % 8+1])),
             'val': np.flatnonzero(good & (subjects == fold % 8+1)),
             'test': np.flatnonzero(good & (subjects == fold))}
    return {key: array_digest(value) for key, value in parts.items()}


def read_json(path, inventory):
    inventory.verify(path)
    with Path(path).open(encoding='utf-8-sig') as stream:
        return json.load(stream)


def verify_source(source, inventory):
    for section in ('code', 'data'):
        if not isinstance(source.get(section), dict) or not source[section]:
            raise ValueError('Missing model source/data identity inventory')
        for path, expected in source[section].items():
            inventory.verify(path, expected)
    if 'plan_sha256' in source:
        inventory.verify(HERE.parents[1]/'docs/superpowers/plans/2026-09-28-uwb-new-routes.md', source['plan_sha256'])


def verify_model_run(root, fold, arm, labels, subjects, allow_smoke, inventory):
    root = Path(root)
    directory = root/f'fold{fold}'/('warmup' if arm == 'warmup' else '')
    row = read_json(directory/'result.json', inventory)
    identity = row['identity']
    manifest = read_json(root/'manifest.json', inventory)
    config = identity['config']
    require_mode(config, allow_smoke)
    if (identity['fold'] != fold or identity['arm'] != arm
            or manifest != {'source': identity['source'], 'config': config}
            or identity['partitions'] != partition_hashes(labels, subjects, fold)):
        raise ValueError('Model fold, source manifest, or whole-user split identity differs')
    required = {'seed': 2026, 'attention': 'MATH', 'weight_decay': 1e-4,
                'smoke_steps': 3 if allow_smoke else 0}
    if arm == 'warmup':
        required.update(warmup_epochs=1 if allow_smoke else 40, warmup_min=1 if allow_smoke else 12,
                        warmup_patience=8, warmup_batch=128, warmup_microbatch=32, warmup_lr=1e-4)
        files = {'right.pt', 'best.pt', 'progress.pt', 'predictions.npz'}
    elif arm == 'multiview':
        required.update(fusion_seed=2026, batch=128, microbatch=16, lr=1e-4, eval_batch=16,
                        epochs=1 if allow_smoke else 40, min_epochs=1 if allow_smoke else 12,
                        patience=8, criterion='cross_entropy only', subsets=[list(s) for s in SUBSETS])
        files = {'best.pt', 'initial.pt', 'progress.pt', 'predictions.npz', 'split.json'}
    else:
        raise ValueError('Unknown measured model arm')
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError('Model training configuration differs from the frozen protocol')
    verify_source(identity['source'], inventory)
    if not files.issubset(row['files']):
        raise ValueError('Incomplete model checkpoint artifact inventory')
    verify_files(directory, row['files'], inventory)
    if arm == 'multiview':
        initial = identity['initial_checkpoint']
        if (initial['model_identity']['fold'] != fold or initial['model_identity']['arm'] != 'warmup'
                or initial['model_identity']['partitions'] != identity['partitions']):
            raise ValueError('Multiview initializer fold/split lineage mismatch')
        inventory.verify(initial['right_path'], initial['right_sha256'])
        initializer_dir = Path(initial['right_path']).parent
        inventory.verify(initializer_dir/'result.json', initial['result_sha256'])
        initializer_root = initializer_dir.parents[1]
        inventory.verify(initializer_root/'manifest.json', initial['manifest_sha256'])
        verify_source(initial['model_identity']['source'], inventory)
    return row


def policy_requirement(summary, allow_smoke, mode, complete):
    if not allow_smoke:
        if (not summary or summary.get('state') != 'complete' or summary.get('opportunity_decision')
                not in ('STOP_NO_ORACLE_ROOM', 'POLICY_PILOT_WARRANTED')):
            raise ValueError('Formal latency needs the complete frozen eight-fold budget gate')
        if summary['opportunity_decision'] == 'STOP_NO_ORACLE_ROOM':
            if mode == 'required':
                raise ValueError('Value policy cannot be required after the frozen oracle STOP')
            return False
    if mode == 'off':
        return False
    if complete:
        return True
    if not allow_smoke or mode == 'required':
        raise ValueError('Requested value-policy artifacts are not complete')
    return False


def measure(reader, route, synchronize):
    synchronize()
    started = time.perf_counter_ns()
    result = route(reader)
    synchronize()
    elapsed = (time.perf_counter_ns()-started)/1e6
    count = int(result['costs'][0])
    if (len(reader.raw_accesses) != count or len(set(reader.raw_accesses)) != count
            or reader.raw_accesses != reader.encoded_views):
        raise ValueError('Timed execution repeated or silently acquired an unrequested view')
    stages = dict(reader.stages)
    stages.update(getattr(reader.backend, 'stages', {}))
    stages['routing_and_harness_ms'] = max(0., elapsed-reader.stages['reader_total_ms'])
    return result, {'elapsed_ms': elapsed, 'actual_views': count,
                    'raw_accesses': list(reader.raw_accesses), 'encoded_views': list(reader.encoded_views),
                    'requested_subsets': list(reader.requested_subsets), 'stages': stages}


class TorchBackend:
    """Identical [1,1,90,378] encoder shape for every newly acquired view.

    Host stage timers do not add CUDA synchronization inside the measured path.
    Thus enqueue timings are descriptive; output_cpu_ms includes pending GPU
    work. The separately synchronized end-to-end elapsed_ms is authoritative.
    """
    def __init__(self, model, device, baseline=False):
        import torch
        self.torch, self.model = torch, model
        self.device, self.baseline = torch.device(device), baseline
        self.stages = {name: 0. for name in ('input_transfer_host_ms', 'encode_host_ms',
                                            'fuse_host_ms', 'output_cpu_ms')}

    def timed(self, name, function):
        started = time.perf_counter_ns()
        value = function()
        self.stages[name] += (time.perf_counter_ns()-started)/1e6
        return value

    def encode(self, packed):
        torch = self.torch
        dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        x = self.timed('input_transfer_host_ms', lambda: torch.from_numpy(packed).to(self.device, dtype=dtype))
        return x if self.baseline else self.timed('encode_host_ms', lambda: self.model.encode(x))

    def fuse(self, features, indices):
        if self.baseline:
            if tuple(indices) != (2,) or len(features) != 1:
                raise ValueError('Independent baselineRight processes physical Right only')
            z = self.timed('encode_host_ms', lambda: self.model(features[0])[0])
        else:
            z = self.timed('fuse_host_ms', lambda: self.model.fuse(self.torch.cat(features, dim=1), indices))
        return self.timed('output_cpu_ms', lambda: z.detach().float().cpu().numpy())


def fixed_selections(selection):
    accuracy = np.asarray(selection['validation_fixed_accuracy'], dtype=float)
    if accuracy.shape != (7,) or not np.isfinite(accuracy).all() or ((accuracy < 0) | (accuracy > 1)).any():
        raise ValueError('Seven finite validation accuracies in0..1 required')
    return {str(cost): max((i for i, s in enumerate(SUBSETS) if len(s) == cost),
                          key=lambda i: (round(float(accuracy[i]), 12), -i)) for cost in (1, 2, 3)}


def method_specs(selection, policies, policy_selection):
    specs = {'baselineRight': {'model': 'right', 'route': lambda r: single_route(r, 2)}}
    for cost, subset in fixed_selections(selection).items():
        specs['fixed'+cost] = {'model': 'shared', 'route': lambda r, s=subset: single_route(r, s)}
    params = selection['curves']['2.0']['params']
    specs['frozen_budget2_cascade'] = {'model': 'shared', 'route': lambda r: causal_cascade(r, params)}
    if policies is not None:
        from value_policy import route_policy
        if set(policies) != {'class_gain', 'extra_trees'}:
            raise ValueError('Complete class_gain and extra_trees policies required')
        for method in ('class_gain', 'extra_trees'):
            policy = policies[method]
            price = policy_selection['selected_prices'][method]['price']
            if policy.first != params['first'] or policy.method != method:
                raise ValueError('Policy object differs from the frozen starting view or method')
            specs[method] = {'model': 'shared',
                             'route': lambda r, p=policy, c=price: route_policy(r, p, c, n_rows=1)}
    return specs


def dump(path, value):
    path = Path(path)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(encoded, encoding='utf-8')
    temporary.replace(path)


def make_report(rows, environment, allow_smoke, folds, command, inventory):
    methods = summarize(rows)
    for method, summary in methods.items():
        matching = [row for row in rows if row['method'] == method]
        summary['offline_batch_prediction_differences'] = sum(bool(r.get('offline_prediction_differs')) for r in matching)
        summary['offline_batch_route_differences'] = sum(bool(r.get('offline_path_differs')) for r in matching)
    desktop_context = bool(environment.get('allowed_idle_gpu_identities'))
    classification = 'desktop_context_gpu_latency' if desktop_context else 'isolated_gpu_latency'
    return {'state': 'complete', 'classification': 'execution_smoke_only' if allow_smoke else classification,
            'scientific_latency_claim': False, 'folds': list(folds), 'methods': methods,
            'included': INCLUDED, 'excluded': EXCLUDED, 'environment': environment,
            'batch_one_replay': 'bitwise_equal', 'hash_inventory': inventory,
            'command_argv': command, 'command_powershell': '' if not command else
                '& '+ ' '.join("'"+str(arg).replace("'", "''")+"'" for arg in command),
            'stage_interpretation': 'Host wall-clock stages, not independent CUDA kernel durations. '
                'Output transfer waits for outstanding GPU work; reader_total overlaps its sub-stages. '
                'Routing_and_harness includes CPU policy, bookkeeping and final synchronization.',
            'limits': '279 representative held-out clips in a complete run; latency samples do not replace '
                'the full target-user performance evaluation. Batch-one numerical changes are logged without '
                'retuning thresholds or prices. No causal-prefix, acquisition-time or hardware-power claim.' +
                (' GPU is not exclusive: an explicitly allowed desktop context remains. Idle sampling '
                 'does not guarantee absence of desktop interference throughout timing.' if desktop_context else '')}


def raw_preflight(args, inventory):
    manifest = read_json(args.raw_manifest, inventory)
    hashes = manifest['hashes']
    def expected(name):
        matches = [sha for path, sha in hashes.items() if Path(path).name == name]
        if len(matches) != 1:
            raise ValueError('Raw manifest missing a unique input identity: '+name)
        return matches[0]
    inventory.verify(args.raw, expected('X_clutterremoved.npy'))
    for name in ('y', 'subj'):
        inventory.verify(args.raw.parent/(name+'_clutterremoved.npy'), expected(name+'_clutterremoved.npy'))
        inventory.verify(args.prepared/(name+'.npy'), expected(name+'_clutterremoved.npy'))
    labels, subjects = (np.load(args.prepared/(name+'.npy'), allow_pickle=False) for name in ('y', 'subj'))
    if len(labels) != 9593:
        raise ValueError('Frozen raw corpus must contain 9593 clips')
    all_selected = select_clips(labels, subjects, range(1, 9))
    all_ids = np.concatenate(list(all_selected.values()))
    reference = read_json(OUTPUT/'LATENCY_PREFLIGHT.json', inventory)
    selected_sha = hashlib.sha256(all_ids.tobytes()).hexdigest()
    if (reference['sampled_clips'] != 279 or reference['per_view_checks'] != 837
            or selected_sha != reference['selected_ids_sha256']):
        raise ValueError('Sampled IDs differ from the frozen latency preflight')
    inventory.verify(OLD/'pipeline.py')
    inventory.verify(OLD/'vendor/uwb_osr/data.py', expected('data.py'))
    processor = Preprocessor()
    raw = np.load(args.raw, mmap_mode='r', allow_pickle=False)
    amplitude = np.load(args.prepared/'off.npy', mmap_mode='r', allow_pickle=False)
    phase = np.load(args.prepared/'only.npy', mmap_mode='r', allow_pickle=False)
    if (list(raw.shape) != reference['raw_shape'] or amplitude.shape != raw.shape or phase.shape != raw.shape
            or amplitude.dtype != np.float16 or phase.dtype != np.float16):
        raise ValueError('Raw/prepared tensor shape or storage dtype differs')
    selected = {fold: all_selected[fold] for fold in args.folds}
    resident, checked = {}, 0
    for fold, ids in selected.items():
        # This whole-host-copy is outside every timed path. The per-view reader
        # below indexes only a requested physical view from this resident array.
        resident[fold] = np.array(raw[ids], copy=True, order='C')
        for offset, clip_id in enumerate(ids):
            for view in range(3):
                packed = processor(resident[fold][offset, view])
                if (not np.array_equal(packed[0, 0, :, :189], amplitude[clip_id, view])
                        or not np.array_equal(packed[0, 0, :, 189:], phase[clip_id, view])):
                    raise ValueError(f'Prepared input bitwise replay differs: clip={clip_id}, view={view}')
                checked += 1
    return {'labels': labels, 'subjects': subjects, 'selected': selected, 'resident': resident,
            'processor': processor, 'audit': {'prepared_replay': 'bitwise_equal', 'sampled_clips': checked//3,
                'per_view_checks': checked, 'selected_ids_sha256': selected_sha,
                'selected_ids_sha256_scope': 'all eight folds, int64 bytes in fold/class/choice order',
                'selected_ids': {str(f): ids.tolist() for f, ids in selected.items()},
                'raw_shape': list(raw.shape), 'host_resident_bytes': sum(a.nbytes for a in resident.values())}}


def artifact_preflight(args, data, inventory):
    contexts = {}
    summary = None if args.allow_smoke else read_json(args.budget/'summary.json', inventory)
    policy_complete = all((args.policies/f'fold{fold}'/'result.json').is_file() for fold in args.folds)
    include_policy = policy_requirement(summary, args.allow_smoke, args.policy, policy_complete)
    policy_manifest = read_json(args.policies/'manifest.json', inventory) if include_policy else None
    if include_policy and not args.allow_smoke:
        policy_summary = read_json(args.policies/'summary.json', inventory)
        if policy_summary.get('state') != 'complete' or policy_summary.get('users') != 8:
            raise ValueError('Formal policy benchmark requires all eight complete policy artifacts')
    for fold in args.folds:
        baseline = verify_model_run(args.baseline, fold, 'warmup', data['labels'], data['subjects'], args.allow_smoke, inventory)
        shared = verify_model_run(args.models, fold, 'multiview', data['labels'], data['subjects'], args.allow_smoke, inventory)
        for row, filenames in ((baseline, ('train_baseline.py',)), (shared, ('train_baseline.py', 'multiview.py'))):
            if any(str(HERE/name) not in row['identity']['source']['code'] for name in filenames):
                raise ValueError('Model identity does not cover the actual inference implementation')
            for name in ('off.npy', 'only.npy', 'y.npy', 'subj.npy'):
                if str((args.prepared/name).resolve()) not in row['identity']['source']['data']:
                    raise ValueError('Prepared path is not covered by the model source identity')
        directory = args.budget/f'fold{fold}'
        budget = read_json(directory/'result.json', inventory)
        selection = read_json(directory/'selection.json', inventory)
        uid = budget['identity']
        require_mode(uid, args.allow_smoke)
        if (budget['fold'] != fold or uid['predictions_sha256'] != inventory.verify(args.models/f'fold{fold}'/'predictions.npz')
                or uid['upstream_result_sha256'] != inventory.verify(args.models/f'fold{fold}'/'result.json')
                or uid['manifest_sha256'] != inventory.verify(args.models/'manifest.json')
                or selection['identity'] != uid or selection.get('uses_target_labels') is not False):
            raise ValueError('Frozen budget source/selection identity differs from the measured model')
        inventory.verify(HERE/'budget_analysis.py', uid['analysis_sha256'])
        inventory.verify(HERE/'run_calibration.py', uid['statistics_sha256'])
        if not {'selection.json', 'cascade_predictions.npz'}.issubset(budget['files']):
            raise ValueError('Incomplete frozen budget artifacts')
        verify_files(directory, budget['files'], inventory)
        if budget['fixed_validation_selected'] != fixed_selections(selection):
            raise ValueError('Fixed model choices differ from frozen validation selection')
        # Verify validation-only numbers without selecting anything from target
        # performance. No archive target row is opened by this calculation.
        with np.load(args.models/f'fold{fold}'/'predictions.npz', allow_pickle=False) as archive:
            from budget_analysis import subset_accuracies
            val_acc = subset_accuracies(archive['val_logits'], archive['val_y'])
            if not np.allclose(val_acc, selection['validation_fixed_accuracy'], rtol=0, atol=1e-12):
                raise ValueError('Validation fixed-subset accuracy does not reproduce frozen choices')
        policy_selection = None
        if include_policy:
            pdir = args.policies/f'fold{fold}'
            policy_row = read_json(pdir/'result.json', inventory)
            policy_selection = read_json(pdir/'selection.json', inventory)
            pid = policy_row['identity']
            require_mode(pid, args.allow_smoke)
            require_mode(policy_manifest, args.allow_smoke)
            expected_summary = None if args.allow_smoke else inventory.verify(args.budget/'summary.json')
            if (pid['fold'] != fold or policy_row['fold'] != fold or policy_selection['identity'] != pid
                    or pid['budget_result_sha256'] != inventory.verify(directory/'result.json')
                    or pid['budget_selection_sha256'] != inventory.verify(directory/'selection.json')
                    or pid['budget_identity'] != uid or policy_selection.get('uses_target_labels') is not False
                    or policy_selection['frozen_cascade_params'] != selection['curves']['2.0']['params']
                    or pid['opportunity_summary_sha256'] != expected_summary
                    or policy_manifest['summary_sha256'] != expected_summary
                    or policy_manifest['source'] != pid['source'] or policy_manifest['config'] != pid['config']
                    or Path(policy_manifest['models']).resolve() != args.models.resolve()
                    or Path(policy_manifest['budget']).resolve() != args.budget.resolve()):
                raise ValueError('Value-policy source, price selection or frozen upstream identity differs')
            if set(policy_row['files']) != {'selection.json', 'models.pkl', 'predictions.npz', 'fit_log.json'}:
                raise ValueError('Incomplete policy artifact inventory')
            verify_files(pdir, policy_row['files'], inventory)
            for path, sha in pid['source']['files'].items():
                inventory.verify(path, sha)
            if str(HERE/'value_policy.py') not in pid['source']['files']:
                raise ValueError('Policy identity does not cover the actual runtime router')
        contexts[fold] = {'baseline': baseline, 'shared': shared, 'selection': selection,
                          'policy_selection': policy_selection}
    return contexts


def configure_torch(device):
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {'device': device, 'python': platform.python_version(), 'executable': sys.executable,
              'platform': platform.platform(), 'numpy': np.__version__, 'torch': torch.__version__,
              'cuda_runtime': torch.version.cuda, 'threads': torch.get_num_threads(),
              'interop_threads': torch.get_num_interop_threads(), 'batch': 1,
              'weights': 'float32', 'packed_inputs': 'float16',
              'inference_inputs': 'float16' if device == 'cuda' else 'float32',
              'autocast': 'bfloat16' if device == 'cuda' else 'disabled CPU smoke',
              'attention': 'MATH', 'tf32': False, 'cudnn_benchmark': False,
              'cudnn_deterministic': True, 'timer': 'perf_counter_ns; explicit device synchronize before/after'}
    if device == 'cuda':
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError('Frozen CUDA BF16 configuration is unavailable')
        properties = torch.cuda.get_device_properties(0)
        result.update(gpu_name=properties.name, gpu_total_memory=properties.total_memory,
                      capability=list(torch.cuda.get_device_capability(0)))
        info = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,name,power.limit', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=20)
        result['nvidia_smi_environment'] = info.stdout.strip()
    return result


def load_models(args, fold, context):
    # All model/source inventories are validated before this function is used.
    import torch
    from multiview import SubsetModel
    from train_baseline import T
    bdir, mdir = args.baseline/f'fold{fold}'/'warmup', args.models/f'fold{fold}'
    best_right = torch.load(bdir/'best.pt', map_location='cpu', weights_only=True)
    right_state = torch.load(bdir/'right.pt', map_location='cpu', weights_only=True)
    best_shared = torch.load(mdir/'best.pt', map_location='cpu', weights_only=True)
    for best, key in ((best_right, 'baseline'), (best_shared, 'shared')):
        if T.state.state_hash(best['state']) != context[key]['selected_state_sha256']:
            raise ValueError('Selected checkpoint tensor state differs from result identity')
    extracted = {k[len('right.'):]: v for k, v in best_right['state'].items() if k.startswith('right.')}
    if set(extracted) != set(right_state) or any(not torch.equal(v, extracted[k]) for k, v in right_state.items()):
        raise ValueError('Independent Right checkpoint differs from its selected wrapper state')
    right = T.models.Classifier([2], 'concat', 'mean')
    right.load_state_dict(right_state, strict=True)
    shared = SubsetModel()
    shared.load_state_dict(best_shared['state'], strict=True)
    return {'right': right.to(args.device).eval(), 'shared': shared.to(args.device).eval()}


def new_reader(clip, processor, model, device, baseline=False):
    return IncrementalReader(lambda view: clip[view], processor, TorchBackend(model, device, baseline))


def load_offline(args, fold, selected):
    output = {}
    for name, path in (('right', args.baseline/f'fold{fold}'/'warmup'/'predictions.npz'),
                       ('shared', args.models/f'fold{fold}'/'predictions.npz')):
        with np.load(path, allow_pickle=False) as archive:
            ids = archive['ids']
            if len(np.unique(ids)) != len(ids):
                raise ValueError('Offline comparison has duplicate IDs')
            lookup = {int(clip_id): offset for offset, clip_id in enumerate(ids)}
            if any(int(clip_id) not in lookup for clip_id in selected):
                raise ValueError('Selected held-out sample missing from offline archive')
            rows = np.array([lookup[int(clip_id)] for clip_id in selected])
            if name == 'shared' and not np.all(archive['role'][rows] == 2):
                raise ValueError('Offline comparison attempted to leave the held-out role')
            output[name] = archive['logits'][rows]
    return output


def run_fold(args, fold, data, context):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    models = load_models(args, fold, context)
    policies = None
    if context['policy_selection'] is not None:
        import pickle
        import value_policy  # register canonical class module before verified unpickling
        with (args.policies/f'fold{fold}'/'models.pkl').open('rb') as stream:
            policies = pickle.load(stream)
    specs = method_specs(context['selection'], policies, context['policy_selection'])
    selected, resident = data['selected'][fold], data['resident'][fold]
    offline = load_offline(args, fold, selected)
    synchronize = torch.cuda.synchronize if args.device == 'cuda' else lambda: None
    precision = torch.autocast('cuda', dtype=torch.bfloat16) if args.device == 'cuda' else nullcontext()
    rng = np.random.default_rng(args.order_seed+fold)
    observations = []
    def reader_for(clip, model):
        return new_reader(clip, data['processor'], models[model], args.device, baseline=model == 'right')
    # Every possible shared causal feature path is warmed, including direct
    # singleton->triple actions that a policy may never select on warmup clips.
    paths = [(first, pair, 6) for first in range(3) for pair in range(3, 6) if first in SUBSETS[pair]]
    paths += [(first, 6) for first in range(3)]
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), precision:
        for repeat in range(args.warmup):
            clip = resident[repeat % len(resident)]
            for spec in specs.values():
                spec['route'](reader_for(clip, spec['model']))
            for path in paths:
                reader = reader_for(clip, 'shared')
                for subset in path:
                    reader(np.array([0], dtype=np.int64), subset)
        synchronize()
        for offset, clip_id in enumerate(selected):
            clip = resident[offset]
            measured = []
            for repeat in range(args.repeats):
                order = rng.permutation(list(specs)).tolist()
                for position, name in enumerate(order):
                    spec = specs[name]
                    reader = reader_for(clip, spec['model'])
                    route, row = measure(reader, spec['route'], synchronize)
                    row.update(fold=fold, clip_id=int(clip_id), label=int(data['labels'][clip_id]),
                               method=name, repeat=repeat, order_position=position,
                               method_order=order, subset_index=int(route['subset_indices'][0]),
                               subset_name=NAMES[int(route['subset_indices'][0])],
                               path=route['path'][0].tolist(), prediction=int(route['prediction'][0]))
                    measured.append((reader, route, row))
            # Build genuinely separate replay caches after all measured paths.
            shared_cache = all_subset_cache(reader_for(clip, 'shared'))
            right_replay = reader_for(clip, 'right')
            right_route = single_route(right_replay, 2)
            right_cache = np.zeros((1, 7, 12), dtype=np.float32)
            right_cache[:, 2] = right_replay.logits[2]
            for reader, route, row in measured:
                spec = specs[row['method']]
                baseline = spec['model'] == 'right'
                cache = right_cache if baseline else shared_cache
                replay = right_route if baseline else spec['route'](ArrayReader(cache))
                row['replay'] = verify_replay(reader, route, cache, replay)
                reference = np.zeros((1, 7, 12), dtype=np.float32) if baseline else offline['shared'][offset:offset+1]
                if baseline:
                    reference[:, 2] = offline['right'][offset]
                original_route = spec['route'](ArrayReader(reference))
                row['offline_prediction_differs'] = bool(route['prediction'][0] != original_route['prediction'][0])
                row['offline_path_differs'] = bool(not np.array_equal(route['path'], original_route['path']))
                row['offline_subset_index'] = int(original_route['subset_indices'][0])
                row['offline_prediction'] = int(original_route['prediction'][0])
                row['offline_path'] = original_route['path'][0].tolist()
                row['visited_logits_sha256'] = {str(s): array_digest(z) for s, z in reader.logits.items()}
                row['final_logits'] = reader.logits[row['subset_index']][0].tolist()
                row['offline_max_abs_logit_difference_on_visited_subsets'] = max(
                    float(np.abs(z-reference[:, s]).max()) for s, z in reader.logits.items())
                observations.append(row)
            if (offset+1) % 6 == 0 or offset == len(selected)-1:
                print(f'LATENCY fold={fold} clips={offset+1}/{len(selected)} replay=bitwise_equal', flush=True)
    warmup = {'iterations_per_method': args.warmup, 'iterations_per_forced_path': args.warmup,
              'forced_shared_paths': [list(path) for path in paths], 'methods': list(specs)}
    del models, policies, measured, reader, shared_cache, right_cache
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    return observations, warmup


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--baseline', type=Path, default=OUTPUT/'baseline_s2026')
    result.add_argument('--models', type=Path, default=OUTPUT/'multiview_s2026')
    result.add_argument('--budget', type=Path, default=OUTPUT/'budget_s2026')
    result.add_argument('--policies', type=Path, default=OUTPUT/'value_policy_s2026')
    result.add_argument('--policy', choices=('auto', 'off', 'required'), default='auto',
                        help='auto includes complete policies; formal oracle STOP skips them')
    result.add_argument('--out', type=Path, default=OUTPUT/'latency_s2026')
    result.add_argument('--folds', type=int, nargs='+', default=list(range(1, 9)))
    result.add_argument('--allow-smoke', action='store_true')
    result.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    result.add_argument('--training-finished', action='store_true',
                        help='explicit assertion from the operator after GPU training has ended')
    result.add_argument('--training-pid', type=int, action='append', default=[])
    result.add_argument('--allow-idle-gpu-pid', type=int, action='append', default=[],
                        help='explicit desktop PID exception, with fixed idle sampling and identity rechecks; not exclusive GPU use')
    result.add_argument('--warmup', type=int, default=10)
    result.add_argument('--repeats', type=int, default=1)
    result.add_argument('--order-seed', type=int, default=45000)
    result.add_argument('--preflight-only', action='store_true', help='verify inputs on CPU without loading models or timing')
    result.add_argument('--raw', type=Path, default=OLD.parents[1]/'cache/X_clutterremoved.npy')
    result.add_argument('--raw-manifest', type=Path, default=OLD/'results/data_manifest.json')
    result.add_argument('--prepared', type=Path, default=OLD/'prepared')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    validate_run(args.allow_smoke, args.folds, args.out, [args.baseline, args.models, args.budget, args.policies], args.device, args.warmup)
    if args.repeats < 1:
        raise ValueError('--repeats must be positive')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Use a new empty output directory; latency artifacts are immutable')
    # Reject incomplete model queues before hashing large raw/prepared arrays.
    for fold in args.folds:
        for path in (args.baseline/f'fold{fold}'/'warmup'/'result.json',
                     args.models/f'fold{fold}'/'result.json', args.budget/f'fold{fold}'/'result.json'):
            if not path.is_file():
                raise ValueError('Incomplete latency input artifacts: '+str(path))
    idle_checks, allowed_identities = [], None
    def verify_environment(phase, fold=None):
        check = {'phase': phase, 'fold': fold}
        try:
            check.update(check_gpu_idle(args.device, args.training_finished, args.training_pid,
                                       allow_idle_gpu_pids=args.allow_idle_gpu_pid,
                                       allowed_identities=allowed_identities))
        except GpuCheckError as error:
            check.update(error.audit)
            raise
        finally:
            idle_checks.append(check)
            # Preserve partial/failed samples too, before any model or timing.
            dump(args.out/'gpu_idle_checks.json', idle_checks)
        return check
    idle = verify_environment('preflight')
    allowed_identities = idle.get('allowed_idle_gpu_identities', [])
    inventory = HashInventory()
    for path in (Path(__file__), HERE/'test_latency.py', OUTPUT/'LATENCY_PROTOCOL.md'):
        inventory.verify(path)
    environment_note = OUTPUT/'LATENCY_ENVIRONMENT_NOTE.md'
    if environment_note.is_file():
        inventory.verify(environment_note)
    data = raw_preflight(args, inventory)
    contexts = artifact_preflight(args, data, inventory)
    command = [sys.executable, str(Path(__file__).resolve())]+(sys.argv[1:] if argv is None else list(argv))
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    audit = {'state': 'verified', 'data': data['audit'], 'configuration': configuration,
             'gpu_idle': idle, 'hash_inventory': inventory.files, 'command_argv': command}
    dump(args.out/'preflight.json', audit)
    if args.preflight_only:
        print('LATENCY_PREFLIGHT_COMPLETE; no model loading or CUDA inference', flush=True)
        return audit
    environment = configure_torch(args.device)
    environment.update(allowed_idle_gpu_identities=allowed_identities,
                       gpu_exclusive=False, gpu_environment_limitations=idle.get('limitations'))
    try:
        from threadpoolctl import threadpool_limits, threadpool_info
        pool_context = threadpool_limits(limits=4)
        environment['threadpools'] = threadpool_info()
    except ImportError:
        pool_context = nullcontext()
        environment['threadpools'] = 'threadpoolctl unavailable; environment limits recorded'
    observations, warmed = [], {}
    try:
        with pool_context:
            for fold in args.folds:
                verify_environment('before_fold', fold)
                rows, warmup = run_fold(args, fold, data, contexts[fold])
                verify_environment('after_fold', fold)
                dump(args.out/f'fold{fold}'/'observations.json', rows)
                dump(args.out/f'fold{fold}'/'summary.json', summarize(rows))
                observations.extend(rows)
                warmed[str(fold)] = warmup
                print(f'LATENCY_FOLD_COMPLETE fold={fold}; {len(rows)} observations', flush=True)
        # Ensure live inference sources did not change during the run. Frozen
        # input files were hashed before model loading; record every identity.
        for path, expected in list(inventory.files.items()):
            inventory.verify(path, expected)
        report = make_report(observations, environment, args.allow_smoke, args.folds, command, inventory.files)
        report.update(configuration=configuration, warmup=warmed, gpu_idle_checks=idle_checks,
                      sample_preflight=data['audit'], order_seed_rule='order_seed + fold; permutation per clip/repetition')
        dump(args.out/'summary.json', report)
        print('LATENCY_COMPLETE; inspect total latency and replay before any speed claim', flush=True)
        return report
    except BaseException as error:
        dump(args.out/'status.json', {'state': 'failed', 'error_type': type(error).__name__, 'error': str(error),
                                     'scientific_latency_claim': False, 'gpu_idle_checks': idle_checks})
        raise


if __name__ == '__main__':
    main()
