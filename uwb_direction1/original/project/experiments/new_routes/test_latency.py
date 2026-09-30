"""CPU contract tests for actual on-demand latency measurement (no CUDA)."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def api():
    path = Path(__file__).with_name('benchmark_latency.py')
    assert path.is_file(), 'Missing the on-demand latency implementation'
    name = 'benchmark_latency'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class TinyBackend:
    """A real deterministic NumPy encoder/fuser; no cached answers."""
    def __init__(self):
        self.inputs = []

    def encode(self, packed):
        self.inputs.append(np.array(packed, copy=True))
        return np.asarray(packed).reshape(1, -1).mean(1, keepdims=True)

    def fuse(self, features, indices):
        scores = np.zeros((1, 12), dtype=np.float32)
        for value, view in zip(features, indices):
            scores[0, view] = float(np.asarray(value).sum()) + view
        return scores


def reader_for(values=(1., 2., 3.)):
    b = api()
    accesses, prepared = [], []
    def read(view):
        accesses.append(view)
        return np.array([values[view]], dtype=np.float32)
    def prepare(raw):
        prepared.append(float(raw[0]))
        return raw.astype(np.float16)
    backend = TinyBackend()
    return b.IncrementalReader(read, prepare, backend), accesses, prepared, backend


def test_incremental_reader_accesses_only_new_requested_views_once():
    reader, accesses, prepared, backend = reader_for()
    first = reader(np.array([0]), 2)
    assert accesses == [2] and len(backend.inputs) == 1
    pair = reader(np.array([0]), 4)
    reader(np.array([0]), 4)
    assert accesses == [2, 0] and len(prepared) == 2 and len(backend.inputs) == 2
    full = reader(np.array([0]), 6)
    assert accesses == [2, 0, 1] and len(backend.inputs) == 3
    assert reader.raw_accesses == reader.encoded_views == accesses
    assert reader.requested_subsets == [2, 4, 4, 6]
    assert first.argmax(1).item() == pair.argmax(1).item() == full.argmax(1).item() == 2
    assert np.array_equal(pair, backend.fuse([np.array([[1]]), np.array([[3]])], (0, 2)))


def test_reader_rejects_backwards_routes_and_other_batch_rows_before_access():
    reader, accesses, _, _ = reader_for()
    reader(np.array([0]), 2)
    for rows, subset in [(np.array([1]), 6), (np.array([0, 0]), 6), (np.array([0.]), 6),
                         (np.array([0]), 0), (np.array([0]), 7), (np.array([0]), True)]:
        with pytest.raises(ValueError):
            reader(rows, subset)
    assert accesses == [2]


@pytest.mark.parametrize('threshold1,threshold2,path,count', [
    (2., 2., [2, -1, -1], 1), (-1., 2., [2, 4, -1], 2),
    (-1., -1., [2, 4, 6], 3)])
def test_cascade_is_causal_and_replays_exact_actions(threshold1, threshold2, path, count):
    b = api()
    reader, accesses, _, _ = reader_for()
    params = dict(first=2, pair=4, mode='confidence', threshold1=threshold1, threshold2=threshold2)
    direct = b.causal_cascade(reader, params)
    assert direct['path'].tolist() == [path]
    assert len(accesses) == int(direct['costs'][0]) == count
    independent, offline_accesses, _, _ = reader_for()
    cache = b.all_subset_cache(independent)
    assert offline_accesses == [0, 1, 2]
    replay = b.causal_cascade(b.ArrayReader(cache), params)
    assert b.verify_replay(reader, direct, cache, replay)['bitwise_equal']
    altered = cache.copy()
    altered[0, path[0], 0] += 1
    with pytest.raises(ValueError, match='logits'):
        b.verify_replay(reader, direct, altered, replay)


def test_value_policy_callback_direct_triple_and_pair_routes_replay():
    b = api()
    from value_policy import GainPolicy, route_policy
    class Constant:
        def __init__(self, values):
            self.values = values
        def predict(self, x):
            return np.tile(self.values, (len(x), 1))
    for gains, expected in [([0., 0., 3.], [2, 6, -1]), ([2., 0., 0.], [2, 4, 6])]:
        policy = GainPolicy('class_gain', 2, {2: Constant(gains), 4: Constant([1.]), 5: Constant([1.])})
        reader, accesses, _, _ = reader_for()
        route = route_policy(reader, policy, 0., n_rows=1)
        assert route['path'].tolist() == [expected]
        assert accesses == [2, 0, 1]
        independent, _, _, _ = reader_for()
        cache = b.all_subset_cache(independent)
        replay = route_policy(b.ArrayReader(cache), policy, 0., n_rows=1)
        assert b.verify_replay(reader, route, cache, replay)['bitwise_equal']


def test_sample_selection_is_held_user_balanced_and_has_quality_exclusions():
    b = api()
    subjects = np.repeat(np.arange(1, 9), 12*6)
    labels = np.tile(np.repeat(np.arange(12), 6), 8)
    selected = b.select_clips(labels, subjects, range(1, 9))
    assert sum(map(len, selected.values())) == 279
    assert len(selected[2]) == 30 and len(selected[6]) == 33
    for fold, ids in selected.items():
        assert np.all(subjects[ids] == fold)
        present, counts = np.unique(labels[ids], return_counts=True)
        assert np.all(counts == 3)
        for label in present:
            candidates = np.flatnonzero((subjects == fold) & (labels == label))
            expected = np.random.default_rng(44000+fold*100+int(label)).choice(candidates, 3, replace=False)
            assert set(ids[labels[ids] == label]) == set(expected)
    assert not np.isin(labels[selected[2]], [10, 11]).any()
    assert 3 not in labels[selected[6]]
    with pytest.raises(ValueError):
        b.select_clips(labels[:-1], subjects, [1])
    with pytest.raises(ValueError):
        b.select_clips(labels, subjects, [1, 1])


def test_latency_summary_uses_ms_and_equal_user_mean():
    b = api()
    rows = [dict(method='fixed1', fold=f, elapsed_ms=t, actual_views=1)
            for f, t in [(1, 1.), (1, 3.), (2, 10.)]]
    result = b.summarize(rows)['fixed1']
    assert result['unit'] == 'ms'
    assert result['mean_ms'] == pytest.approx(14/3)
    assert result['median_ms'] == 3
    assert result['p95_ms'] == pytest.approx(9.3)
    assert result['per_user_macro_mean_ms'] == 6
    assert result['mean_actual_views'] == 1
    for bad in ([], [dict(rows[0], elapsed_ms=float('nan'))], [dict(rows[0], elapsed_ms=-1)]):
        with pytest.raises(ValueError):
            b.summarize(bad)


def test_inventory_rejects_tampered_missing_and_escaping_inputs(tmp_path):
    b = api()
    file = tmp_path/'weights.pt'
    file.write_bytes(b'weights')
    expected = b.digest(file)
    inv = b.HashInventory()
    assert inv.verify(file, expected) == expected
    file.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash'):
        inv.verify(file, expected)
    with pytest.raises(ValueError, match='Missing'):
        inv.verify(tmp_path/'missing', '0'*64)
    with pytest.raises(ValueError, match='inventory'):
        b.verify_files(tmp_path, {'../outside.pt': '0'*64}, inv)


def test_mode_and_cuda_conflict_guards_do_not_initialize_cuda(tmp_path):
    b = api()
    with pytest.raises(ValueError, match='training-finished'):
        b.check_gpu_idle('cuda', False, [], processes=[])
    with pytest.raises(ValueError, match='training'):
        b.check_gpu_idle('cuda', True, [], processes=[{'pid': 123, 'cmdline': 'python multiview.py --folds 1'}])
    with pytest.raises(ValueError, match='training'):
        b.check_gpu_idle('cuda', True, [456], processes=[{'pid': 456, 'cmdline': 'python worker.py'}])
    assert b.check_gpu_idle('cpu', False, [], processes=[])['device'] == 'cpu'
    with pytest.raises(ValueError, match='Smoke'):
        b.validate_run(True, [1], b.OUTPUT/'latency_s2026', [tmp_path/'models'], 'cpu', 10)
    with pytest.raises(ValueError, match='eight'):
        b.validate_run(False, [1], tmp_path/'latency', [tmp_path/'models'], 'cuda', 10)
    with pytest.raises(ValueError, match='10'):
        b.validate_run(True, [1], tmp_path/'latency', [tmp_path/'models'], 'cpu', 9)
    with pytest.raises(ValueError, match='smoke'):
        b.require_mode({'smoke': False}, True)


def test_ast_extraction_ignores_source_top_level_execution(tmp_path):
    b = api()
    source = tmp_path/'unsafe_import.py'
    source.write_text("raise RuntimeError('must never execute')\ndef safe(x):\n    return x + 1\n", encoding='utf-8')
    namespace = {}
    b.extract_functions(source, ['safe'], namespace)
    assert namespace['safe'](2) == 3
    with pytest.raises(ValueError, match='missing'):
        b.extract_functions(source, ['absent'], {})


def test_original_preprocessing_is_float16_and_matches_separate_prepared_modes():
    b = api()
    processor = b.Preprocessor()
    raw = np.random.default_rng(4).normal(size=(90, 189)).astype(np.float32)
    packed = processor(raw)
    x = raw[None, None]
    transform = processor.namespace['transform']
    expected = np.concatenate((transform(x, 'off').astype(np.float16),
                               transform(x, 'only').astype(np.float16)), -1)
    assert packed.shape == (1, 1, 90, 378) and packed.dtype == np.float16
    assert np.array_equal(packed, expected)
    for bad in (raw[:, :-1], np.full_like(raw, np.nan)):
        with pytest.raises(ValueError):
            processor(bad)


def test_model_metadata_verifies_source_before_any_checkpoint_loading(tmp_path):
    b = api()
    labels = np.tile(np.repeat(np.arange(12), 3), 8)
    subjects = np.repeat(np.arange(1, 9), 36)
    source_file = tmp_path/'source.py'
    source_file.write_text('VERSION = 1', encoding='utf-8')
    source = {'code': {str(source_file): b.digest(source_file)}, 'data': {str(source_file): b.digest(source_file)}}
    config = {'smoke': True, 'seed': 2026, 'attention': 'MATH', 'warmup_epochs': 1,
              'warmup_min': 1, 'warmup_patience': 8, 'warmup_batch': 128, 'warmup_microbatch': 32,
              'warmup_lr': 1e-4, 'weight_decay': 1e-4, 'smoke_steps': 3}
    root = tmp_path/'baseline'
    directory = root/'fold1'/'warmup'
    directory.mkdir(parents=True)
    identity = {'source': source, 'config': config, 'arm': 'warmup', 'fold': 1,
                'partitions': b.partition_hashes(labels, subjects, 1)}
    files = {}
    for name in ('right.pt', 'best.pt', 'progress.pt', 'predictions.npz'):
        (directory/name).write_bytes(b'not loaded by metadata preflight')
        files[name] = b.digest(directory/name)
    (root/'manifest.json').write_text(__import__('json').dumps({'source': source, 'config': config}))
    (directory/'result.json').write_text(__import__('json').dumps({'identity': identity, 'files': files}))
    row = b.verify_model_run(root, 1, 'warmup', labels, subjects, True, b.HashInventory())
    assert row['identity'] == identity
    source_file.write_text('VERSION = 2', encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        b.verify_model_run(root, 1, 'warmup', labels, subjects, True, b.HashInventory())


def test_formal_policy_gate_stop_needs_no_policy_artifacts():
    b = api()
    assert b.policy_requirement({'state': 'complete', 'opportunity_decision': 'STOP_NO_ORACLE_ROOM'},
                                False, 'auto', False) is False
    with pytest.raises(ValueError, match='complete'):
        b.policy_requirement({'state': 'complete', 'opportunity_decision': 'POLICY_PILOT_WARRANTED'},
                             False, 'auto', False)
    assert b.policy_requirement(None, True, 'off', False) is False
    with pytest.raises(ValueError):
        b.policy_requirement({'state': 'running'}, False, 'auto', False)


def test_measurement_tracks_milliseconds_without_unrequested_input():
    b = api()
    reader, accesses, _, _ = reader_for()
    calls = []
    result, observation = b.measure(reader, lambda r: b.single_route(r, 2), lambda: calls.append('sync'))
    assert calls == ['sync', 'sync']
    assert accesses == [2] and result['costs'][0] == 1
    assert observation['elapsed_ms'] >= 0
    assert observation['raw_accesses'] == observation['encoded_views'] == [2]
    assert observation['actual_views'] == 1


def test_torch_backend_cpu_preserves_single_view_kernel_shape():
    b = api()
    import torch
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shapes = []
        def encode(self, x):
            self.shapes.append(tuple(x.shape))
            return x.mean((2, 3), keepdim=False)[:, :, None].expand(-1, -1, 256)
        def fuse(self, features, indices):
            scores = features.new_zeros((1, 12))
            scores[:, list(indices)] = features.mean(-1)
            return scores
    model = Model()
    backend = b.TorchBackend(model, 'cpu')
    reader = b.IncrementalReader(lambda v: np.full((90, 189), v+1, dtype=np.float32),
                                lambda x: np.concatenate((x, x), -1)[None, None].astype(np.float16), backend)
    reader(np.array([0]), 2)
    reader(np.array([0]), 6)
    assert model.shapes == [(1, 1, 90, 378)]*3
    assert reader.encoded_views == [2, 0, 1]
    assert reader.logits[6].argmax(1).item() == 2
    assert set(backend.stages) == {'input_transfer_host_ms', 'encode_host_ms', 'fuse_host_ms', 'output_cpu_ms'}


def test_methods_use_frozen_fixed_and_cascade_choices():
    b = api()
    selection = {'validation_fixed_accuracy': [0.1, .4, .7, .8, .6, .5, .9],
                 'curves': {'2.0': {'params': dict(first=2, pair=4, mode='entropy', threshold1=2., threshold2=2.)}}}
    specs = b.method_specs(selection, None, None)
    assert set(specs) == {'baselineRight', 'fixed1', 'fixed2', 'fixed3', 'frozen_budget2_cascade'}
    for name, expected in [('baselineRight', 2), ('fixed1', 2), ('fixed2', 3), ('fixed3', 6),
                           ('frozen_budget2_cascade', 2)]:
        reader, _, _, _ = reader_for()
        assert specs[name]['route'](reader)['subset_indices'].tolist() == [expected]


def test_report_serialization_is_strict_and_labels_smoke(tmp_path):
    b = api()
    output = tmp_path/'summary.json'
    rows = [dict(method='fixed1', fold=1, elapsed_ms=2., actual_views=1)]
    report = b.make_report(rows, {'device': 'cpu'}, True, [1], [], {})
    b.dump(output, report)
    saved = __import__('json').loads(output.read_text(encoding='utf-8'))
    assert saved['scientific_latency_claim'] is False
    assert saved['classification'] == 'execution_smoke_only'
    assert 'prior static-clutter removal' in saved['excluded']
    assert saved['methods']['fixed1']['mean_ms'] == 2
    with pytest.raises(ValueError):
        b.dump(output, {'invalid': float('nan')})


def test_cuda_preflight_still_requires_training_finished(tmp_path, monkeypatch):
    b = api()
    baseline, models, budget = (tmp_path/name for name in ('baseline', 'models', 'budget'))
    for path in (baseline/'fold1/warmup/result.json', models/'fold1/result.json', budget/'fold1/result.json'):
        path.parent.mkdir(parents=True)
        path.write_text('{}')
    def must_not_read(*args):
        raise AssertionError('CUDA readiness was not checked before input reads')
    monkeypatch.setattr(b, 'raw_preflight', must_not_read)
    with pytest.raises(ValueError, match='training-finished'):
        b.main(['--allow-smoke', '--folds', '1', '--device', 'cuda', '--preflight-only',
                '--baseline', str(baseline), '--models', str(models), '--budget', str(budget),
                '--out', str(tmp_path/'latency')])


def desktop_process(pid=52016):
    return dict(pid=pid, name='QQ.exe', executable=r'C:\Program Files\Tencent\QQNT\QQ.exe',
                creation_time='2026-09-20T08:00:00Z', cmdline='"C:\\Program Files\\Tencent\\QQNT\\QQ.exe"')


def fake_gpu_environment(monkeypatch, *, rows=None, compute=None, samples=None):
    """Replace OS reads only; exercise the actual environment guard, on CPU."""
    b = api()
    rows = [desktop_process()] if rows is None else rows
    compute = [(r['pid'], r['executable']) for r in rows] if compute is None else compute
    samples = iter(['0'] * 5 if samples is None else samples)
    sleeps, queries = [], []
    monkeypatch.setattr(b, 'process_inventory', lambda: [dict(r) for r in rows])
    monkeypatch.setattr(b.time, 'sleep', lambda seconds: sleeps.append(seconds))
    def run(command, **kwargs):
        queries.append(command)
        if '--query-compute-apps=pid,process_name' in command:
            output = '\n'.join(f'{pid}, {name}' for pid, name in compute)
        elif '--query-gpu=utilization.gpu' in command:
            output = next(samples)
        else:
            raise AssertionError('Unexpected external command: '+str(command))
        return SimpleNamespace(stdout=output, stderr='', returncode=0)
    monkeypatch.setattr(b.subprocess, 'run', run)
    return rows, compute, sleeps, queries


def test_windows_process_inventory_uses_strict_utf8_and_keeps_identity(monkeypatch):
    b = api()
    monkeypatch.setattr(b.os, 'name', 'nt')
    expected = desktop_process()
    original = dict(ProcessId=expected['pid'], Name=expected['name'],
                    ExecutablePath=expected['executable'], CreationDate=expected['creation_time'],
                    CommandLine='QQ.exe --name="中文用户 🛰"')
    def run(command, **kwargs):
        assert '[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()' in command[-1]
        assert kwargs['encoding'] == 'utf-8' and kwargs['errors'] == 'strict'
        assert kwargs['check'] is True
        # Simulate Windows output bytes containing characters absent from GBK.
        encoded = json.dumps(original, ensure_ascii=False).encode('utf-8')
        return SimpleNamespace(stdout=encoded.decode(kwargs['encoding'], kwargs['errors']))
    monkeypatch.setattr(b.subprocess, 'run', run)
    expected['cmdline'] = original['CommandLine']
    assert b.process_inventory() == [expected]


@pytest.mark.parametrize('output', ['', 'null', '[]', '{broken json'])
def test_windows_process_inventory_failure_never_becomes_empty(monkeypatch, output):
    b = api()
    monkeypatch.setattr(b.os, 'name', 'nt')
    monkeypatch.setattr(b.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=output))
    with pytest.raises(ValueError):
        b.process_inventory()


def test_gpu_default_still_rejects_other_compute_process(monkeypatch):
    b = api()
    _, _, sleeps, _ = fake_gpu_environment(monkeypatch)
    with pytest.raises(ValueError, match='Other GPU compute'):
        b.check_gpu_idle('cuda', True, [])
    assert sleeps == []


def test_explicit_idle_desktop_pid_records_five_samples_without_exclusivity(monkeypatch):
    b = api()
    _, _, sleeps, queries = fake_gpu_environment(monkeypatch, samples=['80', '30', '0', '5', '2'])
    audit = b.check_gpu_idle('cuda', True, [], allow_idle_gpu_pids=[52016])
    assert audit['allowed_idle_gpu_identities'] == [desktop_process()]
    assert [s['utilization_percent'] for s in audit['gpu_utilization_samples']] == [[80.], [30.], [0.], [5.], [2.]]
    assert sleeps == [1.] * 5
    assert sum('--query-compute-apps=pid,process_name' in q for q in queries) == 2
    assert audit['gpu_exclusive'] is False
    assert 'sampling' in audit['limitations'].lower()


@pytest.mark.parametrize('samples', [['0', '0', '6', '0', '0'], ['0', '0', '0', '0', '99']])
def test_explicit_desktop_pid_does_not_excuse_gpu_activity(monkeypatch, samples):
    b = api()
    fake_gpu_environment(monkeypatch, samples=samples)
    with pytest.raises(ValueError, match='utilization') as rejected:
        b.check_gpu_idle('cuda', True, [], allow_idle_gpu_pids=[52016])
    assert len(rejected.value.audit['gpu_utilization_samples']) == 5


@pytest.mark.parametrize('bad_sample', ['N/A', 'nan', 'inf', '-1', '101', ''])
def test_gpu_idle_sampling_fails_closed_on_unavailable_values(monkeypatch, bad_sample):
    b = api()
    fake_gpu_environment(monkeypatch, samples=['0', '0', bad_sample, '0', '0'])
    with pytest.raises(ValueError, match='utilization') as rejected:
        b.check_gpu_idle('cuda', True, [], allow_idle_gpu_pids=[52016])
    assert rejected.value.audit['gpu_utilization_samples'][-1]['raw'] == bad_sample


def test_explicit_desktop_pid_does_not_excuse_an_extra_compute_pid(monkeypatch):
    b = api()
    fake_gpu_environment(monkeypatch, compute=[(52016, desktop_process()['executable']), (60000, 'unknown.exe')])
    with pytest.raises(ValueError, match='Other GPU compute'):
        b.check_gpu_idle('cuda', True, [], allow_idle_gpu_pids=[52016])


@pytest.mark.parametrize('known_training,process', [
    ([52016], desktop_process()),
    ([], dict(desktop_process(), name='python.exe', executable=r'C:\Python\python.exe', cmdline='python worker.py')),
    ([], dict(desktop_process(), cmdline='python multiview.py --folds 1')),
])
def test_training_process_cannot_be_explicitly_allowed(monkeypatch, known_training, process):
    b = api()
    fake_gpu_environment(monkeypatch, rows=[process])
    with pytest.raises(ValueError, match='training|Python'):
        b.check_gpu_idle('cuda', True, known_training, allow_idle_gpu_pids=[52016])


def test_own_pid_cannot_be_explicitly_allowed(monkeypatch):
    b = api()
    fake_gpu_environment(monkeypatch)
    with pytest.raises(ValueError, match='own PID'):
        b.check_gpu_idle('cuda', True, [], allow_idle_gpu_pids=[b.os.getpid()])


@pytest.mark.parametrize('changes', [dict(creation_time='2026-09-29T00:00:00Z'),
                                   dict(executable=r'C:\Other\QQ.exe'), dict(name='renamed.exe')])
def test_allowed_desktop_identity_is_rechecked_between_folds(monkeypatch, changes):
    b = api()
    initial = [desktop_process()]
    fake_gpu_environment(monkeypatch, rows=[dict(desktop_process(), **changes)])
    with pytest.raises(ValueError, match='identity'):
        b.check_gpu_idle('cuda', True, [], allow_idle_gpu_pids=[52016], allowed_identities=initial)


def test_allowed_desktop_identity_is_rechecked_after_sampling(monkeypatch):
    b = api()
    rows, _, _, _ = fake_gpu_environment(monkeypatch)
    def changed_after_sleep(seconds):
        rows[0]['creation_time'] = '2026-09-29T00:00:00Z'
    monkeypatch.setattr(b.time, 'sleep', changed_after_sleep)
    with pytest.raises(ValueError, match='identity'):
        b.check_gpu_idle('cuda', True, [], allow_idle_gpu_pids=[52016])


@pytest.mark.parametrize('field', ['name', 'executable', 'creation_time'])
def test_desktop_allowance_requires_complete_process_identity(monkeypatch, field):
    b = api()
    process = desktop_process()
    process[field] = ''
    fake_gpu_environment(monkeypatch, rows=[process])
    with pytest.raises(ValueError, match='identity'):
        b.check_gpu_idle('cuda', True, [], allow_idle_gpu_pids=[52016])


def test_desktop_allowance_requires_matching_gpu_image_path(monkeypatch):
    b = api()
    fake_gpu_environment(monkeypatch, compute=[(52016, r'C:\Other\QQ.exe')])
    with pytest.raises(ValueError, match='image path'):
        b.check_gpu_idle('cuda', True, [], allow_idle_gpu_pids=[52016])


def test_parser_records_only_explicit_pid_allowances():
    b = api()
    assert b.parser().parse_args([]).allow_idle_gpu_pid == []
    assert b.parser().parse_args(['--allow-idle-gpu-pid', '52016', '--allow-idle-gpu-pid', '60000']).allow_idle_gpu_pid == [52016, 60000]
    with pytest.raises(SystemExit):
        b.parser().parse_args(['--gpu-idle-threshold', '99'])


def test_allowed_desktop_report_does_not_claim_gpu_isolation():
    b = api()
    rows = [dict(method='fixed1', fold=1, elapsed_ms=2., actual_views=1)]
    report = b.make_report(rows, {'device': 'cuda', 'allowed_idle_gpu_identities': [desktop_process()]},
                           False, list(range(1, 9)), [], {})
    assert report['classification'] == 'desktop_context_gpu_latency'
    assert 'not exclusive' in report['limits'].lower()


def test_gpu_sampling_command_failure_keeps_rejection_evidence(monkeypatch):
    b = api()
    fake_gpu_environment(monkeypatch)
    normal_run = b.subprocess.run
    def fail_utilization(command, **kwargs):
        if '--query-gpu=utilization.gpu' in command:
            return SimpleNamespace(returncode=1, stdout='', stderr='driver unavailable')
        return normal_run(command, **kwargs)
    monkeypatch.setattr(b.subprocess, 'run', fail_utilization)
    with pytest.raises(ValueError, match='utilization') as rejected:
        b.check_gpu_idle('cuda', True, [], allow_idle_gpu_pids=[52016])
    sample = rejected.value.audit['gpu_utilization_samples'][-1]
    assert sample['returncode'] == 1 and sample['stderr'] == 'driver unavailable'


def fake_latency_inputs(tmp_path, monkeypatch, folds=(1,)):
    b = api()
    baseline, models, budget = (tmp_path/name for name in ('baseline', 'models', 'budget'))
    for fold in folds:
        for path in (baseline/f'fold{fold}/warmup/result.json', models/f'fold{fold}/result.json',
                     budget/f'fold{fold}/result.json'):
            path.parent.mkdir(parents=True)
            path.write_text('{}', encoding='utf-8')
    reference = tmp_path/'reference'
    reference.mkdir()
    for name in ('LATENCY_PROTOCOL.md', 'LATENCY_ENVIRONMENT_NOTE.md'):
        (reference/name).write_text(name, encoding='utf-8')
    monkeypatch.setattr(b, 'OUTPUT', reference)
    monkeypatch.setattr(b, 'raw_preflight', lambda *a: {'audit': {'test_inputs': True}})
    monkeypatch.setattr(b, 'artifact_preflight', lambda *a: {fold: {} for fold in folds})
    monkeypatch.setattr(b, 'configure_torch', lambda device: {'device': device})
    command = ['--allow-smoke', '--training-finished', '--allow-idle-gpu-pid', '52016',
               '--folds', *map(str, folds), '--device', 'cuda', '--baseline', str(baseline),
               '--models', str(models), '--budget', str(budget), '--out', str(tmp_path/'latency')]
    return command, reference


def test_main_records_each_fold_guard_and_environment_note_hash(tmp_path, monkeypatch):
    b = api()
    command, reference = fake_latency_inputs(tmp_path, monkeypatch, folds=(1, 2))
    fake_gpu_environment(monkeypatch, samples=['0'] * 25)
    calls = []
    def run_fold(args, fold, data, context):
        calls.append(fold)
        saved = json.loads((args.out/'gpu_idle_checks.json').read_text(encoding='utf-8'))
        assert saved[-1]['phase'] == 'before_fold' and saved[-1]['fold'] == fold
        return [dict(method='fixed1', fold=fold, elapsed_ms=2., actual_views=1)], {}
    monkeypatch.setattr(b, 'run_fold', run_fold)
    report = b.main(command)
    checks = report['gpu_idle_checks']
    assert calls == [1, 2]
    assert [(c['phase'], c.get('fold')) for c in checks] == [
        ('preflight', None), ('before_fold', 1), ('after_fold', 1), ('before_fold', 2), ('after_fold', 2)]
    assert all(len(c['gpu_utilization_samples']) == 5 for c in checks)
    assert report['environment']['allowed_idle_gpu_identities'] == [desktop_process()]
    note = reference/'LATENCY_ENVIRONMENT_NOTE.md'
    assert report['hash_inventory'][str(note.resolve())] == b.digest(note)
    assert json.loads((tmp_path/'latency/gpu_idle_checks.json').read_text(encoding='utf-8')) == checks


def test_main_rejects_changed_identity_after_fold_and_saves_samples(tmp_path, monkeypatch):
    b = api()
    command, _ = fake_latency_inputs(tmp_path, monkeypatch)
    rows, _, _, _ = fake_gpu_environment(monkeypatch, samples=['0'] * 10)
    def run_fold(args, fold, data, context):
        rows[0]['creation_time'] = '2026-09-29T00:00:00Z'
        return [dict(method='fixed1', fold=fold, elapsed_ms=2., actual_views=1)], {}
    monkeypatch.setattr(b, 'run_fold', run_fold)
    with pytest.raises(ValueError, match='identity'):
        b.main(command)
    checks = json.loads((tmp_path/'latency/gpu_idle_checks.json').read_text(encoding='utf-8'))
    assert checks[-1]['phase'] == 'after_fold' and checks[-1]['state'] == 'rejected'
    assert sum(len(c['gpu_utilization_samples']) for c in checks) == 10
    assert not (tmp_path/'latency/summary.json').exists()
