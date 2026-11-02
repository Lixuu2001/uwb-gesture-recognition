"""CPU contracts for the fixed, training-only statistics intervention."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope='module')
def style():
    path = HERE / 'style.py'
    if not path.exists():
        return SimpleNamespace()
    spec = importlib.util.spec_from_file_location('_test_mild_style', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata():
    # Unequal instance counts catch sampling rows instead of sampling users first.
    labels = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1])
    subjects = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 5])
    return np.arange(9, dtype=np.int64), labels, subjects


def reference_schedule(ids, labels, subjects, seed, probability=.25):
    ascending = np.sort(ids)
    rng = np.random.default_rng(seed)
    masks = rng.random(len(ascending)) < probability
    donors = []
    for row in ascending:
        candidates = ascending[(labels[ascending] == labels[row]) &
                               (subjects[ascending] != subjects[row])]
        user = rng.choice(np.unique(subjects[candidates]))
        donors.append(rng.choice(candidates[subjects[candidates] == user]))
    location = {int(row): k for k, row in enumerate(ascending)}
    order = [location[int(row)] for row in ids]
    return np.asarray(donors, dtype=np.int64)[order], masks[order]


def test_schedule_is_canonical_same_class_other_user_and_train_only(style):
    ids, labels, subjects = metadata()
    shuffled = ids[[5, 2, 8, 0, 3, 7, 1, 4, 6]]
    donors, masks = style.schedule(shuffled, labels, subjects, 20260909)
    expected = reference_schedule(shuffled, labels, subjects, 20260909)
    assert np.array_equal(donors, expected[0])
    assert np.array_equal(masks, expected[1])
    assert donors.dtype == np.int64 and masks.dtype == np.bool_
    assert np.isin(donors, ids).all()
    assert np.array_equal(labels[donors], labels[shuffled])
    assert np.all(subjects[donors] != subjects[shuffled])


def test_schedule_and_sampler_seed_replay_without_global_rng(style):
    ids, labels, subjects = metadata()
    sampler = np.random.default_rng(48)
    state = {'rng': copy.deepcopy(sampler.bit_generator.state)}
    saved = copy.deepcopy(state)
    payload = json.dumps({'sampler': state['rng'], 'fold': 3,
                          'augmentation_seed': 20260909}, sort_keys=True)
    expected = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'little')
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    assert style.sampler_seed(state, 3) == expected
    a = style.schedule(ids, labels, subjects, expected)
    b = style.schedule(ids, labels, subjects, style.sampler_seed(saved, 3))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert state == saved
    assert np.array_equal(np.random.get_state()[1], np_before[1])
    assert np.random.get_state()[2:] == np_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


@pytest.mark.parametrize('case', ['duplicate', 'negative', 'outside', 'float_ids',
                                  'nonvector', 'metadata', 'missing_donor', 'probability'])
def test_schedule_rejects_invalid_inputs(style, case):
    ids, labels, subjects = metadata()
    probability = .25
    if case == 'duplicate': ids = np.array([0, 0, 3])
    if case == 'negative': ids[0] = -1
    if case == 'outside': ids[0] = len(labels)
    if case == 'float_ids': ids = ids.astype(float)
    if case == 'nonvector': ids = ids[:, None]
    if case == 'metadata': subjects = subjects[:-1]
    if case == 'missing_donor': subjects[:] = 1
    if case == 'probability': probability = float('nan')
    with pytest.raises(ValueError):
        style.schedule(ids, labels, subjects, 1, probability)


def test_style_identity_statistics_and_stopped_statistics_gradient(style):
    source = torch.tensor([[[1., 3.], [3., 7.], [5., 11.]],
                           [[2., 5.], [5., 7.], [9., 13.]]], requires_grad=True)
    donor = (source.detach() * 2 + 4).requires_grad_()
    mask = torch.tensor([True, False])
    out = style.style_mix(source, donor, mask)
    assert torch.equal(out[1], source[1])
    assert torch.allclose(out[0].mean(0), .9 * source[0].mean(0) + .1 * donor[0].mean(0))
    assert torch.allclose(out[0].std(0, unbiased=False),
                          .9 * source[0].std(0, unbiased=False) + .1 * donor[0].std(0, unbiased=False))
    out.sum().backward()
    assert donor.grad is None
    assert torch.allclose(source.grad[0], torch.full_like(source[0], 1.1))
    assert torch.equal(source.grad[1], torch.ones_like(source[1]))
    assert style.style_mix(source, donor, mask, share=0) is source
    assert style.style_mix(source, donor, torch.zeros(2, dtype=torch.bool)) is source


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_style_constant_sequence_is_finite_and_preserves_dtype(style, dtype):
    source = torch.full((2, 1, 4), 3., dtype=dtype, requires_grad=True)
    donor = torch.full_like(source, 8.)
    out = style.style_mix(source, donor, torch.tensor([True, False]))
    assert out.dtype == dtype and out.shape == source.shape
    assert torch.isfinite(out).all()
    assert torch.equal(out[0], torch.full_like(out[0], 3.5))
    assert torch.equal(out[1], source[1])
    out.float().sum().backward()
    assert torch.isfinite(source.grad).all()


@pytest.mark.parametrize('case', ['source_rank', 'donor_shape', 'mask_shape', 'mask_type',
                                  'empty_sequence', 'source_integer', 'share'])
def test_style_rejects_shape_and_parameter_errors(style, case):
    source = torch.ones(2, 3, 4)
    donor = source.clone()
    mask = torch.tensor([True, False])
    share = .1
    if case == 'source_rank': source = source[0]
    if case == 'donor_shape': donor = donor[:1]
    if case == 'mask_shape': mask = mask[:, None]
    if case == 'mask_type': mask = mask.long()
    if case == 'empty_sequence': source = source[:, :0]; donor = donor[:, :0]
    if case == 'source_integer': source = source.long()
    if case == 'share': share = -1
    with pytest.raises(ValueError):
        style.style_mix(source, donor, mask, share)


class GuardedTinyReader:
    def __init__(self):
        ids, self.labels, self.subjects = metadata()
        self.partitions = {'train': ids, 'val': np.array([9]), 'test': np.array([10])}
        values = np.random.default_rng(8).normal(size=(11, 1, 5, 7)).astype(np.float32)
        self.values = torch.tensor(values) * torch.arange(1, 12)[:, None, None, None]
        self.calls = []
        self.loaded = []

    def load_ids(self, ids, role, mode, device='cpu'):
        self.calls.append((np.asarray(ids).copy(), role, mode))
        ids = np.asarray(ids)
        if role not in self.partitions or mode not in ('R', 'MV'):
            raise ValueError('base guard: role or mode')
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids) == 0:
            raise ValueError('base guard: integer vector')
        if not np.isin(ids, self.partitions[role]).all():
            raise ValueError('base guard: partition')
        value = self.values[ids.copy()].clone().to(device).requires_grad_()
        self.loaded.append(value)
        return value


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        path = HERE.parent / 'vendor' / 'uwb_osr' / 'model.py'
        spec = importlib.util.spec_from_file_location('_style_real_encoder', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = SimpleNamespace(dim=4, T=5, M_in=7, pool='flatten', depth=1,
                              heads=1, mlp_ratio=1., dropout=.2, d_key=4, n_views=1)
        self.right = nn.Module()
        self.right.encoder = module.SharedDualPathEncoder(cfg)

    def forward(self, x):
        return self.right.encoder(x)[0]


def reader_model(style, enabled=True):
    base = GuardedTinyReader()
    reader = style.StyleReader(base, base.labels, base.subjects, enabled=enabled)
    reader.configure(20260909)
    return reader, base, TinyModel()


def trace_digest(ids, donors, masks):
    return hashlib.sha256(np.column_stack([ids, donors, masks]).astype(np.int64).tobytes()).hexdigest()


def test_reader_projects_only_selected_donors_before_position_and_stops_donor_gradient(style):
    reader, base, model = reader_model(style)
    ids = base.partitions['train'][::-1]
    donors, masks = reference_schedule(ids, base.labels, base.subjects, 20260909)
    assert masks.any() and (~masks).any()
    original_state = set(model.state_dict())
    captured = {}
    with reader.epoch_context(model):
        source = reader.load_ids(ids, 'train', 'R', device='cpu')
        # Observer runs after the style hook and sees exactly the pre-position output.
        observers = [path.proj.register_forward_hook(
            lambda module, args, output, name=name: captured.__setitem__(name, output))
            for name, path in [('temporal', model.right.encoder.temporal),
                               ('spatial', model.right.encoder.spatial)]]
        result = model(source)
        result.sum().backward()
        for observer in observers: observer.remove()
        assert len(base.calls) == 2
        assert np.array_equal(base.calls[1][0], donors[masks])
        assert base.loaded[1].grad is None
        assert source.grad is not None and torch.isfinite(source.grad).all()
        for name in ('temporal', 'spatial'):
            projection = getattr(model.right.encoder, name).proj
            raw = source.detach().reshape(len(ids), 5, 7)
            raw_donor = base.values[donors].reshape(len(ids), 5, 7)
            if name == 'spatial': raw = raw.transpose(1, 2); raw_donor = raw_donor.transpose(1, 2)
            clean = F.linear(raw, projection.weight, projection.bias)
            expected = style.style_mix(clean, F.linear(raw_donor, projection.weight, projection.bias),
                                       torch.tensor(masks))
            assert torch.equal(captured[name][~masks], clean[~masks])
            assert torch.allclose(captured[name], expected, atol=2e-6)
            assert projection.weight.grad is not None
            assert torch.isfinite(projection.weight.grad).all()
    assert set(model.state_dict()) == original_state
    assert not model.right.encoder.temporal.proj._forward_hooks
    assert not model.right.encoder.spatial.proj._forward_hooks
    assert reader.trace.hexdigest() == trace_digest(ids, donors, masks)
    assert reader.counts == {'clean': int((~masks).sum()), 'style': int(masks.sum())}


def test_reader_disabled_is_bitwise_clean_and_preserves_dropout_rng(style):
    reader, base, model = reader_model(style, enabled=False)
    ids = base.partitions['train']
    donors, _ = reference_schedule(ids, base.labels, base.subjects, 20260909)
    rng = torch.get_rng_state()
    clean = model(base.values[ids])
    after_clean = torch.get_rng_state()
    torch.set_rng_state(rng)
    with reader.epoch_context(model):
        actual = model(reader.load_ids(ids, 'train', 'R', device='cpu'))
    assert torch.equal(clean, actual)
    assert torch.equal(torch.get_rng_state(), after_clean)
    assert len(base.calls) == 1
    assert reader.trace.hexdigest() == trace_digest(ids, donors, np.zeros(len(ids), dtype=bool))
    assert reader.counts == {'clean': len(ids), 'style': 0}


def test_enabled_hooks_use_no_extra_randomness_and_replay(style):
    reader, base, model = reader_model(style)
    ids = base.partitions['train']
    rng = torch.get_rng_state()
    model(base.values[ids])
    expected_rng = torch.get_rng_state()
    outputs = []
    traces = []
    for _ in range(2):
        reader.configure(20260909)
        torch.set_rng_state(rng)
        with reader.epoch_context(model):
            outputs.append(model(reader.load_ids(ids, 'train', 'R', device='cpu')))
        traces.append(reader.trace.hexdigest())
        assert torch.equal(torch.get_rng_state(), expected_rng)
    assert torch.equal(outputs[0], outputs[1])
    assert traces[0] == traces[1]


def test_reader_clean_outside_epoch_and_validation_even_inside_epoch(style):
    reader, base, model = reader_model(style)
    empty = reader.trace.hexdigest()
    ids = base.partitions['train']
    assert torch.equal(reader.load_ids(ids, 'train', 'R', device='cpu'), base.values[ids])
    with reader.epoch_context(model):
        for role in ('val', 'test'):
            x = reader.load_ids(base.partitions[role], role, 'R', device='cpu')
            rng = torch.get_rng_state()
            actual = model(x)
            torch.set_rng_state(rng)
            clean = model(base.values[base.partitions[role]])
            assert torch.equal(actual, clean)
    assert len(base.calls) == 3
    assert reader.trace.hexdigest() == empty
    assert reader.counts == {'clean': 0, 'style': 0}


def test_base_guard_runs_before_wrapper_checks_and_exception_removes_hooks(style):
    reader, base, model = reader_model(style)
    with pytest.raises(RuntimeError, match='deliberate'):
        with reader.epoch_context(model):
            with pytest.raises(ValueError, match='base guard: partition'):
                reader.load_ids(np.array([9]), 'train', 'MV', device='cpu')
            raise RuntimeError('deliberate')
    assert len(base.calls) == 1
    assert not model.right.encoder.temporal.proj._forward_hooks
    assert not model.right.encoder.spatial.proj._forward_hooks


def test_training_mv_rejected_after_base_guard_without_trace_or_donor(style):
    reader, base, model = reader_model(style)
    with reader.epoch_context(model):
        with pytest.raises(ValueError, match='Right'):
            reader.load_ids(base.partitions['train'], 'train', 'MV', device='cpu')
    assert len(base.calls) == 1
    assert reader.counts == {'clean': 0, 'style': 0}


def test_ordered_microbatch_trace_includes_repeated_anchor_visits(style):
    reader, base, model = reader_model(style)
    ids = base.partitions['train']
    donors, masks = reference_schedule(ids, base.labels, base.subjects, 20260909)
    visits = np.array([7, 1, 4, 0, 8, 1, 5], dtype=np.int64)
    with reader.epoch_context(model):
        for micro in (visits[:3], visits[3:5], visits[5:]):
            model(reader.load_ids(micro, 'train', 'R', device='cpu'))
    assert reader.trace.hexdigest() == trace_digest(visits, donors[visits], masks[visits])
    assert reader.counts == {'clean': int((~masks[visits]).sum()), 'style': int(masks[visits].sum())}
    reader.configure(20260909)
    assert reader.trace.hexdigest() == hashlib.sha256().hexdigest()
    assert reader.counts == {'clean': 0, 'style': 0}


def test_enabled_but_all_clean_batch_does_not_read_donors(style):
    reader, base, model = reader_model(style)
    ids = base.partitions['train']
    donors, masks = reference_schedule(ids, base.labels, base.subjects, 20260909)
    clean_ids = ids[~masks]
    rng = torch.get_rng_state()
    expected = model(base.values[clean_ids])
    torch.set_rng_state(rng)
    with reader.epoch_context(model):
        actual = model(reader.load_ids(clean_ids, 'train', 'R', device='cpu'))
    assert torch.equal(actual, expected)
    assert len(base.calls) == 1
    assert reader.trace.hexdigest() == trace_digest(clean_ids, donors[~masks], masks[~masks])


def test_eval_model_training_role_bypasses_style_and_trace(style):
    reader, base, model = reader_model(style)
    model.eval()
    ids = base.partitions['train']
    with reader.epoch_context(model):
        actual = model(reader.load_ids(ids, 'train', 'R', device='cpu'))
        expected = model(base.values[ids])
    assert torch.equal(actual, expected)
    assert len(base.calls) == 1
    assert reader.trace.hexdigest() == hashlib.sha256().hexdigest()
    assert reader.counts == {'clean': 0, 'style': 0}


def test_context_failure_clears_loaded_donors_and_keeps_preexisting_hooks(style):
    reader, base, model = reader_model(style)
    temporal = model.right.encoder.temporal.proj
    existing = temporal.register_forward_hook(lambda module, args, result: result)
    original_hook_ids = set(temporal._forward_hooks)
    try:
        with pytest.raises(RuntimeError, match='after donor'):
            with reader.epoch_context(model):
                reader.load_ids(base.partitions['train'], 'train', 'R', device='cpu')
                assert reader._pending is not None
                raise RuntimeError('after donor load')
        assert reader._pending is None and reader._model is None
        assert set(temporal._forward_hooks) == original_hook_ids
        assert not model.right.encoder.spatial.proj._forward_hooks
    finally:
        existing.remove()


def test_source_shape_mismatch_fails_and_removes_hooks(style):
    reader, base, model = reader_model(style)
    with pytest.raises(ValueError, match='Projection input'):
        with reader.epoch_context(model):
            source = reader.load_ids(base.partitions['train'], 'train', 'R', device='cpu')
            model(source[:1])
    assert reader._pending is None
    assert not model.right.encoder.temporal.proj._forward_hooks
    assert not model.right.encoder.spatial.proj._forward_hooks
