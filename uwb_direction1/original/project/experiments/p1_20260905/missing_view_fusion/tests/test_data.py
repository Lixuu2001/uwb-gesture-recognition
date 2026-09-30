import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def module():
    assert (HERE / 'data.py').exists(), 'missing frozen feature extraction implementation'
    spec = importlib.util.spec_from_file_location('fusion_data', HERE / 'data.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pool_preserves_amplitude_phase_halves():
    m = module()
    t = torch.zeros(2, 1, 90, 32)
    s = torch.cat([torch.ones(2, 1, 189, 32), 7*torch.ones(2, 1, 189, 32)], 2)
    z = m.pool_tokens(t, s)
    assert z.shape == (2, 1, 1664)
    torch.testing.assert_close(z[..., 320:992], torch.ones(2, 1, 672))
    torch.testing.assert_close(z[..., 992:], 7*torch.ones(2, 1, 672))


def test_fit_scale_uses_only_training_rows():
    m = module()
    x = torch.tensor([[[1., 2.]], [[3., 4.]], [[999., 999.]]])
    mu, sd = m.fit_scale(x[:2])
    torch.testing.assert_close(mu, torch.tensor([[[2., 3.]]]))
    torch.testing.assert_close(sd, torch.ones_like(sd))


def test_split_contract_rejects_overlap_and_wrong_subject():
    m = module()
    y = np.zeros(4, dtype=int)
    s = np.array([1, 1, 2, 2])
    m.validate_split(np.array([0]), np.array([1]), np.array([2, 3]), y, s, 2)
    with pytest.raises(AssertionError):
        m.validate_split(np.array([0, 2]), np.array([1]), np.array([2, 3]), y, s, 2)


def test_prototype_banks_have_training_class_means():
    m = module()
    x = torch.arange(4*3*2).float().reshape(4, 3, 2)
    banks = m.prototype_banks(x, torch.tensor([0, 0, 1, 1]), classes=2)
    torch.testing.assert_close(banks[:, 0], x[:2].mean(0))
    torch.testing.assert_close(banks[:, 1], x[2:].mean(0))
    with pytest.raises(ValueError):
        m.prototype_banks(x, torch.tensor([0, 0, 0, 0]), classes=2)


def test_raw_identity_detects_same_length_content_change(tmp_path):
    m = module()
    p = tmp_path/'raw.npy'
    p.write_bytes(b'1234')
    first = m.raw_identity([p])
    p.write_bytes(b'1235')
    assert m.raw_identity([p]) != first
