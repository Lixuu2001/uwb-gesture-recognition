import importlib.util
from pathlib import Path
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]


def module():
    assert (HERE/'data_access.py').exists(), 'missing raw fold data access'
    spec = importlib.util.spec_from_file_location('e2e_data_access', HERE/'data_access.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_hidden_views_rejected_outside_training():
    m = module()
    a = np.arange(5*3*2*2).reshape(5, 3, 2, 2).astype('float16')
    reader = m.ViewReader(a, -a, {'train': np.array([0, 1, 2]), 'val': np.array([3]), 'test': np.array([4])})
    with pytest.raises(ValueError):
        reader.load('test', 'MV', device='cpu')
    with pytest.raises(ValueError):
        reader.load('val', 'MV', device='cpu')
    x = reader.load('train', 'MV', device='cpu').numpy()
    np.testing.assert_array_equal(x[..., :2], a[:3])
    np.testing.assert_array_equal(x[..., 2:], -a[:3])


def test_right_reader_ignores_hidden_nan_values():
    m = module()
    a = np.ones((4, 3, 2, 2), dtype='float16')
    a[:, :2] = np.nan
    reader = m.ViewReader(a, a, {'train': np.array([0, 1]), 'val': np.array([2]), 'test': np.array([3])})
    for role in ('train', 'val', 'test'):
        x = reader.load(role, 'R', device='cpu')
        assert x.shape[1] == 1 and np.isfinite(x.numpy()).all()


def test_source_partitions_cannot_overlap():
    m = module()
    a = np.ones((4, 3, 2, 2), dtype='float16')
    with pytest.raises(ValueError):
        m.ViewReader(a, a, {'train': np.array([0, 1]), 'val': np.array([1]), 'test': np.array([3])})
