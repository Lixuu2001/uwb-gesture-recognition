import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location('_cross_user_protocol_tests', HERE / 'protocol.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture():
    subjects = np.repeat(np.arange(1, 9), 12 * 4)
    labels = np.tile(np.repeat(np.arange(12), 4), 8)
    return labels, subjects


def test_subject_split_excludes_whole_users_and_known_bad_rows():
    m = module(); y, s = fixture()
    for h in range(1, 9):
        parts = m.subject_split(y, s, h)
        assert set(s[parts['test']]) == {h}
        assert set(s[parts['val']]) == {h % 8 + 1}
        assert len(set(s[parts['train']])) == 6
        assert set(y[parts['train']]) == set(range(12))
        ids = np.concatenate(list(parts.values()))
        assert len(ids) == len(set(ids))
        np.testing.assert_array_equal(np.sort(ids), np.where(m.quality_mask(y, s))[0])


def test_invalid_or_missing_subject_is_rejected():
    m = module(); y, s = fixture()
    with pytest.raises(ValueError): m.subject_split(y, s, 0)
    with pytest.raises(ValueError): m.subject_split(y[s != 8], s[s != 8], 1)


def test_episode_roles_are_disjoint_balanced_and_replayable():
    m = module(); y, s = fixture(); parts = m.subject_split(y, s, 1)
    sampler = m.EpisodeSampler(parts['train'], y, s, seed=17, per_user=3)
    seen = []
    for _ in range(6):
        e = sampler.next()
        seen.append(e['query_subject'])
        assert len(e['support_ids']) == 15 and len(e['query_ids']) == 3
        assert not set(s[e['support_ids']]) & set(s[e['query_ids']])
        assert set(np.r_[e['support_ids'], e['query_ids']]) <= set(parts['train'])
        assert np.unique(s[e['support_ids']], return_counts=True)[1].tolist() == [3]*5
    assert set(seen) == set(s[parts['train']])
    state = sampler.state_dict(); wanted = sampler.next()
    other = m.EpisodeSampler(parts['train'], y, s, seed=999, per_user=3)
    other.load_state_dict(state); actual = other.next()
    for key in ('support_ids', 'query_ids'): np.testing.assert_array_equal(actual[key], wanted[key])
    assert actual['query_subject'] == wanted['query_subject']


def test_sampler_refuses_wrong_partition_or_batch_resume():
    m = module(); y, s = fixture(); parts = m.subject_split(y, s, 1)
    a = m.EpisodeSampler(parts['train'], y, s, seed=1, per_user=3)
    b = m.EpisodeSampler(parts['train'], y, s, seed=1, per_user=4)
    with pytest.raises(ValueError): b.load_state_dict(a.state_dict())


def test_reader_enforces_views_and_partition_before_touching_raw_arrays():
    m = module(); y, s = fixture(); p = m.subject_split(y, s, 1)
    a = np.zeros((len(y), 3, 90, 189), dtype=np.float16)
    b = np.ones_like(a)
    reader = m.SubjectData(a, b, p)
    out = reader.load_ids(p['val'][:2], 'val', 'R', device='cpu')
    assert out.shape == (2, 1, 90, 378)
    assert torch.equal(out[..., :189], torch.zeros_like(out[..., :189]))
    assert torch.equal(out[..., 189:], torch.ones_like(out[..., 189:]))
    with pytest.raises(ValueError): reader.load_ids(p['val'][:2], 'val', 'MV', device='cpu')
    with pytest.raises(ValueError): reader.load_ids(p['val'][:2], 'train', 'R', device='cpu')
    with pytest.raises(ValueError): reader.load_ids(np.array([-1]), 'train', 'R', device='cpu')
    a[p['train'][0], 0] = np.nan
    assert torch.isfinite(reader.load_ids(p['train'][:1], 'train', 'R', device='cpu')).all()
    with pytest.raises(ValueError): reader.load_ids(p['train'][:1], 'train', 'MV', device='cpu')
