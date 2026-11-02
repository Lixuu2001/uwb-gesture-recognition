import importlib.util
from pathlib import Path
import random
import numpy as np
import pytest
import torch

P = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tested_meta_state', P/'train_state.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_selection_first_minimum_and_stop():
    state = m.Selection(3.)
    assert not state.observe(1, 3.)
    assert state.observe(2, 2.)
    assert not state.observe(3, 2.)
    assert not state.should_stop(3, minimum=4, patience=1)
    assert state.should_stop(4, minimum=4, patience=1)
    assert state.best_epoch == 2
    with pytest.raises(ValueError): state.observe(4, float('nan'))


def test_resume_rejects_identity_before_mutation_and_replays(tmp_path):
    torch.manual_seed(7); np.random.seed(8); random.seed(9)
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    sampler = m.ShuffleSampler(np.arange(7), seed=3)
    opt.zero_grad(); model(torch.ones(1, 2)).sum().backward(); opt.step()
    sampler.next_epoch()
    path = tmp_path/'progress.pt'
    m.save_progress(path, model, opt, sampler, {'identity': {'fold': 1}})
    expected = (sampler.next_epoch(), torch.rand(4), np.random.rand(4), random.random())
    old = model.weight.detach().clone()
    with pytest.raises(ValueError):
        m.load_progress(path, model, opt, sampler, {'fold': 2})
    assert torch.equal(old, model.weight)
    with torch.no_grad(): model.weight.add_(1)
    m.load_progress(path, model, opt, sampler, {'fold': 1})
    assert torch.equal(old, model.weight)
    assert np.array_equal(expected[0], sampler.next_epoch())
    assert torch.equal(expected[1], torch.rand(4))
    assert np.array_equal(expected[2], np.random.rand(4))
    assert expected[3] == random.random()


def test_next_optimizer_update_is_identical_after_resume(tmp_path):
    torch.manual_seed(12)
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Dropout(.3), torch.nn.Linear(3, 1)).train()
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    sampler = m.ShuffleSampler(np.arange(4), seed=11)
    def step():
        rows = sampler.next_epoch()
        x = torch.tensor(np.stack((rows, rows+1), axis=1), dtype=torch.float32)
        opt.zero_grad(); model(x).square().mean().backward(); opt.step()
    step()
    path = tmp_path/'progress.pt'
    m.save_progress(path, model, opt, sampler, {'identity': {'run': 'exact'}})
    step(); expected = {k:v.clone() for k,v in model.state_dict().items()}
    m.load_progress(path, model, opt, sampler, {'run': 'exact'})
    step()
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in expected.items())
