import importlib.util
from pathlib import Path
import random
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]


def module():
    assert (HERE/'checkpointing.py').exists(), 'missing resumable training state'
    spec = importlib.util.spec_from_file_location('e2e_checkpointing', HERE/'checkpointing.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_resume_reproduces_optimizer_and_random_streams(tmp_path):
    c = module()
    torch.manual_seed(13); np.random.seed(14); random.seed(15)
    m = torch.nn.Linear(2, 1)
    opt = torch.optim.Adam(m.parameters(), lr=.01)
    sampler = torch.Generator().manual_seed(16)
    def step(model, optimizer, order):
        x = torch.randn(5, 2)[torch.randperm(5, generator=order)]
        loss = (model(x)-float(np.random.random()+random.random())).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    step(m, opt, sampler)
    path = tmp_path/'progress.pt'
    c.save_progress(path, m, opt, sampler, {'epoch': 1})
    step(m, opt, sampler)
    expected = {k: v.clone() for k, v in m.state_dict().items()}
    n = torch.nn.Linear(2, 1); other = torch.optim.Adam(n.parameters(), lr=.01)
    other_sampler = torch.Generator()
    info = c.load_progress(path, n, other, other_sampler)
    assert info['epoch'] == 1
    step(n, other, other_sampler)
    for k, v in n.state_dict().items():
        torch.testing.assert_close(v, expected[k], rtol=0, atol=0)


def test_atomic_snapshot_roundtrip(tmp_path):
    c = module()
    path = tmp_path/'state.pt'
    c.atomic_save(path, {'value': torch.tensor([3])})
    assert torch.load(path, weights_only=False)['value'].item() == 3
    assert not list(tmp_path.glob('*.tmp'))


def test_wrong_run_is_rejected_before_restoring_model_or_rng(tmp_path):
    import pytest
    c = module()
    m = torch.nn.Linear(2, 1); opt = torch.optim.Adam(m.parameters())
    sampler = torch.Generator().manual_seed(42)
    path = tmp_path/'wrong.pt'
    c.save_progress(path, m, opt, sampler, {'epoch': 2, 'run_identity': 'fold1-R'})
    m.weight.data.add_(7)
    expected = m.weight.clone(); rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='identity'):
        c.load_progress(path, m, opt, sampler, expected_identity='fold2-MV')
    torch.testing.assert_close(m.weight, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
