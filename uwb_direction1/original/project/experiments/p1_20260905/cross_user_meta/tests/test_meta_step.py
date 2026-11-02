"""Exact-gradient checks use double-precision nonlinear toy objectives."""
import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(4)
MODULE = Path(__file__).resolve().parents[1] / 'meta_step.py'


def load_step():
    assert MODULE.exists(), 'meta_step.py implementation is missing'
    spec = importlib.util.spec_from_file_location('_test_cross_user_meta_step', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Parameter(torch.tensor(.7, dtype=torch.double))
        self.aux = nn.Parameter(torch.tensor(.4, dtype=torch.double))
        self.unused = nn.Parameter(torch.tensor(.9, dtype=torch.double))

    def forward(self, x):
        return torch.tanh(x * self.encoder)

    def support_outputs(self, x, mode):
        right = x[:, -1:]
        return self(right), self.aux * torch.tanh(x.mean(1, keepdim=True) * self.encoder) if mode == 'MV' else None


class MetaStepTests(unittest.TestCase):
    def setUp(self):
        self.step = load_step()
        self.model = Toy()
        self.sx = torch.tensor([[.2, .5, .9], [-.6, .8, .4]], dtype=torch.double)
        self.sy = torch.tensor([[.2], [-.3]], dtype=torch.double)
        self.qx = torch.tensor([[.3], [-.2]], dtype=torch.double)
        self.qy = torch.tensor([[.1], [.6]], dtype=torch.double)

    def loss(self, arm, alpha=.03):
        sx = self.sx if arm in ('A2', 'A3') else self.sx[:, 2:]
        return self.step.episode_loss(self.model, sx, self.sy, self.qx, self.qy, arm, F.mse_loss, alpha=alpha)

    def test_exact_outer_gradient_matches_finite_difference(self):
        loss, _ = self.loss('A3')
        exact = torch.autograd.grad(loss, self.model.encoder)[0].item()
        center = self.model.encoder.item()
        eps = 1e-5
        with torch.no_grad():
            self.model.encoder.fill_(center + eps)
        plus = self.loss('A3')[0].item()
        with torch.no_grad():
            self.model.encoder.fill_(center - eps)
        minus = self.loss('A3')[0].item()
        self.assertAlmostEqual(exact, (plus - minus)/(2*eps), places=8)

    def test_alpha_zero_matches_ordinary_loss_and_gradients(self):
        for ordinary, meta in [('A0', 'A1'), ('A2', 'A3')]:
            a, _ = self.loss(ordinary, alpha=0)
            b, _ = self.loss(meta, alpha=0)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            ga = torch.autograd.grad(a, self.model.encoder)[0]
            gb = torch.autograd.grad(b, self.model.encoder)[0]
            torch.testing.assert_close(ga, gb, rtol=1e-12, atol=1e-12)

    def test_no_original_parameter_or_grad_mutation(self):
        old = {k: v.clone() for k, v in self.model.state_dict().items()}
        _, diag = self.loss('A3')
        for k, v in self.model.state_dict().items():
            torch.testing.assert_close(old[k], v, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertTrue(all(not isinstance(x, torch.Tensor) or not x.requires_grad for x in diag.values()))

    def test_query_gradient_reaches_auxiliary_via_update(self):
        loss, _ = self.loss('A3')
        r, mv = self.model.support_outputs(self.sx, 'MV')
        support = F.mse_loss(r, self.sy) + .5 * F.mse_loss(mv, self.sy)
        query_only = loss - support
        grad = torch.autograd.grad(query_only, self.model.aux)[0]
        self.assertGreater(abs(grad.item()), 1e-8)

    def test_real_hidden_views_affect_a3_but_not_a1(self):
        before_a1, _ = self.loss('A1')
        before_a3, _ = self.loss('A3')
        self.sx[:, :2] += 2
        after_a1, _ = self.loss('A1')
        after_a3, _ = self.loss('A3')
        torch.testing.assert_close(before_a1, after_a1, rtol=0, atol=0)
        self.assertGreater(abs((before_a3-after_a3).item()), 1e-5)

    def test_unused_parameter_stays_identical_in_virtual_update(self):
        loss = self.model(torch.ones(1, 1, dtype=torch.double)).sum()
        params = dict(self.model.named_parameters())
        updated = self.step.differentiable_update(params, loss, .1)
        self.assertIs(updated['unused'], params['unused'])
        self.assertIs(updated['aux'], params['aux'])


if __name__ == '__main__':
    unittest.main()
