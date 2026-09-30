"""Small CPU architecture and deployment-boundary checks."""
import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn

torch.set_num_threads(4)
MODULE = Path(__file__).resolve().parents[1] / 'model.py'


class MetaModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert MODULE.exists(), 'model.py implementation is missing'
        spec = importlib.util.spec_from_file_location('_test_cross_user_meta_model', MODULE)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)
        cls.model = cls.mod.build_model(seed=2026, device='cpu').eval()

    def test_exact_original_right_state_and_logits(self):
        torch.manual_seed(2026)
        original = self.mod.Classifier([2], 'concat', 'mean').eval()
        for name, value in original.state_dict().items():
            torch.testing.assert_close(self.model.right.state_dict()[name], value, rtol=0, atol=0)
        x = torch.randn(1, 1, 90, 378)
        with torch.no_grad():
            expected = original(x)[0]
            actual = self.model(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(self.model.right.encoder.feat_dim, 14976)
        self.assertEqual(self.model.right.head.net[0].out_features, 2048)
        self.assertTrue(all(p.requires_grad for p in self.model.right.parameters()))

    def test_strict_right_shape(self):
        for shape in [(1, 3, 90, 378), (1, 90, 378), (1, 1, 90, 189)]:
            with self.assertRaises(ValueError):
                self.model(torch.empty(shape))

    def test_real_architecture_exact_query_gradient_reaches_auxiliary(self):
        from torch.nn import functional as F
        from torch.func import functional_call
        step_path = MODULE.with_name('meta_step.py')
        spec = importlib.util.spec_from_file_location('_real_model_meta_step', step_path)
        step = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step)
        torch.manual_seed(10)
        # train() prevents eval-only fused Transformer kernels, which do not
        # implement higher-order backward on all supported PyTorch versions.
        self.model.train()
        sx = torch.randn(1, 3, 90, 378)
        qx = torch.randn(1, 1, 90, 378)
        label = torch.tensor([3])
        right, mv = self.model.support_outputs(sx, 'MV')
        support = F.cross_entropy(right, label) + .5*F.cross_entropy(mv, label)
        fast = step.differentiable_update(dict(self.model.named_parameters()), support, 1e-4)
        query = F.cross_entropy(functional_call(self.model, fast, (qx,)), label)
        aux_grads = torch.autograd.grad(query, tuple(self.model.fuser.parameters()), allow_unused=True)
        self.assertGreater(sum(g.abs().sum().item() for g in aux_grads if g is not None), 0)
        self.assertTrue(all(g is None or torch.isfinite(g).all() for g in aux_grads))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.model.eval()

    def test_support_is_single_encoder_call_and_hidden_views_reach_only_aux(self):
        calls = []
        handle = self.model.right.encoder.register_forward_hook(lambda m, i, o: calls.append(i[0].shape))
        x = torch.randn(1, 3, 90, 378)
        with torch.no_grad():
            r, aux = self.model.support_outputs(x, 'MV')
            changed = x.clone()
            changed[:, :2] += 5
            r2, aux2 = self.model.support_outputs(changed, 'MV')
            rr, no_aux = self.model.support_outputs(x[:, 2:], 'R')
        handle.remove()
        self.assertEqual(calls, [torch.Size([1, 3, 90, 378])]*3)
        torch.testing.assert_close(r, r2, rtol=0, atol=0)
        torch.testing.assert_close(r, rr, rtol=0, atol=0)
        self.assertIsNone(no_aux)
        self.assertEqual(aux.shape, (1, 12))
        self.assertGreater((aux-aux2).abs().max().item(), 1e-5)
        self.assertFalse(any(isinstance(m, nn.Dropout) for m in self.model.fuser.modules()))


if __name__ == '__main__':
    unittest.main()
