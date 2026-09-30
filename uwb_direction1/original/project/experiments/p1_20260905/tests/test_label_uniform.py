import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from label_uniform import label_target, label_loss
try:
    import label_global
except ImportError:
    label_global = None


class LabelControlTests(unittest.TestCase):
    def test_global_control_matches_sample_weighted_smoothing_mass(self):
        self.assertIsNotNone(label_global, 'global label-smoothing control is not implemented')
        u=torch.tensor([[.9,.05,.05],[.2,.6,.2],[.1,.1,.8]])
        counts=torch.tensor([1.,2.,1.])
        g=label_global.global_diagonal_control(u,counts)
        expected_diag=(torch.diag(u)*counts).sum()/counts.sum()
        torch.testing.assert_close(torch.diag(g),expected_diag.repeat(3))
        torch.testing.assert_close(g.sum(1),torch.ones(3))
        expected_mass=((1-torch.diag(u))*counts).sum()/counts.sum()
        actual_mass=((1-torch.diag(g))*counts).sum()/counts.sum()
        torch.testing.assert_close(actual_mass,expected_mass)

    def test_distribution_and_exact_definition(self):
        y=torch.tensor([0,2]); u=torch.tensor([[.8,.1,.1],[.2,.6,.2],[.05,.05,.9]])
        q=label_target(y,u)
        torch.testing.assert_close(q,torch.tensor([[.9,.05,.05],[.025,.025,.95]]))
        torch.testing.assert_close(q.sum(1),torch.ones(2))

    def test_identity_one_hot(self):
        y=torch.tensor([0,1,2]); q=label_target(y,torch.eye(3))
        torch.testing.assert_close(q,torch.eye(3))

    def test_gradient_student_only(self):
        s=torch.randn(3,3,requires_grad=True); u=torch.eye(3,requires_grad=True)
        loss=label_loss(s,torch.tensor([0,1,2]),u)
        loss.backward()
        self.assertIsNone(u.grad); self.assertGreater(s.grad.abs().sum().item(),0)
        self.assertTrue(torch.isfinite(loss))


if __name__=='__main__':unittest.main()
